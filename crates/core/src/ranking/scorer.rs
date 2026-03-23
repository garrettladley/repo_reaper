use std::{collections::HashMap, path::PathBuf, str::FromStr};

use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

use crate::{
    index::{InvertedIndex, TermDocument},
    query::Query,
    ranking::{BM25, BM25HyperParams, CosineSimilarity, TFIDF, get_configuration},
};

#[derive(Debug)]
pub struct Scored(pub Vec<Score>);

impl std::fmt::Display for Scored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();

        self.0.iter().for_each(|score| {
            output.push_str(&format!("{}\n", score));
        });

        write!(f, "{}", output)
    }
}

#[derive(Debug)]
pub struct Score {
    pub doc_path: PathBuf,
    pub score: f64,
}

impl std::fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Path: {} Score: {}", self.doc_path.display(), self.score)
    }
}

#[derive(Debug, Clone)]
pub enum RankingAlgo {
    CosineSimilarity,
    BM25(BM25HyperParams),
    TFIDF,
}

pub trait RankingAlgorithm {
    fn score(&self, inverted_index: &InvertedIndex, query: &Query) -> Scored;
}

pub trait Scorer: Send + Sync {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        query: &Query,
        documents: &HashMap<PathBuf, TermDocument>,
        scores: &DashMap<PathBuf, f64>,
    );
}

impl RankingAlgorithm for RankingAlgo {
    fn score(&self, inverted_index: &InvertedIndex, query: &Query) -> Scored {
        let scorer: Box<dyn Scorer> = match self {
            RankingAlgo::CosineSimilarity => Box::new(CosineSimilarity),
            RankingAlgo::BM25(hyper_params) => Box::new(BM25 {
                hyper_params: hyper_params.to_owned(),
            }),
            RankingAlgo::TFIDF => Box::new(TFIDF),
        };

        let scores = DashMap::new();

        query.0.par_iter().for_each(|term| {
            if let Some(documents) = inverted_index.0.get(term) {
                scorer.score(inverted_index, query, documents, &scores)
            }
        });

        Scored(
            scores
                .iter()
                .par_bridge()
                .map(|score| Score {
                    doc_path: score.key().to_owned(),
                    score: *score.value(),
                })
                .collect(),
        )
    }
}

impl RankingAlgo {
    pub fn rank(
        &self,
        inverted_index: &InvertedIndex,
        query: &Query,
        top_n: usize,
    ) -> Option<Scored> {
        if query.0.is_empty() {
            return None;
        }

        let mut ranking = self.score(inverted_index, query).0;

        ranking.sort_by(|a, b| b.score.total_cmp(&a.score));
        ranking.truncate(top_n);

        if ranking.is_empty() {
            None
        } else {
            Some(Scored(ranking))
        }
    }
}

impl FromStr for RankingAlgo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosim" => Ok(RankingAlgo::CosineSimilarity),
            "bm25" => {
                let hyper_params =
                    get_configuration().map_err(|e| format!("failed to load BM25 config: {e}"))?;

                Ok(RankingAlgo::BM25(hyper_params))
            }
            "tfidf" => Ok(RankingAlgo::TFIDF),
            _ => Err(format!("{} is not a valid ranking algorithm", s)),
        }
    }
}
