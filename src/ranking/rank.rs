use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;

use chrono::Utc;
use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

use crate::inverted_index::TermDocument;
use crate::{inverted_index::InvertedIndex, text_transform::Query};

use crate::ranking::{get_configuration, BM25HyperParams, CosineSimilarity, BM25, TFIDF};

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
        write!(
            f,
            "Path: {} Score: {}",
            self.doc_path.to_str().unwrap(),
            self.score
        )
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
        let ranking = if query.0.is_empty() {
            None
        } else {
            let mut ranking = self.score(inverted_index, query).0;

            ranking.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            ranking.truncate(top_n);
            match ranking.is_empty() {
                true => None,
                false => Some(Scored(ranking)),
            }
        };

        let mut query_log = HashMap::new();

        query_log.insert("query".to_string(), query.to_string());
        query_log.insert("top_n".to_string(), top_n.to_string());

        match &ranking {
            Some(ranking) => {
                query_log.insert("ranking".to_string(), format!("{:?}", ranking));
            }
            None => {
                query_log.insert("ranking".to_string(), "".to_string());
            }
        }

        query_log.insert(
            "ranking_algo".to_string(),
            format!("{:?}", self).to_string(),
        );

        query_log.insert(
            "timestamp".to_string(),
            Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        );

        let query_log = serde_json::to_string(&query_log).unwrap();

        let mut file = OpenOptions::new()
            .append(true)
            .create(true)
            .open("./query_log.txt")
            .unwrap();

        file.write_all(query_log.as_bytes()).unwrap();

        ranking
    }
}

impl FromStr for RankingAlgo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosim" => Ok(RankingAlgo::CosineSimilarity),
            "bm25" => {
                let hyper_params = get_configuration().unwrap();

                Ok(RankingAlgo::BM25(hyper_params))
            }
            "tfidf" => Ok(RankingAlgo::TFIDF),
            _ => Err(format!("{} is not a valid ranking algorithm", s)),
        }
    }
}
