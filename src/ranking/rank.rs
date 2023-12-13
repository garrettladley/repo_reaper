use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::str::FromStr;

use crate::{inverted_index::InvertedIndex, text_transform::Query};

use crate::ranking::{BM25HyperParams, CosineSimilarity, BM25, TFIDF};

use super::bm25::get_configuration;

#[derive(Debug)]
pub struct Ranking(pub Vec<Rank>);

impl std::fmt::Display for Ranking {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();

        self.0.iter().for_each(|rank| {
            output.push_str(&format!("{}\n", rank));
        });

        write!(f, "{}", output)
    }
}

#[derive(Debug)]
pub struct Rank {
    pub doc_path: PathBuf,
    pub score: f64,
}

impl std::fmt::Display for Rank {
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
pub enum RankingAlgos {
    CosineSimilarity,
    BM25(BM25HyperParams),
    TFIDF,
}

pub trait RankingAlgorithm {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Option<Ranking>;
}

impl RankingAlgorithm for RankingAlgos {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Option<Ranking> {
        let algo: Box<dyn RankingAlgorithm> = match self {
            RankingAlgos::CosineSimilarity => Box::new(CosineSimilarity),
            RankingAlgos::BM25(hyper_params) => Box::new(BM25 {
                hyper_params: BM25HyperParams {
                    k1: hyper_params.k1,
                    b: hyper_params.b,
                },
            }),
            RankingAlgos::TFIDF => Box::new(TFIDF),
        };

        let mut query_log = HashMap::new();

        query_log.insert("query".to_string(), query.to_string());
        query_log.insert("top_n".to_string(), top_n.to_string());

        let ranking = algo.rank(inverted_index, query, top_n);

        match &ranking {
            Some(ranking) => {
                query_log.insert("ranking".to_string(), format!("{:?}", ranking));
            }
            None => {
                query_log.insert("ranking".to_string(), "".to_string());
            }
        }

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

impl FromStr for RankingAlgos {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosine" => Ok(RankingAlgos::CosineSimilarity),
            "bm25" => {
                let hyper_params = get_configuration().unwrap();

                Ok(RankingAlgos::BM25(hyper_params))
            }
            "tfidf" => Ok(RankingAlgos::TFIDF),
            _ => Err(format!("{} is not a valid ranking algorithm", s)),
        }
    }
}
