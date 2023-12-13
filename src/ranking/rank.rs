use std::path::PathBuf;
use std::str::FromStr;

use crate::{inverted_index::InvertedIndex, text_transform::Query};

use crate::ranking::{BM25HyperParams, CosineSimilarity, BM25, TFIDF};

use super::bm25::get_configuration;

#[derive(Debug)]
pub struct Ranking(pub Vec<Rank>);

#[derive(Debug)]
pub struct Rank {
    pub doc_path: PathBuf,
    pub score: f64,
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

        algo.rank(inverted_index, query, top_n)
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
