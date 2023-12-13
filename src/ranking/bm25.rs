use std::collections::HashMap;

use std::sync::{Arc, Mutex};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, Rank, RankingAlgorithm};
use crate::text_transform::Query;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct BM25HyperParams {
    pub k1: f64,
    pub b: f64,
}

pub fn get_configuration() -> Result<BM25HyperParams, config::ConfigError> {
    let base_path = std::env::current_dir().expect("Failed to determine the current directory");
    let configuration_directory = base_path.join("configuration");

    let settings = config::Config::builder()
        .add_source(config::File::from(configuration_directory.join("bm25.yml")))
        .build()?;

    settings.try_deserialize::<BM25HyperParams>()
}

pub struct BM25 {
    pub hyper_params: BM25HyperParams,
}

impl RankingAlgorithm for BM25 {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Vec<Rank> {
        let avgdl = inverted_index.avg_doc_length();
        let scores = Arc::new(Mutex::new(HashMap::new()));

        query.0.par_iter().for_each(|term| {
            if let Some(documents) = inverted_index.0.get(term) {
                let idf = idf(inverted_index.num_docs(), documents.len());

                documents.par_iter().for_each(|(doc_path, term_doc)| {
                    let tf = term_doc.term_freq as f64;
                    let doc_length = term_doc.length as f64;
                    let score = idf * (tf * (self.hyper_params.k1 + 1.0))
                        / (tf
                            + self.hyper_params.k1
                                * (1.0 - self.hyper_params.b
                                    + self.hyper_params.b * doc_length / avgdl));

                    let mut scores_lock = scores.lock().unwrap();
                    *scores_lock.entry(doc_path.clone()).or_insert(0.0) += score;
                });
            }
        });

        let scores = Arc::try_unwrap(scores).unwrap().into_inner().unwrap();

        let mut scores: Vec<Rank> = scores
            .par_iter()
            .map(|(doc_path, score)| Rank {
                doc_path: doc_path.clone(),
                score: *score,
            })
            .collect();

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        scores.truncate(top_n);

        scores
    }
}
