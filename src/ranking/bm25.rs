use std::collections::HashMap;
use std::path::PathBuf;

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, Rank, RankingAlgorithm};
use crate::text_transform::Query;

#[derive(serde::Deserialize)]
pub struct BM25HyperParams {
    pub k1: f64,
    pub b: f64,
}

pub struct BM25 {
    pub hyper_params: BM25HyperParams,
}

impl RankingAlgorithm for BM25 {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Vec<Rank> {
        let mut scores: HashMap<PathBuf, f64> = HashMap::new();
        let avgdl = inverted_index.avg_doc_length();

        for term in &query.0 {
            if let Some(documents) = inverted_index.0.get(term) {
                let idf = idf(inverted_index.num_docs(), documents.len());

                for (doc_path, term_doc) in documents {
                    let tf = term_doc.term_freq as f64;
                    let doc_length = term_doc.length as f64;
                    let score = idf * (tf * (self.hyper_params.k1 + 1.0))
                        / (tf
                            + self.hyper_params.k1
                                * (1.0 - self.hyper_params.b
                                    + self.hyper_params.b * doc_length / avgdl));

                    *scores.entry(doc_path.clone()).or_insert(0.0) += score;
                }
            }
        }

        let mut scores: Vec<Rank> = scores
            .into_iter()
            .map(|(doc_path, score)| Rank { doc_path, score })
            .collect();

        scores.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        scores.truncate(top_n);

        scores
    }
}
