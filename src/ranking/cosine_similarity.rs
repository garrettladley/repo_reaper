use std::collections::HashMap;
use std::path::PathBuf;

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, Rank, RankingAlgorithm};
use crate::text_transform::Query;

pub struct CosineSimilarity;

impl RankingAlgorithm for CosineSimilarity {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Vec<Rank> {
        let mut scores: HashMap<PathBuf, f64> = HashMap::new();

        let query_magnitude = query
            .0
            .iter()
            .map(|term| {
                let idf = idf(inverted_index.num_docs(), inverted_index.doc_freq(term));
                idf * idf
            })
            .sum::<f64>()
            .sqrt();

        for term in &query.0 {
            if let Some(documents) = inverted_index.0.get(term) {
                let idf = idf(inverted_index.num_docs(), documents.len());

                for (doc_path, term_doc) in documents {
                    let tf = term_doc.term_freq as f64;
                    let score = tf * idf;

                    let doc_magnitude = inverted_index.cosim_doc_magnitude(doc_path);
                    let cosine_similarity = score / (query_magnitude * doc_magnitude);

                    *scores.entry(doc_path.clone()).or_insert(0.0) += cosine_similarity;
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
