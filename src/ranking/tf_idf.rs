use std::collections::HashMap;
use std::path::PathBuf;

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, Rank, RankingAlgorithm};
use crate::text_transform::Query;

pub struct TFIDF;

impl RankingAlgorithm for TFIDF {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Vec<Rank> {
        let mut scores: HashMap<PathBuf, f64> = HashMap::new();

        for term in &query.0 {
            if let Some(documents) = inverted_index.0.get(term) {
                for (doc_path, term_doc) in documents {
                    let tf = term_doc.term_freq as f64 / term_doc.length as f64;
                    let idf = idf(inverted_index.num_docs(), documents.len());
                    *scores.entry(doc_path.clone()).or_insert(0.0) += tf * idf;
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
