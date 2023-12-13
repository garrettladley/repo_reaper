use std::collections::HashMap;

use std::sync::{Arc, Mutex};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, Rank, Ranking, RankingAlgorithm};
use crate::text_transform::Query;

pub struct TFIDF;

impl RankingAlgorithm for TFIDF {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Option<Ranking> {
        if query.0.is_empty() {
            return None;
        }

        let scores = Arc::new(Mutex::new(HashMap::new()));

        query.0.par_iter().for_each(|term| {
            if let Some(documents) = inverted_index.0.get(term) {
                documents.par_iter().for_each(|(doc_path, term_doc)| {
                    let tf = term_doc.term_freq as f64 / term_doc.length as f64;
                    let idf = idf(inverted_index.num_docs(), documents.len());

                    *scores
                        .lock()
                        .unwrap()
                        .entry(doc_path.clone())
                        .or_insert(0.0) += tf * idf;
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

        match scores.is_empty() {
            true => None,
            false => Some(Ranking(scores)),
        }
    }
}
