use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, Rank, Ranking, RankingAlgorithm};
use crate::text_transform::Query;

pub struct CosineSimilarity;

impl RankingAlgorithm for CosineSimilarity {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Option<Ranking> {
        let query_magnitude = query
            .0
            .par_iter()
            .map(|term| {
                let idf = idf(inverted_index.num_docs(), inverted_index.doc_freq(term));
                idf * idf
            })
            .sum::<f64>()
            .sqrt();

        let scores = Arc::new(Mutex::new(HashMap::new()));

        query.0.par_iter().for_each(|term| {
            if let Some(documents) = inverted_index.0.get(term) {
                let idf = idf(inverted_index.num_docs(), documents.len());

                documents.par_iter().for_each(|(doc_path, term_doc)| {
                    let tf = term_doc.term_freq as f64;
                    let score = tf * idf;

                    let doc_magnitude = inverted_index
                        .0
                        .par_iter()
                        .filter_map(|(_, doc_map)| {
                            doc_map
                                .par_iter()
                                .find_any(|(path, _)| *path == doc_path)
                                .map(|(_, term_doc)| {
                                    let tf = term_doc.term_freq as f64;
                                    tf * tf
                                })
                        })
                        .sum::<f64>()
                        .sqrt();
                    let cosine_similarity = score / (query_magnitude * doc_magnitude);

                    *scores
                        .lock()
                        .unwrap()
                        .entry(doc_path.clone())
                        .or_insert(0.0) += cosine_similarity;
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
