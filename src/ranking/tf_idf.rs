use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, RankingAlgorithm, Score, Scored};
use crate::text_transform::Query;

pub struct TFIDF;

impl RankingAlgorithm for TFIDF {
    fn score(&self, inverted_index: &InvertedIndex, query: &Query) -> Scored {
        let num_docs = inverted_index.num_docs();

        let scores = DashMap::new();

        query.0.par_iter().for_each(|term| {
            if let Some(documents) = inverted_index.0.get(term) {
                documents
                    .iter()
                    .par_bridge()
                    .for_each(|(doc_path, term_doc)| {
                        let tf = term_doc.term_freq as f64 / term_doc.length as f64;
                        let idf = idf(num_docs, documents.len());

                        *scores.entry(doc_path.to_owned()).or_insert(0.0) += tf * idf;
                    });
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
