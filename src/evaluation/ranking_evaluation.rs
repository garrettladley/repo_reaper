use std::{default, path::PathBuf};

use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{inverted_index::InvertedIndex, ranking::rank::RankingAlgo, text_transform::Query};

pub struct TestSet {
    pub ranking_algorithm: RankingAlgo,
    pub queries: Vec<TestQuery>,
}

pub struct TestQuery {
    pub query: Query,
    pub relevant_docs: Vec<PathBuf>,
}

#[derive(Debug)]
pub struct Evaluation {
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub mrr: f64,
}

impl default::Default for Evaluation {
    fn default() -> Self {
        Evaluation {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            mrr: 0.0,
        }
    }
}

impl TestSet {
    pub fn evaluate(&self, inverted_index: &InvertedIndex, top_n: usize) -> Evaluation {
        self.queries
            .par_iter()
            .map(|query| {
                let ranked_docs =
                    match self
                        .ranking_algorithm
                        .rank(inverted_index, &query.query, top_n)
                    {
                        Some(ranking) => ranking.0,
                        None => return Evaluation::default(),
                    };

                let true_positives = ranked_docs
                    .par_iter()
                    .filter(|doc| query.relevant_docs.contains(&doc.doc_path))
                    .count();

                let false_positives = ranked_docs.len() - true_positives;

                let false_negatives = query.relevant_docs.len() - true_positives;

                let precision = true_positives as f64 / (true_positives + false_positives) as f64;

                let recall = true_positives as f64 / (true_positives + false_negatives) as f64;

                let f1_score = if precision + recall > 0.0 {
                    2.0 * (precision * recall) / (precision + recall)
                } else {
                    0.0
                };

                let mrr = ranked_docs
                    .par_iter()
                    .enumerate()
                    .filter(|(_, doc)| query.relevant_docs.contains(&doc.doc_path))
                    .map(|(i, _)| 1.0 / (i + 1) as f64)
                    .collect::<Vec<_>>()
                    .first()
                    .cloned()
                    .unwrap_or(0.0);

                Evaluation {
                    precision,
                    recall,
                    f1_score,
                    mrr,
                }
            })
            .reduce(Evaluation::default, |acc, evaluation| Evaluation {
                precision: acc.precision + evaluation.precision,
                recall: acc.recall + evaluation.recall,
                f1_score: acc.f1_score + evaluation.f1_score,
                mrr: acc.mrr + evaluation.mrr,
            })
    }
}
