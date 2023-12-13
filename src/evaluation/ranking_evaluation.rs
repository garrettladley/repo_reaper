use std::{collections::HashSet, default, path::PathBuf};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{inverted_index::Term, ranking::Ranking, text_transform::Query};

pub struct TestSet {
    pub ranking_algorithm: Box<dyn Fn(&Query) -> Ranking + Send + Sync>,
    pub queries: Vec<TestQuery>,
}

pub struct TestQuery {
    pub query: Query,
    pub relevant_docs: Vec<PathBuf>,
}

pub struct Evaluation {
    pub precision: f64,
    pub recall: f64,
    pub accuracy: f64,
    pub f1_score: f64,
}

impl default::Default for Evaluation {
    fn default() -> Self {
        Evaluation {
            precision: 0.0,
            recall: 0.0,
            accuracy: 0.0,
            f1_score: 0.0,
        }
    }
}

impl TestSet {
    pub fn evaluate<F>(&self) -> Evaluation
    where
        F: Fn(&str) -> HashSet<Term> + Sync,
    {
        self.queries
            .par_iter()
            .map(|query| {
                let ranked_docs = (self.ranking_algorithm)(&query.query).0;

                let true_positives = ranked_docs
                    .par_iter()
                    .filter(|doc| query.relevant_docs.contains(&doc.doc_path))
                    .count();

                let false_positives = ranked_docs
                    .par_iter()
                    .filter(|doc| !query.relevant_docs.contains(&doc.doc_path))
                    .count();

                let false_negatives = query
                    .relevant_docs
                    .par_iter()
                    .filter(|doc| {
                        !ranked_docs
                            .par_iter()
                            .any(|ranked| ranked.doc_path == **doc)
                    })
                    .count();

                let true_negatives = query.relevant_docs.len() - true_positives;

                let precision = true_positives as f64 / (true_positives + false_positives) as f64;
                let recall = true_positives as f64 / (true_positives + false_negatives) as f64;
                let accuracy = (true_positives + true_negatives) as f64
                    / (true_positives + true_negatives + false_positives + false_negatives) as f64;

                let f1_score = if precision + recall > 0.0 {
                    2.0 * (precision * recall) / (precision + recall)
                } else {
                    0.0
                };

                Evaluation {
                    precision,
                    recall,
                    accuracy,
                    f1_score,
                }
            })
            .reduce(Evaluation::default, |acc, evaluation| Evaluation {
                precision: acc.precision + evaluation.precision,
                recall: acc.recall + evaluation.recall,
                accuracy: acc.accuracy + evaluation.accuracy,
                f1_score: acc.f1_score + evaluation.f1_score,
            })
    }
}
