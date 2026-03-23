use std::path::PathBuf;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    index::InvertedIndex,
    query::Query,
    ranking::{RankingAlgo, Score},
};

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

impl Default for Evaluation {
    fn default() -> Self {
        Evaluation {
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            mrr: 0.0,
        }
    }
}

pub fn evaluate_query(ranked_docs: &[Score], relevant_docs: &[PathBuf]) -> Evaluation {
    let true_positives = ranked_docs
        .iter()
        .filter(|doc| relevant_docs.contains(&doc.doc_path))
        .count();

    let false_positives = ranked_docs.len() - true_positives;
    let false_negatives = relevant_docs.len() - true_positives;

    let precision = true_positives as f64 / (true_positives + false_positives) as f64;
    let recall = true_positives as f64 / (true_positives + false_negatives) as f64;

    let f1_score = if precision + recall > 0.0 {
        2.0 * (precision * recall) / (precision + recall)
    } else {
        0.0
    };

    let mrr = ranked_docs
        .iter()
        .enumerate()
        .find(|(_, doc)| relevant_docs.contains(&doc.doc_path))
        .map(|(i, _)| 1.0 / (i + 1) as f64)
        .unwrap_or(0.0);

    Evaluation {
        precision,
        recall,
        f1_score,
        mrr,
    }
}

pub fn average_evaluations(evaluations: &[Evaluation]) -> Evaluation {
    if evaluations.is_empty() {
        return Evaluation::default();
    }

    let n = evaluations.len() as f64;
    let sum = evaluations
        .iter()
        .fold(Evaluation::default(), |acc, eval| Evaluation {
            precision: acc.precision + eval.precision,
            recall: acc.recall + eval.recall,
            f1_score: acc.f1_score + eval.f1_score,
            mrr: acc.mrr + eval.mrr,
        });

    Evaluation {
        precision: sum.precision / n,
        recall: sum.recall / n,
        f1_score: sum.f1_score / n,
        mrr: sum.mrr / n,
    }
}

impl TestSet {
    pub fn evaluate(&self, inverted_index: &InvertedIndex, top_n: usize) -> Evaluation {
        let evaluations: Vec<Evaluation> = self
            .queries
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

                evaluate_query(&ranked_docs, &query.relevant_docs)
            })
            .collect();

        average_evaluations(&evaluations)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Evaluation, average_evaluations, evaluate_query};
    use crate::ranking::Score;

    fn scored(paths: &[&str]) -> Vec<Score> {
        paths
            .iter()
            .enumerate()
            .map(|(i, p)| Score {
                doc_path: PathBuf::from(p),
                score: (paths.len() - i) as f64,
            })
            .collect()
    }

    fn relevant(paths: &[&str]) -> Vec<PathBuf> {
        paths.iter().map(PathBuf::from).collect()
    }

    #[test]
    fn precision_is_fraction_of_results_that_are_relevant() {
        // 2 of 4 results are relevant → precision = 0.5
        let ranked = scored(&["a.rs", "b.rs", "c.rs", "d.rs"]);
        let eval = evaluate_query(&ranked, &relevant(&["a.rs", "c.rs"]));

        assert!((eval.precision - 0.5).abs() < 1e-10);
    }

    #[test]
    fn recall_is_fraction_of_relevant_docs_retrieved() {
        // 1 of 3 relevant docs retrieved → recall = 1/3
        let ranked = scored(&["a.rs", "x.rs"]);
        let eval = evaluate_query(&ranked, &relevant(&["a.rs", "b.rs", "c.rs"]));

        assert!((eval.recall - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn f1_is_harmonic_mean_of_precision_and_recall() {
        let ranked = scored(&["a.rs", "x.rs"]);
        let eval = evaluate_query(&ranked, &relevant(&["a.rs", "b.rs"]));

        // precision = 1/2, recall = 1/2, f1 = 2*(0.5*0.5)/(0.5+0.5) = 0.5
        assert!((eval.f1_score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn mrr_uses_rank_of_first_relevant_doc() {
        // First relevant doc is at rank 3 (0-indexed: 2) → MRR = 1/3
        let ranked = scored(&["x.rs", "y.rs", "a.rs", "b.rs"]);
        let eval = evaluate_query(&ranked, &relevant(&["a.rs", "b.rs"]));

        assert!((eval.mrr - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn mrr_is_one_when_first_result_is_relevant() {
        let ranked = scored(&["a.rs", "x.rs"]);
        let eval = evaluate_query(&ranked, &relevant(&["a.rs"]));

        assert!((eval.mrr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn mrr_is_zero_when_no_relevant_docs_in_results() {
        let ranked = scored(&["x.rs", "y.rs"]);
        let eval = evaluate_query(&ranked, &relevant(&["a.rs"]));

        assert!((eval.mrr).abs() < 1e-10);
    }

    #[test]
    fn average_evaluations_divides_by_query_count() {
        let evals = vec![
            Evaluation {
                precision: 1.0,
                recall: 0.5,
                f1_score: 0.6,
                mrr: 1.0,
            },
            Evaluation {
                precision: 0.5,
                recall: 1.0,
                f1_score: 0.4,
                mrr: 0.5,
            },
        ];
        let avg = average_evaluations(&evals);

        assert!((avg.precision - 0.75).abs() < 1e-10);
        assert!((avg.recall - 0.75).abs() < 1e-10);
        assert!((avg.f1_score - 0.5).abs() < 1e-10);
        assert!((avg.mrr - 0.75).abs() < 1e-10);
    }

    #[test]
    fn average_of_empty_returns_zeros() {
        let avg = average_evaluations(&[]);

        assert!((avg.precision).abs() < 1e-10);
        assert!((avg.recall).abs() < 1e-10);
    }

    #[test]
    fn all_metrics_bounded_zero_to_one() {
        let ranked = scored(&["a.rs", "b.rs", "c.rs"]);
        let eval = evaluate_query(&ranked, &relevant(&["a.rs", "d.rs"]));

        for val in [eval.precision, eval.recall, eval.f1_score, eval.mrr] {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&val),
                "metric must be in [0, 1], got {val}"
            );
        }
    }
}
