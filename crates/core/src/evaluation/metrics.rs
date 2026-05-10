use std::{collections::HashSet, fmt, path::PathBuf};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    index::InvertedIndex,
    query::AnalyzedQuery,
    ranking::{RankingAlgo, Score},
};

pub struct TestSet {
    pub ranking_algorithm: RankingAlgo,
    pub queries: Vec<TestQuery>,
}

pub struct TestQuery {
    pub query: AnalyzedQuery,
    pub relevant_docs: Vec<PathBuf>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Evaluation {
    pub k: usize,
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub mean_average_precision: f64,
    pub mean_reciprocal_rank: f64,
    pub normalized_discounted_cumulative_gain: f64,
}

impl Evaluation {
    pub fn zero(k: usize) -> Self {
        Evaluation {
            k,
            precision_at_k: 0.0,
            recall_at_k: 0.0,
            mean_average_precision: 0.0,
            mean_reciprocal_rank: 0.0,
            normalized_discounted_cumulative_gain: 0.0,
        }
    }
}

impl Default for Evaluation {
    fn default() -> Self {
        Self::zero(0)
    }
}

pub fn evaluate_query_at_k(
    ranked_docs: &[Score],
    relevant_docs: &[PathBuf],
    k: usize,
) -> Evaluation {
    if k == 0 || relevant_docs.is_empty() {
        return Evaluation::zero(k);
    }

    let relevant_set: HashSet<&PathBuf> = relevant_docs.iter().collect();
    let cutoff_docs = &ranked_docs[..ranked_docs.len().min(k)];
    let relevant_retrieved = cutoff_docs
        .iter()
        .filter(|doc| relevant_set.contains(&doc.doc_path))
        .count();

    Evaluation {
        k,
        precision_at_k: relevant_retrieved as f64 / k as f64,
        recall_at_k: relevant_retrieved as f64 / relevant_docs.len() as f64,
        mean_average_precision: average_precision_at_k(
            cutoff_docs,
            &relevant_set,
            relevant_docs.len(),
        ),
        mean_reciprocal_rank: reciprocal_rank_at_k(cutoff_docs, &relevant_set),
        normalized_discounted_cumulative_gain: ndcg_at_k(
            cutoff_docs,
            &relevant_set,
            relevant_docs.len(),
            k,
        ),
    }
}

fn average_precision_at_k(
    ranked_docs: &[Score],
    relevant_docs: &HashSet<&PathBuf>,
    total_relevant: usize,
) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }

    let mut relevant_seen = 0;
    let precision_sum = ranked_docs
        .iter()
        .enumerate()
        .filter_map(|(rank, doc)| {
            if relevant_docs.contains(&doc.doc_path) {
                relevant_seen += 1;
                Some(relevant_seen as f64 / (rank + 1) as f64)
            } else {
                None
            }
        })
        .sum::<f64>();

    precision_sum / total_relevant as f64
}

fn reciprocal_rank_at_k(ranked_docs: &[Score], relevant_docs: &HashSet<&PathBuf>) -> f64 {
    ranked_docs
        .iter()
        .enumerate()
        .find(|(_, doc)| relevant_docs.contains(&doc.doc_path))
        .map(|(rank, _)| 1.0 / (rank + 1) as f64)
        .unwrap_or(0.0)
}

fn ndcg_at_k(
    ranked_docs: &[Score],
    relevant_docs: &HashSet<&PathBuf>,
    total_relevant: usize,
    k: usize,
) -> f64 {
    let dcg = ranked_docs
        .iter()
        .enumerate()
        .filter(|(_, doc)| relevant_docs.contains(&doc.doc_path))
        .map(|(rank, _)| discount(rank))
        .sum::<f64>();

    let ideal_relevant = total_relevant.min(k);
    if ideal_relevant == 0 {
        return 0.0;
    }

    let idcg = (0..ideal_relevant).map(discount).sum::<f64>();
    dcg / idcg
}

fn discount(zero_based_rank: usize) -> f64 {
    1.0 / ((zero_based_rank + 2) as f64).log2()
}

pub fn average_evaluations(evaluations: &[Evaluation]) -> Evaluation {
    if evaluations.is_empty() {
        return Evaluation::default();
    }

    let n = evaluations.len() as f64;
    let sum = evaluations
        .iter()
        .fold(Evaluation::default(), |acc, eval| Evaluation {
            k: acc.k.max(eval.k),
            precision_at_k: acc.precision_at_k + eval.precision_at_k,
            recall_at_k: acc.recall_at_k + eval.recall_at_k,
            mean_average_precision: acc.mean_average_precision + eval.mean_average_precision,
            mean_reciprocal_rank: acc.mean_reciprocal_rank + eval.mean_reciprocal_rank,
            normalized_discounted_cumulative_gain: acc.normalized_discounted_cumulative_gain
                + eval.normalized_discounted_cumulative_gain,
        });

    Evaluation {
        k: sum.k,
        precision_at_k: sum.precision_at_k / n,
        recall_at_k: sum.recall_at_k / n,
        mean_average_precision: sum.mean_average_precision / n,
        mean_reciprocal_rank: sum.mean_reciprocal_rank / n,
        normalized_discounted_cumulative_gain: sum.normalized_discounted_cumulative_gain / n,
    }
}

impl fmt::Display for Evaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "P@{k}: {p_at_k:.4}\nR@{k}: {r_at_k:.4}\nMAP@{k}: {map:.4}\nMRR@{k}: {mrr:.4}\nNDCG@{k}: {ndcg:.4}",
            k = self.k,
            p_at_k = self.precision_at_k,
            r_at_k = self.recall_at_k,
            map = self.mean_average_precision,
            mrr = self.mean_reciprocal_rank,
            ndcg = self.normalized_discounted_cumulative_gain,
        )
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
                        None => return Evaluation::zero(top_n),
                    };

                evaluate_query_at_k(&ranked_docs, &query.relevant_docs, top_n)
            })
            .collect();

        average_evaluations(&evaluations)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Evaluation, average_evaluations, evaluate_query_at_k};
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
    fn precision_at_k_uses_cutoff_denominator() {
        let ranked = scored(&["a.rs", "x.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs"]), 4);

        assert!((eval.precision_at_k - 0.25).abs() < 1e-10);
    }

    #[test]
    fn recall_at_k_uses_relevant_document_count() {
        let ranked = scored(&["a.rs", "x.rs", "b.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs", "c.rs"]), 2);

        assert!((eval.recall_at_k - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn mean_average_precision_averages_precision_at_relevant_ranks() {
        let ranked = scored(&["a.rs", "x.rs", "b.rs", "c.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs", "missing.rs"]), 4);

        // AP = (P@1 + P@3) / 3 relevant docs = (1 + 2/3) / 3
        let expected = (1.0 + 2.0 / 3.0) / 3.0;
        assert!((eval.mean_average_precision - expected).abs() < 1e-10);
    }

    #[test]
    fn mean_reciprocal_rank_ignores_relevant_docs_after_cutoff() {
        let ranked = scored(&["x.rs", "y.rs", "a.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs"]), 2);

        assert!((eval.mean_reciprocal_rank).abs() < 1e-10);
    }

    #[test]
    fn normalized_discounted_cumulative_gain_is_one_for_ideal_binary_ranking() {
        let ranked = scored(&["a.rs", "b.rs", "x.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs"]), 3);

        assert!((eval.normalized_discounted_cumulative_gain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalized_discounted_cumulative_gain_discounts_late_relevant_results() {
        let ranked = scored(&["x.rs", "a.rs", "b.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs"]), 3);

        let dcg = 1.0 / 3.0_f64.log2() + 1.0 / 4.0_f64.log2();
        let idcg = 1.0 + 1.0 / 3.0_f64.log2();
        assert!((eval.normalized_discounted_cumulative_gain - dcg / idcg).abs() < 1e-10);
    }

    #[test]
    fn average_evaluations_divides_each_metric_by_query_count() {
        let evals = vec![
            Evaluation {
                k: 2,
                precision_at_k: 0.5,
                recall_at_k: 0.25,
                mean_average_precision: 0.75,
                mean_reciprocal_rank: 1.0,
                normalized_discounted_cumulative_gain: 0.9,
            },
            Evaluation {
                k: 2,
                precision_at_k: 1.0,
                recall_at_k: 0.5,
                mean_average_precision: 0.25,
                mean_reciprocal_rank: 0.5,
                normalized_discounted_cumulative_gain: 0.7,
            },
        ];
        let avg = average_evaluations(&evals);

        assert_eq!(avg.k, 2);
        assert!((avg.precision_at_k - 0.75).abs() < 1e-10);
        assert!((avg.recall_at_k - 0.375).abs() < 1e-10);
        assert!((avg.mean_average_precision - 0.5).abs() < 1e-10);
        assert!((avg.mean_reciprocal_rank - 0.75).abs() < 1e-10);
        assert!((avg.normalized_discounted_cumulative_gain - 0.8).abs() < 1e-10);
    }

    #[test]
    fn average_of_empty_returns_zeros() {
        let avg = average_evaluations(&[]);

        assert_eq!(avg.k, 0);
        assert!((avg.precision_at_k).abs() < 1e-10);
        assert!((avg.recall_at_k).abs() < 1e-10);
        assert!((avg.mean_average_precision).abs() < 1e-10);
        assert!((avg.mean_reciprocal_rank).abs() < 1e-10);
        assert!((avg.normalized_discounted_cumulative_gain).abs() < 1e-10);
    }

    #[test]
    fn all_metrics_are_bounded_zero_to_one() {
        let ranked = scored(&["a.rs", "b.rs", "c.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "d.rs"]), 3);

        for val in [
            eval.precision_at_k,
            eval.recall_at_k,
            eval.mean_average_precision,
            eval.mean_reciprocal_rank,
            eval.normalized_discounted_cumulative_gain,
        ] {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&val),
                "metric must be in [0, 1], got {val}"
            );
        }
    }
}
