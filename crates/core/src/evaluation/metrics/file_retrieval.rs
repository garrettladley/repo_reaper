use std::{collections::BTreeMap, path::PathBuf};

use super::{Evaluation, EvaluationSlice};
use crate::{evaluation::dataset::QueryShape, ranking::Score};

pub fn evaluate_query_at_k(
    ranked_docs: &[Score],
    relevant_docs: &[PathBuf],
    k: usize,
) -> Evaluation {
    if k == 0 || relevant_docs.is_empty() {
        return Evaluation::zero(k);
    }

    let relevant_set: std::collections::HashSet<&PathBuf> = relevant_docs.iter().collect();
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
    relevant_docs: &std::collections::HashSet<&PathBuf>,
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

fn reciprocal_rank_at_k(
    ranked_docs: &[Score],
    relevant_docs: &std::collections::HashSet<&PathBuf>,
) -> f64 {
    ranked_docs
        .iter()
        .enumerate()
        .find(|(_, doc)| relevant_docs.contains(&doc.doc_path))
        .map(|(rank, _)| 1.0 / (rank + 1) as f64)
        .unwrap_or(0.0)
}

fn ndcg_at_k(
    ranked_docs: &[Score],
    relevant_docs: &std::collections::HashSet<&PathBuf>,
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

pub(crate) fn evaluation_slices(evaluations: &[(QueryShape, Evaluation)]) -> Vec<EvaluationSlice> {
    let mut by_shape: BTreeMap<QueryShape, Vec<Evaluation>> = BTreeMap::new();

    evaluations.iter().for_each(|(shape, evaluation)| {
        by_shape.entry(*shape).or_default().push(evaluation.clone());
    });

    by_shape
        .into_iter()
        .map(|(query_shape, evaluations)| EvaluationSlice {
            query_shape,
            metric_family: query_shape.metric_family(),
            query_count: evaluations.len(),
            metrics: average_evaluations(&evaluations),
        })
        .collect()
}
