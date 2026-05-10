use std::{collections::HashSet, fmt, path::PathBuf};

use super::{GroundednessResult, util::ratio};
use crate::{evaluation::dataset::EvidenceSpan, ranking::Score};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct GroundednessEvaluation {
    pub k: usize,
    pub expected_evidence: usize,
    pub cited_evidence: usize,
    pub matching_evidence: usize,
    pub grounded_citations: usize,
    pub total_citations: usize,
    pub highlight_recall: f64,
    pub highlight_precision: f64,
    pub citation_precision: f64,
}

impl GroundednessEvaluation {
    pub fn zero(k: usize) -> Self {
        GroundednessEvaluation {
            k,
            ..GroundednessEvaluation::default()
        }
    }

    fn from_counts(
        k: usize,
        expected_evidence: usize,
        cited_evidence: usize,
        matching_evidence: usize,
        grounded_citations: usize,
        total_citations: usize,
    ) -> Self {
        GroundednessEvaluation {
            k,
            expected_evidence,
            cited_evidence,
            matching_evidence,
            grounded_citations,
            total_citations,
            highlight_recall: ratio(matching_evidence, expected_evidence),
            highlight_precision: ratio(matching_evidence, cited_evidence),
            citation_precision: ratio(grounded_citations, total_citations),
        }
    }
}

pub fn evaluate_groundedness_at_k(
    ranked_docs: &[Score],
    results: &[GroundednessResult],
    k: usize,
) -> GroundednessEvaluation {
    if k == 0 {
        return GroundednessEvaluation::zero(k);
    }

    let expected_evidence = expected_evidence(results);
    if expected_evidence.is_empty() {
        return GroundednessEvaluation::zero(k);
    }

    let cutoff_docs = &ranked_docs[..ranked_docs.len().min(k)];
    let cited_evidence = cited_evidence(cutoff_docs, results);
    let matching_evidence = cited_evidence.intersection(&expected_evidence).count();

    let grounded_citations = cutoff_docs
        .iter()
        .filter(|doc| {
            results.iter().any(|result| {
                result.path == doc.doc_path
                    && result.evidence.iter().any(|span| {
                        expected_evidence.contains(&(result.path.clone(), span.clone()))
                    })
            })
        })
        .count();
    let total_citations = cutoff_docs
        .iter()
        .filter(|doc| {
            results
                .iter()
                .any(|result| result.path == doc.doc_path && !result.evidence.is_empty())
        })
        .count();

    GroundednessEvaluation::from_counts(
        k,
        expected_evidence.len(),
        cited_evidence.len(),
        matching_evidence,
        grounded_citations,
        total_citations,
    )
}

fn expected_evidence(results: &[GroundednessResult]) -> HashSet<(PathBuf, EvidenceSpan)> {
    results
        .iter()
        .filter(|result| result.relevant)
        .flat_map(|result| {
            result
                .evidence
                .iter()
                .cloned()
                .map(|span| (result.path.clone(), span))
        })
        .collect()
}

fn cited_evidence(
    ranked_docs: &[Score],
    results: &[GroundednessResult],
) -> HashSet<(PathBuf, EvidenceSpan)> {
    ranked_docs
        .iter()
        .flat_map(|doc| {
            results
                .iter()
                .filter(move |result| result.path == doc.doc_path)
                .flat_map(|result| {
                    result
                        .evidence
                        .iter()
                        .cloned()
                        .map(|span| (result.path.clone(), span))
                })
        })
        .collect()
}

pub fn aggregate_groundedness(evaluations: &[GroundednessEvaluation]) -> GroundednessEvaluation {
    if evaluations.is_empty() {
        return GroundednessEvaluation::default();
    }

    let counts = evaluations
        .iter()
        .fold(GroundednessEvaluation::zero(0), |acc, evaluation| {
            GroundednessEvaluation {
                k: acc.k.max(evaluation.k),
                expected_evidence: acc.expected_evidence + evaluation.expected_evidence,
                cited_evidence: acc.cited_evidence + evaluation.cited_evidence,
                matching_evidence: acc.matching_evidence + evaluation.matching_evidence,
                grounded_citations: acc.grounded_citations + evaluation.grounded_citations,
                total_citations: acc.total_citations + evaluation.total_citations,
                highlight_recall: 0.0,
                highlight_precision: 0.0,
                citation_precision: 0.0,
            }
        });

    GroundednessEvaluation::from_counts(
        counts.k,
        counts.expected_evidence,
        counts.cited_evidence,
        counts.matching_evidence,
        counts.grounded_citations,
        counts.total_citations,
    )
}

impl fmt::Display for GroundednessEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "highlight recall@{k}: {highlight_recall:.4}\nhighlight precision@{k}: {highlight_precision:.4}\ncitation precision@{k}: {citation_precision:.4}",
            k = self.k,
            highlight_recall = self.highlight_recall,
            highlight_precision = self.highlight_precision,
            citation_precision = self.citation_precision,
        )
    }
}
