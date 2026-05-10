use std::{collections::HashSet, fmt, path::PathBuf};

use super::{
    GroundednessResult,
    util::{estimate_tokens_from_bytes, ratio},
};
use crate::{index::InvertedIndex, ranking::Score};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct TokenEfficiencyEvaluation {
    pub k: usize,
    pub returned_bytes: usize,
    pub estimated_returned_tokens: usize,
    pub relevant_evidence_bytes: usize,
    pub estimated_relevant_evidence_tokens: usize,
    pub evidence_density: f64,
}

impl TokenEfficiencyEvaluation {
    pub fn zero(k: usize) -> Self {
        TokenEfficiencyEvaluation {
            k,
            ..TokenEfficiencyEvaluation::default()
        }
    }

    fn from_counts(k: usize, returned_bytes: usize, relevant_evidence_bytes: usize) -> Self {
        TokenEfficiencyEvaluation {
            k,
            returned_bytes,
            estimated_returned_tokens: estimate_tokens_from_bytes(returned_bytes),
            relevant_evidence_bytes,
            estimated_relevant_evidence_tokens: estimate_tokens_from_bytes(relevant_evidence_bytes),
            evidence_density: ratio(relevant_evidence_bytes, returned_bytes),
        }
    }
}

pub fn evaluate_token_efficiency_at_k(
    ranked_docs: &[Score],
    results: &[GroundednessResult],
    inverted_index: &InvertedIndex,
    k: usize,
) -> TokenEfficiencyEvaluation {
    if k == 0 {
        return TokenEfficiencyEvaluation::zero(k);
    }

    let cutoff_docs = &ranked_docs[..ranked_docs.len().min(k)];
    let returned_docs = cutoff_docs
        .iter()
        .filter_map(|doc| inverted_index.doc_id(&doc.doc_path))
        .filter_map(|doc_id| inverted_index.document(doc_id))
        .map(|metadata| (&metadata.path, metadata.file_size_bytes as usize))
        .collect::<Vec<_>>();
    let returned_bytes = returned_docs.iter().map(|(_, bytes)| bytes).sum();
    let retrieved_paths: HashSet<&PathBuf> = returned_docs.iter().map(|(path, _)| *path).collect();
    let relevant_evidence_bytes = results
        .iter()
        .filter(|result| result.relevant && retrieved_paths.contains(&result.path))
        .flat_map(|result| &result.evidence)
        .map(|span| span.snippet.len())
        .sum();

    TokenEfficiencyEvaluation::from_counts(k, returned_bytes, relevant_evidence_bytes)
}

pub fn aggregate_token_efficiency(
    evaluations: &[TokenEfficiencyEvaluation],
) -> TokenEfficiencyEvaluation {
    if evaluations.is_empty() {
        return TokenEfficiencyEvaluation::default();
    }

    let (k, returned_bytes, relevant_evidence_bytes) =
        evaluations
            .iter()
            .fold((0, 0, 0), |(max_k, returned, evidence), evaluation| {
                (
                    max_k.max(evaluation.k),
                    returned + evaluation.returned_bytes,
                    evidence + evaluation.relevant_evidence_bytes,
                )
            });

    TokenEfficiencyEvaluation::from_counts(k, returned_bytes, relevant_evidence_bytes)
}

impl fmt::Display for TokenEfficiencyEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "returned bytes@{k}: {returned_bytes}\nestimated returned tokens@{k}: {returned_tokens}\nrelevant evidence bytes@{k}: {evidence_bytes}\nestimated relevant evidence tokens@{k}: {evidence_tokens}\nevidence density@{k}: {evidence_density:.4}",
            k = self.k,
            returned_bytes = self.returned_bytes,
            returned_tokens = self.estimated_returned_tokens,
            evidence_bytes = self.relevant_evidence_bytes,
            evidence_tokens = self.estimated_relevant_evidence_tokens,
            evidence_density = self.evidence_density,
        )
    }
}
