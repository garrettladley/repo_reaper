use std::collections::{BTreeSet, HashMap};

use super::{RegexCandidateDiagnostics, RegexCandidatePlan, RegexCandidateSelection, Trigram};
use crate::index::DocId;

const MAX_SELECTED_TRIGRAMS: usize = 8;

pub(crate) fn plan_candidates(
    plan: &RegexCandidatePlan,
    postings: &HashMap<Trigram, BTreeSet<DocId>>,
    all_doc_ids: &[DocId],
) -> RegexCandidateSelection {
    match plan {
        RegexCandidatePlan::All => all_candidates(all_doc_ids),
        RegexCandidatePlan::And(trigrams) => plan_conjunction(trigrams, postings, all_doc_ids),
        RegexCandidatePlan::Or(branches) => plan_disjunction(branches, postings, all_doc_ids),
    }
}

fn all_candidates(all_doc_ids: &[DocId]) -> RegexCandidateSelection {
    let candidates = all_doc_ids.iter().copied().collect::<BTreeSet<_>>();
    selection(candidates, all_doc_ids.len(), Vec::new(), true)
}

fn plan_disjunction(
    branches: &[RegexCandidatePlan],
    postings: &HashMap<Trigram, BTreeSet<DocId>>,
    all_doc_ids: &[DocId],
) -> RegexCandidateSelection {
    let mut candidates = BTreeSet::new();
    let mut selected_trigrams = Vec::new();
    let mut fell_back_to_full_scan = false;

    for branch in branches {
        let branch_selection = plan_candidates(branch, postings, all_doc_ids);
        candidates.extend(branch_selection.candidates);
        selected_trigrams.extend(branch_selection.diagnostics.selected_trigrams);
        fell_back_to_full_scan |= branch_selection.diagnostics.fell_back_to_full_scan;
    }

    selection(
        candidates,
        all_doc_ids.len(),
        selected_trigrams,
        fell_back_to_full_scan,
    )
}

fn plan_conjunction(
    trigrams: &[Trigram],
    postings: &HashMap<Trigram, BTreeSet<DocId>>,
    all_doc_ids: &[DocId],
) -> RegexCandidateSelection {
    if trigrams.is_empty() {
        return all_candidates(all_doc_ids);
    }

    let mut posting_sizes = Vec::with_capacity(trigrams.len());
    for trigram in trigrams {
        let Some(posting) = postings.get(trigram) else {
            return selection(
                BTreeSet::new(),
                all_doc_ids.len(),
                vec![trigram.clone()],
                false,
            );
        };
        posting_sizes.push((trigram, posting.len()));
    }

    posting_sizes.sort_by(|(left_trigram, left_len), (right_trigram, right_len)| {
        left_len
            .cmp(right_len)
            .then_with(|| left_trigram.cmp(right_trigram))
    });

    let mut estimated_work = 0;
    let mut selected_trigrams = Vec::new();
    for (trigram, posting_len) in posting_sizes {
        if selected_trigrams.len() >= MAX_SELECTED_TRIGRAMS {
            break;
        }
        if estimated_work + posting_len >= all_doc_ids.len() {
            break;
        }

        estimated_work += posting_len;
        selected_trigrams.push(trigram.clone());
    }

    if selected_trigrams.is_empty() {
        return all_candidates(all_doc_ids);
    }

    let candidates = intersect_postings(&selected_trigrams, postings);
    selection(candidates, all_doc_ids.len(), selected_trigrams, false)
}

fn intersect_postings(
    trigrams: &[Trigram],
    postings: &HashMap<Trigram, BTreeSet<DocId>>,
) -> BTreeSet<DocId> {
    let Some((first, rest)) = trigrams.split_first() else {
        return BTreeSet::new();
    };

    let mut candidates = postings.get(first).cloned().unwrap_or_else(BTreeSet::new);

    for trigram in rest {
        let Some(posting) = postings.get(trigram) else {
            return BTreeSet::new();
        };
        candidates = candidates.intersection(posting).copied().collect();
    }

    candidates
}

fn selection(
    candidates: BTreeSet<DocId>,
    full_corpus_count: usize,
    selected_trigrams: Vec<Trigram>,
    fell_back_to_full_scan: bool,
) -> RegexCandidateSelection {
    RegexCandidateSelection {
        diagnostics: RegexCandidateDiagnostics {
            full_corpus_count,
            candidate_count: candidates.len(),
            selected_trigram_count: selected_trigrams.len(),
            selected_trigrams,
            fell_back_to_full_scan,
        },
        candidates,
    }
}
