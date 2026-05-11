use std::{
    collections::{BTreeSet, HashMap},
    convert::Infallible,
};

use super::{RegexCandidateDiagnostics, RegexCandidatePlan, RegexCandidateSelection, Trigram};
use crate::index::DocId;

const MAX_SELECTED_TRIGRAMS: usize = 8;

pub(crate) trait RegexPostingSource {
    type Error;

    fn posting_len(&self, trigram: &Trigram) -> Result<Option<usize>, Self::Error>;
    fn postings(&self, trigram: &Trigram) -> Result<Option<BTreeSet<DocId>>, Self::Error>;
}

impl RegexPostingSource for HashMap<Trigram, BTreeSet<DocId>> {
    type Error = Infallible;

    fn posting_len(&self, trigram: &Trigram) -> Result<Option<usize>, Self::Error> {
        Ok(self.get(trigram).map(BTreeSet::len))
    }

    fn postings(&self, trigram: &Trigram) -> Result<Option<BTreeSet<DocId>>, Self::Error> {
        Ok(self.get(trigram).cloned())
    }
}

pub(crate) fn plan_candidates(
    plan: &RegexCandidatePlan,
    postings: &HashMap<Trigram, BTreeSet<DocId>>,
    all_doc_ids: &[DocId],
) -> RegexCandidateSelection {
    try_plan_candidates(plan, postings, all_doc_ids).unwrap_or_else(|never| match never {})
}

pub(crate) fn try_plan_candidates<S>(
    plan: &RegexCandidatePlan,
    postings: &S,
    all_doc_ids: &[DocId],
) -> Result<RegexCandidateSelection, S::Error>
where
    S: RegexPostingSource,
{
    match plan {
        RegexCandidatePlan::All => Ok(all_candidates(all_doc_ids)),
        RegexCandidatePlan::And(trigrams) => plan_conjunction(trigrams, postings, all_doc_ids),
        RegexCandidatePlan::Or(branches) => plan_disjunction(branches, postings, all_doc_ids),
    }
}

fn all_candidates(all_doc_ids: &[DocId]) -> RegexCandidateSelection {
    let candidates = all_doc_ids.iter().copied().collect::<BTreeSet<_>>();
    selection(candidates, all_doc_ids.len(), Vec::new(), true)
}

fn plan_disjunction<S>(
    branches: &[RegexCandidatePlan],
    postings: &S,
    all_doc_ids: &[DocId],
) -> Result<RegexCandidateSelection, S::Error>
where
    S: RegexPostingSource,
{
    let mut candidates = BTreeSet::new();
    let mut selected_trigrams = Vec::new();
    let mut fell_back_to_full_scan = false;

    for branch in branches {
        let branch_selection = try_plan_candidates(branch, postings, all_doc_ids)?;
        candidates.extend(branch_selection.candidates);
        selected_trigrams.extend(branch_selection.diagnostics.selected_trigrams);
        fell_back_to_full_scan |= branch_selection.diagnostics.fell_back_to_full_scan;
    }

    Ok(selection(
        candidates,
        all_doc_ids.len(),
        selected_trigrams,
        fell_back_to_full_scan,
    ))
}

fn plan_conjunction<S>(
    trigrams: &[Trigram],
    postings: &S,
    all_doc_ids: &[DocId],
) -> Result<RegexCandidateSelection, S::Error>
where
    S: RegexPostingSource,
{
    if trigrams.is_empty() {
        return Ok(all_candidates(all_doc_ids));
    }

    let mut posting_sizes = Vec::with_capacity(trigrams.len());
    for trigram in trigrams {
        let Some(posting_len) = postings.posting_len(trigram)? else {
            return Ok(selection(
                BTreeSet::new(),
                all_doc_ids.len(),
                vec![trigram.clone()],
                false,
            ));
        };
        posting_sizes.push((trigram, posting_len));
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
        return Ok(all_candidates(all_doc_ids));
    }

    let candidates = intersect_postings(&selected_trigrams, postings)?;
    Ok(selection(
        candidates,
        all_doc_ids.len(),
        selected_trigrams,
        false,
    ))
}

fn intersect_postings<S>(trigrams: &[Trigram], postings: &S) -> Result<BTreeSet<DocId>, S::Error>
where
    S: RegexPostingSource,
{
    let Some((first, rest)) = trigrams.split_first() else {
        return Ok(BTreeSet::new());
    };

    let mut candidates = postings.postings(first)?.unwrap_or_default();

    for trigram in rest {
        let Some(posting) = postings.postings(trigram)? else {
            return Ok(BTreeSet::new());
        };
        candidates = candidates.intersection(&posting).copied().collect();
    }

    Ok(candidates)
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
