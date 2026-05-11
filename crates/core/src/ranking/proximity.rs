use std::collections::{HashMap, HashSet};

use crate::{
    index::{DocId, DocumentField, InvertedIndex, Term, TermDocument},
    query::{AnalyzedQuery, QueryPhrase},
};

const POSITIONAL_FIELDS: [DocumentField; 4] = [
    DocumentField::Content,
    DocumentField::Identifier,
    DocumentField::Comment,
    DocumentField::StringLiteral,
];

#[derive(Debug, Clone, PartialEq)]
pub struct ProximityConfig {
    pub window: u32,
    pub phrase_boost: f64,
    pub proximity_boost: f64,
}

impl Default for ProximityConfig {
    fn default() -> Self {
        Self {
            window: 8,
            phrase_boost: 2.0,
            proximity_boost: 0.75,
        }
    }
}

pub fn positional_bonus(
    index: &InvertedIndex,
    query: &AnalyzedQuery,
    doc_id: DocId,
    config: &ProximityConfig,
) -> f64 {
    let phrase_bonus = query
        .phrases()
        .iter()
        .map(|phrase| phrase_match_count(index, doc_id, phrase) as f64 * config.phrase_boost)
        .sum::<f64>();

    phrase_bonus + proximity_bonus(index, query, doc_id, config)
}

pub fn phrase_match_count(index: &InvertedIndex, doc_id: DocId, phrase: &QueryPhrase) -> usize {
    if phrase.terms.len() < 2 {
        return 0;
    }

    POSITIONAL_FIELDS
        .into_iter()
        .map(|field| phrase_match_count_in_field(index, doc_id, phrase, field))
        .sum()
}

fn phrase_match_count_in_field(
    index: &InvertedIndex,
    doc_id: DocId,
    phrase: &QueryPhrase,
    field: DocumentField,
) -> usize {
    let Some(first_doc) = term_doc(index, &phrase.terms[0], doc_id) else {
        return 0;
    };
    let Some(first_positions) = first_doc.field_positions(field) else {
        return 0;
    };

    let rest_positions = phrase.terms[1..]
        .iter()
        .filter_map(|term| {
            term_doc(index, term, doc_id)
                .and_then(|term_doc| term_doc.field_positions(field))
                .map(|positions| positions.positions())
        })
        .collect::<Vec<_>>();

    if rest_positions.len() != phrase.terms.len() - 1 {
        return 0;
    }

    first_positions
        .positions()
        .into_iter()
        .filter(|start| {
            rest_positions
                .iter()
                .enumerate()
                .all(|(offset, positions)| {
                    positions
                        .binary_search(&(*start + offset as u32 + 1))
                        .is_ok()
                })
        })
        .count()
}

fn proximity_bonus(
    index: &InvertedIndex,
    query: &AnalyzedQuery,
    doc_id: DocId,
    config: &ProximityConfig,
) -> f64 {
    let unique_terms = query
        .terms()
        .map(|(term, _)| term.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    if unique_terms.len() < 2 {
        return 0.0;
    }

    POSITIONAL_FIELDS
        .into_iter()
        .filter_map(|field| minimum_covering_span(index, doc_id, &unique_terms, field))
        .filter(|span| *span <= config.window)
        .map(|span| {
            config.proximity_boost * (config.window - span + 1) as f64 / config.window as f64
        })
        .sum()
}

fn minimum_covering_span(
    index: &InvertedIndex,
    doc_id: DocId,
    terms: &[Term],
    field: DocumentField,
) -> Option<u32> {
    let mut occurrences = Vec::new();
    for (term_idx, term) in terms.iter().enumerate() {
        let positions = term_doc(index, term, doc_id)?
            .field_positions(field)?
            .positions();
        for position in positions {
            occurrences.push((position, term_idx));
        }
    }

    occurrences.sort_unstable();
    let mut counts: HashMap<usize, usize> = HashMap::new();
    let mut covered_terms = 0;
    let mut left = 0;
    let mut best = None;

    for right in 0..occurrences.len() {
        let right_term = occurrences[right].1;
        let count = counts.entry(right_term).or_insert(0);
        if *count == 0 {
            covered_terms += 1;
        }
        *count += 1;

        while covered_terms == terms.len() {
            let span = occurrences[right].0 - occurrences[left].0;
            best = Some(best.map_or(span, |current: u32| current.min(span)));

            let left_term = occurrences[left].1;
            let count = counts
                .get_mut(&left_term)
                .expect("left occurrence term should be counted");
            *count -= 1;
            if *count == 0 {
                covered_terms -= 1;
            }
            left += 1;
        }
    }

    best
}

fn term_doc<'a>(index: &'a InvertedIndex, term: &Term, doc_id: DocId) -> Option<&'a TermDocument> {
    index.get_postings(term)?.get(&doc_id)
}
