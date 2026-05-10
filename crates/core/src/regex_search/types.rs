use std::{
    collections::BTreeSet,
    ops::{Range, RangeInclusive},
    path::PathBuf,
};

use crate::index::DocId;

#[derive(Debug, thiserror::Error)]
pub enum RegexSearchError {
    #[error("invalid regex pattern")]
    InvalidPattern(#[source] regex::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexSearchMatch {
    pub path: PathBuf,
    pub byte_range: Range<usize>,
    pub line_range: RangeInclusive<usize>,
    pub matched_text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Trigram(pub(crate) String);

impl Trigram {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Trigram {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiteralSearchResult {
    pub candidate_count: usize,
    pub matches: Vec<RegexSearchMatch>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexCandidateSelection {
    pub candidates: BTreeSet<DocId>,
    pub diagnostics: RegexCandidateDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexCandidateDiagnostics {
    pub full_corpus_count: usize,
    pub candidate_count: usize,
    pub selected_trigram_count: usize,
    pub selected_trigrams: Vec<Trigram>,
    pub fell_back_to_full_scan: bool,
}
