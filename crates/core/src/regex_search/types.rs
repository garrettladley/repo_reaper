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

#[derive(Debug, thiserror::Error)]
pub enum RegexPostingsError {
    #[error("regex postings io failed")]
    Io(#[from] std::io::Error),
    #[error("invalid regex postings format")]
    InvalidFormat,
    #[error("regex postings file is too large")]
    FileTooLarge,
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
pub struct ExperimentalSparseNgramComparison {
    /// Number of documents selected by the default trigram candidate path.
    pub classic_candidate_count: usize,
    /// Number of documents selected by the experimental sparse n-gram path.
    pub sparse_candidate_count: usize,
    /// Number of default trigram posting lists read for this literal.
    pub classic_posting_lookups: usize,
    /// Number of sparse n-gram posting lists read by the covering query.
    pub sparse_posting_lookups: usize,
    /// Number of distinct classic trigram keys in the current index.
    pub classic_index_key_count: usize,
    /// Number of distinct sparse n-gram keys in the current experimental index.
    pub sparse_index_key_count: usize,
    /// Number of classic trigram tokens emitted if this literal were indexed.
    pub classic_update_token_count: usize,
    /// Number of sparse n-gram tokens emitted if this literal were indexed.
    pub sparse_update_token_count: usize,
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
