mod corpus;
mod engine;
mod planner;
mod regex_query;
mod trigram_index;
mod types;

use std::ops::{Range, RangeInclusive};

pub use corpus::{CorpusDocument, FileSystemCorpus, RegexCorpus};
pub use engine::RegexSearchEngine;
pub use regex_query::RegexCandidatePlan;
pub use trigram_index::{TrigramIndex, trigrams};
pub use types::{
    LiteralSearchResult, RegexCandidateDiagnostics, RegexCandidateSelection, RegexSearchError,
    RegexSearchMatch, Trigram,
};

pub(crate) fn line_range_for_match(
    content: &str,
    byte_range: Range<usize>,
) -> RangeInclusive<usize> {
    let start_line = line_number_at_byte(content, byte_range.start);
    let end_line = if byte_range.is_empty() {
        start_line
    } else {
        line_number_at_byte(content, byte_range.end)
    };

    start_line..=end_line
}

fn line_number_at_byte(content: &str, byte_index: usize) -> usize {
    content.as_bytes()[..byte_index]
        .iter()
        .filter(|&&byte| byte == b'\n')
        .count()
        + 1
}

#[cfg(test)]
mod tests;
