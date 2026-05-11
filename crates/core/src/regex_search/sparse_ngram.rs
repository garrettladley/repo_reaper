use std::collections::BTreeSet;

/// Advanced learning-track sparse n-gram primitives.
///
/// This module intentionally sits beside the classic trigram planner instead
/// of replacing it. The weight function is a small stable in-repo hash so tests
/// and future on-disk experiments are not tied to std hasher implementation
/// details.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct SparseNgram(pub(crate) String);

pub(crate) fn sparse_ngrams(content: &str) -> Vec<SparseNgram> {
    let chars = content.chars().collect::<Vec<_>>();
    let weights = pair_weights(&chars);
    if weights.is_empty() {
        return Vec::new();
    }

    let mut ngrams = BTreeSet::new();
    for start in 0..weights.len() {
        for end in start..weights.len() {
            if interval_is_sparse(&weights, start, end) {
                ngrams.insert(chars[start..=end + 1].iter().collect::<String>());
            }
        }
    }

    ngrams.into_iter().map(SparseNgram).collect()
}

pub(crate) fn sparse_covering_ngrams(content: &str) -> Vec<SparseNgram> {
    let chars = content.chars().collect::<Vec<_>>();
    let weights = pair_weights(&chars);
    if weights.is_empty() {
        return Vec::new();
    }

    let mut ngrams = BTreeSet::new();
    let mut start = 0;
    while start < weights.len() {
        let end = (start..weights.len())
            .rev()
            .find(|&end| interval_is_sparse(&weights, start, end))
            .unwrap_or(start);
        ngrams.insert(chars[start..=end + 1].iter().collect::<String>());
        start = end + 1;
    }
    ngrams.into_iter().map(SparseNgram).collect()
}

fn interval_is_sparse(weights: &[u32], start: usize, end: usize) -> bool {
    if end <= start + 1 {
        return true;
    }

    let edge_floor = weights[start].min(weights[end]);
    weights[start + 1..end]
        .iter()
        .all(|&interior| edge_floor > interior)
}

fn pair_weights(chars: &[char]) -> Vec<u32> {
    chars
        .windows(2)
        .map(|pair| pair_weight(pair[0], pair[1]))
        .collect()
}

fn pair_weight(left: char, right: char) -> u32 {
    let mut hash = 0x811c_9dc5_u32;
    hash = mix_char(hash, left);
    mix_char(hash, right)
}

fn mix_char(mut hash: u32, char_: char) -> u32 {
    hash ^= char_ as u32;
    hash = hash.wrapping_mul(0x0100_0193);
    hash ^ (hash >> 16)
}
