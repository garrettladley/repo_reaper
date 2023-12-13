use std::path::PathBuf;

use crate::{inverted_index::InvertedIndex, text_transform::Query};

#[derive(Debug)]
pub struct Rank {
    pub doc_path: PathBuf,
    pub score: f64,
}

pub trait RankingAlgorithm {
    fn rank(&self, inverted_index: &InvertedIndex, query: &Query, top_n: usize) -> Vec<Rank>;
}
