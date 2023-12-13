use std::collections::HashSet;

use rust_stemmers::Stemmer;

pub struct Globals {
    pub n_grams: usize,
    pub stemmer: Stemmer,
    pub stop_words: HashSet<String>,
}
