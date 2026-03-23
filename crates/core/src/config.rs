use std::collections::HashSet;

use rust_stemmers::Stemmer;

pub struct Config {
    pub n_grams: usize,
    pub stemmer: Stemmer,
    pub stop_words: HashSet<String>,
}
