use std::collections::HashSet;

use rust_stemmers::Stemmer;

use crate::inverted_index::Term;
use crate::text_transform::n_gram_transform;

pub struct Query(pub HashSet<Term>);

impl Query {
    pub fn new(
        query: &str,
        stemmer: &Stemmer,
        stop_words: &HashSet<String>,
        n_grams: usize,
    ) -> Self {
        Self(n_gram_transform(query, stemmer, stop_words, n_grams))
    }
}
