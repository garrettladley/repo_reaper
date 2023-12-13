use rust_stemmers::Stemmer;

use crate::inverted_index::Term;
use crate::text_transform::n_gram_transform;

pub struct Query(pub Vec<Term>);

impl Query {
    pub fn new(query: &str, stemmer: &Stemmer, n_grams: usize) -> Self {
        Query(
            n_gram_transform(query, stemmer, n_grams)
                .into_iter()
                .map(Term)
                .collect::<Vec<_>>(),
        )
    }
}
