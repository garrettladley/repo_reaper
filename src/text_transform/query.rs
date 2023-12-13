use rust_stemmers::Stemmer;

use crate::inverted_index::Term;
use crate::text_transform::transform;

pub struct Query(pub Vec<Term>);

impl Query {
    pub fn new(query: &str, stemmer: &Stemmer) -> Self {
        Query(
            transform(query, stemmer)
                .into_iter()
                .map(Term)
                .collect::<Vec<_>>(),
        )
    }
}
