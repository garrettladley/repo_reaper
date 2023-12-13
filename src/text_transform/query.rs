use std::collections::HashSet;

use crate::globals::Globals;
use crate::inverted_index::Term;
use crate::text_transform::n_gram_transform;

#[derive(Clone)]
pub struct Query(pub HashSet<Term>);

impl Query {
    pub fn new(query: &str, globals: &Globals) -> Self {
        Self(n_gram_transform(query, globals))
    }
}
