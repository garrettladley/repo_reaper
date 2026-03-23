use std::collections::HashSet;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{config::Config, index::Term, tokenizer::n_gram_transform};

#[derive(Clone)]
pub struct Query(pub HashSet<Term>);

impl Query {
    pub fn new(query: &str, config: &Config) -> Self {
        Self(n_gram_transform(query, config))
    }
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            self.0
                .par_iter()
                .map(|term| term.0.clone())
                .collect::<Vec<_>>()
                .join(" ")
        )
    }
}
