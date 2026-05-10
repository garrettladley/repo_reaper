use std::collections::HashMap;

use crate::{config::Config, index::Term, tokenizer::n_gram_transform};

#[derive(Clone, Debug)]
pub struct AnalyzedQuery {
    original_text: String,
    terms: HashMap<Term, QueryTerm>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryTerm {
    pub frequency: u32,
    pub weight: f64,
}

impl AnalyzedQuery {
    pub fn new(query: &str, config: &Config) -> Self {
        Self::from_frequencies(query.to_string(), n_gram_transform(query, config))
    }

    pub fn from_frequencies(
        original_text: impl Into<String>,
        frequencies: HashMap<Term, u32>,
    ) -> Self {
        let terms = frequencies
            .into_iter()
            .filter(|(_, frequency)| *frequency > 0)
            .map(|(term, frequency)| {
                (
                    term,
                    QueryTerm {
                        frequency,
                        weight: frequency as f64,
                    },
                )
            })
            .collect();

        Self {
            original_text: original_text.into(),
            terms,
        }
    }

    pub fn from_weights(original_text: impl Into<String>, weights: HashMap<Term, f64>) -> Self {
        let terms = weights
            .into_iter()
            .filter(|(_, weight)| *weight > 0.0)
            .map(|(term, weight)| {
                (
                    term,
                    QueryTerm {
                        frequency: 1,
                        weight,
                    },
                )
            })
            .collect();

        Self {
            original_text: original_text.into(),
            terms,
        }
    }

    pub fn original_text(&self) -> &str {
        &self.original_text
    }

    pub fn terms(&self) -> impl Iterator<Item = (&Term, &QueryTerm)> {
        self.terms.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

impl std::fmt::Display for AnalyzedQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut terms: Vec<&str> = self.terms.keys().map(|term| term.0.as_str()).collect();
        terms.sort_unstable();

        write!(f, "{}", terms.join(" "))
    }
}
