use std::collections::HashMap;

use crate::{
    config::Config,
    index::Term,
    tokenizer::{AnalyzerField, AnalyzerProfile, FileType, n_gram_transform},
};

#[derive(Clone, Debug)]
pub struct AnalyzedQuery {
    original_text: String,
    terms: HashMap<Term, QueryTerm>,
    phrases: Vec<QueryPhrase>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryTerm {
    pub frequency: u32,
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryPhrase {
    pub raw: String,
    pub terms: Vec<Term>,
}

impl AnalyzedQuery {
    pub fn new(query: &str, config: &Config) -> Self {
        let phrase_analyzer = |phrase: &str| {
            content_tokens_for_query(
                AnalyzerProfile::for_file_type(FileType::UnknownText),
                phrase,
                config,
            )
        };
        Self::from_frequencies_and_phrases(
            query.to_string(),
            n_gram_transform(query, config),
            parse_quoted_phrases(query, phrase_analyzer),
        )
    }

    pub fn new_code_search(query: &str, config: &Config) -> Self {
        let profile = AnalyzerProfile::for_file_type(FileType::Rust);
        let frequencies = profile
            .analyze(AnalyzerField::Content, query, config)
            .into_iter()
            .fold(HashMap::new(), |mut acc, token| {
                *acc.entry(Term(token)).or_insert(0) += 1;
                acc
            });

        Self::from_frequencies_and_phrases(
            query.to_string(),
            frequencies,
            parse_quoted_phrases(query, |phrase| {
                content_tokens_for_query(profile, phrase, config)
            }),
        )
    }

    pub fn from_frequencies(
        original_text: impl Into<String>,
        frequencies: HashMap<Term, u32>,
    ) -> Self {
        Self::from_frequencies_and_phrases(original_text, frequencies, Vec::new())
    }

    pub fn from_frequencies_and_phrases(
        original_text: impl Into<String>,
        frequencies: HashMap<Term, u32>,
        phrases: Vec<QueryPhrase>,
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
            phrases,
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
            phrases: Vec::new(),
        }
    }

    pub fn original_text(&self) -> &str {
        &self.original_text
    }

    pub fn terms(&self) -> impl Iterator<Item = (&Term, &QueryTerm)> {
        self.terms.iter()
    }

    pub fn phrases(&self) -> &[QueryPhrase] {
        &self.phrases
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

fn parse_quoted_phrases(query: &str, analyzer: impl Fn(&str) -> Vec<String>) -> Vec<QueryPhrase> {
    let mut phrases = Vec::new();
    let mut remaining = query;

    while let Some(start) = remaining.find('"') {
        let after_start = &remaining[start + 1..];
        let Some(end) = after_start.find('"') else {
            break;
        };

        let raw = &after_start[..end];
        let terms = analyzer(raw).into_iter().map(Term).collect::<Vec<_>>();
        if terms.len() > 1 {
            phrases.push(QueryPhrase {
                raw: raw.to_string(),
                terms,
            });
        }

        remaining = &after_start[end + 1..];
    }

    phrases
}

fn content_tokens_for_query(
    profile: AnalyzerProfile,
    phrase: &str,
    config: &Config,
) -> Vec<String> {
    profile.analyze(AnalyzerField::Content, phrase, config)
}

impl std::fmt::Display for AnalyzedQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut terms: Vec<&str> = self.terms.keys().map(|term| term.0.as_str()).collect();
        terms.sort_unstable();

        write!(f, "{}", terms.join(" "))
    }
}
