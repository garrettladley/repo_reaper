use std::collections::HashMap;

use crate::{
    config::Config,
    index::Term,
    tokenizer::{AnalyzerField, AnalyzerProfile, FileType, n_gram_transform},
};

#[derive(Clone, Debug)]
pub struct AnalyzedQuery {
    original_text: String,
    intent: QueryIntent,
    terms: HashMap<Term, QueryTerm>,
    phrases: Vec<QueryPhrase>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueryTerm {
    pub frequency: u32,
    pub weight: f64,
    pub provenance: QueryTermProvenance,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryIntent {
    Path,
    Identifier,
    NaturalLanguage,
    ErrorMessage,
    Config,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryTermProvenance {
    Original,
    ControlledExpansion,
    Feedback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct QueryExpansionConfig {
    pub controlled: bool,
    pub feedback: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryPhrase {
    pub raw: String,
    pub terms: Vec<Term>,
}

impl AnalyzedQuery {
    pub fn new(query: &str, config: &Config) -> Self {
        let profile = AnalyzerProfile::for_file_type(FileType::UnknownText);
        Self::from_frequencies_and_phrases_with_intent(
            query.to_string(),
            classify_query_intent(query),
            n_gram_transform(query, config),
            parse_quoted_phrases(query, |phrase| {
                content_tokens_for_query(profile, phrase, config)
            }),
        )
    }

    pub fn new_code_search(query: &str, config: &Config) -> Self {
        Self::new_code_search_with_expansion(query, config, QueryExpansionConfig::default())
    }

    pub fn new_code_search_with_expansion(
        query: &str,
        config: &Config,
        expansion: QueryExpansionConfig,
    ) -> Self {
        let intent = classify_query_intent(query);
        let profile = AnalyzerProfile::for_file_type(FileType::Rust);
        let frequencies = profile
            .analyze(AnalyzerField::Content, query, config)
            .into_iter()
            .fold(HashMap::new(), |mut acc, token| {
                *acc.entry(Term(token)).or_insert(0) += 1;
                acc
            });

        let mut query = Self::from_frequencies_and_phrases_with_intent(
            query.to_string(),
            intent,
            frequencies,
            parse_quoted_phrases(query, |phrase| {
                content_tokens_for_query(profile, phrase, config)
            }),
        );
        if expansion.controlled && query.intent.allows_controlled_expansion() {
            let additions = controlled_expansions(query.terms.keys().map(|term| term.0.as_str()));
            query.add_weighted_terms(additions, QueryTermProvenance::ControlledExpansion);
        }

        query
    }

    pub fn from_frequencies(
        original_text: impl Into<String>,
        frequencies: HashMap<Term, u32>,
    ) -> Self {
        let original_text = original_text.into();
        Self::from_frequencies_and_phrases_with_intent(
            original_text.clone(),
            classify_query_intent(&original_text),
            frequencies,
            Vec::new(),
        )
    }

    pub fn from_frequencies_and_phrases(
        original_text: impl Into<String>,
        frequencies: HashMap<Term, u32>,
        phrases: Vec<QueryPhrase>,
    ) -> Self {
        let original_text = original_text.into();
        Self::from_frequencies_and_phrases_with_intent(
            original_text.clone(),
            classify_query_intent(&original_text),
            frequencies,
            phrases,
        )
    }

    pub fn from_frequencies_and_phrases_with_intent(
        original_text: impl Into<String>,
        intent: QueryIntent,
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
                        provenance: QueryTermProvenance::Original,
                    },
                )
            })
            .collect();

        Self {
            original_text: original_text.into(),
            intent,
            terms,
            phrases,
        }
    }

    pub fn from_weights(original_text: impl Into<String>, weights: HashMap<Term, f64>) -> Self {
        let original_text = original_text.into();
        Self::from_weights_with_intent(
            original_text.clone(),
            classify_query_intent(&original_text),
            weights,
        )
    }

    pub fn from_weights_with_intent(
        original_text: impl Into<String>,
        intent: QueryIntent,
        weights: HashMap<Term, f64>,
    ) -> Self {
        let terms = weights
            .into_iter()
            .filter(|(_, weight)| *weight > 0.0)
            .map(|(term, weight)| {
                (
                    term,
                    QueryTerm {
                        frequency: 1,
                        weight,
                        provenance: QueryTermProvenance::Original,
                    },
                )
            })
            .collect();

        Self {
            original_text: original_text.into(),
            intent,
            terms,
            phrases: Vec::new(),
        }
    }

    pub fn original_text(&self) -> &str {
        &self.original_text
    }

    pub fn intent(&self) -> QueryIntent {
        self.intent
    }

    pub fn terms(&self) -> impl Iterator<Item = (&Term, &QueryTerm)> {
        self.terms.iter()
    }

    pub fn phrases(&self) -> &[QueryPhrase] {
        &self.phrases
    }

    pub fn add_feedback_terms(&mut self, weights: HashMap<Term, f64>) {
        self.add_weighted_terms(weights, QueryTermProvenance::Feedback);
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn len(&self) -> usize {
        self.terms.len()
    }

    fn add_weighted_terms(&mut self, weights: HashMap<Term, f64>, provenance: QueryTermProvenance) {
        for (term, weight) in weights {
            if weight <= 0.0 {
                continue;
            }

            self.terms
                .entry(term)
                .and_modify(|existing| existing.weight += weight)
                .or_insert(QueryTerm {
                    frequency: 1,
                    weight,
                    provenance,
                });
        }
    }
}

impl QueryIntent {
    pub fn allows_controlled_expansion(self) -> bool {
        matches!(
            self,
            Self::NaturalLanguage | Self::ErrorMessage | Self::Config
        )
    }
}

impl QueryTermProvenance {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::ControlledExpansion => "controlled_expansion",
            Self::Feedback => "feedback",
        }
    }
}

pub fn classify_query_intent(query: &str) -> QueryIntent {
    let trimmed = query.trim();
    let lower = trimmed.to_ascii_lowercase();

    if looks_config_like(trimmed, &lower) {
        return QueryIntent::Config;
    }
    if looks_path_like(trimmed, &lower) {
        return QueryIntent::Path;
    }
    if looks_error_like(trimmed, &lower) {
        return QueryIntent::ErrorMessage;
    }
    if looks_identifier_like(trimmed) {
        return QueryIntent::Identifier;
    }

    QueryIntent::NaturalLanguage
}

fn looks_path_like(query: &str, lower: &str) -> bool {
    query.contains('/')
        || query.contains('\\')
        || lower.starts_with("./")
        || lower.starts_with("../")
        || lower.ends_with(".rs")
        || lower.ends_with(".toml")
        || lower.ends_with(".json")
        || lower.ends_with(".yaml")
        || lower.ends_with(".yml")
}

fn looks_identifier_like(query: &str) -> bool {
    !query.contains(char::is_whitespace)
        && query.chars().all(|character| {
            character.is_ascii_alphanumeric() || character == '_' || character == '-'
        })
        && query
            .chars()
            .any(|character| character == '_' || character == '-' || character.is_ascii_uppercase())
}

fn looks_error_like(query: &str, lower: &str) -> bool {
    query.contains('"')
        || lower.contains("error:")
        || lower.contains("panic")
        || lower.contains("failed to")
        || lower.contains("exception")
        || lower.contains("not found")
        || query.split_whitespace().count() >= 6
            && query
                .chars()
                .any(|character| matches!(character, ':' | '\'' | '`' | '(' | ')'))
}

fn looks_config_like(query: &str, lower: &str) -> bool {
    lower.contains("cargo.toml")
        || lower.contains("package.json")
        || lower.contains("config")
        || lower.contains("configuration")
        || lower.contains("setting")
        || lower.contains("env")
        || query.contains('=')
        || query.contains(':') && !query.contains(char::is_whitespace)
}

fn controlled_expansions<'a>(terms: impl Iterator<Item = &'a str>) -> HashMap<Term, f64> {
    let mut expansions = HashMap::new();

    for term in terms {
        for expansion in expansions_for(term) {
            expansions.insert(Term(expansion.to_string()), 0.35);
        }
    }

    expansions
}

fn expansions_for(term: &str) -> &'static [&'static str] {
    match term {
        "db" => &["database"],
        "auth" => &["authentication", "authorization"],
        "cfg" | "config" => &["configuration"],
        "repo" => &["repository"],
        "err" => &["error"],
        "msg" => &["message"],
        "req" => &["request"],
        "res" => &["response", "result"],
        _ => &[],
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
        let mut terms = self
            .terms
            .keys()
            .map(|term| term.0.as_str())
            .collect::<Vec<_>>();
        terms.sort_unstable();

        write!(f, "{}", terms.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use super::{AnalyzedQuery, QueryExpansionConfig, QueryIntent, classify_query_intent};
    use crate::{config::Config, index::Term};

    fn test_config() -> Config {
        Config {
            n_grams: 1,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: stop_words::get(stop_words::LANGUAGE::English)
                .par_iter()
                .map(|word| word.to_string())
                .collect::<HashSet<String>>(),
        }
    }

    #[test]
    fn classifies_query_intents() {
        assert_eq!(
            classify_query_intent("src/ranking/bm25.rs"),
            QueryIntent::Path
        );
        assert_eq!(
            classify_query_intent("BM25HyperParams"),
            QueryIntent::Identifier
        );
        assert_eq!(
            classify_query_intent("failed to deserialize evaluation JSON data"),
            QueryIntent::ErrorMessage
        );
        assert_eq!(
            classify_query_intent("cargo.toml dependencies"),
            QueryIntent::Config
        );
        assert_eq!(
            classify_query_intent("how bm25 scoring works"),
            QueryIntent::NaturalLanguage
        );
    }

    #[test]
    fn controlled_expansion_is_downweighted_and_gated_by_intent() {
        let config = test_config();
        let expanded = AnalyzedQuery::new_code_search_with_expansion(
            "auth config",
            &config,
            QueryExpansionConfig {
                controlled: true,
                feedback: false,
            },
        );
        let auth_weight = expanded
            .terms()
            .find(|(term, _)| **term == Term("auth".to_string()))
            .unwrap()
            .1
            .weight;
        let authentication_weight = expanded
            .terms()
            .find(|(term, _)| **term == Term("authentication".to_string()))
            .unwrap()
            .1
            .weight;

        assert!(authentication_weight < auth_weight);

        let path_query = AnalyzedQuery::new_code_search_with_expansion(
            "src/auth.rs",
            &config,
            QueryExpansionConfig {
                controlled: true,
                feedback: false,
            },
        );
        assert!(
            path_query
                .terms()
                .all(|(term, _)| term.0 != "authentication")
        );
    }

    #[test]
    fn quoted_phrases_are_preserved_for_proximity_scoring() {
        let query = AnalyzedQuery::new_code_search("\"ranked search\" bm25", &test_config());

        assert_eq!(query.phrases()[0].raw, "ranked search");
        assert_eq!(query.phrases()[0].terms.len(), 2);
    }
}
