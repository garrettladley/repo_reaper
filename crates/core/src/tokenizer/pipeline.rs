use std::collections::HashMap;

use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::{config::Config, index::Term, tokenizer::tokenize_identifier};

pub fn n_gram_transform(content: &str, config: &Config) -> HashMap<Term, u32> {
    content_tokens(content, config)
        .par_windows(config.n_grams)
        .map(|window| window.join(" "))
        .map(Term)
        .fold(HashMap::new, |mut acc: HashMap<Term, u32>, term| {
            *acc.entry(term).or_insert(0) += 1;
            acc
        })
        .reduce(HashMap::new, |mut a, b| {
            for (term, count) in b {
                *a.entry(term).or_insert(0) += count;
            }
            a
        })
}

pub fn content_tokens(content: &str, config: &Config) -> Vec<String> {
    content
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
        .filter(|s| !s.is_empty())
        .flat_map(|token| content_tokens_for_lexeme(token, config))
        .collect()
}

fn content_tokens_for_lexeme(lexeme: &str, config: &Config) -> Vec<String> {
    let Some(identifier) = tokenize_identifier(lexeme) else {
        return Vec::new();
    };

    let has_identifier_shape = identifier.parts.len() > 1 || lexeme.contains(['_', '-']);
    let mut tokens = Vec::new();

    if has_identifier_shape {
        push_unique(&mut tokens, identifier.exact);
        push_unique(&mut tokens, identifier.compound);
    }

    for part in identifier.parts {
        if config.stop_words.contains(&part) {
            continue;
        }

        push_unique(&mut tokens, config.stemmer.stem(&part).to_string());
    }

    tokens
}

fn push_unique(tokens: &mut Vec<String>, token: String) {
    if !tokens.iter().any(|existing| existing == &token) {
        tokens.push(token);
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use crate::{config::Config, index::Term, tokenizer::n_gram_transform};

    fn test_config(n_grams: usize) -> Config {
        Config {
            n_grams,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: stop_words::get(stop_words::LANGUAGE::English)
                .par_iter()
                .map(|word| word.to_string())
                .collect(),
        }
    }

    fn term_map(pairs: &[(&str, u32)]) -> HashMap<Term, u32> {
        pairs
            .iter()
            .map(|(s, c)| (Term(s.to_string()), *c))
            .collect()
    }

    #[test]
    fn unigrams_removes_stop_words_and_stems() {
        let result = n_gram_transform("The quick brown fox", &test_config(1));
        assert_eq!(result, term_map(&[("quick", 1), ("brown", 1), ("fox", 1)]));
    }

    #[test]
    fn unigrams_splits_on_punctuation_and_digits() {
        let result = n_gram_transform("Jumps over the lazy dog!123", &test_config(1));
        assert_eq!(
            result,
            term_map(&[("jump", 1), ("lazi", 1), ("dog", 1), ("123", 1)])
        );
    }

    #[test]
    fn unigrams_strips_special_characters() {
        let result = n_gram_transform("Rust 2023! @#%^&*", &test_config(1));
        assert_eq!(result, term_map(&[("rust", 1), ("2023", 1)]));
    }

    #[test]
    fn empty_input_returns_empty_map() {
        assert_eq!(n_gram_transform("", &test_config(1)), HashMap::new());
    }

    #[test]
    fn bigrams_produces_sliding_window_pairs() {
        let result = n_gram_transform("The quick brown fox", &test_config(2));
        assert_eq!(result, term_map(&[("quick brown", 1), ("brown fox", 1)]));
    }

    #[test]
    fn ngram_larger_than_token_count_returns_empty() {
        // "The quick" → 1 token after stop word removal → no 3-grams possible
        assert_eq!(
            n_gram_transform("The quick", &test_config(3)),
            HashMap::new()
        );
    }

    #[test]
    fn empty_input_bigrams_returns_empty() {
        assert_eq!(n_gram_transform("", &test_config(2)), HashMap::new());
    }

    #[test]
    fn repeated_words_produce_correct_frequencies() {
        // "rust" x3 stems to "rust", "systems" stems to "system"
        let result = n_gram_transform("rust rust rust systems", &test_config(1));
        assert_eq!(result, term_map(&[("rust", 3), ("system", 1)]));
    }

    #[test]
    fn stop_words_excluded_before_frequency_counting() {
        let result = n_gram_transform("the the the quick brown", &test_config(1));
        assert_eq!(result, term_map(&[("quick", 1), ("brown", 1)]));
    }

    #[test]
    fn identifiers_emit_split_tokens_and_compounds() {
        let result = n_gram_transform(
            "let value = HTTPSConnection::parse2Json();",
            &test_config(1),
        );

        assert!(result.contains_key(&Term("httpsconnection".to_string())));
        assert!(result.contains_key(&Term("https".to_string())));
        assert!(result.contains_key(&Term("connect".to_string())));
        assert!(result.contains_key(&Term("parse2json".to_string())));
        assert!(result.contains_key(&Term("pars".to_string())));
        assert!(result.contains_key(&Term("2".to_string())));
        assert!(result.contains_key(&Term("json".to_string())));
    }

    #[test]
    fn split_query_terms_match_compound_identifiers() {
        let document = n_gram_transform("fn repo_reaper() {}", &test_config(1));
        let query = n_gram_transform("reaper", &test_config(1));

        for term in query.keys() {
            assert!(
                document.contains_key(term),
                "expected document tokens to contain query term {term:?}"
            );
        }
    }
}
