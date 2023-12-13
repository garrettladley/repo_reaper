use std::collections::HashSet;

use rayon::{iter::ParallelIterator, slice::ParallelSlice};

use crate::{globals::Globals, inverted_index::Term};

pub fn n_gram_transform(content: &str, globals: &Globals) -> HashSet<Term> {
    content
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|token| !globals.stop_words.contains(&token.to_string()))
        .map(|token| globals.stemmer.stem(token).to_string())
        .collect::<Vec<_>>()
        .par_windows(globals.n_grams)
        .map(|window| window.join(" "))
        .map(Term)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{globals::Globals, inverted_index::Term, text_transform::n_gram_transform};
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    fn get_stemmer() -> Stemmer {
        Stemmer::create(Algorithm::English)
    }

    fn get_stop_words() -> HashSet<String> {
        stop_words::get(stop_words::LANGUAGE::English)
            .par_iter()
            .map(|word| word.to_string())
            .collect()
    }

    fn to_hash_set<T>(vec: Vec<T>) -> HashSet<T>
    where
        T: std::hash::Hash + std::cmp::Eq,
    {
        vec.into_iter().collect()
    }

    #[test]
    fn n_gram_transform_simple_sentence() {
        assert_eq!(
            n_gram_transform(
                "The quick brown fox",
                &Globals {
                    n_grams: 1,
                    stemmer: get_stemmer(),
                    stop_words: get_stop_words(),
                }
            ),
            to_hash_set(vec![
                Term("quick".to_string()),
                Term("brown".to_string()),
                Term("fox".to_string())
            ])
        );
    }

    #[test]
    fn n_gram_transform_with_punctuation() {
        assert_eq!(
            n_gram_transform(
                "Jumps over the lazy dog!123",
                &Globals {
                    n_grams: 1,
                    stemmer: get_stemmer(),
                    stop_words: get_stop_words(),
                }
            ),
            to_hash_set(vec![
                Term("jump".to_string()),
                Term("lazi".to_string()),
                Term("dog".to_string()),
                Term("123".to_string())
            ])
        );
    }

    #[test]
    fn n_gram_transform_with_special_characters() {
        assert_eq!(
            n_gram_transform(
                "Rust 2023! @#%^&*",
                &Globals {
                    n_grams: 1,
                    stemmer: get_stemmer(),
                    stop_words: get_stop_words(),
                }
            ),
            to_hash_set(vec![Term("rust".to_string()), Term("2023".to_string())])
        );
    }

    #[test]
    fn n_gram_transform_empty_string() {
        assert_eq!(
            n_gram_transform(
                "",
                &Globals {
                    n_grams: 1,
                    stemmer: get_stemmer(),
                    stop_words: get_stop_words(),
                }
            ),
            HashSet::<Term>::new()
        );
    }

    #[test]
    fn n_gram_transform_bi_grams() {
        assert_eq!(
            n_gram_transform(
                "The quick brown fox",
                &Globals {
                    n_grams: 2,
                    stemmer: get_stemmer(),
                    stop_words: get_stop_words(),
                }
            ),
            to_hash_set(vec![
                Term("quick brown".to_string()),
                Term("brown fox".to_string())
            ])
        );
    }

    #[test]
    fn n_gram_transform_n_larger_than_words() {
        assert_eq!(
            n_gram_transform(
                "The quick",
                &Globals {
                    n_grams: 3,
                    stemmer: get_stemmer(),
                    stop_words: get_stop_words(),
                }
            ),
            HashSet::<Term>::new()
        );
    }

    #[test]
    fn n_gram_transform_empty_string_bi_gram() {
        assert_eq!(
            n_gram_transform(
                "",
                &Globals {
                    n_grams: 2,
                    stemmer: get_stemmer(),
                    stop_words: get_stop_words(),
                }
            ),
            HashSet::<Term>::new()
        );
    }
}
