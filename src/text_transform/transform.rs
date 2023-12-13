use rayon::{iter::ParallelIterator, slice::ParallelSlice};
use rust_stemmers::Stemmer;

use crate::inverted_index::Term;

pub fn n_gram_transform(
    content: &str,
    stemmer: &Stemmer,
    stop_words: &[String],
    n: usize,
) -> Vec<Term> {
    content
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .filter(|token| !stop_words.contains(&token.to_string()))
        .map(|token| stemmer.stem(token).to_string())
        .collect::<Vec<_>>()
        .par_windows(n)
        .map(|window| window.join(" "))
        .map(Term)
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::{inverted_index::Term, text_transform::n_gram_transform};
    use rust_stemmers::{Algorithm, Stemmer};

    fn get_stemmer() -> Stemmer {
        Stemmer::create(Algorithm::English)
    }

    fn get_stop_words() -> Vec<String> {
        stop_words::get(stop_words::LANGUAGE::English)
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
            to_hash_set(n_gram_transform(
                "The quick brown fox",
                &get_stemmer(),
                &get_stop_words(),
                1
            )),
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
            to_hash_set(n_gram_transform(
                "Jumps over the lazy dog!123",
                &get_stemmer(),
                &get_stop_words(),
                1
            )),
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
            to_hash_set(n_gram_transform(
                "Rust 2023! @#%^&*",
                &get_stemmer(),
                &get_stop_words(),
                1
            )),
            to_hash_set(vec![Term("rust".to_string()), Term("2023".to_string())])
        );
    }

    #[test]
    fn n_gram_transform_empty_string() {
        assert_eq!(
            to_hash_set(n_gram_transform("", &get_stemmer(), &get_stop_words(), 1)),
            HashSet::<Term>::new()
        );
    }

    #[test]
    fn n_gram_transform_bi_grams() {
        assert_eq!(
            to_hash_set(n_gram_transform(
                "The quick brown fox",
                &get_stemmer(),
                &get_stop_words(),
                2
            )),
            to_hash_set(vec![
                Term("quick brown".to_string()),
                Term("brown fox".to_string())
            ])
        );
    }

    #[test]
    fn n_gram_transform_n_larger_than_words() {
        assert_eq!(
            to_hash_set(n_gram_transform(
                "The quick",
                &get_stemmer(),
                &get_stop_words(),
                3
            )),
            HashSet::<Term>::new()
        );
    }

    #[test]
    fn n_gram_transform_empty_string_bi_gram() {
        assert_eq!(
            to_hash_set(n_gram_transform("", &get_stemmer(), &get_stop_words(), 2)),
            HashSet::<Term>::new()
        );
    }
}
