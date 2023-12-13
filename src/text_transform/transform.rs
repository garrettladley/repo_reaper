use rayon::{iter::ParallelIterator, slice::ParallelSlice, str::ParallelString};
use rust_stemmers::Stemmer;

pub fn n_gram_transform(content: &str, stemmer: &Stemmer, n: usize) -> Vec<String> {
    content
        .to_lowercase()
        .par_split_whitespace()
        .map(|s| s.chars().filter(|c| c.is_alphabetic()).collect::<String>())
        .filter(|s| !s.is_empty())
        .map(|token| stemmer.stem(&token).to_string())
        .collect::<Vec<_>>()
        .par_windows(n)
        .map(|window| window.join(" "))
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use crate::text_transform::n_gram_transform;
    use rust_stemmers::{Algorithm, Stemmer};

    fn get_stemmer() -> Stemmer {
        Stemmer::create(Algorithm::English)
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
            to_hash_set(n_gram_transform("The quick brown fox", &get_stemmer(), 1)),
            to_hash_set(vec![
                "the".to_string(),
                "quick".to_string(),
                "brown".to_string(),
                "fox".to_string()
            ])
        );
    }

    #[test]
    fn n_gram_transform_with_punctuation() {
        assert_eq!(
            to_hash_set(n_gram_transform(
                "Jumps over the lazy dog!123",
                &get_stemmer(),
                1
            )),
            to_hash_set(vec![
                "jump".to_string(),
                "over".to_string(),
                "the".to_string(),
                "lazi".to_string(),
                "dog".to_string()
            ])
        );
    }

    #[test]
    fn n_gram_transform_with_special_characters() {
        assert_eq!(
            to_hash_set(n_gram_transform("Rust 2023! @#%^&*", &get_stemmer(), 1)),
            to_hash_set(vec!["rust".to_string()])
        );
    }

    #[test]
    fn n_gram_transform_empty_string() {
        assert_eq!(
            to_hash_set(n_gram_transform("", &get_stemmer(), 1)),
            HashSet::<String>::new()
        );
    }

    #[test]
    fn n_gram_transform_bi_grams() {
        assert_eq!(
            to_hash_set(n_gram_transform("The quick brown fox", &get_stemmer(), 2)),
            to_hash_set(vec![
                "the quick".to_string(),
                "quick brown".to_string(),
                "brown fox".to_string()
            ])
        );
    }

    #[test]
    fn n_gram_transform_n_larger_than_words() {
        let stemmer = get_stemmer();
        assert_eq!(
            to_hash_set(n_gram_transform("The quick", &stemmer, 3)),
            HashSet::<String>::new()
        );
    }

    #[test]
    fn n_gram_transform_empty_string_bi_gram() {
        let stemmer = get_stemmer();
        assert_eq!(
            to_hash_set(n_gram_transform("", &stemmer, 2)),
            HashSet::<String>::new()
        );
    }
}
