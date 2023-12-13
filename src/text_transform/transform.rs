use rust_stemmers::Stemmer;

fn transform(content: &str, stemmer: &Stemmer) -> Vec<String> {
    content
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.chars().filter(|c| c.is_alphabetic()).collect::<String>())
        .filter(|s| !s.is_empty())
        .map(|token| stemmer.stem(&token).to_string())
        .collect::<Vec<_>>()
}

pub fn n_gram_transform(content: &str, stemmer: &Stemmer, n: usize) -> Vec<String> {
    let mut tokens = transform(content, stemmer);

    if n == 1 {
        return tokens;
    }

    let mut n_grams = Vec::new();

    while tokens.len() >= n {
        n_grams.push(tokens[..n].join(" "));
        tokens = tokens[1..].to_vec();
    }

    n_grams
}

#[cfg(test)]
mod tests {
    use crate::text_transform::n_gram_transform;
    use rust_stemmers::{Algorithm, Stemmer};

    fn get_stemmer() -> Stemmer {
        Stemmer::create(Algorithm::English)
    }

    #[test]
    fn n_gram_transform_simple_sentence() {
        let stemmer = get_stemmer();
        assert_eq!(
            n_gram_transform("The quick brown fox", &stemmer, 1),
            vec!["the", "quick", "brown", "fox"]
        );
    }

    #[test]
    fn n_gram_transform_with_punctuation() {
        let stemmer = get_stemmer();
        assert_eq!(
            n_gram_transform("Jumps over the lazy dog!123", &stemmer, 1),
            vec!["jump", "over", "the", "lazi", "dog"]
        );
    }

    #[test]
    fn n_gram_transform_with_special_characters() {
        let stemmer = get_stemmer();
        assert_eq!(
            n_gram_transform("Rust 2023! @#%^&*", &stemmer, 1),
            vec!["rust"]
        );
    }

    #[test]
    fn n_gram_transform_empty_string() {
        let stemmer = get_stemmer();
        assert_eq!(n_gram_transform("", &stemmer, 1), Vec::<String>::new());
    }

    #[test]
    fn n_gram_transform_bi_grams() {
        let stemmer = get_stemmer();
        assert_eq!(
            n_gram_transform("The quick brown fox", &stemmer, 2),
            vec!["the quick", "quick brown", "brown fox"]
        );
    }

    #[test]
    fn n_gram_transform_n_larger_than_words() {
        let stemmer = get_stemmer();
        assert_eq!(
            n_gram_transform("The quick", &stemmer, 3),
            Vec::<String>::new()
        );
    }

    #[test]
    fn n_gram_transform_empty_string_bi_gram() {
        let stemmer = get_stemmer();
        assert_eq!(n_gram_transform("", &stemmer, 2), Vec::<String>::new());
    }
}
