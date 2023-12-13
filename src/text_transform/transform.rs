use rust_stemmers::Stemmer;

pub fn transform(content: &str, stemmer: &Stemmer) -> Vec<String> {
    content
        .to_lowercase()
        .split_whitespace()
        .map(|s| s.chars().filter(|c| c.is_alphabetic()).collect::<String>())
        .filter(|s| !s.is_empty())
        .map(|token| stemmer.stem(&token).to_string())
        .collect::<Vec<_>>()
}
