#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentifierTokens {
    pub exact: String,
    pub compound: String,
    pub parts: Vec<String>,
}

pub fn tokenize_identifier(identifier: &str) -> Option<IdentifierTokens> {
    let exact = identifier.trim().to_lowercase();
    if exact.is_empty() {
        return None;
    }

    let parts = identifier
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
        .flat_map(split_identifier_segment)
        .filter(|part| !part.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>();

    if parts.is_empty() {
        return None;
    }

    let compound = parts.join("");

    Some(IdentifierTokens {
        exact,
        compound,
        parts,
    })
}

pub fn identifier_token_stream(identifier: &str) -> Vec<String> {
    let Some(tokens) = tokenize_identifier(identifier) else {
        return Vec::new();
    };

    let mut stream = Vec::with_capacity(tokens.parts.len() + 2);
    push_unique(&mut stream, tokens.exact);
    push_unique(&mut stream, tokens.compound);
    for part in tokens.parts {
        push_unique(&mut stream, part);
    }

    stream
}

fn split_identifier_segment(segment: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let chars = segment.char_indices().collect::<Vec<_>>();

    for idx in 1..chars.len() {
        let previous = chars[idx - 1].1;
        let current = chars[idx].1;
        let next = chars.get(idx + 1).map(|(_, c)| *c);

        if previous == '_' || previous == '-' {
            start = chars[idx].0;
            continue;
        }

        if current == '_' || current == '-' {
            if start < chars[idx].0 {
                parts.push(&segment[start..chars[idx].0]);
            }
            start = chars[idx].0 + current.len_utf8();
            continue;
        }

        if is_identifier_boundary(previous, current, next) {
            parts.push(&segment[start..chars[idx].0]);
            start = chars[idx].0;
        }
    }

    if start < segment.len() {
        parts.push(&segment[start..]);
    }

    parts
}

fn is_identifier_boundary(previous: char, current: char, next: Option<char>) -> bool {
    if previous.is_ascii_lowercase() && current.is_ascii_uppercase() {
        return true;
    }

    if previous.is_ascii_alphabetic() && current.is_ascii_digit() {
        return true;
    }

    if previous.is_ascii_digit() && current.is_ascii_alphabetic() {
        return true;
    }

    previous.is_ascii_uppercase()
        && current.is_ascii_uppercase()
        && next.is_some_and(|next| next.is_ascii_lowercase())
}

fn push_unique(tokens: &mut Vec<String>, token: String) {
    if !tokens.iter().any(|existing| existing == &token) {
        tokens.push(token);
    }
}

#[cfg(test)]
mod tests {
    use super::identifier_token_stream;

    #[test]
    fn splits_https_connection() {
        assert_eq!(
            identifier_token_stream("HTTPSConnection"),
            vec!["httpsconnection", "https", "connection"]
        );
    }

    #[test]
    fn splits_camel_case() {
        assert_eq!(
            identifier_token_stream("camelCase"),
            vec!["camelcase", "camel", "case"]
        );
    }

    #[test]
    fn preserves_snake_case_exact_form() {
        assert_eq!(
            identifier_token_stream("repo_reaper"),
            vec!["repo_reaper", "reporeaper", "repo", "reaper"]
        );
    }

    #[test]
    fn splits_digits_inside_identifier() {
        assert_eq!(
            identifier_token_stream("parse2Json"),
            vec!["parse2json", "parse", "2", "json"]
        );
    }

    #[test]
    fn splits_acronym_digit_identifier() {
        assert_eq!(
            identifier_token_stream("BM25Score"),
            vec!["bm25score", "bm", "25", "score"]
        );
    }

    #[test]
    fn preserves_kebab_case_exact_form() {
        assert_eq!(
            identifier_token_stream("query-id"),
            vec!["query-id", "queryid", "query", "id"]
        );
    }
}
