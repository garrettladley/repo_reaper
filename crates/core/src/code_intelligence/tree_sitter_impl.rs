use std::{path::Path, str::Utf8Error};

use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

use super::{ByteSpan, CodeIntelligence, ExtractedField};
use crate::{index::DocumentField, tokenizer::FileType};

#[derive(Debug, thiserror::Error)]
pub enum CodeIntelligenceError {
    #[error("unsupported tree-sitter language: {0:?}")]
    UnsupportedLanguage(FileType),
    #[error("failed to configure tree-sitter parser for {file_type:?}: {source}")]
    ParserLanguage {
        file_type: FileType,
        source: tree_sitter::LanguageError,
    },
    #[error("failed to parse {file_type:?} source")]
    Parse { file_type: FileType },
    #[error("failed to compile {file_type:?} tree-sitter query: {source}")]
    Query {
        file_type: FileType,
        source: tree_sitter::QueryError,
    },
    #[error("tree-sitter captured invalid utf-8 for {file_type:?}: {source}")]
    CaptureText {
        file_type: FileType,
        source: Utf8Error,
    },
}

pub fn extract(
    file_type: FileType,
    path: &Path,
    content: &str,
) -> Result<Option<CodeIntelligence>, CodeIntelligenceError> {
    let Some(language) = language_for(file_type, path) else {
        return Ok(None);
    };
    let query_source =
        query_source_for(file_type).ok_or(CodeIntelligenceError::UnsupportedLanguage(file_type))?;
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|source| CodeIntelligenceError::ParserLanguage { file_type, source })?;

    let Some(tree) = parser.parse(content, None) else {
        return Err(CodeIntelligenceError::Parse { file_type });
    };
    let query = compile_query(file_type, &language, query_source)?;
    let mut fields = query_fields(file_type, &query, tree.root_node(), content.as_bytes())?;

    if file_type == FileType::Markdown {
        fields.extend(markdown_frontmatter(content));
    }

    Ok(Some(CodeIntelligence::new(fields)))
}

pub fn compile_language_query(file_type: FileType) -> Result<(), CodeIntelligenceError> {
    let Some(language) = language_for(file_type, Path::new("")) else {
        return Ok(());
    };
    let query_source =
        query_source_for(file_type).ok_or(CodeIntelligenceError::UnsupportedLanguage(file_type))?;
    compile_query(file_type, &language, query_source).map(|_| ())
}

fn compile_query(
    file_type: FileType,
    language: &Language,
    query_source: &str,
) -> Result<Query, CodeIntelligenceError> {
    Query::new(language, query_source)
        .map_err(|source| CodeIntelligenceError::Query { file_type, source })
}

fn query_fields(
    file_type: FileType,
    query: &Query,
    root: tree_sitter::Node<'_>,
    source: &[u8],
) -> Result<Vec<ExtractedField>, CodeIntelligenceError> {
    let capture_names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, root, source);
    let mut fields = Vec::new();

    while let Some(query_match) = matches.next() {
        for capture in query_match.captures {
            let name = capture_names[capture.index as usize];
            let Some(field) = field_for_capture(name) else {
                continue;
            };
            let text = capture
                .node
                .utf8_text(source)
                .map_err(|source| CodeIntelligenceError::CaptureText { file_type, source })?
                .trim()
                .to_string();
            if text.is_empty() {
                continue;
            }
            fields.push(ExtractedField {
                field,
                text,
                span: Some(ByteSpan {
                    start: capture.node.start_byte(),
                    end: capture.node.end_byte(),
                }),
            });
        }
    }

    Ok(fields)
}

fn field_for_capture(name: &str) -> Option<DocumentField> {
    if name.starts_with("symbol.") {
        return Some(DocumentField::Symbol);
    }

    match name {
        "import" => Some(DocumentField::Import),
        "comment" => Some(DocumentField::Comment),
        "string" => Some(DocumentField::StringLiteral),
        "frontmatter" => Some(DocumentField::Frontmatter),
        _ => None,
    }
}

fn language_for(file_type: FileType, path: &Path) -> Option<Language> {
    match file_type {
        FileType::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
        FileType::Python => Some(tree_sitter_python::LANGUAGE.into()),
        FileType::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
        FileType::TypeScript => {
            if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| matches!(extension, "tsx" | "jsx"))
            {
                Some(tree_sitter_typescript::LANGUAGE_TSX.into())
            } else {
                Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into())
            }
        }
        FileType::Go => Some(tree_sitter_go::LANGUAGE.into()),
        FileType::Markdown => Some(tree_sitter_md_025::LANGUAGE.into()),
        _ => None,
    }
}

fn query_source_for(file_type: FileType) -> Option<&'static str> {
    match file_type {
        FileType::Rust => Some(include_str!("queries/rust.scm")),
        FileType::Python => Some(include_str!("queries/python.scm")),
        FileType::JavaScript => Some(include_str!("queries/javascript.scm")),
        FileType::TypeScript => Some(include_str!("queries/typescript.scm")),
        FileType::Go => Some(include_str!("queries/go.scm")),
        FileType::Markdown => Some(include_str!("queries/markdown.scm")),
        _ => None,
    }
}

fn markdown_frontmatter(content: &str) -> Vec<ExtractedField> {
    let Some(rest) = content.strip_prefix("---\n") else {
        return Vec::new();
    };
    let Some(end) = rest.find("\n---") else {
        return Vec::new();
    };
    let start = 4;
    let end_byte = start + end;
    let text = content[start..end_byte].trim().to_string();
    if text.is_empty() {
        return Vec::new();
    }

    vec![ExtractedField {
        field: DocumentField::Frontmatter,
        text,
        span: Some(ByteSpan {
            start,
            end: end_byte,
        }),
    }]
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::extract;
    use crate::{index::DocumentField, tokenizer::FileType};

    fn texts_for(field: DocumentField, source: &[super::ExtractedField]) -> Vec<&str> {
        source
            .iter()
            .filter(|extracted| extracted.field == field)
            .map(|extracted| extracted.text.as_str())
            .collect()
    }

    #[test]
    fn rust_extraction_captures_core_code_fields() {
        let source = r#"
use std::collections::HashMap;

// score a document
pub struct SearchHit;
enum RankingAlgo { BM25 }
trait Scorer { fn score(&self); }
impl Scorer for SearchHit {
    fn score(&self) {
        let label = "BM25 score";
    }
}
"#;

        let extracted = extract(FileType::Rust, Path::new("lib.rs"), source)
            .unwrap()
            .unwrap();
        let fields = extracted.fields();
        let symbols = texts_for(DocumentField::Symbol, fields);
        let imports = texts_for(DocumentField::Import, fields);
        let comments = texts_for(DocumentField::Comment, fields);
        let strings = texts_for(DocumentField::StringLiteral, fields);

        assert!(symbols.contains(&"SearchHit"));
        assert!(symbols.contains(&"RankingAlgo"));
        assert!(symbols.contains(&"Scorer"));
        assert!(symbols.contains(&"score"));
        assert!(imports.iter().any(|text| text.contains("HashMap")));
        assert!(
            comments
                .iter()
                .any(|text| text.contains("score a document"))
        );
        assert!(strings.iter().any(|text| text.contains("BM25 score")));
        assert!(fields.iter().all(|field| field.span.is_some()));
    }

    #[test]
    fn unsupported_languages_have_no_parser_fields() {
        assert!(
            extract(FileType::Toml, Path::new("Cargo.toml"), "[package]")
                .unwrap()
                .is_none()
        );
    }
}
