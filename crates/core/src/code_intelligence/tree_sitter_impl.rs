use std::{path::Path, str::Utf8Error, sync::LazyLock};

use tree_sitter::{Language, Parser, Query, QueryCursor, StreamingIterator};

use super::{ByteSpan, CodeIntelligence, DocumentFeature};
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
    #[error("failed to compile {file_type:?} tree-sitter query: {message}")]
    Query {
        file_type: FileType,
        message: String,
    },
    #[error("tree-sitter captured invalid utf-8 for {file_type:?}: {source}")]
    CaptureText {
        file_type: FileType,
        source: Utf8Error,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QueryCompileError {
    file_type: FileType,
    message: String,
}

macro_rules! static_language_query {
    ($name:ident, $file_type:expr, $language:expr, $query_path:literal) => {
        static $name: LazyLock<Result<Query, QueryCompileError>> = LazyLock::new(|| {
            let language: Language = $language.into();
            Query::new(&language, include_str!($query_path)).map_err(|source| QueryCompileError {
                file_type: $file_type,
                message: source.to_string(),
            })
        });
    };
}

static_language_query!(
    RUST_QUERY,
    FileType::Rust,
    tree_sitter_rust::LANGUAGE,
    "queries/rust.scm"
);
static_language_query!(
    PYTHON_QUERY,
    FileType::Python,
    tree_sitter_python::LANGUAGE,
    "queries/python.scm"
);
static_language_query!(
    JAVASCRIPT_QUERY,
    FileType::JavaScript,
    tree_sitter_javascript::LANGUAGE,
    "queries/javascript.scm"
);
static_language_query!(
    TYPESCRIPT_QUERY,
    FileType::TypeScript,
    tree_sitter_typescript::LANGUAGE_TYPESCRIPT,
    "queries/typescript.scm"
);
static_language_query!(
    TSX_QUERY,
    FileType::TypeScript,
    tree_sitter_typescript::LANGUAGE_TSX,
    "queries/typescript.scm"
);
static_language_query!(
    GO_QUERY,
    FileType::Go,
    tree_sitter_go::LANGUAGE,
    "queries/go.scm"
);
static_language_query!(
    MARKDOWN_QUERY,
    FileType::Markdown,
    tree_sitter_md_025::LANGUAGE,
    "queries/markdown.scm"
);

pub fn extract(
    file_type: FileType,
    path: &Path,
    content: &str,
) -> Result<Option<CodeIntelligence>, CodeIntelligenceError> {
    let Some(language) = language_for(file_type, path) else {
        return Ok(None);
    };
    let Some(query) = query_for(file_type, path)? else {
        return Ok(None);
    };
    let mut parser = Parser::new();
    parser
        .set_language(&language)
        .map_err(|source| CodeIntelligenceError::ParserLanguage { file_type, source })?;

    let Some(tree) = parser.parse(content, None) else {
        return Err(CodeIntelligenceError::Parse { file_type });
    };
    let mut fields = query_fields(file_type, query, tree.root_node(), content.as_bytes())?;

    if file_type == FileType::Markdown {
        fields.extend(markdown_frontmatter(content));
    }

    Ok(Some(CodeIntelligence::new(fields)))
}

pub fn compile_language_query(file_type: FileType) -> Result<(), CodeIntelligenceError> {
    query_for(file_type, Path::new(""))?;
    Ok(())
}

fn query_fields(
    file_type: FileType,
    query: &Query,
    root: tree_sitter::Node<'_>,
    source: &[u8],
) -> Result<Vec<DocumentFeature>, CodeIntelligenceError> {
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
            fields.push(DocumentFeature {
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

fn query_for(
    file_type: FileType,
    path: &Path,
) -> Result<Option<&'static Query>, CodeIntelligenceError> {
    match file_type {
        FileType::Rust => cached_query(&RUST_QUERY).map(Some),
        FileType::Python => cached_query(&PYTHON_QUERY).map(Some),
        FileType::JavaScript => cached_query(&JAVASCRIPT_QUERY).map(Some),
        FileType::TypeScript => {
            if path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| matches!(extension, "tsx" | "jsx"))
            {
                cached_query(&TSX_QUERY).map(Some)
            } else {
                cached_query(&TYPESCRIPT_QUERY).map(Some)
            }
        }
        FileType::Go => cached_query(&GO_QUERY).map(Some),
        FileType::Markdown => cached_query(&MARKDOWN_QUERY).map(Some),
        _ => Ok(None),
    }
}

fn cached_query(
    query: &'static LazyLock<Result<Query, QueryCompileError>>,
) -> Result<&'static Query, CodeIntelligenceError> {
    query
        .as_ref()
        .map_err(|source| CodeIntelligenceError::Query {
            file_type: source.file_type,
            message: source.message.clone(),
        })
}

fn markdown_frontmatter(content: &str) -> Vec<DocumentFeature> {
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

    vec![DocumentFeature {
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

    use super::{compile_language_query, extract};
    use crate::{index::DocumentField, tokenizer::FileType};

    fn texts_for(field: DocumentField, source: &[super::DocumentFeature]) -> Vec<&str> {
        source
            .iter()
            .filter(|extracted| extracted.field == field)
            .map(|extracted| extracted.text.as_str())
            .collect()
    }

    #[test]
    fn language_queries_compile() {
        for file_type in [
            FileType::Rust,
            FileType::Python,
            FileType::JavaScript,
            FileType::TypeScript,
            FileType::Go,
            FileType::Markdown,
        ] {
            compile_language_query(file_type).unwrap();
        }
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
        let fields = extracted.features();
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
