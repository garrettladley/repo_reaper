use crate::index::DocumentField;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedField {
    pub field: DocumentField,
    pub text: String,
    pub span: Option<ByteSpan>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CodeIntelligence {
    fields: Vec<ExtractedField>,
}

impl CodeIntelligence {
    pub fn new(fields: Vec<ExtractedField>) -> Self {
        Self { fields }
    }

    pub fn fields(&self) -> &[ExtractedField] {
        &self.fields
    }
}

#[cfg(feature = "tree-sitter")]
pub use tree_sitter_impl::{CodeIntelligenceError, compile_language_query, extract};

#[cfg(not(feature = "tree-sitter"))]
#[derive(Debug, thiserror::Error)]
pub enum CodeIntelligenceError {
    #[error("tree-sitter support is not enabled")]
    Disabled,
}

#[cfg(not(feature = "tree-sitter"))]
pub fn extract(
    _file_type: crate::tokenizer::FileType,
    _path: &std::path::Path,
    _content: &str,
) -> Result<Option<CodeIntelligence>, CodeIntelligenceError> {
    Ok(None)
}

#[cfg(not(feature = "tree-sitter"))]
pub fn compile_language_query(
    _file_type: crate::tokenizer::FileType,
) -> Result<(), CodeIntelligenceError> {
    Err(CodeIntelligenceError::Disabled)
}

#[cfg(feature = "tree-sitter")]
mod tree_sitter_impl;
