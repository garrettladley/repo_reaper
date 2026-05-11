use crate::{index::DocumentField, tokenizer::FileType};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ByteSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentFeature {
    pub field: DocumentField,
    pub text: String,
    pub span: Option<ByteSpan>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentFeatures {
    pub file_type: FileType,
    features: Vec<DocumentFeature>,
}

impl DocumentFeatures {
    pub fn new(file_type: FileType, features: Vec<DocumentFeature>) -> Self {
        Self {
            file_type,
            features,
        }
    }

    pub fn features(&self) -> &[DocumentFeature] {
        &self.features
    }

    pub fn into_features(self) -> Vec<DocumentFeature> {
        self.features
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CodeIntelligence {
    features: Vec<DocumentFeature>,
}

impl CodeIntelligence {
    pub fn new(features: Vec<DocumentFeature>) -> Self {
        Self { features }
    }

    pub fn features(&self) -> &[DocumentFeature] {
        &self.features
    }

    pub fn into_features(self) -> Vec<DocumentFeature> {
        self.features
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
    _file_type: FileType,
    _path: &std::path::Path,
    _content: &str,
) -> Result<Option<CodeIntelligence>, CodeIntelligenceError> {
    Ok(None)
}

#[cfg(not(feature = "tree-sitter"))]
pub fn compile_language_query(_file_type: FileType) -> Result<(), CodeIntelligenceError> {
    Err(CodeIntelligenceError::Disabled)
}

#[cfg(feature = "tree-sitter")]
mod tree_sitter_impl;

#[cfg(test)]
mod tests {
    use super::{ByteSpan, DocumentFeature, DocumentFeatures};
    use crate::{index::DocumentField, tokenizer::FileType};

    #[test]
    fn document_features_round_trip_through_json() {
        let features = DocumentFeatures::new(
            FileType::Rust,
            vec![DocumentFeature {
                field: DocumentField::Symbol,
                text: "score_bm25".to_string(),
                span: Some(ByteSpan { start: 3, end: 13 }),
            }],
        );

        let encoded = serde_json::to_string(&features).unwrap();
        let decoded: DocumentFeatures = serde_json::from_str(&encoded).unwrap();

        assert_eq!(decoded, features);
    }
}
