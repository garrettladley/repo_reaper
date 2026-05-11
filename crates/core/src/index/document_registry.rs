use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{
    code_intelligence::DocumentFeature,
    index::{DocumentField, StaticQualitySignals},
    tokenizer::FileType,
};

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct DocId(u32);

impl DocId {
    pub fn from_u32(value: u32) -> Self {
        Self(value)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FieldSpan {
    pub field: DocumentField,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentMetadataUpdate {
    pub path: PathBuf,
    pub token_length: usize,
    pub file_size_bytes: u64,
    pub file_type: FileType,
    pub field_lengths: HashMap<DocumentField, usize>,
    pub field_spans: Vec<FieldSpan>,
    pub features: Vec<DocumentFeature>,
    pub quality_signals: StaticQualitySignals,
}

impl DocumentMetadataUpdate {
    fn unknown_text(path: PathBuf, token_length: usize, file_size_bytes: u64) -> Self {
        let quality_signals = StaticQualitySignals::analyze(&path, "", file_size_bytes);
        Self {
            path,
            token_length,
            file_size_bytes,
            file_type: FileType::UnknownText,
            field_lengths: HashMap::new(),
            field_spans: Vec::new(),
            features: Vec::new(),
            quality_signals,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentMetadata {
    pub id: DocId,
    pub path: PathBuf,
    pub token_length: usize,
    pub file_size_bytes: u64,
    pub file_type: FileType,
    pub field_lengths: HashMap<DocumentField, usize>,
    #[serde(default)]
    pub field_spans: Vec<FieldSpan>,
    #[serde(default)]
    pub features: Vec<DocumentFeature>,
    #[serde(default)]
    pub quality_signals: StaticQualitySignals,
}

impl DocumentMetadata {
    fn from_update(id: DocId, update: DocumentMetadataUpdate) -> Self {
        Self {
            id,
            path: update.path,
            token_length: update.token_length,
            file_size_bytes: update.file_size_bytes,
            file_type: update.file_type,
            field_lengths: update.field_lengths,
            field_spans: update.field_spans,
            features: update.features,
            quality_signals: update.quality_signals,
        }
    }

    pub fn into_update(self) -> DocumentMetadataUpdate {
        DocumentMetadataUpdate {
            path: self.path,
            token_length: self.token_length,
            file_size_bytes: self.file_size_bytes,
            file_type: self.file_type,
            field_lengths: self.field_lengths,
            field_spans: self.field_spans,
            features: self.features,
            quality_signals: self.quality_signals,
        }
    }

    pub fn field_length(&self, field: DocumentField) -> usize {
        self.field_lengths.get(&field).copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DocumentRegistry {
    documents: HashMap<DocId, DocumentMetadata>,
    path_to_id: HashMap<PathBuf, DocId>,
    next_id: u32,
    total_token_length: u64,
}

pub trait DocumentCatalog {
    fn insert_or_update_record(&mut self, update: DocumentMetadataUpdate) -> DocId;

    fn insert_or_update(
        &mut self,
        path: PathBuf,
        token_length: usize,
        file_size_bytes: u64,
    ) -> DocId {
        self.insert_or_update_record(DocumentMetadataUpdate::unknown_text(
            path,
            token_length,
            file_size_bytes,
        ))
    }

    fn insert_or_update_with_fields(
        &mut self,
        path: PathBuf,
        token_length: usize,
        file_size_bytes: u64,
        file_type: FileType,
        field_lengths: HashMap<DocumentField, usize>,
        quality_signals: StaticQualitySignals,
    ) -> DocId {
        self.insert_or_update_record(DocumentMetadataUpdate {
            path,
            token_length,
            file_size_bytes,
            file_type,
            field_lengths,
            field_spans: Vec::new(),
            features: Vec::new(),
            quality_signals,
        })
    }

    fn insert_or_update_with_features(&mut self, update: DocumentMetadataUpdate) -> DocId {
        self.insert_or_update_record(update)
    }

    fn remove(&mut self, path: &Path) -> Option<DocumentMetadata>;
    fn get(&self, id: DocId) -> Option<&DocumentMetadata>;
    fn iter(&self) -> Box<dyn Iterator<Item = &DocumentMetadata> + '_>;
    fn get_by_path(&self, path: &Path) -> Option<&DocumentMetadata>;
    fn doc_id(&self, path: &Path) -> Option<DocId>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn total_token_length(&self) -> u64;
    fn avg_doc_length(&self) -> f64;
    fn avg_field_length(&self, field: DocumentField) -> f64;
}

impl DocumentRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn documents_iter(&self) -> impl Iterator<Item = (&DocId, &DocumentMetadata)> {
        self.documents.iter()
    }
}

impl DocumentCatalog for DocumentRegistry {
    fn insert_or_update_record(&mut self, update: DocumentMetadataUpdate) -> DocId {
        if let Some(&id) = self.path_to_id.get(&update.path) {
            if let Some(existing) = self.documents.get_mut(&id) {
                self.total_token_length -= existing.token_length as u64;
                self.total_token_length += update.token_length as u64;
                *existing = DocumentMetadata::from_update(id, update);
            }
            return id;
        }

        let id = DocId(self.next_id);
        self.next_id += 1;
        let metadata = DocumentMetadata::from_update(id, update);
        self.path_to_id.insert(metadata.path.clone(), id);
        self.total_token_length += metadata.token_length as u64;
        self.documents.insert(id, metadata);
        id
    }

    fn remove(&mut self, path: &Path) -> Option<DocumentMetadata> {
        let id = self.path_to_id.remove(path)?;
        let metadata = self.documents.remove(&id)?;
        self.total_token_length -= metadata.token_length as u64;
        Some(metadata)
    }

    fn get(&self, id: DocId) -> Option<&DocumentMetadata> {
        self.documents.get(&id)
    }

    fn iter(&self) -> Box<dyn Iterator<Item = &DocumentMetadata> + '_> {
        Box::new(self.documents.values())
    }

    fn get_by_path(&self, path: &Path) -> Option<&DocumentMetadata> {
        self.path_to_id
            .get(path)
            .and_then(|&id| self.documents.get(&id))
    }

    fn doc_id(&self, path: &Path) -> Option<DocId> {
        self.path_to_id.get(path).copied()
    }

    fn len(&self) -> usize {
        self.documents.len()
    }

    fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    fn total_token_length(&self) -> u64 {
        self.total_token_length
    }

    fn avg_doc_length(&self) -> f64 {
        if self.documents.is_empty() {
            return 0.0;
        }

        self.total_token_length as f64 / self.documents.len() as f64
    }

    fn avg_field_length(&self, field: DocumentField) -> f64 {
        if self.documents.is_empty() {
            return 0.0;
        }

        let total = self
            .documents
            .values()
            .map(|metadata| metadata.field_length(field))
            .sum::<usize>();

        total as f64 / self.documents.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use super::{DocumentCatalog, DocumentMetadataUpdate, DocumentRegistry, FieldSpan};
    use crate::{
        code_intelligence::{ByteSpan, DocumentFeature},
        index::{DocumentField, StaticQualitySignals},
        tokenizer::FileType,
    };

    #[test]
    fn insert_assigns_compact_document_ids() {
        let mut registry = DocumentRegistry::new();

        let first = registry.insert_or_update(PathBuf::from("a.rs"), 3, 10);
        let second = registry.insert_or_update(PathBuf::from("b.rs"), 5, 20);

        assert_eq!(first.as_u32(), 0);
        assert_eq!(second.as_u32(), 1);
    }

    #[test]
    fn update_preserves_document_id_and_replaces_metadata() {
        let mut registry = DocumentRegistry::new();
        let path = PathBuf::from("a.rs");
        let first = registry.insert_or_update(path.clone(), 3, 10);

        let second = registry.insert_or_update(path.clone(), 7, 30);

        let metadata = registry.get_by_path(&path).unwrap();
        assert_eq!(second, first);
        assert_eq!(metadata.token_length, 7);
        assert_eq!(metadata.file_size_bytes, 30);
        assert_eq!(registry.avg_doc_length(), 7.0);
    }

    #[test]
    fn insert_record_preserves_exportable_features() {
        let mut registry = DocumentRegistry::new();
        let path = PathBuf::from("a.rs");
        let feature = DocumentFeature {
            field: DocumentField::Symbol,
            text: "score_bm25".to_string(),
            span: Some(ByteSpan { start: 3, end: 13 }),
        };
        let span = FieldSpan {
            field: DocumentField::Symbol,
            start_byte: 3,
            end_byte: 13,
        };

        registry.insert_or_update_with_features(DocumentMetadataUpdate {
            path: path.clone(),
            token_length: 7,
            file_size_bytes: 30,
            file_type: FileType::Rust,
            field_lengths: HashMap::new(),
            field_spans: vec![span.clone()],
            features: vec![feature.clone()],
            quality_signals: StaticQualitySignals::default(),
        });

        let metadata = registry.get_by_path(&path).unwrap();
        assert_eq!(metadata.field_spans, vec![span]);
        assert_eq!(metadata.features, vec![feature]);
    }

    #[test]
    fn remove_deletes_metadata_and_updates_stats() {
        let mut registry = DocumentRegistry::new();
        registry.insert_or_update(PathBuf::from("a.rs"), 3, 10);
        registry.insert_or_update(PathBuf::from("b.rs"), 5, 20);

        let removed = registry.remove(PathBuf::from("a.rs").as_path()).unwrap();

        assert_eq!(removed.path, PathBuf::from("a.rs"));
        assert!(
            registry
                .get_by_path(PathBuf::from("a.rs").as_path())
                .is_none()
        );
        assert_eq!(registry.len(), 1);
        assert_eq!(registry.avg_doc_length(), 5.0);
    }
}
