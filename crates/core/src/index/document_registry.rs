use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DocId(u32);

impl DocId {
    pub fn from_u32(value: u32) -> Self {
        Self(value)
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentMetadata {
    pub id: DocId,
    pub path: PathBuf,
    pub token_length: usize,
    pub file_size_bytes: u64,
}

impl DocumentMetadata {
    fn new(id: DocId, path: PathBuf, token_length: usize, file_size_bytes: u64) -> Self {
        Self {
            id,
            path,
            token_length,
            file_size_bytes,
        }
    }
}

#[derive(Debug, Default)]
pub struct DocumentRegistry {
    documents: HashMap<DocId, DocumentMetadata>,
    path_to_id: HashMap<PathBuf, DocId>,
    next_id: u32,
    total_token_length: u64,
}

pub trait DocumentCatalog {
    fn insert_or_update(
        &mut self,
        path: PathBuf,
        token_length: usize,
        file_size_bytes: u64,
    ) -> DocId;
    fn remove(&mut self, path: &Path) -> Option<DocumentMetadata>;
    fn get(&self, id: DocId) -> Option<&DocumentMetadata>;
    fn get_by_path(&self, path: &Path) -> Option<&DocumentMetadata>;
    fn doc_id(&self, path: &Path) -> Option<DocId>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn total_token_length(&self) -> u64;
    fn avg_doc_length(&self) -> f64;
}

impl DocumentRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DocumentCatalog for DocumentRegistry {
    fn insert_or_update(
        &mut self,
        path: PathBuf,
        token_length: usize,
        file_size_bytes: u64,
    ) -> DocId {
        if let Some(&id) = self.path_to_id.get(&path) {
            if let Some(existing) = self.documents.get_mut(&id) {
                self.total_token_length -= existing.token_length as u64;
                self.total_token_length += token_length as u64;
                *existing = DocumentMetadata::new(id, path, token_length, file_size_bytes);
            }
            return id;
        }

        let id = DocId(self.next_id);
        self.next_id += 1;
        let metadata = DocumentMetadata::new(id, path, token_length, file_size_bytes);
        self.path_to_id.insert(metadata.path.clone(), id);
        self.total_token_length += token_length as u64;
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
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{DocumentCatalog, DocumentRegistry};

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
