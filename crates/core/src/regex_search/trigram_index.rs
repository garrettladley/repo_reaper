use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

use walkdir::WalkDir;

use crate::index::{
    DocId,
    document_registry::{DocumentCatalog, DocumentMetadata, DocumentRegistry},
};

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Trigram(String);

impl Trigram {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for Trigram {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

#[derive(Debug)]
pub struct TrigramIndex {
    postings: HashMap<Trigram, BTreeSet<DocId>>,
    documents: DocumentRegistry,
    doc_ids_by_path: Vec<DocId>,
}

impl TrigramIndex {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let mut documents = DocumentRegistry::new();
        let mut postings: HashMap<Trigram, BTreeSet<DocId>> = HashMap::new();
        let mut indexed_files = candidate_files(root.as_ref());

        indexed_files.sort();

        for path in &indexed_files {
            let Ok(content) = fs::read_to_string(path) else {
                continue;
            };

            let doc_id = documents.insert_or_update(
                path.to_path_buf(),
                trigram_count(&content),
                content.len() as u64,
            );

            for trigram in trigrams(&content) {
                postings.entry(trigram).or_default().insert(doc_id);
            }
        }

        let mut doc_ids_by_path = documents_by_path(&documents, &indexed_files);
        doc_ids_by_path.sort();

        Self {
            postings,
            documents,
            doc_ids_by_path,
        }
    }

    pub fn postings(&self, trigram: &Trigram) -> Option<&BTreeSet<DocId>> {
        self.postings.get(trigram)
    }

    pub fn document(&self, doc_id: DocId) -> Option<&DocumentMetadata> {
        self.documents.get(doc_id)
    }

    pub fn doc_id(&self, path: &Path) -> Option<DocId> {
        self.documents.doc_id(path)
    }

    pub fn num_docs(&self) -> usize {
        self.documents.len()
    }

    pub fn candidates_for_literal(&self, literal: &str) -> BTreeSet<DocId> {
        let query_trigrams = trigrams(literal);
        if query_trigrams.is_empty() {
            return self.doc_ids_by_path.iter().copied().collect();
        }

        let mut candidates = match self.postings(&query_trigrams[0]) {
            Some(postings) => postings.clone(),
            None => return BTreeSet::new(),
        };

        for trigram in &query_trigrams[1..] {
            let Some(postings) = self.postings(trigram) else {
                return BTreeSet::new();
            };
            candidates = candidates.intersection(postings).copied().collect();
        }

        candidates
    }
}

pub fn trigrams(content: &str) -> Vec<Trigram> {
    let chars = content.chars().collect::<Vec<_>>();
    chars
        .windows(3)
        .map(|window| Trigram(window.iter().collect()))
        .collect()
}

fn trigram_count(content: &str) -> usize {
    content.chars().count().saturating_sub(2)
}

fn candidate_files(root: &Path) -> Vec<PathBuf> {
    WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.path().to_path_buf())
        .collect()
}

fn documents_by_path(documents: &DocumentRegistry, paths: &[PathBuf]) -> Vec<DocId> {
    paths
        .iter()
        .filter_map(|path| documents.doc_id(path))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{Trigram, TrigramIndex, trigrams};

    #[test]
    fn trigrams_are_overlapping_character_windows() {
        let trigrams = trigrams("abcd");

        let values = trigrams.iter().map(Trigram::as_str).collect::<Vec<_>>();
        assert_eq!(values, ["abc", "bcd"]);
    }

    #[test]
    fn trigrams_return_empty_for_short_strings() {
        assert!(trigrams("").is_empty());
        assert!(trigrams("a").is_empty());
        assert!(trigrams("ab").is_empty());
    }

    #[test]
    fn trigrams_include_punctuation_whitespace_and_preserve_case() {
        let trigrams = trigrams("A b!");

        let values = trigrams.iter().map(Trigram::as_str).collect::<Vec<_>>();
        assert_eq!(values, ["A b", " b!"]);
    }

    #[test]
    fn index_maps_every_indexed_doc_id_back_to_path() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("a.rs");
        let second = temp.path().join("b.rs");
        std::fs::write(&first, "abcdef").unwrap();
        std::fs::write(&second, "uvwxyz").unwrap();

        let index = TrigramIndex::new(temp.path());
        let first_id = index.doc_id(&first).unwrap();
        let second_id = index.doc_id(&second).unwrap();

        assert_eq!(index.num_docs(), 2);
        assert_eq!(index.document(first_id).unwrap().path, first);
        assert_eq!(index.document(second_id).unwrap().path, second);
    }

    #[test]
    fn postings_record_documents_containing_each_trigram() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("a.rs");
        let second = temp.path().join("b.rs");
        std::fs::write(&first, "abcdef").unwrap();
        std::fs::write(&second, "zabczz").unwrap();

        let index = TrigramIndex::new(temp.path());
        let postings = index.postings(&Trigram::from("abc")).unwrap();

        assert!(postings.contains(&index.doc_id(&first).unwrap()));
        assert!(postings.contains(&index.doc_id(&second).unwrap()));
    }

    #[test]
    fn literal_candidates_are_supersets_of_true_matches() {
        let temp = tempfile::tempdir().unwrap();
        let matching = temp.path().join("match.rs");
        let false_positive = temp.path().join("false_positive.rs");
        let missing = temp.path().join("missing.rs");
        std::fs::write(&matching, "xxabcdefxx").unwrap();
        std::fs::write(&false_positive, "abc bcd cde def").unwrap();
        std::fs::write(&missing, "abxde").unwrap();

        let index = TrigramIndex::new(temp.path());
        let candidates = index.candidates_for_literal("abcdef");

        assert!(candidates.contains(&index.doc_id(&matching).unwrap()));
        assert!(candidates.contains(&index.doc_id(&false_positive).unwrap()));
        assert!(!candidates.contains(&index.doc_id(&missing).unwrap()));
    }

    #[test]
    fn short_literal_candidates_include_all_indexed_documents() {
        let temp = tempfile::tempdir().unwrap();
        let first = temp.path().join("a.rs");
        let second = temp.path().join("b.rs");
        std::fs::write(&first, "a").unwrap();
        std::fs::write(&second, "b").unwrap();

        let index = TrigramIndex::new(temp.path());
        let candidates = index.candidates_for_literal("ab");

        assert_eq!(candidates.len(), 2);
        assert!(candidates.contains(&index.doc_id(&first).unwrap()));
        assert!(candidates.contains(&index.doc_id(&second).unwrap()));
    }
}
