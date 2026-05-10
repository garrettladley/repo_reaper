use std::{
    collections::{BTreeSet, HashMap},
    ops::Range,
    path::Path,
};

use super::{
    FileSystemCorpus, LiteralSearchResult, RegexCorpus, RegexSearchMatch, Trigram,
    line_range_for_match,
};
use crate::index::{
    DocId,
    document_registry::{DocumentCatalog, DocumentMetadata, DocumentRegistry},
};

#[derive(Debug)]
pub struct TrigramIndex<C = FileSystemCorpus> {
    corpus: C,
    postings: HashMap<Trigram, BTreeSet<DocId>>,
    documents: DocumentRegistry,
    doc_ids_by_path: Vec<DocId>,
}

impl TrigramIndex<FileSystemCorpus> {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self::with_corpus(FileSystemCorpus::new(root.as_ref()))
    }
}

impl<C> TrigramIndex<C>
where
    C: RegexCorpus,
{
    pub fn with_corpus(corpus: C) -> Self {
        let mut documents = DocumentRegistry::new();
        let mut postings: HashMap<Trigram, BTreeSet<DocId>> = HashMap::new();
        let mut corpus_documents = corpus.documents();
        corpus_documents.sort_by(|left, right| left.path.cmp(&right.path));

        for document in &corpus_documents {
            let doc_id = documents.insert_or_update(
                document.path.clone(),
                trigram_count(&document.content),
                document.content.len() as u64,
            );

            for trigram in trigrams(&document.content) {
                postings.entry(trigram).or_default().insert(doc_id);
            }
        }

        let mut doc_ids_by_path = corpus_documents
            .iter()
            .filter_map(|document| documents.doc_id(&document.path))
            .collect::<Vec<_>>();
        doc_ids_by_path.sort();

        Self {
            corpus,
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

    pub fn search_literal(&self, literal: &str) -> LiteralSearchResult {
        if literal.is_empty() {
            return LiteralSearchResult {
                candidate_count: 0,
                matches: Vec::new(),
            };
        }

        let candidate_ids = self.candidates_for_literal(literal);
        let candidate_count = candidate_ids.len();
        let mut candidate_paths = candidate_ids
            .iter()
            .filter_map(|&doc_id| self.document(doc_id).map(|document| document.path.clone()))
            .collect::<Vec<_>>();
        candidate_paths.sort();

        let mut matches = Vec::new();
        for path in candidate_paths {
            let Some(content) = self.corpus.read_document(&path) else {
                continue;
            };

            matches.extend(verified_literal_matches(&path, &content, literal));
        }

        LiteralSearchResult {
            candidate_count,
            matches,
        }
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

fn verified_literal_matches(path: &Path, content: &str, literal: &str) -> Vec<RegexSearchMatch> {
    content
        .match_indices(literal)
        .map(|(start, matched_text)| {
            literal_match(path, content, start..start + matched_text.len())
        })
        .collect()
}

fn literal_match(path: &Path, content: &str, byte_range: Range<usize>) -> RegexSearchMatch {
    RegexSearchMatch {
        path: path.to_path_buf(),
        line_range: line_range_for_match(content, byte_range.clone()),
        matched_text: content[byte_range.clone()].to_string(),
        byte_range,
    }
}
