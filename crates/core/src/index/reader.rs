use std::{collections::BTreeMap, path::Path};

use crate::index::{DocId, DocumentField, DocumentMetadata, InvertedIndex, Term, TermDocument};

pub trait PostingList {
    type Iter<'a>: Iterator<Item = (DocId, &'a TermDocument)> + Send
    where
        Self: 'a;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn get(&self, doc_id: DocId) -> Option<&TermDocument>;
    fn iter(&self) -> Self::Iter<'_>;
}

pub struct BTreePostingIter<'a>(std::collections::btree_map::Iter<'a, DocId, TermDocument>);

impl<'a> Iterator for BTreePostingIter<'a> {
    type Item = (DocId, &'a TermDocument);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(doc_id, term_doc)| (*doc_id, term_doc))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedPostingList {
    postings: Vec<(DocId, TermDocument)>,
}

impl OwnedPostingList {
    pub fn new(mut postings: Vec<(DocId, TermDocument)>) -> Self {
        postings.sort_by_key(|(doc_id, _)| *doc_id);
        Self { postings }
    }
}

pub struct OwnedPostingIter<'a>(std::slice::Iter<'a, (DocId, TermDocument)>);

impl<'a> Iterator for OwnedPostingIter<'a> {
    type Item = (DocId, &'a TermDocument);

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(doc_id, term_doc)| (*doc_id, term_doc))
    }
}

impl PostingList for OwnedPostingList {
    type Iter<'a>
        = OwnedPostingIter<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        self.postings.len()
    }

    fn get(&self, doc_id: DocId) -> Option<&TermDocument> {
        self.postings
            .binary_search_by_key(&doc_id, |(candidate, _)| *candidate)
            .ok()
            .map(|idx| &self.postings[idx].1)
    }

    fn iter(&self) -> Self::Iter<'_> {
        OwnedPostingIter(self.postings.iter())
    }
}

impl PostingList for BTreeMap<DocId, TermDocument> {
    type Iter<'a>
        = BTreePostingIter<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        BTreeMap::len(self)
    }

    fn get(&self, doc_id: DocId) -> Option<&TermDocument> {
        BTreeMap::get(self, &doc_id)
    }

    fn iter(&self) -> Self::Iter<'_> {
        BTreePostingIter(BTreeMap::iter(self))
    }
}

impl PostingList for &BTreeMap<DocId, TermDocument> {
    type Iter<'a>
        = BTreePostingIter<'a>
    where
        Self: 'a;

    fn len(&self) -> usize {
        BTreeMap::len(self)
    }

    fn get(&self, doc_id: DocId) -> Option<&TermDocument> {
        BTreeMap::get(self, &doc_id)
    }

    fn iter(&self) -> Self::Iter<'_> {
        BTreePostingIter(BTreeMap::iter(self))
    }
}

pub trait RankedIndexReader {
    type Postings<'a>: PostingList + Sync
    where
        Self: 'a;

    fn postings(&self, term: &Term) -> Option<Self::Postings<'_>>;
    fn document(&self, id: DocId) -> Option<&DocumentMetadata>;
    fn doc_id(&self, path: &Path) -> Option<DocId>;
    fn document_norm(&self, doc_id: DocId) -> Option<f64>;
    fn num_docs(&self) -> usize;
    fn avg_doc_length(&self) -> f64;
    fn avg_field_length(&self, field: DocumentField) -> f64;
    fn doc_freq(&self, term: &Term) -> usize;
}

impl RankedIndexReader for InvertedIndex {
    type Postings<'a> = &'a BTreeMap<DocId, TermDocument>;

    fn postings(&self, term: &Term) -> Option<Self::Postings<'_>> {
        InvertedIndex::get_postings(self, term)
    }

    fn document(&self, id: DocId) -> Option<&DocumentMetadata> {
        InvertedIndex::document(self, id)
    }

    fn doc_id(&self, path: &Path) -> Option<DocId> {
        InvertedIndex::doc_id(self, path)
    }

    fn document_norm(&self, doc_id: DocId) -> Option<f64> {
        InvertedIndex::document_norm(self, doc_id)
    }

    fn num_docs(&self) -> usize {
        InvertedIndex::num_docs(self)
    }

    fn avg_doc_length(&self) -> f64 {
        InvertedIndex::avg_doc_length(self)
    }

    fn avg_field_length(&self, field: DocumentField) -> f64 {
        InvertedIndex::avg_field_length(self, field)
    }

    fn doc_freq(&self, term: &Term) -> usize {
        InvertedIndex::doc_freq(self, term)
    }
}

#[cfg(test)]
mod tests {
    use crate::index::{
        DocId, InvertedIndex, OwnedPostingList, PostingList, RankedIndexReader, Term, TermDocument,
    };

    #[test]
    fn in_memory_index_implements_ranked_reader_postings() {
        let index =
            InvertedIndex::from_documents(&[("a.rs", &[("rust", 2)]), ("b.rs", &[("rust", 1)])]);
        let postings = index.postings(&Term("rust".to_string())).unwrap();
        let doc_ids = PostingList::iter(&postings)
            .map(|(doc_id, _)| doc_id)
            .collect::<Vec<_>>();

        assert_eq!(doc_ids, vec![DocId::from_u32(0), DocId::from_u32(1)]);
    }

    #[test]
    fn owned_posting_list_supports_sorted_lookup_and_iteration() {
        let postings = OwnedPostingList::new(vec![
            (DocId::from_u32(9), TermDocument::unfielded(3, 1)),
            (DocId::from_u32(2), TermDocument::unfielded(5, 2)),
        ]);
        let doc_ids = postings
            .iter()
            .map(|(doc_id, _)| doc_id)
            .collect::<Vec<_>>();

        assert_eq!(doc_ids, vec![DocId::from_u32(2), DocId::from_u32(9)]);
        assert_eq!(postings.get(DocId::from_u32(2)).unwrap().term_freq, 2);
    }
}
