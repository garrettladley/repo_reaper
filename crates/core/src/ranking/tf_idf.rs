use std::collections::HashMap;

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, InvertedIndex, TermDocument},
    query::Query,
    ranking::{scorer::Scorer, utils::idf},
};

pub struct TFIDF;

impl Scorer for TFIDF {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        _: &Query,
        documents: &HashMap<DocId, TermDocument>,
        scores: &DashMap<DocId, f64>,
    ) {
        let num_docs = inverted_index.num_docs();

        documents
            .iter()
            .par_bridge()
            .for_each(|(doc_id, term_doc)| {
                let tf = term_doc.term_freq as f64 / term_doc.length as f64;
                let idf = idf(num_docs, documents.len());

                *scores.entry(*doc_id).or_insert(0.0) += tf * idf;
            });
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use dashmap::DashMap;

    use super::TFIDF;
    use crate::{
        index::{DocId, InvertedIndex, Term},
        query::Query,
        ranking::scorer::Scorer,
    };

    fn index_from(docs: &[(&str, &[(&str, u32)])]) -> InvertedIndex {
        InvertedIndex::from_documents(docs)
    }

    fn score_for_path(index: &InvertedIndex, scores: &DashMap<DocId, f64>, path: &str) -> f64 {
        let doc_id = index.doc_id(PathBuf::from(path).as_path()).unwrap();
        *scores.get(&doc_id).unwrap()
    }

    #[test]
    fn higher_term_density_scores_higher() {
        // density = term_freq / doc_length
        let index = index_from(&[
            ("dense.rs", &[("rust", 5), ("other", 1)]),
            ("sparse.rs", &[("rust", 1), ("other", 10)]),
        ]);
        let query = Query(HashMap::from([(Term("rust".to_string()), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&Term("rust".to_string())).unwrap();
        TFIDF.score(&index, &query, docs, &scores);

        let dense = score_for_path(&index, &scores, "dense.rs");
        let sparse = score_for_path(&index, &scores, "sparse.rs");
        assert!(
            dense > sparse,
            "higher term density should score higher: dense={dense}, sparse={sparse}"
        );
    }

    #[test]
    fn scores_are_positive() {
        let index = index_from(&[("a.rs", &[("rust", 3)])]);
        let query = Query(HashMap::from([(Term("rust".to_string()), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&Term("rust".to_string())).unwrap();
        TFIDF.score(&index, &query, docs, &scores);

        let score = score_for_path(&index, &scores, "a.rs");
        assert!(score > 0.0, "TF-IDF score should be positive, got {score}");
    }
}
