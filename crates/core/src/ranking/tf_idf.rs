use std::collections::HashMap;

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, InvertedIndex, Term, TermDocument},
    query::{AnalyzedQuery, QueryTerm},
    ranking::{scorer::Scorer, utils::idf},
};

pub struct TFIDF;

impl Scorer for TFIDF {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        _: &AnalyzedQuery,
        _: &Term,
        query_term: QueryTerm,
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

                *scores.entry(*doc_id).or_insert(0.0) += query_term.weight * tf * idf;
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
        query::AnalyzedQuery,
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
        let term = Term("rust".to_string());
        let query = AnalyzedQuery::from_frequencies("rust", HashMap::from([(term.clone(), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&term).unwrap();
        TFIDF.score(
            &index,
            &query,
            &term,
            *query.terms().next().unwrap().1,
            docs,
            &scores,
        );

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
        let term = Term("rust".to_string());
        let query = AnalyzedQuery::from_frequencies("rust", HashMap::from([(term.clone(), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&term).unwrap();
        TFIDF.score(
            &index,
            &query,
            &term,
            *query.terms().next().unwrap().1,
            docs,
            &scores,
        );

        let score = score_for_path(&index, &scores, "a.rs");
        assert!(score > 0.0, "TF-IDF score should be positive, got {score}");
    }

    #[test]
    fn query_weight_scales_score() {
        let index = index_from(&[("a.rs", &[("rust", 3)])]);
        let term = Term("rust".to_string());
        let low_query = AnalyzedQuery::from_weights("rust", HashMap::from([(term.clone(), 1.0)]));
        let high_query =
            AnalyzedQuery::from_weights("rust^3", HashMap::from([(term.clone(), 3.0)]));
        let docs = index.get_postings(&term).unwrap();

        let low_scores = DashMap::new();
        TFIDF.score(
            &index,
            &low_query,
            &term,
            *low_query.terms().next().unwrap().1,
            docs,
            &low_scores,
        );

        let high_scores = DashMap::new();
        TFIDF.score(
            &index,
            &high_query,
            &term,
            *high_query.terms().next().unwrap().1,
            docs,
            &high_scores,
        );

        let low = score_for_path(&index, &low_scores, "a.rs");
        let high = score_for_path(&index, &high_scores, "a.rs");
        assert!(
            high > low,
            "higher query weight should increase score: {high} vs {low}"
        );
    }
}
