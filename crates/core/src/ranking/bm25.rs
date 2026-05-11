use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, PostingList, RankedIndexReader, Term},
    query::{AnalyzedQuery, QueryTerm},
    ranking::{scorer::Scorer, utils::idf},
};

#[derive(Debug, Clone)]
pub struct BM25HyperParams {
    pub k1: f64,
    pub b: f64,
}

impl Default for BM25HyperParams {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

pub struct BM25 {
    pub hyper_params: BM25HyperParams,
}

impl Scorer for BM25 {
    fn score<I, P>(
        &self,
        index: &I,
        _: &AnalyzedQuery,
        _: &Term,
        query_term: QueryTerm,
        documents: &P,
        scores: &DashMap<DocId, f64>,
    ) where
        I: RankedIndexReader + Sync,
        P: PostingList + Sync,
    {
        let avgdl = index.avg_doc_length();
        let num_docs = index.num_docs();

        let idf = idf(num_docs, documents.len());

        documents
            .iter()
            .par_bridge()
            .for_each(|(doc_id, term_doc)| {
                let tf = term_doc.term_freq as f64;
                let doc_length = term_doc.length as f64;
                let score = self.score_term(query_term.weight, idf, tf, doc_length, avgdl);

                *scores.entry(doc_id).or_insert(0.0) += score;
            });
    }
}

impl BM25 {
    pub fn score_term(
        &self,
        query_weight: f64,
        idf: f64,
        tf: f64,
        doc_length: f64,
        avgdl: f64,
    ) -> f64 {
        if tf == 0.0 || avgdl == 0.0 {
            return 0.0;
        }

        query_weight * idf * (tf * (self.hyper_params.k1 + 1.0))
            / (tf
                + self.hyper_params.k1
                    * (1.0 - self.hyper_params.b + self.hyper_params.b * doc_length / avgdl))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use dashmap::DashMap;

    use super::{BM25, BM25HyperParams};
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

    fn bm25() -> BM25 {
        BM25 {
            hyper_params: BM25HyperParams { k1: 1.2, b: 0.75 },
        }
    }

    #[test]
    fn higher_tf_scores_higher() {
        let index = index_from(&[
            ("high.rs", &[("rust", 10), ("other", 1)]),
            ("low.rs", &[("rust", 1), ("other", 1)]),
        ]);
        let term = Term("rust".to_string());
        let query = AnalyzedQuery::from_frequencies("rust", HashMap::from([(term.clone(), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&term).unwrap();
        bm25().score(
            &index,
            &query,
            &term,
            *query.terms().next().unwrap().1,
            docs,
            &scores,
        );

        let high = score_for_path(&index, &scores, "high.rs");
        let low = score_for_path(&index, &scores, "low.rs");
        assert!(high > low, "higher tf should score higher: {high} vs {low}");
    }

    #[test]
    fn rarer_term_scores_higher_than_common() {
        let index = index_from(&[
            ("a.rs", &[("rare", 1), ("common", 1)]),
            ("b.rs", &[("common", 1)]),
            ("c.rs", &[("common", 1)]),
        ]);
        let query = AnalyzedQuery::from_frequencies(
            "rare common",
            HashMap::from([
                (Term("rare".to_string()), 1),
                (Term("common".to_string()), 1),
            ]),
        );

        let scores = DashMap::new();
        for (term, query_term) in query.terms() {
            if let Some(docs) = index.get_postings(term) {
                bm25().score(&index, &query, term, *query_term, docs, &scores);
            }
        }

        let a = score_for_path(&index, &scores, "a.rs");
        let b = score_for_path(&index, &scores, "b.rs");
        assert!(
            a > b,
            "doc matching rare term should score higher: a={a}, b={b}"
        );
    }

    #[test]
    fn scores_are_positive() {
        let index = index_from(&[("a.rs", &[("rust", 3)])]);
        let term = Term("rust".to_string());
        let query = AnalyzedQuery::from_frequencies("rust", HashMap::from([(term.clone(), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&term).unwrap();
        bm25().score(
            &index,
            &query,
            &term,
            *query.terms().next().unwrap().1,
            docs,
            &scores,
        );

        let score = score_for_path(&index, &scores, "a.rs");
        assert!(score > 0.0, "BM25 score should be positive, got {score}");
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
        bm25().score(
            &index,
            &low_query,
            &term,
            *low_query.terms().next().unwrap().1,
            docs,
            &low_scores,
        );

        let high_scores = DashMap::new();
        bm25().score(
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
