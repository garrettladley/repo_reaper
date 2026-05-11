use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, PostingList, RankedIndexReader, Term},
    query::{AnalyzedQuery, QueryTerm},
    ranking::{scorer::Scorer, utils::idf},
};

pub struct CosineSimilarity;

impl Scorer for CosineSimilarity {
    fn score<I, P>(
        &self,
        index: &I,
        query: &AnalyzedQuery,
        _: &Term,
        query_term: QueryTerm,
        documents: &P,
        scores: &DashMap<DocId, f64>,
    ) where
        I: RankedIndexReader + Sync,
        P: PostingList + Sync,
    {
        let num_docs = index.num_docs();

        let query_magnitude = query
            .terms()
            .par_bridge()
            .map(|(term, query_term)| {
                let idf = idf(num_docs, index.doc_freq(term));
                let weight = query_term.weight * idf;
                weight * weight
            })
            .sum::<f64>()
            .sqrt();

        let term_idf = idf(num_docs, documents.len());

        documents
            .iter()
            .par_bridge()
            .for_each(|(doc_id, term_doc)| {
                let tf = term_doc.term_freq as f64;
                let dot_product = query_term.weight * tf * term_idf * term_idf;

                let Some(doc_magnitude) = index.document_norm(doc_id) else {
                    return;
                };

                if query_magnitude == 0.0 || doc_magnitude == 0.0 {
                    return;
                }

                let cosine_similarity = dot_product / (query_magnitude * doc_magnitude);

                *scores.entry(doc_id).or_insert(0.0) += cosine_similarity;
            });
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use dashmap::DashMap;

    use super::CosineSimilarity;
    use crate::{
        index::{InvertedIndex, Term},
        query::AnalyzedQuery,
        ranking::{scorer::Scorer, utils::idf},
    };

    fn index_from(docs: &[(&str, &[(&str, u32)])]) -> InvertedIndex {
        InvertedIndex::from_documents(docs)
    }

    fn score_query(index: &InvertedIndex, query: &AnalyzedQuery) -> HashMap<PathBuf, f64> {
        let scores = DashMap::new();
        for (term, query_term) in query.terms() {
            if let Some(documents) = index.get_postings(term) {
                CosineSimilarity.score(index, query, term, *query_term, documents, &scores);
            }
        }
        scores
            .into_iter()
            .filter_map(|(doc_id, score)| {
                index
                    .document(doc_id)
                    .map(|metadata| (metadata.path.clone(), score))
            })
            .collect()
    }

    #[test]
    fn doc_magnitude_uses_tfidf_weights_not_raw_tf() {
        let index = index_from(&[
            ("a.rs", &[("rare", 1), ("common", 1)]),
            ("b.rs", &[("common", 1)]),
        ]);
        let query =
            AnalyzedQuery::from_frequencies("rare", HashMap::from([(Term("rare".to_string()), 1)]));

        let scores = score_query(&index, &query);
        let score_a = scores[&PathBuf::from("a.rs")];

        let num_docs = index.num_docs();
        let idf_rare = idf(num_docs, 1);
        let idf_common = idf(num_docs, 2);
        let expected = idf_rare / (idf_rare * idf_rare + idf_common * idf_common).sqrt();

        assert!(
            (score_a - expected).abs() < 1e-10,
            "expected {expected}, got {score_a}"
        );
    }

    #[test]
    fn single_term_query_and_doc_yields_one() {
        let index = index_from(&[("a.rs", &[("rust", 3)])]);
        let query =
            AnalyzedQuery::from_frequencies("rust", HashMap::from([(Term("rust".to_string()), 1)]));

        let scores = score_query(&index, &query);

        assert!(
            (scores[&PathBuf::from("a.rs")] - 1.0).abs() < 1e-10,
            "single-term cosine should be 1.0"
        );
    }

    #[test]
    fn scores_bounded_between_zero_and_one() {
        let index = index_from(&[
            ("a.rs", &[("rust", 5), ("code", 2)]),
            ("b.rs", &[("rust", 1), ("java", 3)]),
        ]);
        let query =
            AnalyzedQuery::from_frequencies("rust", HashMap::from([(Term("rust".to_string()), 1)]));

        for (path, s) in score_query(&index, &query) {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&s),
                "cosine must be in [0, 1], got {s} for {}",
                path.display()
            );
        }
    }

    #[test]
    fn repeated_query_term_shifts_similarity_toward_matching_document() {
        let index = index_from(&[
            ("rust.rs", &[("rust", 2), ("code", 1)]),
            ("code.rs", &[("rust", 1), ("code", 2)]),
        ]);

        let balanced = AnalyzedQuery::from_frequencies(
            "rust code",
            HashMap::from([(Term("rust".to_string()), 1), (Term("code".to_string()), 1)]),
        );
        let rust_weighted = AnalyzedQuery::from_frequencies(
            "rust rust code",
            HashMap::from([(Term("rust".to_string()), 2), (Term("code".to_string()), 1)]),
        );

        let balanced_scores = score_query(&index, &balanced);
        let weighted_scores = score_query(&index, &rust_weighted);
        let rust_path = PathBuf::from("rust.rs");
        let code_path = PathBuf::from("code.rs");

        let balanced_gap = balanced_scores[&rust_path] - balanced_scores[&code_path];
        let weighted_gap = weighted_scores[&rust_path] - weighted_scores[&code_path];

        assert!(
            weighted_gap > balanced_gap,
            "repeated query term should shift cosine score toward matching doc: {weighted_gap} vs {balanced_gap}"
        );
    }
}
