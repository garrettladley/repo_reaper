use std::{collections::HashMap, path::PathBuf};

use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

use crate::{
    index::{InvertedIndex, TermDocument},
    query::Query,
    ranking::{scorer::Scorer, utils::idf},
};

pub struct CosineSimilarity;

impl Scorer for CosineSimilarity {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        query: &Query,
        documents: &HashMap<PathBuf, TermDocument>,
        scores: &DashMap<PathBuf, f64>,
    ) {
        let num_docs = inverted_index.num_docs();

        let query_magnitude = query
            .0
            .par_iter()
            .map(|(term, _)| {
                let idf = idf(num_docs, inverted_index.doc_freq(term));
                idf * idf
            })
            .sum::<f64>()
            .sqrt();

        let term_idf = idf(num_docs, documents.len());

        documents
            .iter()
            .par_bridge()
            .for_each(|(doc_path, term_doc)| {
                let tf = term_doc.term_freq as f64;
                let dot_product = tf * term_idf * term_idf;

                let doc_magnitude = inverted_index
                    .postings_iter()
                    .par_bridge()
                    .filter_map(|(_, doc_map)| {
                        doc_map.get(doc_path).map(|td| {
                            let tf = td.term_freq as f64;
                            let doc_term_idf = idf(num_docs, doc_map.len());
                            let w = tf * doc_term_idf;
                            w * w
                        })
                    })
                    .sum::<f64>()
                    .sqrt();

                let cosine_similarity = dot_product / (query_magnitude * doc_magnitude);

                *scores.entry(doc_path.to_owned()).or_insert(0.0) += cosine_similarity;
            });
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use dashmap::DashMap;

    use super::CosineSimilarity;
    use crate::{
        index::{InvertedIndex, Term, TermDocument},
        query::Query,
        ranking::{scorer::Scorer, utils::idf},
    };

    fn index_from(docs: &[(&str, &[(&str, u32)])]) -> InvertedIndex {
        let mut postings: HashMap<Term, HashMap<PathBuf, TermDocument>> = HashMap::new();
        for &(path, terms) in docs {
            let total_len: usize = terms.iter().map(|(_, c)| *c as usize).sum();
            for &(term, freq) in terms {
                postings.entry(Term(term.to_string())).or_default().insert(
                    PathBuf::from(path),
                    TermDocument {
                        length: total_len,
                        term_freq: freq as usize,
                    },
                );
            }
        }
        InvertedIndex::from_postings(postings)
    }

    fn score_query(index: &InvertedIndex, query: &Query) -> HashMap<PathBuf, f64> {
        let scores = DashMap::new();
        for term in query.0.keys() {
            if let Some(documents) = index.get_postings(term) {
                CosineSimilarity.score(index, query, documents, &scores);
            }
        }
        scores.into_iter().collect()
    }

    #[test]
    fn doc_magnitude_uses_tfidf_weights_not_raw_tf() {
        let index = index_from(&[
            ("a.rs", &[("rare", 1), ("common", 1)]),
            ("b.rs", &[("common", 1)]),
        ]);
        let query = Query(HashMap::from([(Term("rare".to_string()), 1)]));

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
        let query = Query(HashMap::from([(Term("rust".to_string()), 1)]));

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
        let query = Query(HashMap::from([(Term("rust".to_string()), 1)]));

        for (path, s) in score_query(&index, &query) {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&s),
                "cosine must be in [0, 1], got {s} for {}",
                path.display()
            );
        }
    }
}
