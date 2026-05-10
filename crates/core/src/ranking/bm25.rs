use std::collections::HashMap;

use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    error::RankingError,
    index::{DocId, InvertedIndex, TermDocument},
    query::Query,
    ranking::{scorer::Scorer, utils::idf},
};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct BM25HyperParams {
    pub k1: f64,
    pub b: f64,
}

pub fn get_configuration() -> Result<BM25HyperParams, RankingError> {
    let base_path = std::env::current_dir().map_err(RankingError::CurrentDir)?;
    let configuration_directory = base_path.join("configuration");

    let settings = config::Config::builder()
        .add_source(config::File::from(configuration_directory.join("bm25.yml")))
        .build()?;

    Ok(settings.try_deserialize::<BM25HyperParams>()?)
}

pub struct BM25 {
    pub hyper_params: BM25HyperParams,
}

impl Scorer for BM25 {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        _: &Query,
        documents: &HashMap<DocId, TermDocument>,
        scores: &DashMap<DocId, f64>,
    ) {
        let avgdl = inverted_index.avg_doc_length();
        let num_docs = inverted_index.num_docs();

        let idf = idf(num_docs, documents.len());

        documents.par_iter().for_each(|(doc_id, term_doc)| {
            let tf = term_doc.term_freq as f64;
            let doc_length = term_doc.length as f64;
            let score = idf * (tf * (self.hyper_params.k1 + 1.0))
                / (tf
                    + self.hyper_params.k1
                        * (1.0 - self.hyper_params.b + self.hyper_params.b * doc_length / avgdl));

            *scores.entry(*doc_id).or_insert(0.0) += score;
        });
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::PathBuf};

    use dashmap::DashMap;

    use super::{BM25, BM25HyperParams};
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
        let query = Query(HashMap::from([(Term("rust".to_string()), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&Term("rust".to_string())).unwrap();
        bm25().score(&index, &query, docs, &scores);

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
        let query = Query(HashMap::from([
            (Term("rare".to_string()), 1),
            (Term("common".to_string()), 1),
        ]));

        let scores = DashMap::new();
        for term in query.0.keys() {
            if let Some(docs) = index.get_postings(term) {
                bm25().score(&index, &query, docs, &scores);
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
        let query = Query(HashMap::from([(Term("rust".to_string()), 1)]));
        let scores = DashMap::new();

        let docs = index.get_postings(&Term("rust".to_string())).unwrap();
        bm25().score(&index, &query, docs, &scores);

        let score = score_for_path(&index, &scores, "a.rs");
        assert!(score > 0.0, "BM25 score should be positive, got {score}");
    }
}
