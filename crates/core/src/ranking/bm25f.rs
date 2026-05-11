use std::collections::HashMap;

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    error::RankingError,
    index::{DocId, DocumentField, PostingList, RankedIndexReader, Term, TermDocument},
    query::{AnalyzedQuery, QueryIntent, QueryTerm},
    ranking::{scorer::Scorer, utils::idf},
};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct BM25FHyperParams {
    pub k1: f64,
    pub b: f64,
    pub field_weights: HashMap<DocumentField, f64>,
}

impl BM25FHyperParams {
    pub fn code_search_defaults() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            field_weights: HashMap::from([
                (DocumentField::FileName, 8.0),
                (DocumentField::RelativePath, 4.0),
                (DocumentField::Extension, 0.25),
                (DocumentField::Content, 1.0),
                (DocumentField::Identifier, 3.0),
                (DocumentField::Comment, 0.8),
                (DocumentField::StringLiteral, 0.8),
            ]),
        }
    }

    pub fn field_weight(&self, field: DocumentField) -> f64 {
        self.field_weights.get(&field).copied().unwrap_or(1.0)
    }

    pub fn intent_field_weight(&self, field: DocumentField, intent: QueryIntent) -> f64 {
        let base = self.field_weight(field);
        let multiplier = match intent {
            QueryIntent::Path => match field {
                DocumentField::FileName => 1.5,
                DocumentField::RelativePath => 2.2,
                DocumentField::Extension => 1.4,
                DocumentField::Content => 0.45,
                DocumentField::Identifier
                | DocumentField::Comment
                | DocumentField::StringLiteral => 0.6,
            },
            QueryIntent::Identifier => match field {
                DocumentField::FileName => 1.15,
                DocumentField::RelativePath => 1.1,
                DocumentField::Identifier => 1.7,
                DocumentField::Content => 0.75,
                DocumentField::Extension
                | DocumentField::Comment
                | DocumentField::StringLiteral => 0.8,
            },
            QueryIntent::ErrorMessage => match field {
                DocumentField::StringLiteral => 2.0,
                DocumentField::Content => 1.25,
                DocumentField::Comment => 0.9,
                DocumentField::FileName
                | DocumentField::RelativePath
                | DocumentField::Extension
                | DocumentField::Identifier => 0.7,
            },
            QueryIntent::Config => match field {
                DocumentField::FileName => 1.5,
                DocumentField::RelativePath => 1.8,
                DocumentField::Extension => 1.7,
                DocumentField::Identifier => 1.25,
                DocumentField::Content => 0.9,
                DocumentField::Comment | DocumentField::StringLiteral => 0.8,
            },
            QueryIntent::NaturalLanguage => match field {
                DocumentField::Content => 1.1,
                DocumentField::Comment => 1.35,
                DocumentField::Identifier => 0.9,
                DocumentField::FileName
                | DocumentField::RelativePath
                | DocumentField::Extension
                | DocumentField::StringLiteral => 0.8,
            },
        };

        base * multiplier
    }
}

pub fn get_configuration() -> Result<BM25FHyperParams, RankingError> {
    Ok(BM25FHyperParams::code_search_defaults())
}

pub struct BM25F {
    pub hyper_params: BM25FHyperParams,
}

impl BM25F {
    pub fn weighted_term_frequency<I>(
        &self,
        index: &I,
        term_doc: &TermDocument,
        intent: QueryIntent,
    ) -> f64
    where
        I: RankedIndexReader,
    {
        if term_doc.field_frequencies.is_empty() {
            return term_doc.term_freq as f64;
        }

        DocumentField::ALL
            .into_iter()
            .map(|field| {
                let tf = term_doc.field_term_freq(field) as f64;
                if tf == 0.0 {
                    return 0.0;
                }

                let field_length = term_doc.field_length(field) as f64;
                let avg_field_length = index.avg_field_length(field);
                if avg_field_length == 0.0 {
                    return 0.0;
                }

                let normalized_tf = tf
                    / (1.0 - self.hyper_params.b
                        + self.hyper_params.b * field_length / avg_field_length);
                self.hyper_params.intent_field_weight(field, intent) * normalized_tf
            })
            .sum()
    }

    pub fn score_weighted_tf(&self, query_weight: f64, idf: f64, weighted_tf: f64) -> f64 {
        if weighted_tf == 0.0 {
            return 0.0;
        }

        query_weight * idf * (weighted_tf * (self.hyper_params.k1 + 1.0))
            / (weighted_tf + self.hyper_params.k1)
    }
}

impl Scorer for BM25F {
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
        let idf = idf(num_docs, documents.len());

        documents
            .iter()
            .par_bridge()
            .for_each(|(doc_id, term_doc)| {
                let weighted_tf = self.weighted_term_frequency(index, term_doc, query.intent());
                let score = self.score_weighted_tf(query_term.weight, idf, weighted_tf);

                *scores.entry(doc_id).or_insert(0.0) += score;
            });
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        path::{Path, PathBuf},
    };

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use super::{BM25F, BM25FHyperParams};
    use crate::{
        config::Config,
        index::InvertedIndex,
        query::{AnalyzedQuery, QueryIntent},
        ranking::{BM25HyperParams, RankingAlgo},
    };

    fn test_config() -> Config {
        Config {
            n_grams: 1,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: stop_words::get(stop_words::LANGUAGE::English)
                .par_iter()
                .map(|word| word.to_string())
                .collect::<HashSet<String>>(),
        }
    }

    fn write_temp_file(dir: &Path, name: &str, content: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    fn bm25f() -> RankingAlgo {
        RankingAlgo::BM25F(BM25FHyperParams::code_search_defaults())
    }

    #[test]
    fn filename_match_can_outrank_generic_content_match() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "src/inverted_index.rs", "fn unrelated() {}");
        write_temp_file(
            dir.path(),
            "src/generic.rs",
            "inverted index inverted index inverted index",
        );

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let query = AnalyzedQuery::new_code_search("inverted index", &test_config());

        let ranked = bm25f().rank(&index, &query, 2).unwrap().0;

        assert_eq!(ranked[0].doc_path, PathBuf::from("src/inverted_index.rs"));
    }

    #[test]
    fn bm25f_keeps_plain_bm25_available_for_comparison() {
        let index =
            InvertedIndex::from_documents(&[("a.rs", &[("rust", 3)]), ("b.rs", &[("rust", 1)])]);
        let query = AnalyzedQuery::new("rust", &test_config());

        let bm25 = RankingAlgo::BM25(BM25HyperParams { k1: 1.2, b: 0.75 })
            .rank(&index, &query, 2)
            .unwrap();
        let bm25f = bm25f().rank(&index, &query, 2).unwrap();

        assert_eq!(bm25.0[0].doc_path, bm25f.0[0].doc_path);
    }

    #[test]
    fn weighted_term_frequency_uses_field_weights() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "src/query_id.rs", "fn unrelated() {}");

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let doc_id = index.doc_id(Path::new("src/query_id.rs")).unwrap();
        let term_doc = index
            .get_postings(&crate::index::Term("query_id".to_string()))
            .unwrap()
            .get(&doc_id)
            .unwrap();
        let weighted_tf = BM25F {
            hyper_params: BM25FHyperParams::code_search_defaults(),
        }
        .weighted_term_frequency(&index, term_doc, QueryIntent::Identifier);

        assert!(weighted_tf > term_doc.term_freq as f64);
    }
}
