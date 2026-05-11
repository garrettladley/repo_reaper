use std::{collections::HashMap, path::PathBuf, str::FromStr};

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, DocumentField, InvertedIndex, Term, TermDocument},
    query::{AnalyzedQuery, QueryTerm},
    ranking::{
        BM25, BM25F, BM25FHyperParams, BM25HyperParams, CosineSimilarity, FieldContribution,
        ScoreExplanation, ScoreWithExplanation, ScoredWithExplanations, TFIDF, TermExplanation,
        get_configuration, idf,
    },
};

#[derive(Debug, Clone, PartialEq)]
pub struct Scored(pub Vec<Score>);

impl std::fmt::Display for Scored {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut output = String::new();

        self.0.iter().for_each(|score| {
            output.push_str(&format!("{}\n", score));
        });

        write!(f, "{}", output)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Score {
    pub doc_path: PathBuf,
    pub score: f64,
}

impl std::fmt::Display for Score {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Path: {} Score: {}", self.doc_path.display(), self.score)
    }
}

#[derive(Debug, Clone)]
pub enum RankingAlgo {
    CosineSimilarity,
    BM25(BM25HyperParams),
    BM25F(BM25FHyperParams),
    TFIDF,
}

pub trait RankingAlgorithm {
    fn score(&self, inverted_index: &InvertedIndex, query: &AnalyzedQuery) -> Scored;
}

pub trait Scorer: Send + Sync {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        query: &AnalyzedQuery,
        term: &Term,
        query_term: QueryTerm,
        documents: &HashMap<DocId, TermDocument>,
        scores: &DashMap<DocId, f64>,
    );
}

impl RankingAlgorithm for RankingAlgo {
    fn score(&self, inverted_index: &InvertedIndex, query: &AnalyzedQuery) -> Scored {
        match self {
            RankingAlgo::CosineSimilarity => score_with(CosineSimilarity, inverted_index, query),
            RankingAlgo::BM25(hyper_params) => score_with(
                BM25 {
                    hyper_params: hyper_params.clone(),
                },
                inverted_index,
                query,
            ),
            RankingAlgo::BM25F(hyper_params) => score_with(
                BM25F {
                    hyper_params: hyper_params.clone(),
                },
                inverted_index,
                query,
            ),
            RankingAlgo::TFIDF => score_with(TFIDF, inverted_index, query),
        }
    }
}

fn score_with<S>(scorer: S, inverted_index: &InvertedIndex, query: &AnalyzedQuery) -> Scored
where
    S: Scorer,
{
    let scores = DashMap::new();

    query.terms().par_bridge().for_each(|(term, query_term)| {
        if let Some(documents) = inverted_index.get_postings(term) {
            scorer.score(inverted_index, query, term, *query_term, documents, &scores)
        }
    });

    Scored(
        scores
            .iter()
            .par_bridge()
            .filter_map(|score| {
                inverted_index.document(*score.key()).map(|metadata| Score {
                    doc_path: metadata.path.clone(),
                    score: *score.value(),
                })
            })
            .collect(),
    )
}

impl RankingAlgo {
    pub fn rank(
        &self,
        inverted_index: &InvertedIndex,
        query: &AnalyzedQuery,
        top_n: usize,
    ) -> Option<Scored> {
        if query.is_empty() {
            return None;
        }

        let mut ranking = self.score(inverted_index, query).0;

        ranking.sort_by(|a, b| b.score.total_cmp(&a.score));
        ranking.truncate(top_n);

        if ranking.is_empty() {
            None
        } else {
            Some(Scored(ranking))
        }
    }

    pub fn rank_with_explanations(
        &self,
        inverted_index: &InvertedIndex,
        query: &AnalyzedQuery,
        top_n: usize,
    ) -> Option<ScoredWithExplanations> {
        let ranked = self.rank(inverted_index, query, top_n)?;
        let results = ranked
            .0
            .into_iter()
            .filter_map(|score| {
                let doc_id = inverted_index.doc_id(&score.doc_path)?;
                Some(ScoreWithExplanation {
                    explanation: self.explain_doc(inverted_index, query, doc_id, score.score),
                    score,
                })
            })
            .collect::<Vec<_>>();

        Some(ScoredWithExplanations { results })
    }

    fn explain_doc(
        &self,
        inverted_index: &InvertedIndex,
        query: &AnalyzedQuery,
        doc_id: DocId,
        final_score: f64,
    ) -> ScoreExplanation {
        let terms = match self {
            RankingAlgo::BM25(hyper_params) => {
                explain_bm25(inverted_index, query, doc_id, hyper_params)
            }
            RankingAlgo::BM25F(hyper_params) => {
                explain_bm25f(inverted_index, query, doc_id, hyper_params)
            }
            RankingAlgo::CosineSimilarity | RankingAlgo::TFIDF => {
                explain_matched_terms(inverted_index, query, doc_id)
            }
        };

        ScoreExplanation { final_score, terms }
    }

    pub fn needs_fielded_index(&self) -> bool {
        matches!(self, Self::BM25F(_))
    }
}

fn explain_bm25(
    inverted_index: &InvertedIndex,
    query: &AnalyzedQuery,
    doc_id: DocId,
    hyper_params: &BM25HyperParams,
) -> Vec<TermExplanation> {
    let scorer = BM25 {
        hyper_params: hyper_params.clone(),
    };
    let avgdl = inverted_index.avg_doc_length();
    let num_docs = inverted_index.num_docs();

    query
        .terms()
        .filter_map(|(term, query_term)| {
            let documents = inverted_index.get_postings(term)?;
            let term_doc = documents.get(&doc_id)?;
            let term_idf = idf(num_docs, documents.len());
            let contribution = scorer.score_term(
                query_term.weight,
                term_idf,
                term_doc.term_freq as f64,
                term_doc.length as f64,
                avgdl,
            );

            Some(TermExplanation {
                term: term.0.clone(),
                query_weight: query_term.weight,
                term_frequency: term_doc.term_freq,
                document_frequency: documents.len(),
                idf: term_idf,
                matched_fields: field_contributions(term_doc, contribution),
                contribution,
            })
        })
        .collect()
}

fn explain_matched_terms(
    inverted_index: &InvertedIndex,
    query: &AnalyzedQuery,
    doc_id: DocId,
) -> Vec<TermExplanation> {
    query
        .terms()
        .filter_map(|(term, query_term)| {
            let documents = inverted_index.get_postings(term)?;
            let term_doc = documents.get(&doc_id)?;
            Some(TermExplanation {
                term: term.0.clone(),
                query_weight: query_term.weight,
                term_frequency: term_doc.term_freq,
                document_frequency: documents.len(),
                idf: idf(inverted_index.num_docs(), documents.len()),
                matched_fields: field_contributions(term_doc, 0.0),
                contribution: 0.0,
            })
        })
        .collect()
}

fn explain_bm25f(
    inverted_index: &InvertedIndex,
    query: &AnalyzedQuery,
    doc_id: DocId,
    hyper_params: &BM25FHyperParams,
) -> Vec<TermExplanation> {
    let scorer = BM25F {
        hyper_params: hyper_params.clone(),
    };
    let num_docs = inverted_index.num_docs();

    query
        .terms()
        .filter_map(|(term, query_term)| {
            let documents = inverted_index.get_postings(term)?;
            let term_doc = documents.get(&doc_id)?;
            let term_idf = idf(num_docs, documents.len());
            let weighted_tf = scorer.weighted_term_frequency(inverted_index, term_doc);
            let contribution = scorer.score_weighted_tf(query_term.weight, term_idf, weighted_tf);

            Some(TermExplanation {
                term: term.0.clone(),
                query_weight: query_term.weight,
                term_frequency: term_doc.term_freq,
                document_frequency: documents.len(),
                idf: term_idf,
                matched_fields: field_contributions_with_weights(
                    term_doc,
                    contribution,
                    &hyper_params.field_weights,
                ),
                contribution,
            })
        })
        .collect()
}

fn field_contributions(term_doc: &TermDocument, contribution: f64) -> Vec<FieldContribution> {
    field_contributions_with_weights(term_doc, contribution, &HashMap::new())
}

fn field_contributions_with_weights(
    term_doc: &TermDocument,
    contribution: f64,
    field_weights: &HashMap<DocumentField, f64>,
) -> Vec<FieldContribution> {
    let mut fields = DocumentField::ALL
        .into_iter()
        .filter_map(|field| {
            let term_frequency = term_doc.field_term_freq(field);
            if term_frequency == 0 {
                return None;
            }

            let proportional_contribution = if term_doc.term_freq == 0 {
                0.0
            } else {
                contribution * term_frequency as f64 / term_doc.term_freq as f64
            };

            Some(FieldContribution {
                field,
                term_frequency,
                field_length: term_doc.field_length(field),
                field_weight: field_weights.get(&field).copied().unwrap_or(1.0),
                contribution: proportional_contribution,
            })
        })
        .collect::<Vec<_>>();

    fields.sort_by_key(|field| field.field);
    fields
}

impl FromStr for RankingAlgo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cosim" => Ok(RankingAlgo::CosineSimilarity),
            "bm25" => {
                let hyper_params =
                    get_configuration().map_err(|e| format!("failed to load BM25 config: {e}"))?;

                Ok(RankingAlgo::BM25(hyper_params))
            }
            "bm25f" => Ok(RankingAlgo::BM25F(BM25FHyperParams::code_search_defaults())),
            "tfidf" => Ok(RankingAlgo::TFIDF),
            _ => Err(format!("{} is not a valid ranking algorithm", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        path::{Path, PathBuf},
    };

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use super::{BM25HyperParams, RankingAlgo};
    use crate::{
        config::Config,
        index::{DocumentField, InvertedIndex, Term},
        query::AnalyzedQuery,
    };

    fn index() -> InvertedIndex {
        InvertedIndex::from_documents(&[
            ("rust.rs", &[("rust", 5), ("code", 1)]),
            ("code.rs", &[("rust", 1), ("code", 5)]),
        ])
    }

    fn query_with_weights(rust: f64, code: f64) -> AnalyzedQuery {
        AnalyzedQuery::from_weights(
            "weighted query",
            HashMap::from([
                (Term("rust".to_string()), rust),
                (Term("code".to_string()), code),
            ]),
        )
    }

    fn bm25() -> RankingAlgo {
        RankingAlgo::BM25(BM25HyperParams { k1: 1.2, b: 0.75 })
    }

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

    #[test]
    fn bm25_query_weights_change_ranking_order() {
        let index = index();
        let rust_weighted = query_with_weights(4.0, 1.0);
        let code_weighted = query_with_weights(1.0, 4.0);

        let rust_top = bm25().rank(&index, &rust_weighted, 1).unwrap().0;
        let code_top = bm25().rank(&index, &code_weighted, 1).unwrap().0;

        assert_eq!(rust_top[0].doc_path, PathBuf::from("rust.rs"));
        assert_eq!(code_top[0].doc_path, PathBuf::from("code.rs"));
    }

    #[test]
    fn tfidf_query_weights_change_ranking_order() {
        let index = index();
        let rust_weighted = query_with_weights(4.0, 1.0);
        let code_weighted = query_with_weights(1.0, 4.0);

        let rust_top = RankingAlgo::TFIDF
            .rank(&index, &rust_weighted, 1)
            .unwrap()
            .0;
        let code_top = RankingAlgo::TFIDF
            .rank(&index, &code_weighted, 1)
            .unwrap()
            .0;

        assert_eq!(rust_top[0].doc_path, PathBuf::from("rust.rs"));
        assert_eq!(code_top[0].doc_path, PathBuf::from("code.rs"));
    }

    #[test]
    fn bm25_explanations_include_terms_fields_and_contributions() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(
            dir.path(),
            "src/inverted_index.rs",
            "fn unrelated() { let value = \"needle\"; }",
        );

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let query = AnalyzedQuery::from_weights(
            "inverted index",
            HashMap::from([(Term("inverted_index".to_string()), 1.0)]),
        );

        let explanations = bm25().rank_with_explanations(&index, &query, 1).unwrap();
        let result = explanations.results.first().unwrap();
        let term = result.explanation.terms.first().unwrap();

        assert_eq!(
            result.score.doc_path,
            PathBuf::from("src/inverted_index.rs")
        );
        assert_eq!(term.term, "inverted_index");
        assert_eq!(term.document_frequency, 1);
        assert!(term.idf > 0.0);
        assert!(term.contribution > 0.0);
        assert!(
            term.matched_fields
                .iter()
                .any(|field| field.field == DocumentField::FileName)
        );
        assert!(
            term.matched_fields
                .iter()
                .any(|field| field.field == DocumentField::RelativePath)
        );
        assert_eq!(result.explanation.final_score, result.score.score);
    }
}
