use std::{collections::HashMap, path::PathBuf, str::FromStr};

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, DocumentField, InvertedIndex, Term, TermDocument},
    query::{AnalyzedQuery, QueryTerm},
    ranking::{
        BM25, BM25F, BM25FHyperParams, BM25HyperParams, CosineSimilarity, FieldContribution,
        ScoreExplanation, ScoreWithExplanation, ScoredWithExplanations, StaticQualityContribution,
        TFIDF, TermExplanation, get_configuration, idf,
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

        apply_static_quality_priors(inverted_index, &mut ranking);

        ranking.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_path.cmp(&b.doc_path))
        });
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

    pub fn rank_with_feedback(
        &self,
        inverted_index: &InvertedIndex,
        query: &AnalyzedQuery,
        top_n: usize,
        feedback_docs: usize,
        feedback_terms: usize,
    ) -> Option<Scored> {
        let seed = self.rank(inverted_index, query, feedback_docs)?;
        let expanded = crate::ranking::feedback::expand_query_with_feedback(
            inverted_index,
            query,
            &seed.0,
            feedback_terms,
        );

        self.rank(inverted_index, &expanded, top_n)
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

        ScoreExplanation {
            final_score,
            query_intent: query.intent(),
            terms,
            static_quality: static_quality_explanation(inverted_index, doc_id),
        }
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
                provenance: query_term.provenance.as_str().to_string(),
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
                provenance: query_term.provenance.as_str().to_string(),
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
            let weighted_tf =
                scorer.weighted_term_frequency(inverted_index, term_doc, query.intent());
            let contribution = scorer.score_weighted_tf(query_term.weight, term_idf, weighted_tf);

            Some(TermExplanation {
                term: term.0.clone(),
                provenance: query_term.provenance.as_str().to_string(),
                query_weight: query_term.weight,
                term_frequency: term_doc.term_freq,
                document_frequency: documents.len(),
                idf: term_idf,
                matched_fields: field_contributions_with_weights(
                    term_doc,
                    contribution,
                    hyper_params,
                    query.intent(),
                ),
                contribution,
            })
        })
        .collect()
}

fn field_contributions(term_doc: &TermDocument, contribution: f64) -> Vec<FieldContribution> {
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
                field_weight: 1.0,
                contribution: proportional_contribution,
            })
        })
        .collect::<Vec<_>>();

    fields.sort_by_key(|field| field.field);
    fields
}

fn field_contributions_with_weights(
    term_doc: &TermDocument,
    contribution: f64,
    hyper_params: &BM25FHyperParams,
    intent: crate::query::QueryIntent,
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
                field_weight: hyper_params.intent_field_weight(field, intent),
                contribution: proportional_contribution,
            })
        })
        .collect::<Vec<_>>();

    fields.sort_by_key(|field| field.field);
    fields
}

fn apply_static_quality_priors(inverted_index: &InvertedIndex, ranking: &mut [Score]) {
    for score in ranking {
        if let Some(doc_id) = inverted_index.doc_id(&score.doc_path)
            && let Some(metadata) = inverted_index.document(doc_id)
        {
            score.score += metadata.quality_signals.prior_score();
        }
    }
}

fn static_quality_explanation(
    inverted_index: &InvertedIndex,
    doc_id: DocId,
) -> Vec<StaticQualityContribution> {
    let Some(metadata) = inverted_index.document(doc_id) else {
        return Vec::new();
    };
    let signals = &metadata.quality_signals;

    let mut contributions = Vec::new();
    push_quality(&mut contributions, "generated", signals.generated, -0.55);
    push_quality(&mut contributions, "vendor", signals.vendor, -0.45);
    push_quality(&mut contributions, "test", signals.test, -0.08);
    push_quality(&mut contributions, "entry_point", signals.entry_point, 0.28);
    push_quality(&mut contributions, "readme", signals.readme, 0.22);
    push_quality(
        &mut contributions,
        "config_or_manifest",
        signals.config_or_manifest,
        0.20,
    );
    push_quality(
        &mut contributions,
        "public_entry_point",
        signals.public_entry_point,
        0.12,
    );

    let depth_contribution = match signals.file_depth {
        0 | 1 => 0.10,
        2 => 0.04,
        3..=5 => 0.0,
        _ => -0.08,
    };
    contributions.push(StaticQualityContribution {
        signal: "file_depth".to_string(),
        value: signals.file_depth as f64,
        contribution: depth_contribution,
    });

    let size_contribution = if signals.file_size_bytes > 750_000 {
        -0.18
    } else if signals.file_size_bytes > 250_000 {
        -0.08
    } else {
        0.0
    };
    contributions.push(StaticQualityContribution {
        signal: "file_size_bytes".to_string(),
        value: signals.file_size_bytes as f64,
        contribution: size_contribution,
    });

    contributions.push(StaticQualityContribution {
        signal: "reference_count".to_string(),
        value: signals.reference_count as f64,
        contribution: signals.reference_count.min(12) as f64 * 0.015,
    });

    contributions
}

fn push_quality(
    contributions: &mut Vec<StaticQualityContribution>,
    signal: &str,
    value: bool,
    contribution: f64,
) {
    contributions.push(StaticQualityContribution {
        signal: signal.to_string(),
        value: if value { 1.0 } else { 0.0 },
        contribution: if value { contribution } else { 0.0 },
    });
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

    use super::{BM25FHyperParams, BM25HyperParams, RankingAlgo};
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

    #[test]
    fn static_quality_prior_can_penalize_generated_matches() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "src/lib.rs", "needle");
        write_temp_file(
            dir.path(),
            "vendor/generated/client.rs",
            "// generated\nneedle needle",
        );

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let query = AnalyzedQuery::new_code_search("needle", &test_config());

        let ranked = RankingAlgo::BM25F(BM25FHyperParams::code_search_defaults())
            .rank(&index, &query, 2)
            .unwrap()
            .0;

        assert_eq!(ranked[0].doc_path, PathBuf::from("src/lib.rs"));
    }

    #[test]
    fn explanations_include_static_quality_and_intent() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(
            dir.path(),
            "src/main.rs",
            "fn main() { println!(\"needle\"); }",
        );

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let query = AnalyzedQuery::new_code_search("needle", &test_config());

        let explanations = RankingAlgo::BM25F(BM25FHyperParams::code_search_defaults())
            .rank_with_explanations(&index, &query, 1)
            .unwrap();
        let explanation = &explanations.results[0].explanation;

        assert_eq!(explanation.query_intent, query.intent());
        assert!(
            explanation
                .static_quality
                .iter()
                .any(|signal| signal.signal == "entry_point" && signal.contribution > 0.0)
        );
    }
}
