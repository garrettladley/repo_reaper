use std::{path::PathBuf, str::FromStr};

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{DocId, DocumentField, PostingList, RankedIndexReader, Term, TermDocument},
    query::{AnalyzedQuery, QueryTerm},
    ranking::{
        BM25, BM25F, BM25FHyperParams, BM25HyperParams, CosineSimilarity, FieldContribution,
        ProximityConfig, QueryLikelihood, QueryLikelihoodParams, ScoreExplanation,
        ScoreWithExplanation, ScoredWithExplanations, StaticQualityContribution, TFIDF,
        TermExplanation, idf,
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
    BM25Proximity(BM25HyperParams, ProximityConfig),
    QueryLikelihood(QueryLikelihoodParams),
    TFIDF,
}

pub trait RankingAlgorithm {
    fn score<I>(&self, index: &I, query: &AnalyzedQuery) -> Scored
    where
        I: RankedIndexReader + Sync;
}

pub trait Scorer: Send + Sync {
    fn score<I, P>(
        &self,
        index: &I,
        query: &AnalyzedQuery,
        term: &Term,
        query_term: QueryTerm,
        documents: &P,
        scores: &DashMap<DocId, f64>,
    ) where
        I: RankedIndexReader + Sync,
        P: PostingList + Sync;
}

impl RankingAlgorithm for RankingAlgo {
    fn score<I>(&self, index: &I, query: &AnalyzedQuery) -> Scored
    where
        I: RankedIndexReader + Sync,
    {
        match self {
            RankingAlgo::CosineSimilarity => score_with(CosineSimilarity, index, query),
            RankingAlgo::BM25(hyper_params) => score_with(
                BM25 {
                    hyper_params: hyper_params.clone(),
                },
                index,
                query,
            ),
            RankingAlgo::BM25F(hyper_params) => score_with(
                BM25F {
                    hyper_params: hyper_params.clone(),
                },
                index,
                query,
            ),
            RankingAlgo::BM25Proximity(hyper_params, config) => {
                let mut scored = score_with(
                    BM25 {
                        hyper_params: hyper_params.clone(),
                    },
                    index,
                    query,
                );
                for score in &mut scored.0 {
                    if let Some(doc_id) = index.doc_id(&score.doc_path) {
                        score.score += crate::ranking::proximity::positional_bonus(
                            index, query, doc_id, config,
                        );
                    }
                }
                scored
            }
            RankingAlgo::QueryLikelihood(params) => QueryLikelihood {
                params: params.clone(),
            }
            .score(index, query),
            RankingAlgo::TFIDF => score_with(TFIDF, index, query),
        }
    }
}

fn score_with<S, I>(scorer: S, index: &I, query: &AnalyzedQuery) -> Scored
where
    S: Scorer,
    I: RankedIndexReader + Sync,
{
    let scores = DashMap::new();

    query.terms().par_bridge().for_each(|(term, query_term)| {
        if let Some(documents) = index.postings(term) {
            scorer.score(index, query, term, *query_term, &documents, &scores)
        }
    });

    Scored(
        scores
            .iter()
            .par_bridge()
            .filter_map(|score| {
                index.document(*score.key()).map(|metadata| Score {
                    doc_path: metadata.path.clone(),
                    score: *score.value(),
                })
            })
            .collect(),
    )
}

impl RankingAlgo {
    pub fn rank<I>(&self, index: &I, query: &AnalyzedQuery, top_n: usize) -> Option<Scored>
    where
        I: RankedIndexReader + Sync,
    {
        if query.is_empty() {
            return None;
        }

        let mut ranking = self.score(index, query).0;

        apply_static_quality_priors(index, &mut ranking);

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

    pub fn rank_with_explanations<I>(
        &self,
        index: &I,
        query: &AnalyzedQuery,
        top_n: usize,
    ) -> Option<ScoredWithExplanations>
    where
        I: RankedIndexReader + Sync,
    {
        let ranked = self.rank(index, query, top_n)?;
        let results = ranked
            .0
            .into_iter()
            .filter_map(|score| {
                let doc_id = index.doc_id(&score.doc_path)?;
                Some(ScoreWithExplanation {
                    explanation: self.explain_doc(index, query, doc_id, score.score),
                    score,
                })
            })
            .collect::<Vec<_>>();

        Some(ScoredWithExplanations { results })
    }

    pub fn rank_with_feedback<I>(
        &self,
        index: &I,
        query: &AnalyzedQuery,
        top_n: usize,
        feedback_docs: usize,
        feedback_terms: usize,
    ) -> Option<Scored>
    where
        I: crate::ranking::feedback::FeedbackTermSource + Sync,
    {
        let seed = self.rank(index, query, feedback_docs)?;
        let expanded = crate::ranking::feedback::expand_query_with_feedback(
            index,
            query,
            &seed.0,
            feedback_terms,
        );

        self.rank(index, &expanded, top_n)
    }

    fn explain_doc<I>(
        &self,
        index: &I,
        query: &AnalyzedQuery,
        doc_id: DocId,
        final_score: f64,
    ) -> ScoreExplanation
    where
        I: RankedIndexReader + Sync,
    {
        let terms = match self {
            RankingAlgo::BM25(hyper_params) => explain_bm25(index, query, doc_id, hyper_params),
            RankingAlgo::BM25F(hyper_params) => explain_bm25f(index, query, doc_id, hyper_params),
            RankingAlgo::BM25Proximity(hyper_params, _) => {
                explain_bm25(index, query, doc_id, hyper_params)
            }
            RankingAlgo::CosineSimilarity
            | RankingAlgo::QueryLikelihood(_)
            | RankingAlgo::TFIDF => explain_matched_terms(index, query, doc_id),
        };

        ScoreExplanation {
            final_score,
            query_intent: query.intent(),
            terms,
            static_quality: static_quality_explanation(index, doc_id),
        }
    }

    pub fn needs_fielded_index(&self) -> bool {
        matches!(self, Self::BM25F(_) | Self::BM25Proximity(_, _))
    }
}

fn explain_bm25<I>(
    index: &I,
    query: &AnalyzedQuery,
    doc_id: DocId,
    hyper_params: &BM25HyperParams,
) -> Vec<TermExplanation>
where
    I: RankedIndexReader + Sync,
{
    let scorer = BM25 {
        hyper_params: hyper_params.clone(),
    };
    let avgdl = index.avg_doc_length();
    let num_docs = index.num_docs();

    query
        .terms()
        .filter_map(|(term, query_term)| {
            let documents = index.postings(term)?;
            let term_doc = documents.get(doc_id)?;
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

fn explain_matched_terms<I>(index: &I, query: &AnalyzedQuery, doc_id: DocId) -> Vec<TermExplanation>
where
    I: RankedIndexReader + Sync,
{
    query
        .terms()
        .filter_map(|(term, query_term)| {
            let documents = index.postings(term)?;
            let term_doc = documents.get(doc_id)?;
            Some(TermExplanation {
                term: term.0.clone(),
                provenance: query_term.provenance.as_str().to_string(),
                query_weight: query_term.weight,
                term_frequency: term_doc.term_freq,
                document_frequency: documents.len(),
                idf: idf(index.num_docs(), documents.len()),
                matched_fields: field_contributions(term_doc, 0.0),
                contribution: 0.0,
            })
        })
        .collect()
}

fn explain_bm25f<I>(
    index: &I,
    query: &AnalyzedQuery,
    doc_id: DocId,
    hyper_params: &BM25FHyperParams,
) -> Vec<TermExplanation>
where
    I: RankedIndexReader + Sync,
{
    let scorer = BM25F {
        hyper_params: hyper_params.clone(),
    };
    let num_docs = index.num_docs();

    query
        .terms()
        .filter_map(|(term, query_term)| {
            let documents = index.postings(term)?;
            let term_doc = documents.get(doc_id)?;
            let term_idf = idf(num_docs, documents.len());
            let weighted_tf = scorer.weighted_term_frequency(index, term_doc, query.intent());
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

fn apply_static_quality_priors<I>(index: &I, ranking: &mut [Score])
where
    I: RankedIndexReader + Sync,
{
    for score in ranking {
        if let Some(doc_id) = index.doc_id(&score.doc_path)
            && let Some(metadata) = index.document(doc_id)
        {
            score.score += metadata.quality_signals.prior_score();
        }
    }
}

fn static_quality_explanation<I>(index: &I, doc_id: DocId) -> Vec<StaticQualityContribution>
where
    I: RankedIndexReader + Sync,
{
    let Some(metadata) = index.document(doc_id) else {
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
            "bm25" => Ok(RankingAlgo::BM25(BM25HyperParams::default())),
            "bm25f" => Ok(RankingAlgo::BM25F(BM25FHyperParams::code_search_defaults())),
            "bm25-proximity" | "proximity" => Ok(RankingAlgo::BM25Proximity(
                BM25HyperParams::default(),
                ProximityConfig::default(),
            )),
            "ql" | "ql-dirichlet" | "query-likelihood" => Ok(RankingAlgo::QueryLikelihood(
                QueryLikelihoodParams::dirichlet_defaults(),
            )),
            "ql-jm" | "jelinek-mercer" => Ok(RankingAlgo::QueryLikelihood(
                QueryLikelihoodParams::jelinek_mercer_defaults(),
            )),
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

    use super::{BM25FHyperParams, BM25HyperParams, ProximityConfig, RankingAlgo};
    use crate::{
        config::Config,
        index::{
            DocId, DocumentField, DocumentMetadata, InvertedIndex, OwnedPostingList, PostingList,
            RankedIndexReader, Term, TermDocument,
        },
        query::AnalyzedQuery,
        tokenizer::FileType,
    };

    struct OwnedPostingsReader {
        postings: HashMap<Term, OwnedPostingList>,
        documents: HashMap<DocId, DocumentMetadata>,
    }

    impl RankedIndexReader for OwnedPostingsReader {
        type Postings<'a> = OwnedPostingList;

        fn postings(&self, term: &Term) -> Option<Self::Postings<'_>> {
            self.postings.get(term).cloned()
        }

        fn documents(&self) -> Vec<&DocumentMetadata> {
            self.documents.values().collect()
        }

        fn document(&self, id: DocId) -> Option<&DocumentMetadata> {
            self.documents.get(&id)
        }

        fn doc_id(&self, path: &Path) -> Option<DocId> {
            self.documents
                .iter()
                .find_map(|(doc_id, metadata)| (metadata.path == path).then_some(*doc_id))
        }

        fn document_norm(&self, _: DocId) -> Option<f64> {
            None
        }

        fn num_docs(&self) -> usize {
            self.documents.len()
        }

        fn avg_doc_length(&self) -> f64 {
            let total = self
                .documents
                .values()
                .map(|metadata| metadata.token_length)
                .sum::<usize>();
            total as f64 / self.documents.len() as f64
        }

        fn avg_field_length(&self, _: DocumentField) -> f64 {
            0.0
        }

        fn doc_freq(&self, term: &Term) -> usize {
            self.postings.get(term).map_or(0, |postings| postings.len())
        }

        fn total_token_count(&self) -> u64 {
            self.documents
                .values()
                .map(|metadata| metadata.token_length as u64)
                .sum()
        }

        fn vocabulary_size(&self) -> usize {
            self.postings.len()
        }

        fn collection_frequency(&self, term: &Term) -> usize {
            self.postings.get(term).map_or(0, |postings| {
                postings
                    .iter()
                    .map(|(_, term_doc)| term_doc.term_freq)
                    .sum()
            })
        }
    }

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
    fn ranking_uses_reader_trait_with_owned_posting_lists() {
        let rust = Term("rust".to_string());
        let first = DocId::from_u32(0);
        let second = DocId::from_u32(1);
        let reader = OwnedPostingsReader {
            postings: HashMap::from([(
                rust.clone(),
                OwnedPostingList::new(vec![
                    (first, TermDocument::unfielded(2, 2)),
                    (second, TermDocument::unfielded(2, 1)),
                ]),
            )]),
            documents: HashMap::from([
                (
                    first,
                    DocumentMetadata {
                        id: first,
                        path: PathBuf::from("first.rs"),
                        token_length: 2,
                        file_size_bytes: 10,
                        file_type: FileType::Rust,
                        field_lengths: HashMap::new(),
                        field_spans: Vec::new(),
                        features: Vec::new(),
                        quality_signals: Default::default(),
                    },
                ),
                (
                    second,
                    DocumentMetadata {
                        id: second,
                        path: PathBuf::from("second.rs"),
                        token_length: 2,
                        file_size_bytes: 10,
                        file_type: FileType::Rust,
                        field_lengths: HashMap::new(),
                        field_spans: Vec::new(),
                        features: Vec::new(),
                        quality_signals: Default::default(),
                    },
                ),
            ]),
        };
        let query = AnalyzedQuery::from_frequencies("rust", HashMap::from([(rust, 1)]));

        let ranked = bm25().rank(&reader, &query, 2).unwrap();

        assert_eq!(ranked.0[0].doc_path, PathBuf::from("first.rs"));
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

    #[test]
    fn quoted_phrase_requires_adjacent_terms_in_one_field() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "phrase.rs", "// mean average precision");
        write_temp_file(dir.path(), "split.rs", "// mean average\nfn precision() {}");

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let query = AnalyzedQuery::new_code_search("\"mean average precision\"", &test_config());
        let phrase = query.phrases().first().unwrap();

        let phrase_doc = index.doc_id(Path::new("phrase.rs")).unwrap();
        let split_doc = index.doc_id(Path::new("split.rs")).unwrap();

        assert_eq!(
            crate::ranking::proximity::phrase_match_count(&index, phrase_doc, phrase),
            1
        );
        assert_eq!(
            crate::ranking::proximity::phrase_match_count(&index, split_doc, phrase),
            0
        );
    }

    #[test]
    fn proximity_score_promotes_near_terms_over_far_terms() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "near.rs", "// alpha beta gamma");
        write_temp_file(
            dir.path(),
            "far.rs",
            "// alpha filler filler filler filler filler filler beta gamma",
        );

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let query = AnalyzedQuery::new_code_search("alpha gamma", &test_config());
        let ranking = RankingAlgo::BM25Proximity(
            BM25HyperParams { k1: 1.2, b: 0.75 },
            ProximityConfig {
                window: 4,
                phrase_boost: 0.0,
                proximity_boost: 10.0,
            },
        )
        .rank(&index, &query, 2)
        .unwrap()
        .0;

        assert_eq!(ranking[0].doc_path, PathBuf::from("near.rs"));
    }

    #[test]
    fn phrase_matching_does_not_cross_documents() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "alpha.rs", "// alpha");
        write_temp_file(dir.path(), "beta.rs", "// beta");

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let query = AnalyzedQuery::new_code_search("\"alpha beta\"", &test_config());
        let phrase = query.phrases().first().unwrap();

        for metadata in index.documents() {
            assert_eq!(
                crate::ranking::proximity::phrase_match_count(&index, metadata.id, phrase),
                0
            );
        }
    }
}
