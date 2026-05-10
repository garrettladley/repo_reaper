use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    path::PathBuf,
};

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    evaluation::dataset::{EvidenceSpan, QueryShape},
    index::InvertedIndex,
    query::AnalyzedQuery,
    ranking::{RankingAlgo, Score},
};

pub struct TestSet {
    pub ranking_algorithm: RankingAlgo,
    pub queries: Vec<TestQuery>,
}

pub struct TestQuery {
    pub query: AnalyzedQuery,
    pub query_shape: QueryShape,
    pub relevant_docs: Vec<PathBuf>,
    pub groundedness_results: Vec<GroundednessResult>,
    pub evidence_span_count: usize,
}

pub struct GroundednessResult {
    pub path: PathBuf,
    pub relevant: bool,
    pub evidence: Vec<EvidenceSpan>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct Evaluation {
    pub k: usize,
    pub precision_at_k: f64,
    pub recall_at_k: f64,
    pub mean_average_precision: f64,
    pub mean_reciprocal_rank: f64,
    pub normalized_discounted_cumulative_gain: f64,
}

impl Evaluation {
    pub fn zero(k: usize) -> Self {
        Evaluation {
            k,
            precision_at_k: 0.0,
            recall_at_k: 0.0,
            mean_average_precision: 0.0,
            mean_reciprocal_rank: 0.0,
            normalized_discounted_cumulative_gain: 0.0,
        }
    }
}

impl Default for Evaluation {
    fn default() -> Self {
        Self::zero(0)
    }
}

#[derive(Debug, Clone, Copy, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetricFamily {
    ManyRelevantDocuments,
    TopDocument,
}

impl fmt::Display for MetricFamily {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricFamily::ManyRelevantDocuments => write!(f, "many_relevant_documents"),
            MetricFamily::TopDocument => write!(f, "top_document"),
        }
    }
}

impl QueryShape {
    pub fn metric_family(self) -> MetricFamily {
        match self {
            QueryShape::Conceptual => MetricFamily::ManyRelevantDocuments,
            QueryShape::Configuration
            | QueryShape::ErrorMessage
            | QueryShape::Identifier
            | QueryShape::Navigational
            | QueryShape::Path
            | QueryShape::Regex
            | QueryShape::TestFinding => MetricFamily::TopDocument,
        }
    }
}

impl fmt::Display for QueryShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryShape::Conceptual => write!(f, "conceptual"),
            QueryShape::Configuration => write!(f, "configuration"),
            QueryShape::ErrorMessage => write!(f, "error_message"),
            QueryShape::Identifier => write!(f, "identifier"),
            QueryShape::Navigational => write!(f, "navigational"),
            QueryShape::Path => write!(f, "path"),
            QueryShape::Regex => write!(f, "regex"),
            QueryShape::TestFinding => write!(f, "test_finding"),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct EvaluationSlice {
    pub query_shape: QueryShape,
    pub metric_family: MetricFamily,
    pub query_count: usize,
    #[serde(flatten)]
    pub metrics: Evaluation,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct EvaluationReport {
    pub file_retrieval: FileRetrievalReport,
    pub evidence: EvidenceReport,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct FileRetrievalReport {
    #[serde(flatten)]
    pub aggregate: Evaluation,
    #[serde(default)]
    pub slices: Vec<EvaluationSlice>,
    #[serde(default)]
    pub queries: Vec<QueryEvaluation>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct QueryEvaluation {
    pub query: String,
    pub query_shape: QueryShape,
    pub metrics: Evaluation,
    pub relevant_docs: Vec<PathBuf>,
    pub retrieved_docs: Vec<PathBuf>,
    #[serde(default)]
    pub token_efficiency: TokenEfficiencyEvaluation,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct EvidenceReport {
    pub status: EvidenceReportStatus,
    pub queries_with_evidence: usize,
    pub total_evidence_spans: usize,
    #[serde(default)]
    pub groundedness: GroundednessEvaluation,
    #[serde(default)]
    pub token_efficiency: TokenEfficiencyEvaluation,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceReportStatus {
    NotScored,
    Scored,
}

impl fmt::Display for EvidenceReportStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvidenceReportStatus::NotScored => write!(f, "not_scored"),
            EvidenceReportStatus::Scored => write!(f, "scored"),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct GroundednessEvaluation {
    pub k: usize,
    pub expected_evidence: usize,
    pub cited_evidence: usize,
    pub matching_evidence: usize,
    pub grounded_citations: usize,
    pub total_citations: usize,
    pub highlight_recall: f64,
    pub highlight_precision: f64,
    pub citation_precision: f64,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, Default)]
pub struct TokenEfficiencyEvaluation {
    pub k: usize,
    pub returned_bytes: usize,
    pub estimated_returned_tokens: usize,
    pub relevant_evidence_bytes: usize,
    pub estimated_relevant_evidence_tokens: usize,
    pub evidence_density: f64,
}

impl TokenEfficiencyEvaluation {
    pub fn zero(k: usize) -> Self {
        TokenEfficiencyEvaluation {
            k,
            ..TokenEfficiencyEvaluation::default()
        }
    }

    fn from_counts(k: usize, returned_bytes: usize, relevant_evidence_bytes: usize) -> Self {
        TokenEfficiencyEvaluation {
            k,
            returned_bytes,
            estimated_returned_tokens: estimate_tokens_from_bytes(returned_bytes),
            relevant_evidence_bytes,
            estimated_relevant_evidence_tokens: estimate_tokens_from_bytes(relevant_evidence_bytes),
            evidence_density: ratio(relevant_evidence_bytes, returned_bytes),
        }
    }
}

impl GroundednessEvaluation {
    pub fn zero(k: usize) -> Self {
        GroundednessEvaluation {
            k,
            ..GroundednessEvaluation::default()
        }
    }

    fn from_counts(
        k: usize,
        expected_evidence: usize,
        cited_evidence: usize,
        matching_evidence: usize,
        grounded_citations: usize,
        total_citations: usize,
    ) -> Self {
        GroundednessEvaluation {
            k,
            expected_evidence,
            cited_evidence,
            matching_evidence,
            grounded_citations,
            total_citations,
            highlight_recall: ratio(matching_evidence, expected_evidence),
            highlight_precision: ratio(matching_evidence, cited_evidence),
            citation_precision: ratio(grounded_citations, total_citations),
        }
    }
}

pub fn evaluate_token_efficiency_at_k(
    ranked_docs: &[Score],
    results: &[GroundednessResult],
    inverted_index: &InvertedIndex,
    k: usize,
) -> TokenEfficiencyEvaluation {
    if k == 0 {
        return TokenEfficiencyEvaluation::zero(k);
    }

    let cutoff_docs = &ranked_docs[..ranked_docs.len().min(k)];
    let returned_docs = cutoff_docs
        .iter()
        .filter_map(|doc| inverted_index.doc_id(&doc.doc_path))
        .filter_map(|doc_id| inverted_index.document(doc_id))
        .map(|metadata| (&metadata.path, metadata.file_size_bytes as usize))
        .collect::<Vec<_>>();
    let returned_bytes = returned_docs.iter().map(|(_, bytes)| bytes).sum();
    let retrieved_paths: HashSet<&PathBuf> = returned_docs.iter().map(|(path, _)| *path).collect();
    let relevant_evidence_bytes = results
        .iter()
        .filter(|result| result.relevant && retrieved_paths.contains(&result.path))
        .flat_map(|result| &result.evidence)
        .map(|span| span.snippet.len())
        .sum();

    TokenEfficiencyEvaluation::from_counts(k, returned_bytes, relevant_evidence_bytes)
}

pub fn evaluate_query_at_k(
    ranked_docs: &[Score],
    relevant_docs: &[PathBuf],
    k: usize,
) -> Evaluation {
    if k == 0 || relevant_docs.is_empty() {
        return Evaluation::zero(k);
    }

    let relevant_set: HashSet<&PathBuf> = relevant_docs.iter().collect();
    let cutoff_docs = &ranked_docs[..ranked_docs.len().min(k)];
    let relevant_retrieved = cutoff_docs
        .iter()
        .filter(|doc| relevant_set.contains(&doc.doc_path))
        .count();

    Evaluation {
        k,
        precision_at_k: relevant_retrieved as f64 / k as f64,
        recall_at_k: relevant_retrieved as f64 / relevant_docs.len() as f64,
        mean_average_precision: average_precision_at_k(
            cutoff_docs,
            &relevant_set,
            relevant_docs.len(),
        ),
        mean_reciprocal_rank: reciprocal_rank_at_k(cutoff_docs, &relevant_set),
        normalized_discounted_cumulative_gain: ndcg_at_k(
            cutoff_docs,
            &relevant_set,
            relevant_docs.len(),
            k,
        ),
    }
}

pub fn evaluate_groundedness_at_k(
    ranked_docs: &[Score],
    results: &[GroundednessResult],
    k: usize,
) -> GroundednessEvaluation {
    if k == 0 {
        return GroundednessEvaluation::zero(k);
    }

    let expected_evidence = expected_evidence(results);
    if expected_evidence.is_empty() {
        return GroundednessEvaluation::zero(k);
    }

    let cutoff_docs = &ranked_docs[..ranked_docs.len().min(k)];
    let cited_evidence = cited_evidence(cutoff_docs, results);
    let matching_evidence = cited_evidence.intersection(&expected_evidence).count();

    let grounded_citations = cutoff_docs
        .iter()
        .filter(|doc| {
            results.iter().any(|result| {
                result.path == doc.doc_path
                    && result.evidence.iter().any(|span| {
                        expected_evidence.contains(&(result.path.clone(), span.clone()))
                    })
            })
        })
        .count();
    let total_citations = cutoff_docs
        .iter()
        .filter(|doc| {
            results
                .iter()
                .any(|result| result.path == doc.doc_path && !result.evidence.is_empty())
        })
        .count();

    GroundednessEvaluation::from_counts(
        k,
        expected_evidence.len(),
        cited_evidence.len(),
        matching_evidence,
        grounded_citations,
        total_citations,
    )
}

fn expected_evidence(results: &[GroundednessResult]) -> HashSet<(PathBuf, EvidenceSpan)> {
    results
        .iter()
        .filter(|result| result.relevant)
        .flat_map(|result| {
            result
                .evidence
                .iter()
                .cloned()
                .map(|span| (result.path.clone(), span))
        })
        .collect()
}

fn cited_evidence(
    ranked_docs: &[Score],
    results: &[GroundednessResult],
) -> HashSet<(PathBuf, EvidenceSpan)> {
    ranked_docs
        .iter()
        .flat_map(|doc| {
            results
                .iter()
                .filter(move |result| result.path == doc.doc_path)
                .flat_map(|result| {
                    result
                        .evidence
                        .iter()
                        .cloned()
                        .map(|span| (result.path.clone(), span))
                })
        })
        .collect()
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

fn estimate_tokens_from_bytes(bytes: usize) -> usize {
    bytes.div_ceil(4)
}

fn average_precision_at_k(
    ranked_docs: &[Score],
    relevant_docs: &HashSet<&PathBuf>,
    total_relevant: usize,
) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }

    let mut relevant_seen = 0;
    let precision_sum = ranked_docs
        .iter()
        .enumerate()
        .filter_map(|(rank, doc)| {
            if relevant_docs.contains(&doc.doc_path) {
                relevant_seen += 1;
                Some(relevant_seen as f64 / (rank + 1) as f64)
            } else {
                None
            }
        })
        .sum::<f64>();

    precision_sum / total_relevant as f64
}

fn reciprocal_rank_at_k(ranked_docs: &[Score], relevant_docs: &HashSet<&PathBuf>) -> f64 {
    ranked_docs
        .iter()
        .enumerate()
        .find(|(_, doc)| relevant_docs.contains(&doc.doc_path))
        .map(|(rank, _)| 1.0 / (rank + 1) as f64)
        .unwrap_or(0.0)
}

fn ndcg_at_k(
    ranked_docs: &[Score],
    relevant_docs: &HashSet<&PathBuf>,
    total_relevant: usize,
    k: usize,
) -> f64 {
    let dcg = ranked_docs
        .iter()
        .enumerate()
        .filter(|(_, doc)| relevant_docs.contains(&doc.doc_path))
        .map(|(rank, _)| discount(rank))
        .sum::<f64>();

    let ideal_relevant = total_relevant.min(k);
    if ideal_relevant == 0 {
        return 0.0;
    }

    let idcg = (0..ideal_relevant).map(discount).sum::<f64>();
    dcg / idcg
}

fn discount(zero_based_rank: usize) -> f64 {
    1.0 / ((zero_based_rank + 2) as f64).log2()
}

pub fn average_evaluations(evaluations: &[Evaluation]) -> Evaluation {
    if evaluations.is_empty() {
        return Evaluation::default();
    }

    let n = evaluations.len() as f64;
    let sum = evaluations
        .iter()
        .fold(Evaluation::default(), |acc, eval| Evaluation {
            k: acc.k.max(eval.k),
            precision_at_k: acc.precision_at_k + eval.precision_at_k,
            recall_at_k: acc.recall_at_k + eval.recall_at_k,
            mean_average_precision: acc.mean_average_precision + eval.mean_average_precision,
            mean_reciprocal_rank: acc.mean_reciprocal_rank + eval.mean_reciprocal_rank,
            normalized_discounted_cumulative_gain: acc.normalized_discounted_cumulative_gain
                + eval.normalized_discounted_cumulative_gain,
        });

    Evaluation {
        k: sum.k,
        precision_at_k: sum.precision_at_k / n,
        recall_at_k: sum.recall_at_k / n,
        mean_average_precision: sum.mean_average_precision / n,
        mean_reciprocal_rank: sum.mean_reciprocal_rank / n,
        normalized_discounted_cumulative_gain: sum.normalized_discounted_cumulative_gain / n,
    }
}

pub fn aggregate_groundedness(evaluations: &[GroundednessEvaluation]) -> GroundednessEvaluation {
    if evaluations.is_empty() {
        return GroundednessEvaluation::default();
    }

    let counts = evaluations
        .iter()
        .fold(GroundednessEvaluation::zero(0), |acc, evaluation| {
            GroundednessEvaluation {
                k: acc.k.max(evaluation.k),
                expected_evidence: acc.expected_evidence + evaluation.expected_evidence,
                cited_evidence: acc.cited_evidence + evaluation.cited_evidence,
                matching_evidence: acc.matching_evidence + evaluation.matching_evidence,
                grounded_citations: acc.grounded_citations + evaluation.grounded_citations,
                total_citations: acc.total_citations + evaluation.total_citations,
                highlight_recall: 0.0,
                highlight_precision: 0.0,
                citation_precision: 0.0,
            }
        });

    GroundednessEvaluation::from_counts(
        counts.k,
        counts.expected_evidence,
        counts.cited_evidence,
        counts.matching_evidence,
        counts.grounded_citations,
        counts.total_citations,
    )
}

pub fn aggregate_token_efficiency(
    evaluations: &[TokenEfficiencyEvaluation],
) -> TokenEfficiencyEvaluation {
    if evaluations.is_empty() {
        return TokenEfficiencyEvaluation::default();
    }

    let (k, returned_bytes, relevant_evidence_bytes) =
        evaluations
            .iter()
            .fold((0, 0, 0), |(max_k, returned, evidence), evaluation| {
                (
                    max_k.max(evaluation.k),
                    returned + evaluation.returned_bytes,
                    evidence + evaluation.relevant_evidence_bytes,
                )
            });

    TokenEfficiencyEvaluation::from_counts(k, returned_bytes, relevant_evidence_bytes)
}

fn evaluation_slices(evaluations: &[(QueryShape, Evaluation)]) -> Vec<EvaluationSlice> {
    let mut by_shape: BTreeMap<QueryShape, Vec<Evaluation>> = BTreeMap::new();

    evaluations.iter().for_each(|(shape, evaluation)| {
        by_shape.entry(*shape).or_default().push(evaluation.clone());
    });

    by_shape
        .into_iter()
        .map(|(query_shape, evaluations)| EvaluationSlice {
            query_shape,
            metric_family: query_shape.metric_family(),
            query_count: evaluations.len(),
            metrics: average_evaluations(&evaluations),
        })
        .collect()
}

impl fmt::Display for Evaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "P@{k}: {p_at_k:.4}\nR@{k}: {r_at_k:.4}\nMAP@{k}: {map:.4}\nMRR@{k}: {mrr:.4}\nNDCG@{k}: {ndcg:.4}",
            k = self.k,
            p_at_k = self.precision_at_k,
            r_at_k = self.recall_at_k,
            map = self.mean_average_precision,
            mrr = self.mean_reciprocal_rank,
            ndcg = self.normalized_discounted_cumulative_gain,
        )
    }
}

impl fmt::Display for GroundednessEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "highlight recall@{k}: {highlight_recall:.4}\nhighlight precision@{k}: {highlight_precision:.4}\ncitation precision@{k}: {citation_precision:.4}",
            k = self.k,
            highlight_recall = self.highlight_recall,
            highlight_precision = self.highlight_precision,
            citation_precision = self.citation_precision,
        )
    }
}

impl fmt::Display for TokenEfficiencyEvaluation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "returned bytes@{k}: {returned_bytes}\nestimated returned tokens@{k}: {returned_tokens}\nrelevant evidence bytes@{k}: {evidence_bytes}\nestimated relevant evidence tokens@{k}: {evidence_tokens}\nevidence density@{k}: {evidence_density:.4}",
            k = self.k,
            returned_bytes = self.returned_bytes,
            returned_tokens = self.estimated_returned_tokens,
            evidence_bytes = self.relevant_evidence_bytes,
            evidence_tokens = self.estimated_relevant_evidence_tokens,
            evidence_density = self.evidence_density,
        )
    }
}

impl TestSet {
    pub fn evaluate(&self, inverted_index: &InvertedIndex, top_n: usize) -> Evaluation {
        self.evaluate_report(inverted_index, top_n)
            .file_retrieval
            .aggregate
    }

    pub fn evaluate_report(
        &self,
        inverted_index: &InvertedIndex,
        top_n: usize,
    ) -> EvaluationReport {
        let evaluated_queries: Vec<EvaluatedQuery> = self
            .queries
            .par_iter()
            .map(|query| evaluate_test_query(&self.ranking_algorithm, inverted_index, query, top_n))
            .collect();
        let queries = evaluated_queries
            .iter()
            .map(|query| query.file_retrieval.clone())
            .collect::<Vec<_>>();

        let aggregate = average_evaluations(
            &queries
                .iter()
                .map(|query| query.metrics.clone())
                .collect::<Vec<_>>(),
        );

        let queries_with_evidence = self
            .queries
            .iter()
            .filter(|query| query.evidence_span_count > 0)
            .count();
        let total_evidence_spans = self
            .queries
            .iter()
            .map(|query| query.evidence_span_count)
            .sum();
        let slice_inputs = queries
            .iter()
            .map(|query| (query.query_shape, query.metrics.clone()))
            .collect::<Vec<_>>();

        EvaluationReport {
            file_retrieval: FileRetrievalReport {
                aggregate,
                slices: evaluation_slices(&slice_inputs),
                queries,
            },
            evidence: EvidenceReport {
                status: EvidenceReportStatus::Scored,
                queries_with_evidence,
                total_evidence_spans,
                groundedness: aggregate_groundedness(
                    &evaluated_queries
                        .iter()
                        .map(|query| query.groundedness.clone())
                        .collect::<Vec<_>>(),
                ),
                token_efficiency: aggregate_token_efficiency(
                    &evaluated_queries
                        .iter()
                        .map(|query| query.token_efficiency.clone())
                        .collect::<Vec<_>>(),
                ),
            },
        }
    }
}

struct EvaluatedQuery {
    file_retrieval: QueryEvaluation,
    groundedness: GroundednessEvaluation,
    token_efficiency: TokenEfficiencyEvaluation,
}

fn evaluate_test_query(
    ranking_algorithm: &RankingAlgo,
    inverted_index: &InvertedIndex,
    query: &TestQuery,
    top_n: usize,
) -> EvaluatedQuery {
    let ranked_docs = ranking_algorithm
        .rank(inverted_index, &query.query, top_n)
        .map(|ranking| ranking.0)
        .unwrap_or_default();

    let metrics = evaluate_query_at_k(&ranked_docs, &query.relevant_docs, top_n);
    let groundedness = evaluate_groundedness_at_k(&ranked_docs, &query.groundedness_results, top_n);
    let token_efficiency = evaluate_token_efficiency_at_k(
        &ranked_docs,
        &query.groundedness_results,
        inverted_index,
        top_n,
    );
    let retrieved_docs = ranked_docs
        .into_iter()
        .map(|score| score.doc_path)
        .collect::<Vec<_>>();

    EvaluatedQuery {
        file_retrieval: QueryEvaluation {
            query: query.query.original_text().to_string(),
            query_shape: query.query_shape,
            metrics,
            relevant_docs: query.relevant_docs.clone(),
            retrieved_docs,
            token_efficiency: token_efficiency.clone(),
        },
        groundedness,
        token_efficiency,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        Evaluation, GroundednessResult, MetricFamily, TestQuery, TestSet, average_evaluations,
        evaluate_groundedness_at_k, evaluate_query_at_k, evaluate_token_efficiency_at_k,
    };
    use crate::{
        evaluation::dataset::{EvidenceSpan, QueryShape},
        query::AnalyzedQuery,
        ranking::{RankingAlgo, Score},
    };

    fn scored(paths: &[&str]) -> Vec<Score> {
        paths
            .iter()
            .enumerate()
            .map(|(i, p)| Score {
                doc_path: PathBuf::from(p),
                score: (paths.len() - i) as f64,
            })
            .collect()
    }

    fn relevant(paths: &[&str]) -> Vec<PathBuf> {
        paths.iter().map(PathBuf::from).collect()
    }

    fn span(start_line: usize, snippet: &str) -> EvidenceSpan {
        EvidenceSpan {
            start_line,
            end_line: start_line,
            snippet: snippet.to_string(),
        }
    }

    fn grounded_result(
        path: &str,
        relevant: bool,
        evidence: Vec<EvidenceSpan>,
    ) -> GroundednessResult {
        GroundednessResult {
            path: PathBuf::from(path),
            relevant,
            evidence,
        }
    }

    #[test]
    fn precision_at_k_uses_cutoff_denominator() {
        let ranked = scored(&["a.rs", "x.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs"]), 4);

        assert!((eval.precision_at_k - 0.25).abs() < 1e-10);
    }

    #[test]
    fn recall_at_k_uses_relevant_document_count() {
        let ranked = scored(&["a.rs", "x.rs", "b.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs", "c.rs"]), 2);

        assert!((eval.recall_at_k - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn mean_average_precision_averages_precision_at_relevant_ranks() {
        let ranked = scored(&["a.rs", "x.rs", "b.rs", "c.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs", "missing.rs"]), 4);

        // AP = (P@1 + P@3) / 3 relevant docs = (1 + 2/3) / 3
        let expected = (1.0 + 2.0 / 3.0) / 3.0;
        assert!((eval.mean_average_precision - expected).abs() < 1e-10);
    }

    #[test]
    fn mean_reciprocal_rank_ignores_relevant_docs_after_cutoff() {
        let ranked = scored(&["x.rs", "y.rs", "a.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs"]), 2);

        assert!((eval.mean_reciprocal_rank).abs() < 1e-10);
    }

    #[test]
    fn normalized_discounted_cumulative_gain_is_one_for_ideal_binary_ranking() {
        let ranked = scored(&["a.rs", "b.rs", "x.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs"]), 3);

        assert!((eval.normalized_discounted_cumulative_gain - 1.0).abs() < 1e-10);
    }

    #[test]
    fn normalized_discounted_cumulative_gain_discounts_late_relevant_results() {
        let ranked = scored(&["x.rs", "a.rs", "b.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "b.rs"]), 3);

        let dcg = 1.0 / 3.0_f64.log2() + 1.0 / 4.0_f64.log2();
        let idcg = 1.0 + 1.0 / 3.0_f64.log2();
        assert!((eval.normalized_discounted_cumulative_gain - dcg / idcg).abs() < 1e-10);
    }

    #[test]
    fn average_evaluations_divides_each_metric_by_query_count() {
        let evals = vec![
            Evaluation {
                k: 2,
                precision_at_k: 0.5,
                recall_at_k: 0.25,
                mean_average_precision: 0.75,
                mean_reciprocal_rank: 1.0,
                normalized_discounted_cumulative_gain: 0.9,
            },
            Evaluation {
                k: 2,
                precision_at_k: 1.0,
                recall_at_k: 0.5,
                mean_average_precision: 0.25,
                mean_reciprocal_rank: 0.5,
                normalized_discounted_cumulative_gain: 0.7,
            },
        ];
        let avg = average_evaluations(&evals);

        assert_eq!(avg.k, 2);
        assert!((avg.precision_at_k - 0.75).abs() < 1e-10);
        assert!((avg.recall_at_k - 0.375).abs() < 1e-10);
        assert!((avg.mean_average_precision - 0.5).abs() < 1e-10);
        assert!((avg.mean_reciprocal_rank - 0.75).abs() < 1e-10);
        assert!((avg.normalized_discounted_cumulative_gain - 0.8).abs() < 1e-10);
    }

    #[test]
    fn average_of_empty_returns_zeros() {
        let avg = average_evaluations(&[]);

        assert_eq!(avg.k, 0);
        assert!((avg.precision_at_k).abs() < 1e-10);
        assert!((avg.recall_at_k).abs() < 1e-10);
        assert!((avg.mean_average_precision).abs() < 1e-10);
        assert!((avg.mean_reciprocal_rank).abs() < 1e-10);
        assert!((avg.normalized_discounted_cumulative_gain).abs() < 1e-10);
    }

    #[test]
    fn evaluation_slices_average_metrics_by_query_shape() {
        let evaluations = vec![
            (
                QueryShape::Conceptual,
                Evaluation {
                    k: 10,
                    precision_at_k: 0.2,
                    recall_at_k: 0.4,
                    mean_average_precision: 0.6,
                    mean_reciprocal_rank: 0.8,
                    normalized_discounted_cumulative_gain: 1.0,
                },
            ),
            (
                QueryShape::Conceptual,
                Evaluation {
                    k: 10,
                    precision_at_k: 0.4,
                    recall_at_k: 0.6,
                    mean_average_precision: 0.2,
                    mean_reciprocal_rank: 0.4,
                    normalized_discounted_cumulative_gain: 0.6,
                },
            ),
            (
                QueryShape::Identifier,
                Evaluation {
                    k: 10,
                    precision_at_k: 1.0,
                    recall_at_k: 1.0,
                    mean_average_precision: 1.0,
                    mean_reciprocal_rank: 1.0,
                    normalized_discounted_cumulative_gain: 1.0,
                },
            ),
        ];

        let slices = super::evaluation_slices(&evaluations);

        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].query_shape, QueryShape::Conceptual);
        assert_eq!(slices[0].metric_family, MetricFamily::ManyRelevantDocuments);
        assert_eq!(slices[0].query_count, 2);
        assert!((slices[0].metrics.mean_average_precision - 0.4).abs() < 1e-10);
        assert_eq!(slices[1].query_shape, QueryShape::Identifier);
        assert_eq!(slices[1].metric_family, MetricFamily::TopDocument);
        assert_eq!(slices[1].query_count, 1);
        assert!((slices[1].metrics.mean_reciprocal_rank - 1.0).abs() < 1e-10);
    }

    #[test]
    fn all_metrics_are_bounded_zero_to_one() {
        let ranked = scored(&["a.rs", "b.rs", "c.rs"]);
        let eval = evaluate_query_at_k(&ranked, &relevant(&["a.rs", "d.rs"]), 3);

        for val in [
            eval.precision_at_k,
            eval.recall_at_k,
            eval.mean_average_precision,
            eval.mean_reciprocal_rank,
            eval.normalized_discounted_cumulative_gain,
        ] {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&val),
                "metric must be in [0, 1], got {val}"
            );
        }
    }

    #[test]
    fn groundedness_counts_retrieved_relevant_path_with_matching_evidence() {
        let target = span(10, "needle");
        let ranked = scored(&["a.rs"]);
        let results = vec![grounded_result("a.rs", true, vec![target])];

        let eval = evaluate_groundedness_at_k(&ranked, &results, 1);

        assert_eq!(eval.matching_evidence, 1);
        assert!((eval.highlight_recall - 1.0).abs() < 1e-10);
        assert!((eval.highlight_precision - 1.0).abs() < 1e-10);
        assert!((eval.citation_precision - 1.0).abs() < 1e-10);
    }

    #[test]
    fn groundedness_does_not_credit_retrieved_path_without_evidence() {
        let ranked = scored(&["a.rs"]);
        let results = vec![
            grounded_result("a.rs", true, Vec::new()),
            grounded_result("b.rs", true, vec![span(20, "needle")]),
        ];

        let eval = evaluate_groundedness_at_k(&ranked, &results, 1);

        assert_eq!(eval.expected_evidence, 1);
        assert_eq!(eval.matching_evidence, 0);
        assert!((eval.highlight_recall).abs() < 1e-10);
        assert!((eval.highlight_precision).abs() < 1e-10);
        assert!((eval.citation_precision).abs() < 1e-10);
    }

    #[test]
    fn groundedness_recall_drops_when_expected_evidence_is_not_retrieved() {
        let ranked = scored(&["a.rs"]);
        let results = vec![
            grounded_result("a.rs", true, vec![span(10, "first")]),
            grounded_result("b.rs", true, vec![span(20, "second")]),
        ];

        let eval = evaluate_groundedness_at_k(&ranked, &results, 1);

        assert_eq!(eval.expected_evidence, 2);
        assert_eq!(eval.matching_evidence, 1);
        assert!((eval.highlight_recall - 0.5).abs() < 1e-10);
    }

    #[test]
    fn groundedness_precision_drops_for_non_relevant_cited_evidence() {
        let ranked = scored(&["a.rs", "noise.rs"]);
        let results = vec![
            grounded_result("a.rs", true, vec![span(10, "needle")]),
            grounded_result("noise.rs", false, vec![span(30, "distractor")]),
        ];

        let eval = evaluate_groundedness_at_k(&ranked, &results, 2);

        assert_eq!(eval.cited_evidence, 2);
        assert_eq!(eval.matching_evidence, 1);
        assert_eq!(eval.grounded_citations, 1);
        assert_eq!(eval.total_citations, 2);
        assert!((eval.highlight_precision - 0.5).abs() < 1e-10);
        assert!((eval.citation_precision - 0.5).abs() < 1e-10);
    }

    #[test]
    fn token_efficiency_counts_returned_docs_only() {
        let ranked = scored(&["a.rs"]);
        let results = vec![
            grounded_result("a.rs", true, vec![span(10, "needle")]),
            grounded_result("b.rs", true, vec![span(20, "not returned")]),
        ];
        let index = crate::index::InvertedIndex::from_documents_with_sizes(&[
            ("a.rs", &[("needle", 1)], 100),
            ("b.rs", &[("needle", 1)], 400),
        ]);

        let eval = evaluate_token_efficiency_at_k(&ranked, &results, &index, 10);

        assert_eq!(eval.returned_bytes, 100);
        assert_eq!(eval.estimated_returned_tokens, 25);
    }

    #[test]
    fn token_efficiency_counts_retrieved_relevant_evidence_bytes() {
        let ranked = scored(&["a.rs", "b.rs"]);
        let results = vec![
            grounded_result("a.rs", true, vec![span(10, "needle")]),
            grounded_result("b.rs", false, vec![span(20, "distractor")]),
        ];
        let index = crate::index::InvertedIndex::from_documents_with_sizes(&[
            ("a.rs", &[("needle", 1)], 100),
            ("b.rs", &[("noise", 1)], 200),
        ]);

        let eval = evaluate_token_efficiency_at_k(&ranked, &results, &index, 2);

        assert_eq!(eval.relevant_evidence_bytes, "needle".len());
        assert_eq!(eval.estimated_relevant_evidence_tokens, 2);
    }

    #[test]
    fn token_efficiency_density_drops_when_extra_non_evidence_docs_are_returned() {
        let results = vec![grounded_result("a.rs", true, vec![span(10, "needle")])];
        let index = crate::index::InvertedIndex::from_documents_with_sizes(&[
            ("a.rs", &[("needle", 1)], 100),
            ("noise.rs", &[("noise", 1)], 300),
        ]);

        let focused = evaluate_token_efficiency_at_k(&scored(&["a.rs"]), &results, &index, 1);
        let noisy =
            evaluate_token_efficiency_at_k(&scored(&["a.rs", "noise.rs"]), &results, &index, 2);

        assert!(noisy.evidence_density < focused.evidence_density);
    }

    #[test]
    fn token_efficiency_zero_returned_bytes_stays_finite() {
        let ranked = scored(&["missing.rs"]);
        let results = vec![grounded_result(
            "missing.rs",
            true,
            vec![span(10, "needle")],
        )];
        let index = crate::index::InvertedIndex::from_documents_with_sizes(&[]);

        let eval = evaluate_token_efficiency_at_k(&ranked, &results, &index, 1);

        assert_eq!(eval.returned_bytes, 0);
        assert_eq!(eval.relevant_evidence_bytes, 0);
        assert_eq!(eval.evidence_density, 0.0);
        assert!(eval.evidence_density.is_finite());
    }

    #[test]
    fn evaluation_report_deserializes_without_token_efficiency_fields() {
        let content = r#"{
            "file_retrieval": {
                "k": 10,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
                "mean_average_precision": 0.0,
                "mean_reciprocal_rank": 0.0,
                "normalized_discounted_cumulative_gain": 0.0,
                "queries": [{
                    "query": "missing query",
                    "query_shape": "identifier",
                    "metrics": {
                        "k": 10,
                        "precision_at_k": 0.0,
                        "recall_at_k": 0.0,
                        "mean_average_precision": 0.0,
                        "mean_reciprocal_rank": 0.0,
                        "normalized_discounted_cumulative_gain": 0.0
                    },
                    "relevant_docs": ["a.rs"],
                    "retrieved_docs": []
                }]
            },
            "evidence": {
                "status": "scored",
                "queries_with_evidence": 1,
                "total_evidence_spans": 1,
                "groundedness": {
                    "k": 10,
                    "expected_evidence": 0,
                    "cited_evidence": 0,
                    "matching_evidence": 0,
                    "grounded_citations": 0,
                    "total_citations": 0,
                    "highlight_recall": 0.0,
                    "highlight_precision": 0.0,
                    "citation_precision": 0.0
                }
            }
        }"#;

        let report: super::EvaluationReport = serde_json::from_str(content).unwrap();

        assert_eq!(report.evidence.token_efficiency.returned_bytes, 0);
        assert_eq!(
            report.file_retrieval.queries[0]
                .token_efficiency
                .returned_bytes,
            0
        );
    }

    #[test]
    fn evaluate_report_includes_query_metrics_and_evidence_boundary() {
        let evidence = span(10, "needle");
        let queries = vec![TestQuery {
            query: AnalyzedQuery::from_frequencies("missing query", Default::default()),
            query_shape: QueryShape::Identifier,
            relevant_docs: relevant(&["a.rs"]),
            groundedness_results: vec![grounded_result("a.rs", true, vec![evidence])],
            evidence_span_count: 2,
        }];
        let test_set = TestSet {
            ranking_algorithm: RankingAlgo::TFIDF,
            queries,
        };
        let temp = tempfile::tempdir().unwrap();
        let index = crate::index::InvertedIndex::new(
            temp.path(),
            |_| std::collections::HashMap::new(),
            None,
        );

        let report = test_set.evaluate_report(&index, 10);

        assert_eq!(report.file_retrieval.queries[0].query, "missing query");
        assert_eq!(
            report.file_retrieval.slices[0].query_shape,
            QueryShape::Identifier
        );
        assert_eq!(report.evidence.queries_with_evidence, 1);
        assert_eq!(report.evidence.total_evidence_spans, 2);
        assert_eq!(report.evidence.groundedness.expected_evidence, 1);
        assert_eq!(report.evidence.token_efficiency.k, 10);
    }
}
