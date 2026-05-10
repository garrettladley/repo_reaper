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
}

pub struct GroundednessResult {
    pub path: PathBuf,
    pub relevant: bool,
    pub evidence: Vec<EvidenceSpan>,
}

#[derive(Debug, Clone, serde::Serialize)]
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

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationSlice {
    pub query_shape: QueryShape,
    pub metric_family: MetricFamily,
    pub query_count: usize,
    #[serde(flatten)]
    pub metrics: Evaluation,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EvaluationReport {
    #[serde(flatten)]
    pub aggregate: Evaluation,
    pub slices: Vec<EvaluationSlice>,
    pub groundedness: GroundednessEvaluation,
}

#[derive(Debug, Clone, serde::Serialize, Default)]
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

impl TestSet {
    pub fn evaluate(&self, inverted_index: &InvertedIndex, top_n: usize) -> Evaluation {
        let evaluations = self.query_evaluations(inverted_index, top_n);
        let evaluations: Vec<_> = evaluations
            .into_iter()
            .map(|(_, evaluation)| evaluation)
            .collect();

        average_evaluations(&evaluations)
    }

    pub fn evaluate_report(
        &self,
        inverted_index: &InvertedIndex,
        top_n: usize,
    ) -> EvaluationReport {
        let evaluations = self.query_evaluations(inverted_index, top_n);
        let aggregate_evaluations: Vec<_> = evaluations
            .iter()
            .map(|(_, evaluation)| evaluation.clone())
            .collect();

        EvaluationReport {
            aggregate: average_evaluations(&aggregate_evaluations),
            slices: evaluation_slices(&evaluations),
            groundedness: self.evaluate_groundedness(inverted_index, top_n),
        }
    }

    fn evaluate_groundedness(
        &self,
        inverted_index: &InvertedIndex,
        top_n: usize,
    ) -> GroundednessEvaluation {
        let evaluations: Vec<_> = self
            .queries
            .par_iter()
            .map(|query| {
                let ranked_docs =
                    match self
                        .ranking_algorithm
                        .rank(inverted_index, &query.query, top_n)
                    {
                        Some(ranking) => ranking.0,
                        None => Vec::new(),
                    };

                evaluate_groundedness_at_k(&ranked_docs, &query.groundedness_results, top_n)
            })
            .collect();

        aggregate_groundedness(&evaluations)
    }

    fn query_evaluations(
        &self,
        inverted_index: &InvertedIndex,
        top_n: usize,
    ) -> Vec<(QueryShape, Evaluation)> {
        self.queries
            .par_iter()
            .map(|query| {
                let ranked_docs =
                    match self
                        .ranking_algorithm
                        .rank(inverted_index, &query.query, top_n)
                    {
                        Some(ranking) => ranking.0,
                        None => return (query.query_shape, Evaluation::zero(top_n)),
                    };

                (
                    query.query_shape,
                    evaluate_query_at_k(&ranked_docs, &query.relevant_docs, top_n),
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        Evaluation, GroundednessResult, MetricFamily, average_evaluations,
        evaluate_groundedness_at_k, evaluate_query_at_k,
    };
    use crate::{
        evaluation::dataset::{EvidenceSpan, QueryShape},
        ranking::Score,
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
}
