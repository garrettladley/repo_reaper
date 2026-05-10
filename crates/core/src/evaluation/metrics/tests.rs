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

fn grounded_result(path: &str, relevant: bool, evidence: Vec<EvidenceSpan>) -> GroundednessResult {
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

    let slices = super::file_retrieval::evaluation_slices(&evaluations);

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
    let noisy = evaluate_token_efficiency_at_k(&scored(&["a.rs", "noise.rs"]), &results, &index, 2);

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
    let index =
        crate::index::InvertedIndex::new(temp.path(), |_| std::collections::HashMap::new(), None);

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
