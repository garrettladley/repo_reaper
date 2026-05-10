use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use super::{
    Evaluation, EvaluationReport, EvidenceReport, EvidenceReportStatus, FileRetrievalReport,
    GroundednessEvaluation, QueryEvaluation, TestQuery, TestSet, TokenEfficiencyEvaluation,
    aggregate_groundedness, aggregate_token_efficiency, average_evaluations,
    evaluate_groundedness_at_k, evaluate_query_at_k, evaluate_token_efficiency_at_k,
    file_retrieval::evaluation_slices,
};
use crate::{index::InvertedIndex, ranking::RankingAlgo};

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
            .map(|query| evaluate_query(&self.ranking_algorithm, inverted_index, query, top_n))
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

fn evaluate_query(
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
