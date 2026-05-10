mod file_retrieval;
mod groundedness;
mod report;
mod token_efficiency;
mod types;
mod util;

pub use file_retrieval::{average_evaluations, evaluate_query_at_k};
pub use groundedness::{
    GroundednessEvaluation, aggregate_groundedness, evaluate_groundedness_at_k,
};
pub use token_efficiency::{
    TokenEfficiencyEvaluation, aggregate_token_efficiency, evaluate_token_efficiency_at_k,
};
pub use types::{
    Evaluation, EvaluationReport, EvaluationSlice, EvidenceReport, EvidenceReportStatus,
    FileRetrievalReport, GroundednessResult, MetricFamily, QueryEvaluation, TestQuery, TestSet,
};

#[cfg(test)]
mod tests;
