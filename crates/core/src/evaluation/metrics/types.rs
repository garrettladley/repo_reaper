use std::{fmt, path::PathBuf};

use super::{GroundednessEvaluation, TokenEfficiencyEvaluation};
use crate::{
    evaluation::dataset::{EvidenceSpan, QueryShape},
    query::AnalyzedQuery,
    ranking::RankingAlgo,
};

pub struct TestSet {
    pub ranking_algorithm: RankingAlgo,
    pub queries: Vec<TestQuery>,
    pub feedback_expansion: bool,
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
