use crate::{index::DocumentField, query::QueryIntent};

#[derive(Debug, Clone, PartialEq)]
pub struct ScoredWithExplanations {
    pub results: Vec<ScoreWithExplanation>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreWithExplanation {
    pub score: super::Score,
    pub explanation: ScoreExplanation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScoreExplanation {
    pub final_score: f64,
    pub query_intent: QueryIntent,
    pub terms: Vec<TermExplanation>,
    pub static_quality: Vec<StaticQualityContribution>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TermExplanation {
    pub term: String,
    pub provenance: String,
    pub query_weight: f64,
    pub term_frequency: usize,
    pub document_frequency: usize,
    pub idf: f64,
    pub matched_fields: Vec<FieldContribution>,
    pub contribution: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StaticQualityContribution {
    pub signal: String,
    pub value: f64,
    pub contribution: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldContribution {
    pub field: DocumentField,
    pub term_frequency: usize,
    pub field_length: usize,
    pub field_weight: f64,
    pub contribution: f64,
}
