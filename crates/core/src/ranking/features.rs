use std::{
    collections::BTreeMap,
    io::{BufRead, Write},
    path::PathBuf,
};

use crate::{
    evaluation::dataset::{EvaluationData, Relevance},
    index::{DocumentField, InvertedIndex, Term},
    query::{AnalyzedQuery, QueryTermProvenance},
    ranking::{RankingAlgo, Score},
};

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct RankingFeatureRow {
    pub query_id: String,
    pub query: String,
    pub query_intent: String,
    pub doc_path: PathBuf,
    pub rank: usize,
    pub relevance: Option<usize>,
    pub features: BTreeMap<String, f64>,
    pub expansion_provenance: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct PairwisePreferenceRow {
    pub query_id: String,
    pub preferred_doc: PathBuf,
    pub dispreferred_doc: PathBuf,
    pub preferred_relevance: usize,
    pub dispreferred_relevance: usize,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FeatureExportRecord {
    Feature(RankingFeatureRow),
    PairwisePreference(PairwisePreferenceRow),
}

#[derive(Debug, thiserror::Error)]
pub enum FeatureIoError {
    #[error("feature export I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("feature record JSON failed: {0}")]
    Json(#[from] serde_json::Error),
}

pub trait FeatureRecordSink {
    type Error;

    fn write_record(&mut self, record: &FeatureExportRecord) -> Result<(), Self::Error>;
}

pub trait FeatureRecordSource {
    type Error;

    fn read_records(&mut self) -> Result<Vec<FeatureExportRecord>, Self::Error>;
}

pub struct JsonlFeatureSink<W> {
    writer: W,
}

impl<W> JsonlFeatureSink<W>
where
    W: Write,
{
    pub fn new(writer: W) -> Self {
        Self { writer }
    }
}

impl<W> FeatureRecordSink for JsonlFeatureSink<W>
where
    W: Write,
{
    type Error = FeatureIoError;

    fn write_record(&mut self, record: &FeatureExportRecord) -> Result<(), Self::Error> {
        serde_json::to_writer(&mut self.writer, record)?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }
}

pub struct JsonlFeatureSource<R> {
    reader: R,
}

impl<R> JsonlFeatureSource<R>
where
    R: BufRead,
{
    pub fn new(reader: R) -> Self {
        Self { reader }
    }
}

impl<R> FeatureRecordSource for JsonlFeatureSource<R>
where
    R: BufRead,
{
    type Error = FeatureIoError;

    fn read_records(&mut self) -> Result<Vec<FeatureExportRecord>, Self::Error> {
        self.reader
            .by_ref()
            .lines()
            .filter_map(|line| match line {
                Ok(line) if line.trim().is_empty() => None,
                other => Some(other),
            })
            .map(|line| {
                let line = line?;
                serde_json::from_str(&line).map_err(FeatureIoError::from)
            })
            .collect()
    }
}

pub fn export_evaluation_features_to_sink<S>(
    sink: &mut S,
    index: &InvertedIndex,
    ranking_algorithm: &RankingAlgo,
    evaluation_data: &EvaluationData,
    top_n: usize,
) -> Result<(), S::Error>
where
    S: FeatureRecordSink,
{
    for (query_index, example) in evaluation_data.examples.iter().enumerate() {
        let query_id = format!("q{}", query_index + 1);
        let relevance = example
            .results
            .iter()
            .filter_map(|result| match &result.relevance {
                Relevance::Relevant(rank) => Some((result.path.clone(), *rank)),
                Relevance::NonRelevant => None,
            })
            .collect::<BTreeMap<PathBuf, usize>>();

        for row in export_feature_rows(
            index,
            ranking_algorithm,
            &query_id,
            &example.query,
            top_n,
            &relevance,
        ) {
            sink.write_record(&FeatureExportRecord::Feature(row))?;
        }

        for row in export_pairwise_preferences(&query_id, &relevance) {
            sink.write_record(&FeatureExportRecord::PairwisePreference(row))?;
        }
    }

    Ok(())
}

pub fn export_feature_rows(
    index: &InvertedIndex,
    ranking_algorithm: &RankingAlgo,
    query_id: &str,
    query: &AnalyzedQuery,
    top_n: usize,
    relevance: &BTreeMap<PathBuf, usize>,
) -> Vec<RankingFeatureRow> {
    let ranked = ranking_algorithm
        .rank(index, query, top_n)
        .map(|scores| scores.0)
        .unwrap_or_default();

    ranked
        .iter()
        .enumerate()
        .map(|(rank, score)| RankingFeatureRow {
            query_id: query_id.to_string(),
            query: query.original_text().to_string(),
            query_intent: format!("{:?}", query.intent()),
            doc_path: score.doc_path.clone(),
            rank: rank + 1,
            relevance: relevance.get(&score.doc_path).copied(),
            features: features_for_score(index, query, score),
            expansion_provenance: expansion_provenance(query),
        })
        .collect()
}

pub fn export_pairwise_preferences(
    query_id: &str,
    relevance: &BTreeMap<PathBuf, usize>,
) -> Vec<PairwisePreferenceRow> {
    let judged = relevance.iter().collect::<Vec<_>>();
    let mut rows = Vec::new();

    for (preferred_doc, preferred_rank) in &judged {
        for (dispreferred_doc, dispreferred_rank) in &judged {
            if preferred_rank < dispreferred_rank {
                rows.push(PairwisePreferenceRow {
                    query_id: query_id.to_string(),
                    preferred_doc: (*preferred_doc).clone(),
                    dispreferred_doc: (*dispreferred_doc).clone(),
                    preferred_relevance: **preferred_rank,
                    dispreferred_relevance: **dispreferred_rank,
                });
            }
        }
    }

    rows
}

fn features_for_score(
    index: &InvertedIndex,
    query: &AnalyzedQuery,
    score: &Score,
) -> BTreeMap<String, f64> {
    let mut features = BTreeMap::from([("ranking_score".to_string(), score.score)]);
    let Some(doc_id) = index.doc_id(&score.doc_path) else {
        return features;
    };
    let Some(metadata) = index.document(doc_id) else {
        return features;
    };

    for (name, value) in metadata.quality_signals.features() {
        features.insert(format!("static.{name}"), value);
    }

    for field in DocumentField::ALL {
        let count = query
            .terms()
            .filter_map(|(term, _)| {
                index
                    .get_postings(term)
                    .and_then(|docs| docs.get(&doc_id))
                    .map(|term_doc| term_doc.field_term_freq(field))
            })
            .sum::<usize>();
        features.insert(format!("field_match.{}", field.as_str()), count as f64);
    }

    features.insert(
        "path_exact_contains_query".to_string(),
        contains_normalized(&score.doc_path.to_string_lossy(), query.original_text()),
    );
    features.insert(
        "filename_exact_contains_query".to_string(),
        score
            .doc_path
            .file_name()
            .and_then(|name| name.to_str())
            .map_or(0.0, |file_name| {
                contains_normalized(file_name, query.original_text())
            }),
    );
    features.insert(
        "expansion.controlled_terms".to_string(),
        provenance_count(query, QueryTermProvenance::ControlledExpansion) as f64,
    );
    features.insert(
        "expansion.feedback_terms".to_string(),
        provenance_count(query, QueryTermProvenance::Feedback) as f64,
    );
    features.insert("regex.literal_evidence".to_string(), 0.0);

    features
}

fn contains_normalized(haystack: &str, needle: &str) -> f64 {
    let haystack = haystack.to_ascii_lowercase();
    let needle = needle.trim().to_ascii_lowercase();
    if !needle.is_empty() && haystack.contains(&needle) {
        1.0
    } else {
        0.0
    }
}

fn provenance_count(query: &AnalyzedQuery, provenance: QueryTermProvenance) -> usize {
    query
        .terms()
        .filter(|(_, query_term)| query_term.provenance == provenance)
        .count()
}

fn expansion_provenance(query: &AnalyzedQuery) -> BTreeMap<String, String> {
    query
        .terms()
        .filter(|(_, query_term)| query_term.provenance != QueryTermProvenance::Original)
        .map(|(term, query_term)| (term.0.clone(), query_term.provenance.as_str().to_string()))
        .collect()
}

#[allow(dead_code)]
fn _term_key(term: &Term) -> &str {
    &term.0
}

#[cfg(test)]
mod tests {
    use std::{
        collections::BTreeMap,
        path::{Path, PathBuf},
    };

    use super::{
        FeatureExportRecord, FeatureRecordSink, FeatureRecordSource, JsonlFeatureSink,
        JsonlFeatureSource, export_feature_rows, export_pairwise_preferences,
    };
    use crate::{
        index::InvertedIndex,
        query::AnalyzedQuery,
        ranking::{BM25HyperParams, RankingAlgo},
    };

    #[test]
    fn exports_inspectable_feature_rows() {
        let index = InvertedIndex::from_documents(&[("src/lib.rs", &[("rust", 2)])]);
        let query = AnalyzedQuery::from_frequencies(
            "rust",
            [(crate::index::Term("rust".to_string()), 1)].into(),
        );

        let rows = export_feature_rows(
            &index,
            &RankingAlgo::BM25(BM25HyperParams { k1: 1.2, b: 0.75 }),
            "q1",
            &query,
            1,
            &BTreeMap::new(),
        );

        assert_eq!(rows[0].query_id, "q1");
        assert!(rows[0].features.contains_key("ranking_score"));
        assert!(rows[0].features.contains_key("static.file_depth"));
    }

    #[test]
    fn exports_pairwise_preferences_from_graded_judgments() {
        let rows = export_pairwise_preferences(
            "q1",
            &BTreeMap::from([
                (PathBuf::from("a.rs"), 1),
                (PathBuf::from("b.rs"), 3),
                (PathBuf::from("c.rs"), 2),
            ]),
        );

        assert_eq!(rows.len(), 3);
        assert!(rows.iter().any(|row| row.preferred_doc == Path::new("a.rs")
            && row.dispreferred_doc == Path::new("b.rs")));
    }

    #[test]
    fn jsonl_sink_and_source_round_trip_feature_records() {
        let mut bytes = Vec::new();
        let record = FeatureExportRecord::PairwisePreference(super::PairwisePreferenceRow {
            query_id: "q1".to_string(),
            preferred_doc: PathBuf::from("a.rs"),
            dispreferred_doc: PathBuf::from("b.rs"),
            preferred_relevance: 1,
            dispreferred_relevance: 2,
        });

        JsonlFeatureSink::new(&mut bytes)
            .write_record(&record)
            .unwrap();
        let mut source = JsonlFeatureSource::new(std::io::Cursor::new(bytes));

        let records = source.read_records().unwrap();

        assert_eq!(records.len(), 1);
        assert!(matches!(
            &records[0],
            FeatureExportRecord::PairwisePreference(row)
                if row.preferred_doc == Path::new("a.rs")
        ));
    }
}
