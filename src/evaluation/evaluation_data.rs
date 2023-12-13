use std::fs;
use std::path::PathBuf;

use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use url::Url;

use crate::globals::Globals;
use crate::ranking::Ranking;
use crate::text_transform::Query;

use crate::evaluation::ranking_evaluation::{TestQuery, TestSet};

#[derive(serde::Deserialize, Debug)]
struct RawEvaluationData {
    pub repo: String,
    pub examples: Vec<RawExample>,
}

pub struct EvaluationData {
    pub repo: Url,
    pub examples: Vec<Example>,
}

impl EvaluationData {
    fn parse(raw: RawEvaluationData, globals: &Globals) -> Self {
        EvaluationData {
            repo: Url::parse(&raw.repo).unwrap(),
            examples: raw
                .examples
                .into_par_iter()
                .map(|example| Example::parse(example, globals))
                .collect(),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct RawExample {
    pub query: String,
    pub narrative: String,
    pub results: Vec<RawResultData>,
}

pub struct Example {
    pub query: Query,
    pub narrative: String,
    pub results: Vec<ResultData>,
}

impl Example {
    fn parse(raw: RawExample, globals: &Globals) -> Self {
        Example {
            query: Query::new(&raw.query, globals),
            narrative: raw.narrative,
            results: raw.results.into_par_iter().map(ResultData::from).collect(),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
struct RawResultData {
    pub path: String,
    pub content: String,
    pub relevance: Relevance,
}

pub struct ResultData {
    pub path: PathBuf,
    pub content: String,
    pub relevance: Relevance,
}

impl From<RawResultData> for ResultData {
    fn from(raw: RawResultData) -> Self {
        ResultData {
            path: PathBuf::from(raw.path),
            content: raw.content,
            relevance: raw.relevance,
        }
    }
}

#[derive(serde::Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Relevance {
    Relevant(usize),
    NonRelevant,
}

pub fn load_evaluation_data(
    path: &str,
    globals: &Globals,
    ranking_algorithm: Box<dyn Fn(&Query) -> Ranking + Send + Sync>,
) -> TestSet {
    let file_content = fs::read_to_string(path).expect("Failed to read file");

    let raw_evaluation_data: RawEvaluationData =
        serde_json::from_str(&file_content).expect("Failed to deserialize JSON data");

    let evaluation_data = EvaluationData::parse(raw_evaluation_data, globals);

    let queries = evaluation_data
        .examples
        .par_iter()
        .map(|example| {
            let relevant_docs = example
                .results
                .iter()
                .filter(|result| matches!(result.relevance, Relevance::Relevant(_)))
                .map(|result| result.path.clone())
                .collect();

            TestQuery {
                query: example.query.clone(),
                relevant_docs,
            }
        })
        .collect::<Vec<TestQuery>>();

    TestSet {
        ranking_algorithm,
        queries,
    }
}
