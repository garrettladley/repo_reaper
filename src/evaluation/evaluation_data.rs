use std::path::PathBuf;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use url::Url;

use crate::globals::Globals;

use crate::text_transform::Query;

#[derive(serde::Deserialize, Debug)]
pub struct RawEvaluationData {
    pub repo: String,
    pub commit: String,
    pub examples: Vec<RawExample>,
}

pub struct EvaluationData {
    pub repo: Url,
    pub commit: String,
    pub examples: Vec<Example>,
}

impl EvaluationData {
    pub fn parse(raw: RawEvaluationData, globals: &Globals) -> Self {
        EvaluationData {
            repo: Url::parse(&raw.repo).unwrap(),
            commit: raw.commit,
            examples: raw
                .examples
                .into_par_iter()
                .map(|example| Example::parse(example, globals))
                .collect(),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct RawExample {
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
    pub fn parse(raw: RawExample, globals: &Globals) -> Self {
        Example {
            query: Query::new(&raw.query, globals),
            narrative: raw.narrative,
            results: raw.results.into_par_iter().map(ResultData::from).collect(),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct RawResultData {
    pub path: String,
    pub content: String,
    pub relevant: bool,
    pub rank: Option<usize>,
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
            relevance: match raw.relevant {
                true => Relevance::Relevant(raw.rank.unwrap()),
                false => Relevance::NonRelevant,
            },
        }
    }
}

pub enum Relevance {
    Relevant(usize),
    NonRelevant,
}
