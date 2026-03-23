use std::path::PathBuf;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use url::Url;

use crate::{config::Config, error::EvalError, query::Query};

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
    pub fn parse(raw: RawEvaluationData, config: &Config) -> Result<Self, EvalError> {
        let repo = Url::parse(&raw.repo).map_err(|_| EvalError::InvalidUrl(raw.repo.clone()))?;

        let examples: Result<Vec<_>, _> = raw
            .examples
            .into_par_iter()
            .map(|example| Example::parse(example, config))
            .collect();

        Ok(EvaluationData {
            repo,
            commit: raw.commit,
            examples: examples?,
        })
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
    pub fn parse(raw: RawExample, config: &Config) -> Result<Self, EvalError> {
        let results: Result<Vec<_>, _> = raw
            .results
            .into_par_iter()
            .map(ResultData::try_from)
            .collect();

        Ok(Example {
            query: Query::new(&raw.query, config),
            narrative: raw.narrative,
            results: results?,
        })
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

impl TryFrom<RawResultData> for ResultData {
    type Error = EvalError;

    fn try_from(raw: RawResultData) -> Result<Self, Self::Error> {
        let relevance = if raw.relevant {
            Relevance::Relevant(raw.rank.ok_or(EvalError::MissingRank)?)
        } else {
            Relevance::NonRelevant
        };

        Ok(ResultData {
            path: PathBuf::from(raw.path),
            content: raw.content,
            relevance,
        })
    }
}

pub enum Relevance {
    Relevant(usize),
    NonRelevant,
}
