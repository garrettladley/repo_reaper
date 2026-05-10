use std::path::PathBuf;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{config::Config, error::EvalError, query::Query};

#[derive(serde::Deserialize, Debug)]
pub struct RawEvaluationData {
    pub repo: Option<String>,
    pub repo_path: Option<PathBuf>,
    pub commit: Option<String>,
    pub local_root: Option<PathBuf>,
    pub examples: Vec<RawExample>,
}

pub struct EvaluationData {
    pub corpus: EvaluationCorpus,
    pub examples: Vec<Example>,
}

impl EvaluationData {
    pub fn parse(raw: RawEvaluationData, config: &Config) -> Result<Self, EvalError> {
        let corpus = EvaluationCorpus::parse(raw.local_root, raw.repo, raw.repo_path, raw.commit)?;

        let examples: Result<Vec<_>, _> = raw
            .examples
            .into_par_iter()
            .map(|example| Example::parse(example, config))
            .collect();

        Ok(EvaluationData {
            corpus,
            examples: examples?,
        })
    }
}

pub enum EvaluationCorpus {
    Git {
        repo: String,
        commit: String,
        root: PathBuf,
    },
    Tree {
        root: PathBuf,
    },
}

impl EvaluationCorpus {
    fn parse(
        local_root: Option<PathBuf>,
        repo: Option<String>,
        repo_path: Option<PathBuf>,
        commit: Option<String>,
    ) -> Result<Self, EvalError> {
        let repo = repo.or_else(|| repo_path.map(|path| path.display().to_string()));

        match (repo, commit) {
            (Some(repo), Some(commit)) => {
                let root = local_root.unwrap_or_else(|| PathBuf::from("."));
                Ok(EvaluationCorpus::Git { repo, commit, root })
            }
            (None, Some(_)) => Err(EvalError::MissingCorpus),
            (None, None) => local_root
                .map(|root| EvaluationCorpus::Tree { root })
                .ok_or(EvalError::MissingCorpus),
            (Some(_), None) => Err(EvalError::MissingCorpus),
        }
    }
}

#[derive(serde::Deserialize, Debug)]
pub struct RawExample {
    pub query: String,
    pub narrative: String,
    pub query_shape: Option<QueryShape>,
    pub results: Vec<RawResultData>,
}

pub struct Example {
    pub query: Query,
    pub narrative: String,
    pub query_shape: Option<QueryShape>,
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
            query_shape: raw.query_shape,
            results: results?,
        })
    }
}

#[derive(serde::Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum QueryShape {
    Conceptual,
    Configuration,
    ErrorMessage,
    Identifier,
    Navigational,
    Path,
    TestFinding,
}

#[derive(serde::Deserialize, Debug)]
pub struct RawResultData {
    pub path: String,
    pub content: String,
    pub relevant: bool,
    pub rank: Option<usize>,
    #[serde(default)]
    pub evidence: Vec<EvidenceSpan>,
}

#[derive(serde::Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct EvidenceSpan {
    pub start_line: usize,
    pub end_line: usize,
    pub snippet: String,
}

pub struct ResultData {
    pub path: PathBuf,
    pub content: String,
    pub relevance: Relevance,
    pub evidence: Vec<EvidenceSpan>,
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
            evidence: raw.evidence,
        })
    }
}

pub enum Relevance {
    Relevant(usize),
    NonRelevant,
}

#[cfg(test)]
mod tests {
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use super::{EvaluationCorpus, EvaluationData, QueryShape, RawEvaluationData};
    use crate::config::Config;

    fn test_config() -> Config {
        Config {
            n_grams: 1,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: stop_words::get(stop_words::LANGUAGE::English)
                .par_iter()
                .map(|word| word.to_string())
                .collect(),
        }
    }

    #[test]
    fn parse_accepts_local_root_with_query_shape_and_evidence() {
        let raw: RawEvaluationData = serde_json::from_str(
            r#"{
                "local_root": "crates",
                "examples": [{
                    "query": "bm25 score",
                    "narrative": "find bm25 scoring implementation",
                    "query_shape": "conceptual",
                    "results": [{
                        "path": "core/src/ranking/bm25.rs",
                        "content": "bm25 score implementation",
                        "relevant": true,
                        "rank": 1,
                        "evidence": [{
                            "start_line": 35,
                            "end_line": 50,
                            "snippet": "impl Scorer for BM25"
                        }]
                    }]
                }]
            }"#,
        )
        .unwrap();

        let parsed = EvaluationData::parse(raw, &test_config()).unwrap();

        assert!(matches!(parsed.corpus, EvaluationCorpus::Tree { .. }));
        assert_eq!(parsed.examples[0].query_shape, Some(QueryShape::Conceptual));
        assert_eq!(parsed.examples[0].results[0].evidence[0].start_line, 35);
    }

    #[test]
    fn parse_pins_git_corpus_to_commit_and_subdirectory() {
        let raw: RawEvaluationData = serde_json::from_str(
            r#"{
                "repo": ".",
                "commit": "abc123",
                "local_root": "crates",
                "examples": []
            }"#,
        )
        .unwrap();

        let parsed = EvaluationData::parse(raw, &test_config()).unwrap();

        assert!(matches!(
            parsed.corpus,
            EvaluationCorpus::Git {
                ref repo,
                ref commit,
                ref root,
            } if repo == "." && commit == "abc123" && root == &std::path::PathBuf::from("crates")
        ));
    }
}
