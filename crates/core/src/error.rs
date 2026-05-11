#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("failed to parse evaluation data")]
    Parse(#[source] serde_json::Error),
    #[error("evaluation data must provide local_root, or commit with repo/repo_path")]
    MissingCorpus,
    #[error("relevant result missing rank field")]
    MissingRank,
}
