#[derive(Debug, thiserror::Error)]
pub enum RankingError {
    #[error("failed to load BM25 configuration")]
    ConfigLoad(#[from] config::ConfigError),
    #[error("failed to determine current directory")]
    CurrentDir(#[source] std::io::Error),
}

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("failed to parse evaluation data")]
    Parse(#[source] serde_json::Error),
    #[error("evaluation data must provide local_root, or commit with repo/repo_path")]
    MissingCorpus,
    #[error("relevant result missing rank field")]
    MissingRank,
}
