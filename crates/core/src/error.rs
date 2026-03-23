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
    #[error("invalid repository URL: {0}")]
    InvalidUrl(String),
    #[error("relevant result missing rank field")]
    MissingRank,
}
