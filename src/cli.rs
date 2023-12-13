use std::path::PathBuf;

use clap::Parser;

use crate::ranking::rank::RankingAlgos;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The directory to watch
    #[clap(short, long, default_value = ".")]
    pub directory: PathBuf,
    /// Query log location
    #[clap(short, long, default_value = "query_logs.json")]
    pub query_log: PathBuf,
    /// n-grams
    #[clap(short, long, default_value = "1")]
    pub n_grams: usize,
    /// Ranking algorithm
    #[clap(short, long, default_value = "bm25")]
    pub ranking_algorithm: RankingAlgos,
    /// Number of results to return
    #[clap(short, long, default_value = "10")]
    pub top_n: usize,
}
