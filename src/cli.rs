use std::path::PathBuf;

use clap::Parser;
use spinners::Spinner;

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
}

pub fn spin_it<F: FnOnce() -> T, T>(func: F, spinner: &mut Spinner) -> T {
    let result = func();
    spinner.stop();
    result
}
