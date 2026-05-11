use std::{
    collections::HashSet,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use clap::{Parser, Subcommand};
use eval::{EvalOutputFormat, evaluate_training};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use repo_reaper_core::{
    config::Config as ReaperConfig,
    index::{CorpusStats, InvertedIndex},
    ranking::RankingAlgo,
    regex_search::{RegexSearchEngine, RegexSearchMatch},
    tokenizer::n_gram_transform,
};
use rust_stemmers::{Algorithm, Stemmer};

mod eval;
mod live_search;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The directory to watch
    #[clap(short, long, default_value = ".", global = true)]
    directory: PathBuf,
    /// n-grams
    #[clap(short, long, default_value = "1")]
    n_grams: usize,
    /// Ranking algorithm
    #[clap(short, long, default_value = "bm25")]
    ranking_algorithm: RankingAlgo,
    /// Number of results to return
    #[clap(short, long, default_value = "10")]
    top_n: usize,
    /// Evaluate the training set
    #[clap(short, long, default_value = "false")]
    evaluate: bool,
    /// Evaluation output format
    #[clap(long, value_enum, default_value = "pretty")]
    eval_format: EvalOutputFormat,
    /// Compare the current evaluation report against a saved JSON baseline
    #[clap(long)]
    eval_compare: Option<PathBuf>,
    /// Allowed negative delta before eval comparison fails
    #[clap(long, default_value = "0")]
    eval_regression_threshold: f64,
    /// Evaluation data JSON
    #[clap(long, default_value = "./data/train.json")]
    eval_data: PathBuf,
    /// Directory used for a cloned Git evaluation corpus
    #[clap(long, default_value = "./data/repo")]
    eval_workdir: PathBuf,
    /// Remove and reclone the Git evaluation workdir before evaluating
    #[clap(long, default_value = "false")]
    fresh: bool,
    /// Print corpus statistics for the indexed directory and exit
    #[clap(long, default_value = "false")]
    stats: bool,
    /// Enable controlled abbreviation query expansion
    #[clap(long, default_value = "false")]
    query_expansion: bool,
    /// Enable experimental pseudo-relevance feedback expansion
    #[clap(long, default_value = "false")]
    feedback_expansion: bool,
    /// Write ranking feature export JSONL while evaluating
    #[clap(long)]
    export_features: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Search files with a regular expression and exact verification
    Regex {
        /// Regex pattern to search for
        pattern: String,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    if let Some(Commands::Regex { pattern }) = &args.command {
        run_regex_search(&args.directory, pattern)?;
        return Ok(());
    }

    let config = Arc::new(ReaperConfig {
        n_grams: args.n_grams,
        stemmer: Stemmer::create(Algorithm::English),
        stop_words: stop_words::get(stop_words::LANGUAGE::English)
            .par_iter()
            .map(|word| word.to_string())
            .collect::<HashSet<String>>(),
    });

    if args.evaluate {
        evaluate_training(&args, &config)?;
        return Ok(());
    }

    if args.stats {
        print_directory_stats(&args.directory, &config);
        return Ok(());
    }

    live_search::run(
        args.directory,
        config,
        args.ranking_algorithm,
        args.top_n,
        args.query_expansion,
        args.feedback_expansion,
    )
}

fn print_directory_stats(directory: &Path, config: &ReaperConfig) {
    println!("Indexing files in {}", directory.display());
    let index = InvertedIndex::new(
        directory,
        |content: &str| n_gram_transform(content, config),
        None::<&Path>,
    );
    println!("Successfully indexed {} files", index.num_docs());
    print_corpus_stats(&index.corpus_stats(10));
}

fn print_corpus_stats(stats: &CorpusStats) {
    println!("documents: {}", stats.document_count);
    println!("total tokens: {}", stats.total_token_count);
    println!("vocabulary size: {}", stats.vocabulary_size);
    println!("singleton terms: {}", stats.singleton_term_count);
    println!("high-frequency terms:");

    if stats.high_frequency_terms.is_empty() {
        println!("  none");
        return;
    }

    for summary in &stats.high_frequency_terms {
        println!(
            "  {}: collection_frequency={}, document_frequency={}",
            summary.term, summary.collection_frequency, summary.document_frequency
        );
    }
}

fn run_regex_search(directory: &Path, pattern: &str) -> Result<()> {
    let matches = RegexSearchEngine::new(directory).search(pattern)?;

    if matches.is_empty() {
        println!("No regex matches found");
        return Ok(());
    }

    for match_ in &matches {
        print_regex_match(match_);
    }

    Ok(())
}

fn print_regex_match(match_: &RegexSearchMatch) {
    let line_start = match_.line_range.start();
    let line_end = match_.line_range.end();
    let matched_text = match_
        .matched_text
        .chars()
        .flat_map(char::escape_debug)
        .collect::<String>();

    println!(
        "{path}:bytes {byte_start}..{byte_end}:lines {line_start}..{line_end}: {matched_text}",
        path = match_.path.display(),
        byte_start = match_.byte_range.start,
        byte_end = match_.byte_range.end,
    );
}
