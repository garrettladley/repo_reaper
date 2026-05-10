use std::{
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process::Command,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use clap::{Parser, ValueEnum};
use notify::{
    Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
    event::{ModifyKind, RemoveKind},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use repo_reaper_core::{
    config::Config as ReaperConfig,
    evaluation::{
        EvaluationCorpus, EvaluationData, RawEvaluationData, TestSet, dataset::Relevance,
        metrics::TestQuery,
    },
    index::InvertedIndex,
    query::AnalyzedQuery,
    ranking::RankingAlgo,
    tokenizer::n_gram_transform,
};
use rust_stemmers::{Algorithm, Stemmer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The directory to watch
    #[clap(short, long, default_value = ".")]
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
    /// Evaluation data JSON
    #[clap(long, default_value = "./data/train.json")]
    eval_data: PathBuf,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EvalOutputFormat {
    Pretty,
    Json,
}

fn main() -> Result<()> {
    let args = Args::parse();

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

    let config_clone = Arc::clone(&config);

    let algo = args.ranking_algorithm;

    let transformer = Arc::new(move |content: &str| n_gram_transform(content, &config_clone));
    let transformer_clone = Arc::clone(&transformer);

    let path_clone = args.directory.clone();

    println!("Indexing files in {}", args.directory.display());

    let inverted_index = Arc::new(Mutex::new(InvertedIndex::new(
        args.directory,
        transformer_clone.as_ref(),
        None,
    )));

    println!(
        "Successfully indexed {} files",
        inverted_index
            .lock()
            .map_err(|_| anyhow!("index lock poisoned"))?
            .num_docs()
    );

    let (tx, rx) = std::sync::mpsc::channel();
    let rx = Arc::new(Mutex::new(rx));
    let rx_clone = Arc::clone(&rx);

    let inverted_index_clone_for_thread = Arc::clone(&inverted_index);

    thread::spawn(move || {
        let mut watcher = match RecommendedWatcher::new(tx, Config::default()) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("failed to create file watcher: {e}");
                return;
            }
        };

        if let Err(e) = watcher.watch(path_clone.as_ref(), RecursiveMode::Recursive) {
            eprintln!("failed to watch directory: {e}");
            return;
        }

        loop {
            let event = match rx_clone.lock() {
                Ok(rx) => rx.recv(),
                Err(_) => {
                    eprintln!("watch error: event receiver lock poisoned");
                    return;
                }
            };

            match event {
                Ok(event) => match event {
                    Ok(event) => match event.kind {
                        EventKind::Modify(ModifyKind::Metadata(_)) => continue,
                        EventKind::Remove(RemoveKind::File | RemoveKind::Any) => {
                            let Ok(mut index) = inverted_index_clone_for_thread.lock() else {
                                eprintln!("watch error: index lock poisoned");
                                return;
                            };

                            event.paths.iter().for_each(|path| {
                                index.remove_document(path);
                            });
                        }
                        _ => {
                            let Ok(mut index) = inverted_index_clone_for_thread.lock() else {
                                eprintln!("watch error: index lock poisoned");
                                return;
                            };

                            event.paths.iter().for_each(|path| {
                                index.update(path, &transformer.as_ref());
                            });
                        }
                    },
                    Err(error) => eprintln!("watch error: {:?}", error),
                },
                Err(e) => {
                    eprintln!("watch error: {:?}", e);
                }
            }
        }
    });

    loop {
        println!("Enter a query: ");

        let mut query = String::new();
        std::io::stdin()
            .read_line(&mut query)
            .context("failed to read from stdin")?;

        if query.trim() == "exit" {
            break;
        }

        let query = AnalyzedQuery::new(&query, &config);

        let ranking = {
            let index = inverted_index
                .lock()
                .map_err(|_| anyhow!("index lock poisoned"))?;
            algo.rank(&index, &query, args.top_n)
        };

        log_query(&query, &ranking, &algo, args.top_n)?;

        match ranking {
            Some(ranking) => {
                ranking.0.iter().for_each(|rank| {
                    println!("{rank}");
                });
            }
            None => println!("No results found :("),
        }
    }

    Ok(())
}

fn log_query(
    query: &AnalyzedQuery,
    ranking: &Option<repo_reaper_core::ranking::Scored>,
    algo: &RankingAlgo,
    top_n: usize,
) -> Result<()> {
    let mut query_log = HashMap::new();

    query_log.insert("query".to_string(), query.to_string());
    query_log.insert("top_n".to_string(), top_n.to_string());

    match ranking {
        Some(ranking) => {
            query_log.insert("ranking".to_string(), format!("{:?}", ranking));
        }
        None => {
            query_log.insert("ranking".to_string(), "".to_string());
        }
    }

    query_log.insert("ranking_algo".to_string(), format!("{:?}", algo));

    query_log.insert(
        "timestamp".to_string(),
        Utc::now().format("%Y-%m-%d %H:%M:%S").to_string(),
    );

    let query_log = serde_json::to_string(&query_log).context("failed to serialize query log")?;

    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open("./query_log.txt")
        .context("failed to open query log file")?;

    file.write_all(query_log.as_bytes())
        .context("failed to write query log")?;

    Ok(())
}

fn evaluate_training(args: &Args, config: &ReaperConfig) -> Result<()> {
    let file_content = fs::read_to_string(&args.eval_data).with_context(|| {
        format!(
            "failed to read evaluation data from {}",
            args.eval_data.display()
        )
    })?;

    let raw_evaluation_data: RawEvaluationData = serde_json::from_str(&file_content)
        .context("failed to deserialize evaluation JSON data")?;

    let evaluation_data = EvaluationData::parse(raw_evaluation_data, config)
        .context("failed to parse evaluation data")?;

    let index_root = match &evaluation_data.corpus {
        EvaluationCorpus::Tree { root } => root.clone(),
        EvaluationCorpus::Git { repo, commit, root } => {
            let path = PathBuf::from("./data/repo");
            if path.exists() {
                fs::remove_dir_all(&path).context("failed to clean previous evaluation repo")?;
            }

            let clone_output = Command::new("git")
                .arg("clone")
                .arg(repo)
                .arg(&path)
                .output()
                .context("failed to execute git clone")?;

            ensure_command_succeeded("git clone", clone_output)?;

            let checkout_output = Command::new("git")
                .arg("checkout")
                .arg(commit)
                .current_dir(&path)
                .output()
                .context("failed to execute git checkout")?;

            ensure_command_succeeded("git checkout", checkout_output)?;

            path.join(root)
        }
    };

    let inverted_index = InvertedIndex::new(
        index_root.as_path(),
        |content: &str| n_gram_transform(content, config),
        Some(index_root.as_path()),
    );

    let queries = evaluation_data
        .examples
        .par_iter()
        .map(|example| {
            let mut relevant_docs: Vec<_> = example
                .results
                .iter()
                .filter_map(|result| match result.relevance {
                    Relevance::Relevant(rank) => Some((result.path.clone(), rank)),
                    Relevance::NonRelevant => None,
                })
                .collect();

            relevant_docs.sort_by_key(|a| a.1);

            TestQuery {
                query: example.query.clone(),
                relevant_docs: relevant_docs
                    .par_iter()
                    .map(|(path, _)| path.clone())
                    .collect::<Vec<_>>(),
            }
        })
        .collect();

    let evaluation = TestSet {
        ranking_algorithm: args.ranking_algorithm.clone(),
        queries,
    }
    .evaluate(&inverted_index, args.top_n);

    match args.eval_format {
        EvalOutputFormat::Pretty => println!("{evaluation}"),
        EvalOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&evaluation)
                .context("failed to serialize evaluation report")?;
            println!("{json}");
        }
    }

    Ok(())
}

fn ensure_command_succeeded(command: &str, output: std::process::Output) -> Result<()> {
    if output.status.success() {
        return Ok(());
    }

    bail!(
        "{command} failed with status {status}: {stderr}",
        status = output.status,
        stderr = String::from_utf8_lossy(&output.stderr).trim()
    );
}
