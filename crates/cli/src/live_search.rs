use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Context, Result, anyhow};
use chrono::Utc;
use notify::{
    Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
    event::{ModifyKind, RemoveKind},
};
use repo_reaper_core::{
    config::Config as ReaperConfig,
    index::InvertedIndex,
    query::{AnalyzedQuery, QueryExpansionConfig},
    ranking::RankingAlgo,
    tokenizer::n_gram_transform,
};

pub(crate) fn run(
    directory: PathBuf,
    config: Arc<ReaperConfig>,
    algo: RankingAlgo,
    top_n: usize,
    query_expansion: bool,
    feedback_expansion: bool,
) -> Result<()> {
    let config_clone = Arc::clone(&config);
    let transformer = Arc::new(move |content: &str| n_gram_transform(content, &config_clone));
    let path_clone = directory.clone();

    println!("Indexing files in {}", directory.display());

    let inverted_index = if algo.needs_fielded_index() {
        Arc::new(Mutex::new(InvertedIndex::new_fielded(
            directory, &config, None,
        )))
    } else {
        let transformer_clone = Arc::clone(&transformer);
        Arc::new(Mutex::new(InvertedIndex::new(
            directory,
            transformer_clone.as_ref(),
            None,
        )))
    };

    println!(
        "Successfully indexed {} files",
        inverted_index
            .lock()
            .map_err(|_| anyhow!("index lock poisoned"))?
            .num_docs()
    );

    spawn_watcher(
        path_clone,
        Arc::clone(&inverted_index),
        transformer,
        Arc::clone(&config),
        algo.needs_fielded_index(),
    );
    run_repl(
        config,
        algo,
        top_n,
        inverted_index,
        query_expansion,
        feedback_expansion,
    )
}

fn spawn_watcher(
    path: PathBuf,
    inverted_index: Arc<Mutex<InvertedIndex>>,
    transformer: Arc<
        impl Fn(&str) -> HashMap<repo_reaper_core::index::Term, u32> + Send + Sync + 'static,
    >,
    config: Arc<ReaperConfig>,
    fielded: bool,
) {
    let (tx, rx) = std::sync::mpsc::channel();
    let rx = Arc::new(Mutex::new(rx));

    thread::spawn(move || {
        let mut watcher = match RecommendedWatcher::new(tx, Config::default()) {
            Ok(watcher) => watcher,
            Err(error) => {
                eprintln!("failed to create file watcher: {error}");
                return;
            }
        };

        if let Err(error) = watcher.watch(path.as_ref(), RecursiveMode::Recursive) {
            eprintln!("failed to watch directory: {error}");
            return;
        }

        loop {
            let event = match rx.lock() {
                Ok(rx) => rx.recv(),
                Err(_) => {
                    eprintln!("watch error: event receiver lock poisoned");
                    return;
                }
            };

            match event {
                Ok(Ok(event)) => match event.kind {
                    EventKind::Modify(ModifyKind::Metadata(_)) => continue,
                    EventKind::Remove(RemoveKind::File | RemoveKind::Any) => {
                        let Ok(mut index) = inverted_index.lock() else {
                            eprintln!("watch error: index lock poisoned");
                            return;
                        };

                        for path in &event.paths {
                            index.remove_document(path);
                        }
                    }
                    _ => {
                        let Ok(mut index) = inverted_index.lock() else {
                            eprintln!("watch error: index lock poisoned");
                            return;
                        };

                        for path in &event.paths {
                            if fielded {
                                index.update_fielded(path, &config);
                            } else {
                                index.update(path, &transformer.as_ref());
                            }
                        }
                    }
                },
                Ok(Err(error)) => eprintln!("watch error: {error:?}"),
                Err(error) => eprintln!("watch error: {error:?}"),
            }
        }
    });
}

fn run_repl(
    config: Arc<ReaperConfig>,
    algo: RankingAlgo,
    top_n: usize,
    inverted_index: Arc<Mutex<InvertedIndex>>,
    query_expansion: bool,
    feedback_expansion: bool,
) -> Result<()> {
    loop {
        println!("Enter a query: ");

        let mut query = String::new();
        std::io::stdin()
            .read_line(&mut query)
            .context("failed to read from stdin")?;

        if query.trim() == "exit" {
            break;
        }

        let query = if algo.needs_fielded_index() {
            AnalyzedQuery::new_code_search_with_expansion(
                &query,
                &config,
                QueryExpansionConfig {
                    controlled: query_expansion,
                    feedback: feedback_expansion,
                },
            )
        } else {
            AnalyzedQuery::new(&query, &config)
        };

        let ranking = {
            let index = inverted_index
                .lock()
                .map_err(|_| anyhow!("index lock poisoned"))?;
            if feedback_expansion {
                algo.rank_with_feedback(&index, &query, top_n, top_n.min(3), 6)
            } else {
                algo.rank(&index, &query, top_n)
            }
        };

        log_query(&query, &ranking, &algo, top_n)?;

        match ranking {
            Some(ranking) => {
                for rank in &ranking.0 {
                    println!("{rank}");
                }
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
            query_log.insert("ranking".to_string(), format!("{ranking:?}"));
        }
        None => {
            query_log.insert("ranking".to_string(), String::new());
        }
    }

    query_log.insert("ranking_algo".to_string(), format!("{algo:?}"));
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
