use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Context, Result};
use chrono::Utc;
use notify::{
    Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
    event::{ModifyKind, RemoveKind},
};
use repo_reaper_core::{
    config::Config as ReaperConfig,
    index::{
        InvertedIndex, SearchEngine,
        event_log::{IndexEvent, append_event, clear_events, read_events, replay_events},
        inverted_file::InvertedFileLayout,
        snapshot::{load_snapshot, write_snapshot},
    },
    query::AnalyzedQuery,
    ranking::RankingAlgo,
    tokenizer::n_gram_transform,
};

pub(crate) fn run(
    directory: PathBuf,
    config: Arc<ReaperConfig>,
    algo: RankingAlgo,
    top_n: usize,
    index_dir: Option<PathBuf>,
    reindex: bool,
) -> Result<()> {
    let config_clone = Arc::clone(&config);
    let transformer = Arc::new(move |content: &str| n_gram_transform(content, &config_clone));
    let path_clone = directory.clone();

    println!("Indexing files in {}", directory.display());

    let fielded = algo.needs_fielded_index();
    let index = if let Some(index_dir) = index_dir.as_ref().filter(|_| !reindex) {
        match load_snapshot(index_dir, &directory, &config) {
            Ok(mut index) => {
                let events = read_events(index_dir).context("failed to read index event log")?;
                replay_events(&mut index, &events, transformer.as_ref(), &config, fielded);
                index
            }
            Err(error) => {
                eprintln!("rebuilding index because snapshot could not be loaded: {error}");
                build_index(&directory, &config, transformer.as_ref(), fielded)
            }
        }
    } else {
        build_index(&directory, &config, transformer.as_ref(), fielded)
    };

    let engine = SearchEngine::new(index);

    if let Some(index_dir) = &index_dir {
        engine.with_read(|index| {
            write_snapshot(index, index_dir, &directory, &config)?;
            InvertedFileLayout::write(index, index_dir)?;
            clear_events(index_dir)?;
            Ok::<_, anyhow::Error>(())
        })??;
    }

    println!("Successfully indexed {} files", engine.num_docs()?);

    spawn_watcher(
        path_clone,
        engine.clone(),
        transformer,
        Arc::clone(&config),
        fielded,
        index_dir,
    );
    run_repl(config, algo, top_n, engine)
}

fn build_index(
    directory: &PathBuf,
    config: &ReaperConfig,
    transformer: &(impl Fn(&str) -> HashMap<repo_reaper_core::index::Term, u32> + Sync),
    fielded: bool,
) -> InvertedIndex {
    if fielded {
        InvertedIndex::new_fielded(directory, config, None)
    } else {
        InvertedIndex::new(directory, transformer, None)
    }
}

fn spawn_watcher(
    path: PathBuf,
    engine: SearchEngine,
    transformer: Arc<
        impl Fn(&str) -> HashMap<repo_reaper_core::index::Term, u32> + Send + Sync + 'static,
    >,
    config: Arc<ReaperConfig>,
    fielded: bool,
    index_dir: Option<PathBuf>,
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
                        for path in &event.paths {
                            let event = IndexEvent::FileDeleted { path: path.clone() };
                            if let Some(index_dir) = &index_dir
                                && let Err(error) = append_event(index_dir, &event)
                            {
                                eprintln!("watch error: failed to append index event: {error}");
                                return;
                            }
                            if let Err(error) =
                                engine.apply_event(&event, transformer.as_ref(), &config, fielded)
                            {
                                eprintln!("watch error: {error}");
                                return;
                            }
                        }
                    }
                    _ => {
                        for path in &event.paths {
                            let event = if path.exists() {
                                IndexEvent::FileModified { path: path.clone() }
                            } else {
                                IndexEvent::FileDeleted { path: path.clone() }
                            };
                            if let Some(index_dir) = &index_dir
                                && let Err(error) = append_event(index_dir, &event)
                            {
                                eprintln!("watch error: failed to append index event: {error}");
                                return;
                            }
                            if let Err(error) =
                                engine.apply_event(&event, transformer.as_ref(), &config, fielded)
                            {
                                eprintln!("watch error: {error}");
                                return;
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
    engine: SearchEngine,
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
            AnalyzedQuery::new_code_search(&query, &config)
        } else {
            AnalyzedQuery::new(&query, &config)
        };

        let ranking = engine.search(&algo, &query, top_n)?;

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
