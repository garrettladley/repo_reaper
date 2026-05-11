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
        FileSystemIndexCorpus, InvertedIndex, SearchEngine,
        event_log::{IndexEvent, append_event, clear_events, read_events, replay_events},
        inverted_file::InvertedFileLayout,
        snapshot::{load_snapshot, write_snapshot},
    },
    query::{AnalyzedQuery, QueryExpansionConfig},
    ranking::{RankingAlgo, Scored},
    tokenizer::n_gram_transform,
};

pub(crate) struct LiveSearchOptions {
    pub(crate) top_n: usize,
    pub(crate) query_expansion: bool,
    pub(crate) feedback_expansion: bool,
    pub(crate) index_dir: Option<PathBuf>,
    pub(crate) reindex: bool,
    pub(crate) respect_gitignore: bool,
}

pub(crate) fn run(
    directory: PathBuf,
    config: Arc<ReaperConfig>,
    algo: RankingAlgo,
    options: LiveSearchOptions,
) -> Result<()> {
    let prepared = prepare_ranked_search(&directory, Arc::clone(&config), &algo, &options, true)?;
    let transformer = build_transformer(Arc::clone(&config));

    spawn_watcher(
        directory,
        prepared.engine.clone(),
        transformer,
        Arc::clone(&config),
        prepared.fielded,
        options.index_dir,
    );
    run_repl(
        config,
        algo,
        options.top_n,
        prepared.engine,
        options.query_expansion,
        options.feedback_expansion,
    )
}

pub(crate) fn run_once(
    directory: PathBuf,
    config: Arc<ReaperConfig>,
    algo: RankingAlgo,
    options: LiveSearchOptions,
    query: &str,
) -> Result<()> {
    let prepared = prepare_ranked_search(&directory, Arc::clone(&config), &algo, &options, false)?;
    let analyzed_query = analyze_query(
        &config,
        &algo,
        query,
        options.query_expansion,
        options.feedback_expansion,
    );
    let ranking = search_ranked(
        &prepared.engine,
        &algo,
        &analyzed_query,
        options.top_n,
        options.feedback_expansion,
    )?;
    print_one_shot_results(&ranking);
    Ok(())
}

struct PreparedRankedSearch {
    engine: SearchEngine,
    fielded: bool,
}

fn prepare_ranked_search(
    directory: &PathBuf,
    config: Arc<ReaperConfig>,
    algo: &RankingAlgo,
    options: &LiveSearchOptions,
    verbose: bool,
) -> Result<PreparedRankedSearch> {
    let transformer = build_transformer(Arc::clone(&config));

    if verbose {
        println!("Indexing files in {}", directory.display());
    }

    let fielded = algo.needs_fielded_index();
    let index = if let Some(index_dir) = options.index_dir.as_ref().filter(|_| !options.reindex) {
        match load_snapshot(index_dir, directory, &config) {
            Ok(mut index) => {
                let events = read_events(index_dir).context("failed to read index event log")?;
                replay_events(&mut index, &events, transformer.as_ref(), &config, fielded);
                index
            }
            Err(error) => {
                eprintln!("rebuilding index because snapshot could not be loaded: {error}");
                build_index(
                    directory,
                    &config,
                    transformer.as_ref(),
                    fielded,
                    options.respect_gitignore,
                )
            }
        }
    } else {
        build_index(
            directory,
            &config,
            transformer.as_ref(),
            fielded,
            options.respect_gitignore,
        )
    };

    let engine = SearchEngine::new(index);

    if let Some(index_dir) = &options.index_dir {
        engine.with_read(|index| {
            write_snapshot(index, index_dir, directory, &config)?;
            InvertedFileLayout::write(index, index_dir)?;
            clear_events(index_dir)?;
            Ok::<_, anyhow::Error>(())
        })??;
    }

    if verbose {
        println!("Successfully indexed {} files", engine.num_docs()?);
    }

    Ok(PreparedRankedSearch { engine, fielded })
}

fn build_transformer(
    config: Arc<ReaperConfig>,
) -> Arc<impl Fn(&str) -> HashMap<repo_reaper_core::index::Term, u32> + Send + Sync + 'static> {
    Arc::new(move |content: &str| n_gram_transform(content, &config))
}

fn build_index(
    directory: &PathBuf,
    config: &ReaperConfig,
    transformer: &(impl Fn(&str) -> HashMap<repo_reaper_core::index::Term, u32> + Sync),
    fielded: bool,
    respect_gitignore: bool,
) -> InvertedIndex {
    let corpus = FileSystemIndexCorpus::new(directory, None::<&PathBuf>)
        .with_respect_gitignore(respect_gitignore);

    if fielded {
        InvertedIndex::from_corpus_fielded(&corpus, config).index
    } else {
        InvertedIndex::from_corpus(&corpus, transformer).index
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

        let query = analyze_query(&config, &algo, &query, query_expansion, feedback_expansion);
        let ranking = search_ranked(&engine, &algo, &query, top_n, feedback_expansion)?;

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

fn analyze_query(
    config: &ReaperConfig,
    algo: &RankingAlgo,
    query: &str,
    query_expansion: bool,
    feedback_expansion: bool,
) -> AnalyzedQuery {
    if algo.needs_fielded_index() {
        AnalyzedQuery::new_code_search_with_expansion(
            query,
            config,
            QueryExpansionConfig {
                controlled: query_expansion,
                feedback: feedback_expansion,
            },
        )
    } else {
        AnalyzedQuery::new(query, config)
    }
}

fn search_ranked(
    engine: &SearchEngine,
    algo: &RankingAlgo,
    query: &AnalyzedQuery,
    top_n: usize,
    feedback_expansion: bool,
) -> Result<Option<Scored>> {
    if feedback_expansion {
        Ok(engine
            .with_read(|index| algo.rank_with_feedback(index, query, top_n, top_n.min(3), 6))?)
    } else {
        Ok(engine.search(algo, query, top_n)?)
    }
}

fn print_one_shot_results(ranking: &Option<Scored>) {
    match ranking {
        Some(ranking) => {
            for score in &ranking.0 {
                println!("{}\tscore={:.6}", score.doc_path.display(), score.score);
            }
        }
        None => println!("No results found"),
    }
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
