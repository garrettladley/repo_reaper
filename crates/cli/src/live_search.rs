use std::{
    collections::HashMap,
    env,
    fs::OpenOptions,
    io::{IsTerminal, Write},
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
        snapshot::{load_snapshot, snapshot_path, write_snapshot},
    },
    query::{AnalyzedQuery, QueryExpansionConfig},
    ranking::{RankingAlgo, Score, Scored},
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
    let ui = TerminalUi::new();

    if verbose {
        ui.status("index", &format!("{} preparing", directory.display()));
    }

    let fielded = algo.needs_fielded_index();
    let index = if let Some(index_dir) = options.index_dir.as_ref().filter(|_| !options.reindex) {
        let path = snapshot_path(index_dir);
        if path.exists() {
            match load_snapshot(index_dir, directory, &config) {
                Ok(mut index) => {
                    let events =
                        read_events(index_dir).context("failed to read index event log")?;
                    let event_count = events.len();
                    replay_events(&mut index, &events, transformer.as_ref(), &config, fielded);
                    if verbose {
                        ui.status(
                            "cache",
                            &format!(
                                "{} loaded{}",
                                index_dir.display(),
                                if event_count == 0 {
                                    String::new()
                                } else {
                                    format!(" + {event_count} pending updates")
                                }
                            ),
                        );
                    }
                    index
                }
                Err(error) => {
                    if verbose {
                        ui.status(
                            "cache",
                            &format!("{} snapshot unusable; rebuilding", index_dir.display()),
                        );
                        ui.detail(&error.to_string());
                    }
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
            if verbose {
                ui.status(
                    "cache",
                    &format!("{} no snapshot yet; building", index_dir.display()),
                );
            }
            build_index(
                directory,
                &config,
                transformer.as_ref(),
                fielded,
                options.respect_gitignore,
            )
        }
    } else {
        if verbose && options.reindex {
            ui.status("cache", "reindex requested; rebuilding");
        }
        build_index(
            directory,
            &config,
            transformer.as_ref(),
            fielded,
            options.respect_gitignore,
        )
    };

    let engine = SearchEngine::new(index);
    let document_count = engine.num_docs()?;

    if let Some(index_dir) = &options.index_dir {
        engine.with_read(|index| {
            write_snapshot(index, index_dir, directory, &config)?;
            InvertedFileLayout::write(index, index_dir)?;
            clear_events(index_dir)?;
            Ok::<_, anyhow::Error>(())
        })??;
    }

    if verbose {
        ui.status("ready", &format!("{document_count} files indexed"));
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
    let ui = TerminalUi::new();

    loop {
        ui.prompt()?;

        let mut query = String::new();
        let bytes_read = std::io::stdin()
            .read_line(&mut query)
            .context("failed to read from stdin")?;
        if bytes_read == 0 {
            ui.notice("bye");
            break;
        }

        let trimmed_query = query.trim();
        if matches!(trimmed_query, "exit" | "quit") {
            break;
        }
        if trimmed_query.is_empty() {
            println!();
            continue;
        }

        let query = analyze_query(&config, &algo, &query, query_expansion, feedback_expansion);
        let ranking = search_ranked(&engine, &algo, &query, top_n, feedback_expansion)?;

        log_query(&query, &ranking, &algo, top_n)?;

        match ranking {
            Some(ranking) => {
                print_results(&ranking, &ui);
                println!();
            }
            None => {
                ui.notice("no results found");
                println!();
            }
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
    let ui = TerminalUi::new();

    match ranking {
        Some(ranking) => print_results(ranking, &ui),
        None => ui.notice("no results found"),
    }
}

fn print_results(ranking: &Scored, ui: &TerminalUi) {
    for score in &ranking.0 {
        println!("{}", ui.format_score(score));
    }
}

struct TerminalUi {
    stdout_color: bool,
    stderr_color: bool,
}

impl TerminalUi {
    fn new() -> Self {
        let color_allowed = env::var_os("NO_COLOR").is_none();
        Self {
            stdout_color: color_allowed && std::io::stdout().is_terminal(),
            stderr_color: color_allowed && std::io::stderr().is_terminal(),
        }
    }

    fn status(&self, label: &str, message: &str) {
        let style = if label == "ready" {
            Style::GreenBold
        } else {
            Style::CyanBold
        };
        eprintln!(
            "{} {}",
            self.style_stderr(&format!("{label:<6}"), style),
            message
        );
    }

    fn notice(&self, message: &str) {
        println!(
            "{} {}",
            self.style_stdout(&format!("{:<6}", "notice"), Style::YellowBold),
            message
        );
    }

    fn detail(&self, message: &str) {
        eprintln!("       {}", self.style_stderr(message, Style::Dim));
    }

    fn prompt(&self) -> Result<()> {
        print!("{} ", self.style_stdout("query>", Style::CyanBold));
        std::io::stdout().flush().context("failed to flush prompt")
    }

    fn format_score(&self, score: &Score) -> String {
        format!(
            "{} {}",
            self.style_stdout(&score.doc_path.display().to_string(), Style::Path),
            self.style_stdout(&format!("score={:.6}", score.score), Style::Dim),
        )
    }

    fn style_stdout(&self, text: &str, style: Style) -> String {
        style.apply(text, self.stdout_color)
    }

    fn style_stderr(&self, text: &str, style: Style) -> String {
        style.apply(text, self.stderr_color)
    }
}

#[derive(Clone, Copy)]
enum Style {
    CyanBold,
    GreenBold,
    YellowBold,
    Dim,
    Path,
}

impl Style {
    fn apply(self, text: &str, enabled: bool) -> String {
        if !enabled {
            return text.to_string();
        }

        let code = match self {
            Style::CyanBold => "1;36",
            Style::GreenBold => "1;32",
            Style::YellowBold => "1;33",
            Style::Dim => "2",
            Style::Path => "36",
        };
        format!("\x1b[{code}m{text}\x1b[0m")
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
