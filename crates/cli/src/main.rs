use std::{
    collections::{HashMap, HashSet},
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    process::Command as ProcessCommand,
    sync::{Arc, Mutex},
    thread,
};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use notify::{
    Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
    event::{ModifyKind, RemoveKind},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use repo_reaper_core::{
    config::Config as ReaperConfig,
    evaluation::{
        Evaluation, EvaluationCorpus, EvaluationData, EvaluationReport, RawEvaluationData, TestSet,
        dataset::Relevance,
        metrics::{GroundednessResult, TestQuery, TokenEfficiencyEvaluation},
    },
    index::{CorpusStats, InvertedIndex},
    query::AnalyzedQuery,
    ranking::RankingAlgo,
    regex_search::{RegexSearchEngine, RegexSearchMatch},
    tokenizer::n_gram_transform,
};
use rust_stemmers::{Algorithm, Stemmer};

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

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EvalOutputFormat {
    Pretty,
    Json,
}

#[derive(Debug, serde::Serialize)]
struct EvaluationComparison {
    file_retrieval: FileRetrievalComparison,
    evidence: EvidenceComparison,
}

#[derive(Debug, serde::Serialize)]
struct FileRetrievalComparison {
    aggregate: EvaluationDelta,
    queries: Vec<QueryEvaluationDelta>,
    regressions: Vec<MetricRegression>,
}

#[derive(Debug, serde::Serialize)]
struct QueryEvaluationDelta {
    query: String,
    metrics: EvaluationDelta,
}

#[derive(Debug, serde::Serialize)]
struct EvaluationDelta {
    precision_at_k: MetricDelta,
    recall_at_k: MetricDelta,
    mean_average_precision: MetricDelta,
    mean_reciprocal_rank: MetricDelta,
    normalized_discounted_cumulative_gain: MetricDelta,
}

#[derive(Debug, serde::Serialize)]
struct MetricDelta {
    baseline: f64,
    current: f64,
    delta: f64,
}

#[derive(Debug, serde::Serialize)]
struct MetricRegression {
    scope: String,
    metric: String,
    delta: f64,
}

#[derive(Debug, serde::Serialize)]
struct EvidenceComparison {
    status: String,
    baseline_queries_with_evidence: usize,
    current_queries_with_evidence: usize,
    baseline_total_evidence_spans: usize,
    current_total_evidence_spans: usize,
    token_efficiency: TokenEfficiencyComparison,
}

#[derive(Debug, serde::Serialize)]
struct TokenEfficiencyComparison {
    baseline_returned_bytes: usize,
    current_returned_bytes: usize,
    baseline_estimated_returned_tokens: usize,
    current_estimated_returned_tokens: usize,
    baseline_relevant_evidence_bytes: usize,
    current_relevant_evidence_bytes: usize,
    baseline_evidence_density: f64,
    current_evidence_density: f64,
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

    if args.stats {
        let stats = inverted_index
            .lock()
            .map_err(|_| anyhow!("index lock poisoned"))?
            .corpus_stats(10);
        print_corpus_stats(&stats);
        return Ok(());
    }

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
            let path = args.eval_workdir.clone();
            prepare_git_eval_workdir(&path, args.fresh)?;

            let clone_output = ProcessCommand::new("git")
                .arg("clone")
                .arg(repo)
                .arg(&path)
                .output()
                .context("failed to execute git clone")?;

            ensure_command_succeeded("git clone", clone_output)?;

            let checkout_output = ProcessCommand::new("git")
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
                .filter_map(|result| match &result.relevance {
                    Relevance::Relevant(rank) => Some((result.path.clone(), *rank)),
                    Relevance::NonRelevant => None,
                })
                .collect();

            relevant_docs.sort_by_key(|a| a.1);
            let groundedness_results = example
                .results
                .iter()
                .map(|result| GroundednessResult {
                    path: result.path.clone(),
                    relevant: matches!(&result.relevance, Relevance::Relevant(_)),
                    evidence: result.evidence.clone(),
                })
                .collect();

            TestQuery {
                query: example.query.clone(),
                query_shape: example.query_shape,
                relevant_docs: relevant_docs
                    .par_iter()
                    .map(|(path, _)| path.clone())
                    .collect::<Vec<_>>(),
                groundedness_results,
                evidence_span_count: example
                    .results
                    .iter()
                    .map(|result| result.evidence.len())
                    .sum(),
            }
        })
        .collect();

    let evaluation_report = TestSet {
        ranking_algorithm: args.ranking_algorithm.clone(),
        queries,
    }
    .evaluate_report(&inverted_index, args.top_n);

    if let Some(baseline_path) = &args.eval_compare {
        let baseline = read_evaluation_baseline(baseline_path)?;
        let mut comparison = compare_evaluation_reports(&baseline, &evaluation_report);
        comparison.file_retrieval.regressions =
            collect_regressions(&comparison, args.eval_regression_threshold);

        match args.eval_format {
            EvalOutputFormat::Pretty => print_evaluation_comparison(&comparison),
            EvalOutputFormat::Json => {
                let json = serde_json::to_string_pretty(&comparison)
                    .context("failed to serialize evaluation comparison")?;
                println!("{json}");
            }
        }

        if !comparison.file_retrieval.regressions.is_empty() {
            bail!(
                "evaluation regressed beyond threshold {}: {}",
                args.eval_regression_threshold,
                comparison
                    .file_retrieval
                    .regressions
                    .iter()
                    .map(|regression| format!(
                        "{} {} ({:+.4})",
                        regression.scope, regression.metric, regression.delta
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }

        return Ok(());
    }

    match args.eval_format {
        EvalOutputFormat::Pretty => print_pretty_evaluation(&evaluation_report),
        EvalOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&evaluation_report)
                .context("failed to serialize evaluation report")?;
            println!("{json}");
        }
    }

    Ok(())
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

fn print_pretty_evaluation(evaluation: &repo_reaper_core::evaluation::metrics::EvaluationReport) {
    println!("{}", evaluation.file_retrieval.aggregate);
    println!("\ngroundedness:");
    println!("{}", evaluation.evidence.groundedness);
    println!("\ntoken efficiency:");
    println!("{}", evaluation.evidence.token_efficiency);

    if evaluation.file_retrieval.slices.is_empty() {
        return;
    }

    println!("\nquery-shape slices:");
    for slice in &evaluation.file_retrieval.slices {
        println!(
            "  {shape} ({family}, n={count}): MAP@{k}: {map:.4}, MRR@{k}: {mrr:.4}, NDCG@{k}: {ndcg:.4}",
            shape = slice.query_shape,
            family = slice.metric_family,
            count = slice.query_count,
            k = slice.metrics.k,
            map = slice.metrics.mean_average_precision,
            mrr = slice.metrics.mean_reciprocal_rank,
            ndcg = slice.metrics.normalized_discounted_cumulative_gain,
        );
    }
}

fn read_evaluation_baseline(path: &Path) -> Result<EvaluationReport> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read eval baseline from {}", path.display()))?;

    if let Ok(report) = serde_json::from_str::<EvaluationReport>(&content) {
        return Ok(report);
    }

    let aggregate: Evaluation = serde_json::from_str(&content)
        .context("failed to parse eval baseline as an evaluation report or legacy aggregate")?;

    Ok(EvaluationReport {
        file_retrieval: repo_reaper_core::evaluation::metrics::FileRetrievalReport {
            aggregate,
            slices: Vec::new(),
            queries: Vec::new(),
        },
        evidence: repo_reaper_core::evaluation::metrics::EvidenceReport {
            status: repo_reaper_core::evaluation::metrics::EvidenceReportStatus::NotScored,
            queries_with_evidence: 0,
            total_evidence_spans: 0,
            groundedness: repo_reaper_core::evaluation::metrics::GroundednessEvaluation::default(),
            token_efficiency: TokenEfficiencyEvaluation::default(),
        },
    })
}

fn compare_evaluation_reports(
    baseline: &EvaluationReport,
    current: &EvaluationReport,
) -> EvaluationComparison {
    let aggregate = compare_evaluations(
        &baseline.file_retrieval.aggregate,
        &current.file_retrieval.aggregate,
    );
    let queries = current
        .file_retrieval
        .queries
        .iter()
        .filter_map(|current_query| {
            let baseline_query = baseline
                .file_retrieval
                .queries
                .iter()
                .find(|baseline_query| baseline_query.query == current_query.query)?;

            Some(QueryEvaluationDelta {
                query: current_query.query.clone(),
                metrics: compare_evaluations(&baseline_query.metrics, &current_query.metrics),
            })
        })
        .collect::<Vec<_>>();

    EvaluationComparison {
        file_retrieval: FileRetrievalComparison {
            aggregate,
            queries,
            regressions: Vec::new(),
        },
        evidence: EvidenceComparison {
            status: current.evidence.status.to_string(),
            baseline_queries_with_evidence: baseline.evidence.queries_with_evidence,
            current_queries_with_evidence: current.evidence.queries_with_evidence,
            baseline_total_evidence_spans: baseline.evidence.total_evidence_spans,
            current_total_evidence_spans: current.evidence.total_evidence_spans,
            token_efficiency: compare_token_efficiency(
                &baseline.evidence.token_efficiency,
                &current.evidence.token_efficiency,
            ),
        },
    }
}

fn compare_token_efficiency(
    baseline: &TokenEfficiencyEvaluation,
    current: &TokenEfficiencyEvaluation,
) -> TokenEfficiencyComparison {
    TokenEfficiencyComparison {
        baseline_returned_bytes: baseline.returned_bytes,
        current_returned_bytes: current.returned_bytes,
        baseline_estimated_returned_tokens: baseline.estimated_returned_tokens,
        current_estimated_returned_tokens: current.estimated_returned_tokens,
        baseline_relevant_evidence_bytes: baseline.relevant_evidence_bytes,
        current_relevant_evidence_bytes: current.relevant_evidence_bytes,
        baseline_evidence_density: baseline.evidence_density,
        current_evidence_density: current.evidence_density,
    }
}

fn compare_evaluations(baseline: &Evaluation, current: &Evaluation) -> EvaluationDelta {
    EvaluationDelta {
        precision_at_k: metric_delta(baseline.precision_at_k, current.precision_at_k),
        recall_at_k: metric_delta(baseline.recall_at_k, current.recall_at_k),
        mean_average_precision: metric_delta(
            baseline.mean_average_precision,
            current.mean_average_precision,
        ),
        mean_reciprocal_rank: metric_delta(
            baseline.mean_reciprocal_rank,
            current.mean_reciprocal_rank,
        ),
        normalized_discounted_cumulative_gain: metric_delta(
            baseline.normalized_discounted_cumulative_gain,
            current.normalized_discounted_cumulative_gain,
        ),
    }
}

fn metric_delta(baseline: f64, current: f64) -> MetricDelta {
    let raw_delta = current - baseline;
    let delta = if raw_delta.abs() < 1e-12 {
        0.0
    } else {
        raw_delta
    };

    MetricDelta {
        baseline,
        current,
        delta,
    }
}

fn collect_regressions(comparison: &EvaluationComparison, threshold: f64) -> Vec<MetricRegression> {
    let mut regressions = Vec::new();
    push_key_metric_regressions(
        &mut regressions,
        "aggregate",
        &comparison.file_retrieval.aggregate,
        threshold,
    );

    for query in &comparison.file_retrieval.queries {
        push_key_metric_regressions(&mut regressions, &query.query, &query.metrics, threshold);
    }

    regressions
}

fn push_key_metric_regressions(
    regressions: &mut Vec<MetricRegression>,
    scope: &str,
    delta: &EvaluationDelta,
    threshold: f64,
) {
    for (metric, value) in [
        ("recall_at_k", delta.recall_at_k.delta),
        ("mean_average_precision", delta.mean_average_precision.delta),
        ("mean_reciprocal_rank", delta.mean_reciprocal_rank.delta),
        (
            "normalized_discounted_cumulative_gain",
            delta.normalized_discounted_cumulative_gain.delta,
        ),
    ] {
        if value < -threshold {
            regressions.push(MetricRegression {
                scope: scope.to_string(),
                metric: metric.to_string(),
                delta: value,
            });
        }
    }
}

fn print_evaluation_comparison(comparison: &EvaluationComparison) {
    println!("file retrieval aggregate deltas:");
    print_metric_delta("P", &comparison.file_retrieval.aggregate.precision_at_k);
    print_metric_delta("R", &comparison.file_retrieval.aggregate.recall_at_k);
    print_metric_delta(
        "MAP",
        &comparison.file_retrieval.aggregate.mean_average_precision,
    );
    print_metric_delta(
        "MRR",
        &comparison.file_retrieval.aggregate.mean_reciprocal_rank,
    );
    print_metric_delta(
        "NDCG",
        &comparison
            .file_retrieval
            .aggregate
            .normalized_discounted_cumulative_gain,
    );

    if comparison.file_retrieval.queries.is_empty() {
        println!("per-query deltas: no comparable query baselines");
    } else {
        println!("per-query deltas:");
        for query in &comparison.file_retrieval.queries {
            println!("  {}", query.query);
            print_metric_delta("  MAP", &query.metrics.mean_average_precision);
            print_metric_delta("  MRR", &query.metrics.mean_reciprocal_rank);
            print_metric_delta(
                "  NDCG",
                &query.metrics.normalized_discounted_cumulative_gain,
            );
        }
    }

    println!(
        "evidence metrics: {} (queries with evidence: {} -> {}, spans: {} -> {})",
        comparison.evidence.status,
        comparison.evidence.baseline_queries_with_evidence,
        comparison.evidence.current_queries_with_evidence,
        comparison.evidence.baseline_total_evidence_spans,
        comparison.evidence.current_total_evidence_spans
    );
    println!(
        "token efficiency: returned bytes {} -> {}, estimated tokens {} -> {}, evidence density {:.4} -> {:.4}",
        comparison.evidence.token_efficiency.baseline_returned_bytes,
        comparison.evidence.token_efficiency.current_returned_bytes,
        comparison
            .evidence
            .token_efficiency
            .baseline_estimated_returned_tokens,
        comparison
            .evidence
            .token_efficiency
            .current_estimated_returned_tokens,
        comparison
            .evidence
            .token_efficiency
            .baseline_evidence_density,
        comparison
            .evidence
            .token_efficiency
            .current_evidence_density
    );
}

fn print_metric_delta(name: &str, delta: &MetricDelta) {
    println!(
        "  {name}: {baseline:.4} -> {current:.4} ({delta:+.4})",
        baseline = delta.baseline,
        current = delta.current,
        delta = delta.delta
    );
}

fn prepare_git_eval_workdir(path: &Path, fresh: bool) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }

    if !fresh {
        bail!(
            "evaluation workdir {} already exists; pass --fresh to remove and reclone it, or choose a different --eval-workdir",
            path.display()
        );
    }

    if !path.is_dir() {
        bail!(
            "refusing to remove non-directory evaluation workdir {}",
            path.display()
        );
    }

    fs::remove_dir_all(path).with_context(|| {
        format!(
            "failed to clean previous evaluation workdir {}",
            path.display()
        )
    })?;

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

#[cfg(test)]
mod tests {
    use super::prepare_git_eval_workdir;

    #[test]
    fn prepare_git_eval_workdir_allows_missing_directory() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("repo");

        prepare_git_eval_workdir(&path, false).unwrap();

        assert!(!path.exists());
    }

    #[test]
    fn prepare_git_eval_workdir_refuses_existing_directory_without_fresh() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("repo");
        std::fs::create_dir(&path).unwrap();

        let error = prepare_git_eval_workdir(&path, false).unwrap_err();

        assert!(error.to_string().contains("already exists; pass --fresh"));
        assert!(path.exists());
    }

    #[test]
    fn prepare_git_eval_workdir_removes_existing_directory_with_fresh() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("repo");
        std::fs::create_dir(&path).unwrap();
        std::fs::write(path.join("README.md"), "old clone").unwrap();

        prepare_git_eval_workdir(&path, true).unwrap();

        assert!(!path.exists());
    }

    #[test]
    fn prepare_git_eval_workdir_refuses_non_directory_with_fresh() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("repo");
        std::fs::write(&path, "not a directory").unwrap();

        let error = prepare_git_eval_workdir(&path, true).unwrap_err();

        assert!(
            error
                .to_string()
                .contains("refusing to remove non-directory")
        );
        assert!(path.exists());
    }
}
