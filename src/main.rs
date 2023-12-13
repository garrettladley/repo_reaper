use std::{collections::HashSet, fs, process::Command, sync::Mutex, thread};

use clap::Parser;
use notify::{event::ModifyKind, Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use repo_reaper::{
    cli::Args,
    evaluation::{
        evaluation_data::Relevance, ranking_evaluation::TestQuery, EvaluationData,
        RawEvaluationData, TestSet,
    },
    globals::Globals,
    inverted_index::InvertedIndex,
    ranking::RankingAlgorithm,
    text_transform::{n_gram_transform, Query},
};
use rust_stemmers::{Algorithm, Stemmer};

use std::sync::Arc;

fn main() {
    let args = Args::parse();

    let globals = Arc::new(Globals {
        n_grams: args.n_grams,
        stemmer: Stemmer::create(Algorithm::English),
        stop_words: stop_words::get(stop_words::LANGUAGE::English)
            .par_iter()
            .map(|word| word.to_string())
            .collect::<HashSet<String>>(),
    });

    if args.evaluate {
        evaluate_training(&args, &globals);

        return;
    }

    let globals_clone = Arc::clone(&globals);

    let algo = args.ranking_algorithm;

    let transformer = Arc::new(move |content: &str| n_gram_transform(content, &globals_clone));
    let transformer_clone = Arc::clone(&transformer);

    let path_clone = args.directory.clone();

    println!("Indexing files in {}", args.directory.to_str().unwrap());

    let inverted_index = Arc::new(Mutex::new(InvertedIndex::new(
        args.directory,
        transformer_clone.as_ref(),
        None,
    )));

    println!(
        "Successfully indexed {} files",
        inverted_index.lock().unwrap().num_docs()
    );

    let (tx, rx) = std::sync::mpsc::channel();
    let rx = Arc::new(Mutex::new(rx));
    let rx_clone = Arc::clone(&rx);

    let inverted_index_clone_for_thread = Arc::clone(&inverted_index);

    thread::spawn(move || {
        let mut watcher = RecommendedWatcher::new(tx, Config::default()).unwrap();
        watcher
            .watch(path_clone.as_ref(), RecursiveMode::Recursive)
            .unwrap();

        loop {
            match rx_clone.lock().unwrap().recv() {
                Ok(event) => match event {
                    Ok(event) => match event.kind {
                        EventKind::Modify(ModifyKind::Metadata(_)) => continue,
                        _ => {
                            event.paths.par_iter().for_each(|path| {
                                let path_clone = path.clone();
                                inverted_index_clone_for_thread
                                    .lock()
                                    .unwrap()
                                    .update(&path_clone, &transformer.as_ref());
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
        std::io::stdin().read_line(&mut query).unwrap();

        if query.trim() == "exit" {
            break;
        }

        let ranking = algo.rank(
            &inverted_index.lock().unwrap(),
            &Query::new(&query, &globals),
            args.top_n,
        );

        match ranking {
            Some(ranking) => {
                ranking.0.iter().for_each(|rank| {
                    println!("{:?}", rank);
                });
            }
            None => println!("No results found :("),
        }
    }
}

fn evaluate_training(args: &Args, globals: &Globals) {
    let file_content = fs::read_to_string("./data/train.json").expect("Failed to read file");

    let raw_evaluation_data: RawEvaluationData =
        serde_json::from_str(&file_content).expect("Failed to deserialize JSON data");

    let evaluation_data = EvaluationData::parse(raw_evaluation_data, globals);

    let path = "./data/repo";

    Command::new("git")
        .arg("clone")
        .arg(evaluation_data.repo.to_string())
        .arg(path)
        .output()
        .expect("Failed to execute command");

    Command::new("git")
        .arg("checkout")
        .arg(evaluation_data.commit)
        .current_dir(path)
        .output()
        .expect("Failed to execute command");

    let inverted_index = InvertedIndex::new(
        path,
        |content: &str| n_gram_transform(content, globals),
        Some(path),
    );

    let queries = evaluation_data
        .examples
        .par_iter()
        .map(|example| {
            let relevant_docs = example
                .results
                .iter()
                .filter(|result| matches!(result.relevance, Relevance::Relevant(_)))
                .map(|result| result.path.clone())
                .collect();

            TestQuery {
                query: example.query.clone(),
                relevant_docs,
            }
        })
        .collect::<Vec<TestQuery>>();

    println!(
        "Evaluation: {:?}",
        TestSet {
            ranking_algorithm: Box::new(args.ranking_algorithm.clone()),
            queries
        }
        .evaluate(&inverted_index, args.top_n)
    );
}
