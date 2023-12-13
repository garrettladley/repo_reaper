use std::{collections::HashSet, sync::Mutex, thread};

use clap::Parser;
use notify::{event::ModifyKind, Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use repo_reaper::{
    cli::Args,
    inverted_index::InvertedIndex,
    ranking::RankingAlgorithm,
    text_transform::{n_gram_transform, Query},
};
use rust_stemmers::{Algorithm, Stemmer};

use std::sync::Arc;

fn get_stemmer() -> Arc<Stemmer> {
    Arc::new(Stemmer::create(Algorithm::English))
}

fn get_stop_words() -> Arc<HashSet<String>> {
    Arc::new(
        stop_words::get(stop_words::LANGUAGE::English)
            .par_iter()
            .map(|word| word.to_string())
            .collect(),
    )
}

fn main() -> notify::Result<()> {
    let stop_words = get_stop_words();
    let stop_words_clone = Arc::clone(&stop_words);

    let stemmer = get_stemmer();
    let stemmer_clone = Arc::clone(&stemmer);

    let args = Args::parse();

    let algo = args.ranking_algorithm;

    let transformer = Arc::new(move |content: &str| {
        n_gram_transform(content, &stemmer_clone, &stop_words_clone, args.n_grams)
    });
    let transformer_clone = Arc::clone(&transformer);

    let path_clone = args.directory.clone();

    println!("Indexing files in {}", args.directory.to_str().unwrap());

    let inverted_index = Arc::new(Mutex::new(InvertedIndex::new(
        args.directory,
        transformer_clone.as_ref(),
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
                            for path in &event.paths {
                                let path_clone = path.clone();
                                inverted_index_clone_for_thread
                                    .lock()
                                    .unwrap()
                                    .update(&path_clone, &transformer.as_ref());
                            }
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
            &Query::new(&query, &stemmer, &stop_words, args.n_grams),
            args.top_n,
        );

        match ranking {
            Some(ranking) => {
                ranking.iter().for_each(|rank| {
                    println!("{:?}", rank);
                });
            }
            None => println!("No results found :("),
        }
    }

    Ok(())
}
