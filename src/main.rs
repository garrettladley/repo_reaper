use std::{collections::HashSet, sync::Mutex, thread};

use notify::{Config, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use repo_reaper::{
    inverted_index::InvertedIndex,
    ranking::{bm25::get_configuration, RankingAlgorithm, BM25},
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
    let stemmer = get_stemmer();
    let stop_words_clone = Arc::clone(&stop_words);
    let stemmer_clone = Arc::clone(&stemmer);

    let n_grams = 1;

    let path = "./src/";

    let transformer = Arc::new(move |content: &str| {
        n_gram_transform(content, &stemmer_clone, &stop_words_clone, n_grams)
    });
    let transformer_clone = Arc::clone(&transformer);
    let inverted_index = Arc::new(Mutex::new(InvertedIndex::new(
        path,
        transformer_clone.as_ref(),
    )));

    let (tx, rx) = std::sync::mpsc::channel();
    let rx = Arc::new(Mutex::new(rx));
    let rx_clone = Arc::clone(&rx);
    let path_clone = path.clone();

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
                        EventKind::Modify(modify_kind) => match modify_kind {
                            notify::event::ModifyKind::Metadata(_) => continue,
                            _ => {
                                for path in event.paths {
                                    let path_clone = path.clone();
                                    inverted_index_clone_for_thread
                                        .lock()
                                        .unwrap()
                                        .update(&path_clone, &transformer.as_ref());
                                    println!(
                                        "Successfully reindexed {}",
                                        path_clone.to_str().unwrap()
                                    );
                                }
                            }
                        },
                        _ => {
                            println!("Other event: {:?}", event);
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

        let algo = BM25 {
            hyper_params: get_configuration().unwrap(),
        };

        let top_n = 10;

        algo.rank(
            &inverted_index.lock().unwrap(),
            &Query::new(&query, &stemmer, &stop_words, n_grams),
            top_n,
        )
        .iter()
        .for_each(|rank| {
            println!("{:?}", rank);
        });
    }

    Ok(())
}
