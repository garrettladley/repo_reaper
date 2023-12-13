use repo_reaper::{
    cli::spin_it,
    inverted_index::InvertedIndex,
    ranking::{bm25::get_configuration, RankingAlgorithm, BM25},
    text_transform::{n_gram_transform, Query},
};
use rust_stemmers::{Algorithm, Stemmer};
use spinners::{Spinner, Spinners};

fn main() {
    let stemmer = Stemmer::create(Algorithm::English);
    let stop_words = stop_words::get(stop_words::LANGUAGE::English);

    let hyper_params = get_configuration().unwrap();

    println!("k1: {} b: {}", hyper_params.k1, hyper_params.b);

    let n_grams = 1;

    let inverted_index = spin_it(
        || {
            InvertedIndex::new("../", |content: &str| {
                n_gram_transform(content, &stemmer, &stop_words, n_grams)
            })
        },
        &mut Spinner::new(Spinners::Dots, "Indexing Documents".into()),
    );

    println!("\nNumber of documents: {}", inverted_index.num_docs());

    let query = "tokenization and stemming";

    let algo = BM25 { hyper_params };

    let top_n = 10;

    for rank in algo.rank(
        &inverted_index,
        &Query::new(query, &stemmer, &stop_words, n_grams),
        top_n,
    ) {
        println!("{:?}", rank);
    }
}
