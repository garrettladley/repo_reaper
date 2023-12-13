use repo_reaper::{
    cli::spin_it,
    inverted_index::InvertedIndex,
    ranking::{bm25::get_configuration, tf_idf::TFIDF, RankingAlgorithm},
    text_transform::{n_gram_transform, Query},
};
use rust_stemmers::{Algorithm, Stemmer};
use spinners::{Spinner, Spinners};

fn main() {
    let stemmer = Stemmer::create(Algorithm::English);
    let stop_words = stop_words::get(stop_words::LANGUAGE::English);

    let x = get_configuration().unwrap();

    println!("k1: {} b: {}", x.k1, x.b);

    let n_grams = 1;

    let inverted_index = spin_it(
        || {
            InvertedIndex::new("../", |content: &str| {
                n_gram_transform(content, &stemmer, &stop_words, n_grams)
            })
        },
        &mut Spinner::new(Spinners::Dots, "Indexing Documents".into()),
    );

    println!("Number of documents: {}", inverted_index.num_docs());

    let query = "utils";

    let algo = TFIDF;

    let top_n = 10;

    for rank in algo.rank(
        &inverted_index,
        &Query::new(query, &stemmer, &stop_words, n_grams),
        top_n,
    ) {
        println!("{:?}", rank);
    }
}
