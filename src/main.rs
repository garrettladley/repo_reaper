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

    let x = get_configuration().unwrap();

    println!("k1: {} b:{}", x.k1, x.b);

    let n = 1;

    let inverted_index = spin_it(
        || {
            InvertedIndex::new("../", |content: &str| {
                n_gram_transform(content, &stemmer, n)
            })
        },
        &mut Spinner::new(Spinners::Dots, "Indexing Documents".into()),
    );

    println!("Number of documents: {}", inverted_index.num_docs());

    let query = "utils";

    let algo = TFIDF;

    let top_n = 10;

    for rank in algo.rank(&inverted_index, &Query::new(query, &stemmer, n), top_n) {
        println!("{:?}", rank);
    }
}
