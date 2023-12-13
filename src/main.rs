use repo_reaper::{
    inverted_index::InvertedIndex,
    ranking::{bm25::get_configuration, tf_idf::TFIDF, RankingAlgorithm},
    text_transform::{n_gram_transform, Query},
};
use rust_stemmers::{Algorithm, Stemmer};

fn main() {
    let stemmer = Stemmer::create(Algorithm::English);

    let x = get_configuration().unwrap();

    println!("k1: {} b:{}", x.k1, x.b);

    let n = 2;

    let inverted_index = InvertedIndex::new("./src/", |content: &str| {
        n_gram_transform(content, &stemmer, n)
    });

    let query = "utils";

    let algo = TFIDF;

    let top_n = 10;

    for rank in algo.rank(&inverted_index, &Query::new(query, &stemmer, n), top_n) {
        println!("{:?}", rank);
    }
}
