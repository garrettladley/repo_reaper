use repo_reaper::{
    config::get_configuration,
    inverted_index::InvertedIndex,
    ranking::{tf_idf::TFIDF, RankingAlgorithm},
    text_transform::{transform, Query},
};
use rust_stemmers::{Algorithm, Stemmer};

fn main() {
    let stemmer = Stemmer::create(Algorithm::English);

    let x = get_configuration().unwrap();

    println!("k1: {} b:{}", x.k1, x.b);

    let inverted_index = InvertedIndex::new("./src/", |content: &str| transform(content, &stemmer));

    let query = "utils";

    let algo = TFIDF;

    let top_n = 10;

    for rank in algo.rank(&inverted_index, &Query::new(query, &stemmer), top_n) {
        println!("{:?}", rank);
    }
}
