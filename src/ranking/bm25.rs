use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

use crate::inverted_index::InvertedIndex;
use crate::ranking::{utils::idf, RankingAlgorithm, Score, Scored};
use crate::text_transform::Query;

#[derive(serde::Deserialize, Debug, Clone)]
pub struct BM25HyperParams {
    pub k1: f64,
    pub b: f64,
}

pub fn get_configuration() -> Result<BM25HyperParams, config::ConfigError> {
    let base_path = std::env::current_dir().expect("Failed to determine the current directory");
    let configuration_directory = base_path.join("configuration");

    let settings = config::Config::builder()
        .add_source(config::File::from(configuration_directory.join("bm25.yml")))
        .build()?;

    settings.try_deserialize::<BM25HyperParams>()
}

pub struct BM25 {
    pub hyper_params: BM25HyperParams,
}

impl RankingAlgorithm for BM25 {
    fn score(&self, inverted_index: &InvertedIndex, query: &Query) -> Scored {
        let avgdl = inverted_index.avg_doc_length();
        let num_docs = inverted_index.num_docs();

        let scores = DashMap::new();

        query.0.par_iter().for_each(|term| {
            if let Some(documents) = inverted_index.0.get(term) {
                let idf = idf(num_docs, documents.len());

                documents
                    .iter()
                    .par_bridge()
                    .for_each(|(doc_path, term_doc)| {
                        let tf = term_doc.term_freq as f64;
                        let doc_length = term_doc.length as f64;
                        let score = idf * (tf * (self.hyper_params.k1 + 1.0))
                            / (tf
                                + self.hyper_params.k1
                                    * (1.0 - self.hyper_params.b
                                        + self.hyper_params.b * doc_length / avgdl));

                        *scores.entry(doc_path.clone()).or_insert(0.0) += score;
                    });
            }
        });

        Scored(
            scores
                .iter()
                .par_bridge()
                .map(|score| Score {
                    doc_path: score.key().to_owned(),
                    score: *score.value(),
                })
                .collect(),
        )
    }
}
