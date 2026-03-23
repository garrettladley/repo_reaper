use std::{collections::HashMap, path::PathBuf};

use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    error::RankingError,
    index::{InvertedIndex, TermDocument},
    query::Query,
    ranking::{scorer::Scorer, utils::idf},
};

#[derive(serde::Deserialize, Debug, Clone)]
pub struct BM25HyperParams {
    pub k1: f64,
    pub b: f64,
}

pub fn get_configuration() -> Result<BM25HyperParams, RankingError> {
    let base_path = std::env::current_dir().map_err(RankingError::CurrentDir)?;
    let configuration_directory = base_path.join("configuration");

    let settings = config::Config::builder()
        .add_source(config::File::from(configuration_directory.join("bm25.yml")))
        .build()?;

    Ok(settings.try_deserialize::<BM25HyperParams>()?)
}

pub struct BM25 {
    pub hyper_params: BM25HyperParams,
}

impl Scorer for BM25 {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        _: &Query,
        documents: &HashMap<PathBuf, TermDocument>,
        scores: &DashMap<PathBuf, f64>,
    ) {
        let avgdl = inverted_index.avg_doc_length();
        let num_docs = inverted_index.num_docs();

        let idf = idf(num_docs, documents.len());

        documents.par_iter().for_each(|(doc_path, term_doc)| {
            let tf = term_doc.term_freq as f64;
            let doc_length = term_doc.length as f64;
            let score = idf * (tf * (self.hyper_params.k1 + 1.0))
                / (tf
                    + self.hyper_params.k1
                        * (1.0 - self.hyper_params.b + self.hyper_params.b * doc_length / avgdl));

            *scores.entry(doc_path.clone()).or_insert(0.0) += score;
        });
    }
}
