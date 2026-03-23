use std::{collections::HashMap, path::PathBuf};

use dashmap::DashMap;
use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};

use crate::{
    index::{InvertedIndex, TermDocument},
    query::Query,
    ranking::{scorer::Scorer, utils::idf},
};

pub struct CosineSimilarity;

impl Scorer for CosineSimilarity {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        query: &Query,
        documents: &HashMap<PathBuf, TermDocument>,
        scores: &DashMap<PathBuf, f64>,
    ) {
        let num_docs = inverted_index.num_docs();

        let query_magnitude = query
            .0
            .par_iter()
            .map(|(term, _)| {
                let idf = idf(num_docs, inverted_index.doc_freq(term));
                idf * idf
            })
            .sum::<f64>()
            .sqrt();

        let idf = idf(num_docs, documents.len());

        documents
            .iter()
            .par_bridge()
            .for_each(|(doc_path, term_doc)| {
                let tf = term_doc.term_freq as f64;

                let score = tf * idf;

                let doc_magnitude = inverted_index
                    .0
                    .iter()
                    .par_bridge()
                    .filter_map(|(_, doc_map)| {
                        doc_map
                            .par_iter()
                            .find_any(|(path, _)| *path == doc_path)
                            .map(|(_, term_doc)| {
                                let tf = term_doc.term_freq as f64;
                                tf * tf
                            })
                    })
                    .sum::<f64>()
                    .sqrt();

                let cosine_similarity = score / (query_magnitude * doc_magnitude);

                *scores.entry(doc_path.to_owned()).or_insert(0.0) += cosine_similarity;
            });
    }
}
