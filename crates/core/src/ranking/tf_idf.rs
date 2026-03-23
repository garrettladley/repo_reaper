use std::{collections::HashMap, path::PathBuf};

use dashmap::DashMap;
use rayon::iter::{ParallelBridge, ParallelIterator};

use crate::{
    index::{InvertedIndex, TermDocument},
    query::Query,
    ranking::{scorer::Scorer, utils::idf},
};

pub struct TFIDF;

impl Scorer for TFIDF {
    fn score(
        &self,
        inverted_index: &InvertedIndex,
        _: &Query,
        documents: &HashMap<PathBuf, TermDocument>,
        scores: &DashMap<PathBuf, f64>,
    ) {
        let num_docs = inverted_index.num_docs();

        documents
            .iter()
            .par_bridge()
            .for_each(|(doc_path, term_doc)| {
                let tf = term_doc.term_freq as f64 / term_doc.length as f64;
                let idf = idf(num_docs, documents.len());

                *scores.entry(doc_path.to_owned()).or_insert(0.0) += tf * idf;
            });
    }
}
