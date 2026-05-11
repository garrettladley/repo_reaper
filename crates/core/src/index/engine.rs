use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, RwLock},
};

use crate::{
    config::Config,
    index::{InvertedIndex, Term, event_log::IndexEvent},
    query::AnalyzedQuery,
    ranking::{RankingAlgo, Scored},
};

#[derive(Debug, Clone)]
pub struct SearchEngine {
    index: Arc<RwLock<InvertedIndex>>,
}

#[derive(Debug, thiserror::Error)]
pub enum SearchEngineError {
    #[error("index lock poisoned while acquiring shared read access")]
    ReadLockPoisoned,
    #[error("index lock poisoned while acquiring exclusive write access")]
    WriteLockPoisoned,
}

impl SearchEngine {
    pub fn new(index: InvertedIndex) -> Self {
        Self {
            index: Arc::new(RwLock::new(index)),
        }
    }

    pub fn num_docs(&self) -> Result<usize, SearchEngineError> {
        let index = self
            .index
            .read()
            .map_err(|_| SearchEngineError::ReadLockPoisoned)?;
        Ok(index.num_docs())
    }

    pub fn search(
        &self,
        algo: &RankingAlgo,
        query: &AnalyzedQuery,
        top_n: usize,
    ) -> Result<Option<Scored>, SearchEngineError> {
        let index = self
            .index
            .read()
            .map_err(|_| SearchEngineError::ReadLockPoisoned)?;
        Ok(algo.rank(&*index, query, top_n))
    }

    pub fn update<F>(
        &self,
        path: &Path,
        transform_fn: &F,
        config: &Config,
        fielded: bool,
    ) -> Result<(), SearchEngineError>
    where
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let mut index = self
            .index
            .write()
            .map_err(|_| SearchEngineError::WriteLockPoisoned)?;
        if fielded {
            index.update_fielded(path, config);
        } else {
            index.update(path, transform_fn);
        }
        Ok(())
    }

    pub fn apply_event<F>(
        &self,
        event: &IndexEvent,
        transform_fn: &F,
        config: &Config,
        fielded: bool,
    ) -> Result<(), SearchEngineError>
    where
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let mut index = self
            .index
            .write()
            .map_err(|_| SearchEngineError::WriteLockPoisoned)?;
        crate::index::event_log::apply_event(&mut index, event, transform_fn, config, fielded);
        Ok(())
    }

    pub fn with_read<T>(
        &self,
        f: impl FnOnce(&InvertedIndex) -> T,
    ) -> Result<T, SearchEngineError> {
        let index = self
            .index
            .read()
            .map_err(|_| SearchEngineError::ReadLockPoisoned)?;
        Ok(f(&index))
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, sync::Arc, thread};

    use crate::{
        config::Config,
        index::{InvertedIndex, SearchEngine, Term},
        query::AnalyzedQuery,
        ranking::{BM25HyperParams, RankingAlgo},
    };

    fn transform(content: &str) -> HashMap<Term, u32> {
        content
            .split_whitespace()
            .fold(HashMap::new(), |mut acc, word| {
                *acc.entry(Term(word.to_string())).or_insert(0) += 1;
                acc
            })
    }

    fn config() -> Config {
        Config {
            n_grams: 1,
            stemmer: rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English),
            stop_words: Default::default(),
        }
    }

    #[test]
    fn searches_can_hold_shared_access_concurrently() {
        let engine = Arc::new(SearchEngine::new(InvertedIndex::from_documents(&[
            ("a.rs", &[("rust", 2)]),
            ("b.rs", &[("rust", 1)]),
        ])));
        let algo = RankingAlgo::BM25(BM25HyperParams { k1: 1.2, b: 0.75 });
        let query = AnalyzedQuery::new("rust", &config());

        let handles = (0..4)
            .map(|_| {
                let engine = Arc::clone(&engine);
                let algo = algo.clone();
                let query = query.clone();
                thread::spawn(move || engine.search(&algo, &query, 10).unwrap())
            })
            .collect::<Vec<_>>();

        for handle in handles {
            assert!(handle.join().unwrap().is_some());
        }
    }

    #[test]
    fn updates_take_exclusive_access_and_preserve_search_behavior() {
        let source = tempfile::tempdir().unwrap();
        let path = source.path().join("a.rs");
        std::fs::write(&path, "old").unwrap();
        let engine = SearchEngine::new(InvertedIndex::new(
            source.path(),
            transform,
            None::<&std::path::Path>,
        ));
        std::fs::write(&path, "new").unwrap();

        engine.update(&path, &transform, &config(), false).unwrap();

        assert!(
            engine
                .with_read(|index| index.get_postings(&Term("old".to_string())).is_none())
                .unwrap()
        );
    }
}
