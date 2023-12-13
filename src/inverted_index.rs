use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, ParallelBridge, ParallelIterator,
};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};
use walkdir::WalkDir;

#[derive(Debug)]
pub struct TermDocument {
    pub length: usize,
    pub term_freq: usize,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct Term(pub String);

pub struct InvertedIndex(pub HashMap<Term, HashMap<PathBuf, TermDocument>>);

impl InvertedIndex {
    pub fn new<P, F>(root: P, transform_fn: F) -> Self
    where
        P: Into<PathBuf>,
        F: Fn(&str) -> HashSet<Term> + Sync,
    {
        let index = Arc::new(Mutex::new(HashMap::new()));
        let root_path: PathBuf = root.into();

        WalkDir::new(root_path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .par_bridge()
            .for_each(|entry| {
                let entry_path = entry.path().to_path_buf();

                if let Ok(content) = Self::read_document(&entry_path) {
                    let index_clone = Arc::clone(&index);
                    Self::process_document(&index_clone, &entry_path, &content, &transform_fn);
                }
            });

        InvertedIndex(Arc::try_unwrap(index).unwrap().into_inner().unwrap())
    }

    fn read_document(entry_path: &PathBuf) -> Result<String, std::io::Error> {
        fs::read_to_string(entry_path)
    }

    fn process_document<F>(
        index: &Arc<Mutex<HashMap<Term, HashMap<PathBuf, TermDocument>>>>,
        entry_path: &Path,
        content: &str,
        transform_fn: &F,
    ) where
        F: Fn(&str) -> HashSet<Term> + Sync,
    {
        let terms = transform_fn(content);
        transform_fn(content).extend(transform_fn(entry_path.to_str().unwrap()));

        let document_length = terms.len();

        let updates: Vec<_> = terms
            .into_par_iter()
            .map(|term| (term, entry_path.to_path_buf()))
            .collect();

        let mut index_lock = index.lock().unwrap();
        updates.into_iter().for_each(|(term, path)| {
            index_lock
                .entry(term)
                .or_default()
                .entry(path)
                .or_insert_with(|| TermDocument {
                    length: document_length,
                    term_freq: 0,
                })
                .term_freq += 1;
        });
    }
}

impl InvertedIndex {
    pub fn num_docs(&self) -> usize {
        self.0.len()
    }

    pub fn avg_doc_length(&self) -> f64 {
        let total_document_length: usize = self
            .0
            .par_iter()
            .map(|(_, documents)| documents.par_iter().map(|(_, d)| d.length).sum::<usize>())
            .sum();

        total_document_length as f64 / self.num_docs() as f64
    }

    pub fn doc_freq(&self, term: &Term) -> usize {
        self.0.get(term).map_or(0, |documents| documents.len())
    }

    pub fn update<F>(&mut self, path: &PathBuf, transform_fn: &F)
    where
        F: Fn(&str) -> HashSet<Term> + Sync,
    {
        if let Ok(content) = Self::read_document(path) {
            let index = Arc::new(Mutex::new(HashMap::new()));
            let index_clone = Arc::clone(&index);

            Self::process_document(&index_clone, path, &content, transform_fn);

            let mut index_lock = index.lock().unwrap();

            index_lock.drain().for_each(|(term, term_document)| {
                self.0
                    .entry(term)
                    .or_default()
                    .entry(path.clone())
                    .or_insert_with(|| TermDocument {
                        length: term_document.get(path).unwrap().length,
                        term_freq: 0,
                    })
                    .term_freq += term_document.get(path).unwrap().term_freq;
            });
        }
    }
}
