use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Debug)]
pub struct TermDocument {
    pub length: usize,
    pub term_freq: usize,
}

#[derive(Hash, Eq, PartialEq, Debug, Clone)]
pub struct Term(pub String);

#[derive(Debug)]
pub struct InvertedIndex(pub HashMap<Term, HashMap<PathBuf, TermDocument>>);

impl InvertedIndex {
    pub fn new<P, F>(root: P, transform_fn: F, drop_prefix: Option<P>) -> Self
    where
        P: AsRef<Path> + Sync,
        F: Fn(&str) -> HashSet<Term> + Sync,
    {
        let root_path: PathBuf = root.as_ref().to_owned();
        let drop_prefix = drop_prefix.map(|p| p.as_ref().to_owned());

        let index = WalkDir::new(root_path)
            .into_iter()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(|entry| {
                let mut entry_path = entry.path().to_path_buf();

                if let Some(ref prefix) = drop_prefix {
                    entry_path = entry_path
                        .strip_prefix(prefix)
                        .unwrap_or(&entry_path)
                        .to_owned();
                }

                if let Ok(content) = Self::read_document(&entry_path) {
                    Self::process_document(&entry_path, &content, &transform_fn)
                } else {
                    HashMap::new()
                }
            })
            .reduce(HashMap::new, |mut accumulator, local_map| {
                for (term, doc_map) in local_map {
                    accumulator.entry(term).or_default().extend(doc_map);
                }
                accumulator
            });

        InvertedIndex(index)
    }

    fn read_document(entry_path: &PathBuf) -> Result<String, std::io::Error> {
        fs::read_to_string(entry_path)
    }

    fn process_document<F>(
        entry_path: &Path,
        content: &str,
        transform_fn: &F,
    ) -> HashMap<Term, HashMap<PathBuf, TermDocument>>
    where
        F: Fn(&str) -> HashSet<Term> + Sync,
    {
        let terms = transform_fn(content);
        let document_length = terms.len();

        let mut local_map: HashMap<Term, HashMap<PathBuf, TermDocument>> = HashMap::new();

        terms.into_iter().for_each(|term| {
            local_map.entry(term).or_default().insert(
                entry_path.to_path_buf(),
                TermDocument {
                    length: document_length,
                    term_freq: 1,
                },
            );
        });

        local_map
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
            Self::process_document(path, &content, transform_fn)
                .into_iter()
                .for_each(|(term, doc_map)| {
                    self.0.entry(term).or_default().extend(doc_map);
                });
        }
    }
}
