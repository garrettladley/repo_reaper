use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use rayon::iter::{IntoParallelRefIterator, ParallelBridge, ParallelIterator};
use walkdir::WalkDir;

use crate::index::term::Term;

#[derive(Debug)]
pub struct TermDocument {
    pub length: usize,
    pub term_freq: usize,
}

#[derive(Debug)]
pub struct InvertedIndex(pub HashMap<Term, HashMap<PathBuf, TermDocument>>);

impl InvertedIndex {
    pub fn new<P, F>(root: P, transform_fn: F, drop_prefix: Option<P>) -> Self
    where
        P: AsRef<Path> + Sync,
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
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
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let term_frequencies = transform_fn(content);
        let document_length: usize = term_frequencies.values().map(|&c| c as usize).sum();

        let mut local_map: HashMap<Term, HashMap<PathBuf, TermDocument>> = HashMap::new();

        for (term, freq) in term_frequencies {
            local_map.entry(term).or_default().insert(
                entry_path.to_path_buf(),
                TermDocument {
                    length: document_length,
                    term_freq: freq as usize,
                },
            );
        }

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
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
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

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, path::Path};

    use super::InvertedIndex;
    use crate::index::term::Term;

    fn term_freqs(pairs: &[(&str, u32)]) -> HashMap<Term, u32> {
        pairs
            .iter()
            .map(|(s, c)| (Term(s.to_string()), *c))
            .collect()
    }

    fn get_term_doc<'a>(
        result: &'a HashMap<Term, HashMap<std::path::PathBuf, super::TermDocument>>,
        term: &str,
        path: &str,
    ) -> &'a super::TermDocument {
        result
            .get(&Term(term.to_string()))
            .unwrap()
            .get(Path::new(path))
            .unwrap()
    }

    #[test]
    fn term_freq_matches_transform_frequency() {
        let transform_fn =
            |_: &str| -> HashMap<Term, u32> { term_freqs(&[("rust", 3), ("system", 1)]) };

        let result = InvertedIndex::process_document(Path::new("test.rs"), "", &transform_fn);

        assert_eq!(get_term_doc(&result, "rust", "test.rs").term_freq, 3);
        assert_eq!(get_term_doc(&result, "system", "test.rs").term_freq, 1);
    }

    #[test]
    fn document_length_is_sum_of_all_frequencies() {
        // 3 + 1 = 4 total tokens
        let transform_fn =
            |_: &str| -> HashMap<Term, u32> { term_freqs(&[("rust", 3), ("system", 1)]) };

        let result = InvertedIndex::process_document(Path::new("test.rs"), "", &transform_fn);

        assert_eq!(get_term_doc(&result, "rust", "test.rs").length, 4);
    }

    #[test]
    fn document_length_consistent_across_all_terms() {
        // 3 + 1 + 1 = 5 total tokens
        let transform_fn =
            |_: &str| -> HashMap<Term, u32> { term_freqs(&[("foo", 3), ("bar", 1), ("baz", 1)]) };

        let result = InvertedIndex::process_document(Path::new("test.rs"), "", &transform_fn);

        let lengths: Vec<usize> = result
            .values()
            .filter_map(|docs| docs.get(Path::new("test.rs")))
            .map(|td| td.length)
            .collect();

        assert!(lengths.iter().all(|&l| l == 5));
    }
}
