use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use rayon::iter::{ParallelBridge, ParallelIterator};
use walkdir::WalkDir;

use crate::index::term::Term;

#[derive(Debug)]
pub struct TermDocument {
    pub length: usize,
    pub term_freq: usize,
}

#[derive(Debug)]
pub struct InvertedIndex {
    postings: HashMap<Term, HashMap<PathBuf, TermDocument>>,
    doc_count: usize,
    total_doc_length: u64,
}

impl InvertedIndex {
    pub(crate) fn from_postings(postings: HashMap<Term, HashMap<PathBuf, TermDocument>>) -> Self {
        let mut unique_docs: HashMap<&PathBuf, usize> = HashMap::new();
        for doc_map in postings.values() {
            for (path, td) in doc_map {
                unique_docs.entry(path).or_insert(td.length);
            }
        }

        InvertedIndex {
            doc_count: unique_docs.len(),
            total_doc_length: unique_docs.values().map(|&l| l as u64).sum(),
            postings,
        }
    }

    pub fn new<P, F>(root: P, transform_fn: F, drop_prefix: Option<P>) -> Self
    where
        P: AsRef<Path> + Sync,
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let root_path: PathBuf = root.as_ref().to_owned();
        let drop_prefix = drop_prefix.map(|p| p.as_ref().to_owned());

        let postings = WalkDir::new(root_path)
            .into_iter()
            .par_bridge()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
            .map(|entry| {
                let full_path = entry.path().to_path_buf();

                let index_path = if let Some(ref prefix) = drop_prefix {
                    full_path
                        .strip_prefix(prefix)
                        .unwrap_or(&full_path)
                        .to_owned()
                } else {
                    full_path.clone()
                };

                if let Ok(content) = Self::read_document(&full_path) {
                    Self::process_document(&index_path, &content, &transform_fn)
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

        Self::from_postings(postings)
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
    pub fn get_postings(&self, term: &Term) -> Option<&HashMap<PathBuf, TermDocument>> {
        self.postings.get(term)
    }

    pub fn postings_iter(&self) -> impl Iterator<Item = (&Term, &HashMap<PathBuf, TermDocument>)> {
        self.postings.iter()
    }

    pub fn num_docs(&self) -> usize {
        self.doc_count
    }

    pub fn avg_doc_length(&self) -> f64 {
        self.total_doc_length as f64 / self.doc_count as f64
    }

    pub fn doc_freq(&self, term: &Term) -> usize {
        self.postings
            .get(term)
            .map_or(0, |documents| documents.len())
    }

    pub fn update<F>(&mut self, path: &PathBuf, transform_fn: &F)
    where
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        if let Ok(content) = Self::read_document(path) {
            Self::process_document(path, &content, transform_fn)
                .into_iter()
                .for_each(|(term, doc_map)| {
                    self.postings.entry(term).or_default().extend(doc_map);
                });
            self.recalculate_stats();
        }
    }

    fn recalculate_stats(&mut self) {
        let mut unique_docs: HashMap<&PathBuf, usize> = HashMap::new();
        for doc_map in self.postings.values() {
            for (path, td) in doc_map {
                unique_docs.entry(path).or_insert(td.length);
            }
        }
        self.doc_count = unique_docs.len();
        self.total_doc_length = unique_docs.values().map(|&l| l as u64).sum();
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        path::{Path, PathBuf},
    };

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
        let transform_fn =
            |_: &str| -> HashMap<Term, u32> { term_freqs(&[("rust", 3), ("system", 1)]) };

        let result = InvertedIndex::process_document(Path::new("test.rs"), "", &transform_fn);

        assert_eq!(get_term_doc(&result, "rust", "test.rs").length, 4);
    }

    #[test]
    fn document_length_consistent_across_all_terms() {
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

    fn build_index(docs: &[(&str, &[(&str, u32)])]) -> InvertedIndex {
        let mut postings: HashMap<Term, HashMap<PathBuf, super::TermDocument>> = HashMap::new();
        for &(path, terms) in docs {
            let total_len: usize = terms.iter().map(|(_, c)| *c as usize).sum();
            for &(term, freq) in terms {
                postings.entry(Term(term.to_string())).or_default().insert(
                    PathBuf::from(path),
                    super::TermDocument {
                        length: total_len,
                        term_freq: freq as usize,
                    },
                );
            }
        }
        InvertedIndex::from_postings(postings)
    }

    #[test]
    fn num_docs_returns_unique_document_count() {
        let index = build_index(&[
            ("a.rs", &[("foo", 2), ("bar", 1)]),
            ("b.rs", &[("bar", 1), ("baz", 3)]),
        ]);

        assert_eq!(index.num_docs(), 2);
    }

    #[test]
    fn num_docs_single_document_with_many_terms() {
        let index = build_index(&[("a.rs", &[("a", 1), ("b", 1), ("c", 1), ("d", 1)])]);

        assert_eq!(index.num_docs(), 1);
    }

    #[test]
    fn avg_doc_length_counts_each_document_once() {
        let index = build_index(&[
            ("a.rs", &[("foo", 2), ("bar", 1), ("baz", 2)]),
            ("b.rs", &[("bar", 1), ("qux", 2)]),
        ]);

        assert_eq!(index.avg_doc_length(), 4.0);
    }

    #[test]
    fn avg_doc_length_single_document() {
        let index = build_index(&[("a.rs", &[("x", 3), ("y", 7)])]);

        assert_eq!(index.avg_doc_length(), 10.0);
    }
}
