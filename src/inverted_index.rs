use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

pub struct TermDocument {
    pub length: usize,
    pub term_freq: usize,
}

#[derive(Hash, Eq, PartialEq)]
pub struct Term(pub String);

pub struct InvertedIndex(pub HashMap<Term, HashMap<PathBuf, TermDocument>>);

impl InvertedIndex {
    pub fn new<P, F>(root: P, transform_fn: F) -> Self
    where
        P: Into<PathBuf>,
        F: Fn(&str) -> Vec<String>,
    {
        let mut index = HashMap::new();
        let root_path: PathBuf = root.into();

        for entry in WalkDir::new(root_path)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|e| e.file_type().is_file())
        {
            let entry_path = entry.path().to_path_buf();

            if let Ok(content) = Self::read_document(&entry_path) {
                Self::process_document(&mut index, &entry_path, &content, &transform_fn);
            }

            Self::process_document(
                &mut index,
                &entry_path,
                &entry_path.to_string_lossy(),
                &transform_fn,
            );
        }

        InvertedIndex(index)
    }

    fn read_document(entry_path: &PathBuf) -> Result<String, std::io::Error> {
        fs::read_to_string(entry_path)
    }

    fn process_document<F>(
        index: &mut HashMap<Term, HashMap<PathBuf, TermDocument>>,
        entry_path: &Path,
        content: &str,
        transform_fn: &F,
    ) where
        F: Fn(&str) -> Vec<String>,
    {
        let document_length = content.split_whitespace().count();
        let terms = transform_fn(content);

        for term in terms {
            let entry = index.entry(Term(term)).or_default();
            let term_document =
                entry
                    .entry(entry_path.to_path_buf())
                    .or_insert_with(|| TermDocument {
                        length: document_length,
                        term_freq: 0,
                    });
            term_document.term_freq += 1;
        }
    }
}

impl InvertedIndex {
    pub fn num_docs(&self) -> usize {
        self.0.len()
    }

    pub fn avg_doc_length(&self) -> f64 {
        let total_document_length: usize = self
            .0
            .values()
            .map(|documents| documents.values().map(|d| d.length).sum::<usize>())
            .sum();

        total_document_length as f64 / self.num_docs() as f64
    }

    pub fn doc_freq(&self, term: &Term) -> usize {
        self.0.get(term).map_or(0, |documents| documents.len())
    }

    pub fn cosim_doc_magnitude(&self, doc_path: &PathBuf) -> f64 {
        let mut square_sum = 0.0;

        self.0.iter().for_each(|(_, doc_map)| {
            if let Some(term_doc) = doc_map.get(doc_path) {
                let tf = term_doc.term_freq as f64;
                square_sum += tf * tf;
            }
        });

        square_sum.sqrt()
    }
}
