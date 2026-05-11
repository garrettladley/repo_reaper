use std::{
    collections::{BTreeMap, HashMap},
    fs, io,
    path::{Path, PathBuf},
};

use crate::index::{
    DocId, InvertedIndex, Term, TermDocument,
    compression::{CompressedPostingList, DecodedPosting, decode_postings},
    skips::SkipTable,
};

const LEXICON_FILE: &str = "lexicon.json";
const POSTINGS_FILE: &str = "postings.bin";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LexiconEntry {
    pub term: String,
    pub offset: u64,
    pub byte_len: u64,
    pub document_frequency: usize,
    pub collection_frequency: usize,
}

#[derive(Debug)]
pub struct InvertedFileLayout {
    lexicon: BTreeMap<String, LexiconEntry>,
    postings: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum InvertedFileError {
    #[error("inverted file io failed for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("lexicon json failed for {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("compressed posting list for {term} is invalid: {source}")]
    Compression {
        term: String,
        #[source]
        source: crate::index::compression::CompressionError,
    },
}

impl InvertedFileLayout {
    pub fn write(
        index: &InvertedIndex,
        index_dir: &Path,
    ) -> Result<Vec<LexiconEntry>, InvertedFileError> {
        fs::create_dir_all(index_dir).map_err(|source| InvertedFileError::Io {
            path: index_dir.to_path_buf(),
            source,
        })?;

        let mut postings_bytes = Vec::new();
        let mut lexicon = Vec::new();
        let mut terms = index.postings_iter().collect::<Vec<_>>();
        terms.sort_by_key(|(term, _)| *term);

        for (term, documents) in terms {
            let decoded = documents
                .iter()
                .map(|(doc_id, term_document)| DecodedPosting {
                    doc_id: *doc_id,
                    term_freq: term_document.term_freq as u32,
                    positions: Vec::new(),
                })
                .collect::<Vec<_>>();
            let compressed = CompressedPostingList::from_postings(&decoded).map_err(|source| {
                InvertedFileError::Compression {
                    term: term.0.clone(),
                    source,
                }
            })?;
            let offset = postings_bytes.len() as u64;
            postings_bytes.extend_from_slice(compressed.bytes());
            lexicon.push(LexiconEntry {
                term: term.0.clone(),
                offset,
                byte_len: compressed.bytes().len() as u64,
                document_frequency: documents.len(),
                collection_frequency: documents.values().map(|document| document.term_freq).sum(),
            });
        }

        let postings_path = index_dir.join(POSTINGS_FILE);
        fs::write(&postings_path, &postings_bytes).map_err(|source| InvertedFileError::Io {
            path: postings_path,
            source,
        })?;

        let lexicon_path = index_dir.join(LEXICON_FILE);
        let json =
            serde_json::to_vec_pretty(&lexicon).map_err(|source| InvertedFileError::Json {
                path: lexicon_path.clone(),
                source,
            })?;
        fs::write(&lexicon_path, json).map_err(|source| InvertedFileError::Io {
            path: lexicon_path,
            source,
        })?;

        Ok(lexicon)
    }

    pub fn open(index_dir: &Path) -> Result<Self, InvertedFileError> {
        let lexicon_path = index_dir.join(LEXICON_FILE);
        let postings_path = index_dir.join(POSTINGS_FILE);
        let lexicon_entries: Vec<LexiconEntry> =
            serde_json::from_slice(&fs::read(&lexicon_path).map_err(|source| {
                InvertedFileError::Io {
                    path: lexicon_path.clone(),
                    source,
                }
            })?)
            .map_err(|source| InvertedFileError::Json {
                path: lexicon_path,
                source,
            })?;
        let postings = fs::read(&postings_path).map_err(|source| InvertedFileError::Io {
            path: postings_path,
            source,
        })?;
        let lexicon = lexicon_entries
            .into_iter()
            .map(|entry| (entry.term.clone(), entry))
            .collect();

        Ok(Self { lexicon, postings })
    }

    pub fn lexicon_len(&self) -> usize {
        self.lexicon.len()
    }

    pub fn lexicon_entry(&self, term: &str) -> Option<&LexiconEntry> {
        self.lexicon.get(term)
    }

    pub fn postings_for(
        &self,
        term: &str,
    ) -> Result<Option<BTreeMap<DocId, TermDocument>>, InvertedFileError> {
        let Some(entry) = self.lexicon.get(term) else {
            return Ok(None);
        };
        let start = entry.offset as usize;
        let end = start + entry.byte_len as usize;
        let postings = decode_postings(&self.postings[start..end], entry.document_frequency)
            .map_err(|source| InvertedFileError::Compression {
                term: term.to_string(),
                source,
            })?;
        Ok(Some(
            postings
                .into_iter()
                .map(|posting| {
                    (
                        posting.doc_id,
                        TermDocument::unfielded(0, posting.term_freq as usize),
                    )
                })
                .collect(),
        ))
    }

    pub fn skip_table_for(
        &self,
        term: &str,
        block_size: usize,
    ) -> Result<Option<SkipTable>, InvertedFileError> {
        let Some(entry) = self.lexicon.get(term) else {
            return Ok(None);
        };
        let start = entry.offset as usize;
        let end = start + entry.byte_len as usize;
        SkipTable::build(
            &self.postings[start..end],
            entry.document_frequency,
            block_size,
        )
        .map(Some)
        .map_err(|source| InvertedFileError::Compression {
            term: term.to_string(),
            source,
        })
    }
}

pub fn materialize_compressed_postings(
    index: &InvertedIndex,
) -> Result<HashMap<Term, CompressedPostingList>, InvertedFileError> {
    index
        .postings_iter()
        .map(|(term, documents)| {
            let decoded = documents
                .iter()
                .map(|(doc_id, term_document)| DecodedPosting {
                    doc_id: *doc_id,
                    term_freq: term_document.term_freq as u32,
                    positions: Vec::new(),
                })
                .collect::<Vec<_>>();
            CompressedPostingList::from_postings(&decoded)
                .map(|compressed| (term.clone(), compressed))
                .map_err(|source| InvertedFileError::Compression {
                    term: term.0.clone(),
                    source,
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{InvertedFileLayout, materialize_compressed_postings};
    use crate::index::{InvertedIndex, Term};

    #[test]
    fn compressed_postings_are_smaller_than_plain_doc_frequency_pairs() {
        let index =
            InvertedIndex::from_documents(&[("a.rs", &[("rust", 2)]), ("b.rs", &[("rust", 1)])]);
        let compressed = materialize_compressed_postings(&index).unwrap();
        let rust = compressed.get(&Term("rust".to_string())).unwrap();

        assert!(rust.bytes().len() < 2 * std::mem::size_of::<(u32, u32)>());
    }

    #[test]
    fn inverted_file_loads_lexicon_before_posting_lookup() {
        let index_dir = tempfile::tempdir().unwrap();
        let index =
            InvertedIndex::from_documents(&[("a.rs", &[("rust", 2)]), ("b.rs", &[("search", 1)])]);
        InvertedFileLayout::write(&index, index_dir.path()).unwrap();

        let layout = InvertedFileLayout::open(index_dir.path()).unwrap();

        assert_eq!(layout.lexicon_len(), 2);
        assert!(layout.lexicon_entry("rust").is_some());
        assert!(layout.postings_for("rust").unwrap().is_some());
    }
}
