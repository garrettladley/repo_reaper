use std::{
    fs, io,
    path::{Path, PathBuf},
};

use walkdir::WalkDir;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexCorpusDocument {
    pub path: PathBuf,
    pub content: String,
    pub file_size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexCorpusScan {
    pub documents: Vec<IndexCorpusDocument>,
    pub skipped_documents: Vec<SkippedDocument>,
}

impl IndexCorpusScan {
    pub fn indexed_document_count(&self) -> usize {
        self.documents.len()
    }

    pub fn skipped_document_count(&self) -> usize {
        self.skipped_documents.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedDocument {
    pub path: PathBuf,
    pub reason: IndexSkipReason,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IndexSkipReason {
    #[error("failed to walk corpus: {message}")]
    Walk { message: String },
    #[error("failed to read document: {kind:?}: {message}")]
    Read {
        kind: io::ErrorKind,
        message: String,
    },
}

pub trait IndexCorpus {
    fn scan(&self) -> IndexCorpusScan;
}

#[derive(Debug, Clone)]
pub struct FileSystemIndexCorpus {
    root: PathBuf,
    drop_prefix: Option<PathBuf>,
}

impl FileSystemIndexCorpus {
    pub fn new(root: impl AsRef<Path>, drop_prefix: Option<impl AsRef<Path>>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            drop_prefix: drop_prefix.map(|path| path.as_ref().to_path_buf()),
        }
    }

    fn index_path(&self, full_path: &Path) -> PathBuf {
        self.drop_prefix
            .as_ref()
            .and_then(|prefix| full_path.strip_prefix(prefix).ok())
            .unwrap_or(full_path)
            .to_path_buf()
    }
}

impl IndexCorpus for FileSystemIndexCorpus {
    fn scan(&self) -> IndexCorpusScan {
        let mut documents = Vec::new();
        let mut skipped_documents = Vec::new();

        for entry in WalkDir::new(&self.root) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    skipped_documents.push(SkippedDocument {
                        path: error.path().map(Path::to_path_buf).unwrap_or_default(),
                        reason: IndexSkipReason::Walk {
                            message: error.to_string(),
                        },
                    });
                    continue;
                }
            };

            if !entry.file_type().is_file() {
                continue;
            }

            let full_path = entry.path();
            match fs::read_to_string(full_path) {
                Ok(content) => documents.push(IndexCorpusDocument {
                    path: self.index_path(full_path),
                    file_size_bytes: content.len() as u64,
                    content,
                }),
                Err(error) => skipped_documents.push(SkippedDocument {
                    path: full_path.to_path_buf(),
                    reason: IndexSkipReason::Read {
                        kind: error.kind(),
                        message: error.to_string(),
                    },
                }),
            }
        }

        documents.sort_by(|left, right| left.path.cmp(&right.path));

        IndexCorpusScan {
            documents,
            skipped_documents,
        }
    }
}
