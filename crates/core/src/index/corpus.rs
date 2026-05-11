use std::{
    fs, io,
    path::{Path, PathBuf},
};

use crate::fs_walk::filesystem_files;

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
    respect_gitignore: bool,
}

impl FileSystemIndexCorpus {
    pub fn new(root: impl AsRef<Path>, drop_prefix: Option<impl AsRef<Path>>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            drop_prefix: drop_prefix.map(|path| path.as_ref().to_path_buf()),
            respect_gitignore: true,
        }
    }

    pub fn with_respect_gitignore(mut self, respect_gitignore: bool) -> Self {
        self.respect_gitignore = respect_gitignore;
        self
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

        for path in filesystem_files(&self.root, self.respect_gitignore) {
            let full_path = match path {
                Ok(path) => path,
                Err(error) => {
                    skipped_documents.push(SkippedDocument {
                        path: error.path.unwrap_or_default(),
                        reason: IndexSkipReason::Walk {
                            message: error.message,
                        },
                    });
                    continue;
                }
            };

            match fs::read_to_string(&full_path) {
                Ok(content) => documents.push(IndexCorpusDocument {
                    path: self.index_path(&full_path),
                    file_size_bytes: content.len() as u64,
                    content,
                }),
                Err(error) => skipped_documents.push(SkippedDocument {
                    path: full_path,
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

#[cfg(test)]
mod tests {
    use super::{FileSystemIndexCorpus, IndexCorpus};

    #[test]
    fn scan_respects_gitignore_inside_git_repository_by_default() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        std::fs::write(root.path().join(".gitignore"), "node_modules/\n").unwrap();
        let workspace = root.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(workspace.join("src")).unwrap();
        std::fs::create_dir(workspace.join("node_modules")).unwrap();
        std::fs::write(workspace.join("src/lib.rs"), "pub fn indexed() {}").unwrap();
        std::fs::write(workspace.join("node_modules/dep.js"), "ignored").unwrap();

        let scan = FileSystemIndexCorpus::new(&workspace, None::<&std::path::Path>).scan();
        let paths = scan
            .documents
            .into_iter()
            .map(|document| document.path)
            .collect::<Vec<_>>();

        assert_eq!(paths, vec![workspace.join("src/lib.rs")]);
    }

    #[test]
    fn scan_can_opt_out_of_gitignore_filtering() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        std::fs::write(root.path().join(".gitignore"), "node_modules/\n").unwrap();
        let workspace = root.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(workspace.join("src")).unwrap();
        std::fs::create_dir(workspace.join("node_modules")).unwrap();
        std::fs::write(workspace.join("src/lib.rs"), "pub fn indexed() {}").unwrap();
        std::fs::write(workspace.join("node_modules/dep.js"), "indexed").unwrap();

        let scan = FileSystemIndexCorpus::new(&workspace, None::<&std::path::Path>)
            .with_respect_gitignore(false)
            .scan();
        let paths = scan
            .documents
            .into_iter()
            .map(|document| document.path)
            .collect::<Vec<_>>();

        assert_eq!(
            paths,
            vec![
                workspace.join("node_modules/dep.js"),
                workspace.join("src/lib.rs"),
            ]
        );
    }

    #[test]
    fn scan_ignores_gitignore_without_git_repository() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join(".gitignore"), "ignored.txt\n").unwrap();
        std::fs::write(root.path().join("ignored.txt"), "indexed").unwrap();

        let scan = FileSystemIndexCorpus::new(root.path(), None::<&std::path::Path>).scan();
        let paths = scan
            .documents
            .into_iter()
            .map(|document| document.path)
            .collect::<Vec<_>>();

        assert_eq!(
            paths,
            vec![
                root.path().join(".gitignore"),
                root.path().join("ignored.txt")
            ]
        );
    }
}
