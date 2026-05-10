use std::{
    fs,
    path::{Path, PathBuf},
};

use walkdir::WalkDir;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusDocument {
    pub path: PathBuf,
    pub content: String,
}

pub trait RegexCorpus {
    fn documents(&self) -> Vec<CorpusDocument>;
    fn read_document(&self, path: &Path) -> Option<String>;
}

#[derive(Debug, Clone)]
pub struct FileSystemCorpus {
    root: PathBuf,
}

impl FileSystemCorpus {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }
}

impl RegexCorpus for FileSystemCorpus {
    fn documents(&self) -> Vec<CorpusDocument> {
        let mut paths = WalkDir::new(&self.root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.path().to_path_buf())
            .collect::<Vec<_>>();

        paths.sort();
        paths
            .into_iter()
            .filter_map(|path| {
                let content = fs::read_to_string(&path).ok()?;
                Some(CorpusDocument { path, content })
            })
            .collect()
    }

    fn read_document(&self, path: &Path) -> Option<String> {
        fs::read_to_string(path).ok()
    }
}
