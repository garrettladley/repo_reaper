use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::fs_walk::filesystem_files;

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
    respect_gitignore: bool,
}

impl FileSystemCorpus {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            respect_gitignore: true,
        }
    }

    pub fn with_respect_gitignore(mut self, respect_gitignore: bool) -> Self {
        self.respect_gitignore = respect_gitignore;
        self
    }
}

impl RegexCorpus for FileSystemCorpus {
    fn documents(&self) -> Vec<CorpusDocument> {
        let mut paths = filesystem_files(&self.root, self.respect_gitignore)
            .into_iter()
            .filter_map(Result::ok)
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

#[cfg(test)]
mod tests {
    use super::{FileSystemCorpus, RegexCorpus};

    #[test]
    fn documents_respect_gitignore_inside_git_repository_by_default() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        std::fs::write(root.path().join(".gitignore"), "node_modules/\n").unwrap();
        let workspace = root.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(workspace.join("src")).unwrap();
        std::fs::create_dir(workspace.join("node_modules")).unwrap();
        std::fs::write(workspace.join("src/lib.rs"), "pub fn indexed() {}").unwrap();
        std::fs::write(workspace.join("node_modules/dep.js"), "ignored").unwrap();

        let paths = FileSystemCorpus::new(&workspace)
            .documents()
            .into_iter()
            .map(|document| document.path)
            .collect::<Vec<_>>();

        assert_eq!(paths, vec![workspace.join("src/lib.rs")]);
    }

    #[test]
    fn documents_can_opt_out_of_gitignore_filtering() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join(".git")).unwrap();
        std::fs::write(root.path().join(".gitignore"), "node_modules/\n").unwrap();
        let workspace = root.path().join("workspace");
        std::fs::create_dir(&workspace).unwrap();
        std::fs::create_dir(workspace.join("src")).unwrap();
        std::fs::create_dir(workspace.join("node_modules")).unwrap();
        std::fs::write(workspace.join("src/lib.rs"), "pub fn indexed() {}").unwrap();
        std::fs::write(workspace.join("node_modules/dep.js"), "indexed").unwrap();

        let paths = FileSystemCorpus::new(&workspace)
            .with_respect_gitignore(false)
            .documents()
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
}
