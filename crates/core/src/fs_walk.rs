use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

pub(crate) fn filesystem_files(
    root: &Path,
    respect_gitignore: bool,
) -> Vec<Result<PathBuf, WalkError>> {
    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(false)
        .ignore(false)
        .git_global(respect_gitignore)
        .git_ignore(respect_gitignore)
        .git_exclude(respect_gitignore)
        .require_git(true);

    builder
        .build()
        .filter_map(|entry| match entry {
            Ok(entry)
                if entry
                    .file_type()
                    .is_some_and(|file_type| file_type.is_file()) =>
            {
                Some(Ok(entry.path().to_path_buf()))
            }
            Ok(_) => None,
            Err(error) => Some(Err(WalkError {
                path: None,
                message: error.to_string(),
            })),
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WalkError {
    pub(crate) path: Option<PathBuf>,
    pub(crate) message: String,
}
