use std::{
    fs,
    path::{Path, PathBuf},
};

use regex::Regex;

use super::{RegexSearchError, RegexSearchMatch, line_range_for_match};
use crate::fs_walk::filesystem_files;

#[derive(Debug, Clone)]
pub struct RegexSearchEngine {
    root: PathBuf,
    respect_gitignore: bool,
}

impl RegexSearchEngine {
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

    pub fn search(&self, pattern: &str) -> Result<Vec<RegexSearchMatch>, RegexSearchError> {
        let regex = Regex::new(pattern).map_err(RegexSearchError::InvalidPattern)?;
        let mut matches = Vec::new();

        for path in self.candidate_files() {
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };

            matches.extend(verified_matches(&path, &content, &regex));
        }

        Ok(matches)
    }

    fn candidate_files(&self) -> Vec<PathBuf> {
        let mut files = filesystem_files(&self.root, self.respect_gitignore)
            .into_iter()
            .filter_map(Result::ok)
            .collect::<Vec<_>>();

        files.sort();
        files
    }
}

fn verified_matches(path: &Path, content: &str, regex: &Regex) -> Vec<RegexSearchMatch> {
    regex
        .find_iter(content)
        .map(|match_| {
            let byte_range = match_.start()..match_.end();
            RegexSearchMatch {
                path: path.to_path_buf(),
                line_range: line_range_for_match(content, byte_range.clone()),
                matched_text: match_.as_str().to_string(),
                byte_range,
            }
        })
        .collect()
}
