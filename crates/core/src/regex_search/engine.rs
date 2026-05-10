use std::{
    fs,
    path::{Path, PathBuf},
};

use regex::Regex;
use walkdir::WalkDir;

use super::{RegexSearchError, RegexSearchMatch, line_range_for_match};

#[derive(Debug, Clone)]
pub struct RegexSearchEngine {
    root: PathBuf,
}

impl RegexSearchEngine {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
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
        let mut files = WalkDir::new(&self.root)
            .into_iter()
            .filter_map(Result::ok)
            .filter(|entry| entry.file_type().is_file())
            .map(|entry| entry.path().to_path_buf())
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
