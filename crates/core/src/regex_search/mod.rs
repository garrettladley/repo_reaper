use std::{
    fs,
    ops::{Range, RangeInclusive},
    path::{Path, PathBuf},
};

use regex::Regex;
use walkdir::WalkDir;

mod trigram_index;

pub use trigram_index::{Trigram, TrigramIndex};

#[derive(Debug, thiserror::Error)]
pub enum RegexSearchError {
    #[error("invalid regex pattern")]
    InvalidPattern(#[source] regex::Error),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexSearchMatch {
    pub path: PathBuf,
    pub byte_range: Range<usize>,
    pub line_range: RangeInclusive<usize>,
    pub matched_text: String,
}

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

fn line_range_for_match(content: &str, byte_range: Range<usize>) -> RangeInclusive<usize> {
    let start_line = line_number_at_byte(content, byte_range.start);
    let end_line = if byte_range.is_empty() {
        start_line
    } else {
        line_number_at_byte(content, byte_range.end)
    };

    start_line..=end_line
}

fn line_number_at_byte(content: &str, byte_index: usize) -> usize {
    content.as_bytes()[..byte_index]
        .iter()
        .filter(|&&byte| byte == b'\n')
        .count()
        + 1
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::RegexSearchEngine;

    #[test]
    fn search_returns_path_byte_range_line_range_and_matched_text() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("src/lib.rs");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, "alpha\nlet answer = 42;\nomega\n").unwrap();

        let matches = RegexSearchEngine::new(temp.path())
            .search(r"answer = \d+")
            .unwrap();

        assert_eq!(matches.len(), 1);
        let match_ = &matches[0];
        assert_eq!(match_.path, file);
        assert_eq!(match_.byte_range, 10..21);
        assert_eq!(match_.line_range, 2..=2);
        assert_eq!(match_.matched_text, "answer = 42");
    }

    #[test]
    fn search_returns_multiple_matches_in_file_order() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("lib.rs");
        std::fs::write(&file, "todo one\ntodo two\ntodo three\n").unwrap();

        let matches = RegexSearchEngine::new(temp.path())
            .search(r"todo \w+")
            .unwrap();

        let matched_text = matches
            .iter()
            .map(|match_| match_.matched_text.as_str())
            .collect::<Vec<_>>();
        assert_eq!(matched_text, ["todo one", "todo two", "todo three"]);
    }

    #[test]
    fn search_returns_files_in_deterministic_path_order() {
        let temp = tempfile::tempdir().unwrap();
        write_file(temp.path(), "zeta.rs", "needle z").unwrap();
        write_file(temp.path(), "alpha.rs", "needle a").unwrap();
        write_file(temp.path(), "nested/beta.rs", "needle b").unwrap();

        let matches = RegexSearchEngine::new(temp.path())
            .search(r"needle \w")
            .unwrap();

        let names = matches
            .iter()
            .map(|match_| {
                match_
                    .path
                    .strip_prefix(temp.path())
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect::<Vec<_>>();
        assert_eq!(names, ["alpha.rs", "nested/beta.rs", "zeta.rs"]);
    }

    #[test]
    fn search_returns_error_for_invalid_patterns() {
        let temp = tempfile::tempdir().unwrap();

        let error = RegexSearchEngine::new(temp.path()).search("(").unwrap_err();

        assert!(error.to_string().contains("invalid regex pattern"));
    }

    #[test]
    fn search_uses_regex_line_boundary_flags_explicitly() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("lib.rs");
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();

        let matches = RegexSearchEngine::new(temp.path())
            .search(r"(?m)^beta$")
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].path, file);
        assert_eq!(matches[0].byte_range, 6..10);
        assert_eq!(matches[0].line_range, 2..=2);
        assert_eq!(matches[0].matched_text, "beta");
    }

    #[test]
    fn search_reports_multiline_match_line_ranges() {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("lib.rs");
        std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();

        let matches = RegexSearchEngine::new(temp.path())
            .search(r"(?s)alpha.*gamma")
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].path, file);
        assert_eq!(matches[0].byte_range, 0..16);
        assert_eq!(matches[0].line_range, 1..=3);
        assert_eq!(matches[0].matched_text, "alpha\nbeta\ngamma");
    }

    #[test]
    fn search_skips_non_utf8_files() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(temp.path().join("binary.bin"), [0xff, 0xfe, b'n', b'e']).unwrap();
        std::fs::write(temp.path().join("text.txt"), "needle").unwrap();

        let matches = RegexSearchEngine::new(temp.path())
            .search("needle")
            .unwrap();

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].matched_text, "needle");
        assert_eq!(matches[0].path, temp.path().join("text.txt"));
    }

    fn write_file(root: &Path, path: &str, content: &str) -> std::io::Result<()> {
        let path = root.join(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content)
    }
}
