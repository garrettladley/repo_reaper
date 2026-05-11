use std::path::Path;

use crate::{
    config::Config,
    tokenizer::{content_tokens, tokenize_identifier},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum FileType {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Markdown,
    Toml,
    Yaml,
    Json,
    Shell,
    UnknownText,
}

impl FileType {
    pub fn detect(path: &Path) -> Self {
        if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
            match file_name {
                "Cargo.toml" | "pyproject.toml" => return Self::Toml,
                "Dockerfile" | "Makefile" | "Justfile" | "justfile" => return Self::Shell,
                _ => {}
            }
        }

        match path.extension().and_then(|extension| extension.to_str()) {
            Some("rs") => Self::Rust,
            Some("py") => Self::Python,
            Some("js" | "jsx" | "mjs" | "cjs") => Self::JavaScript,
            Some("ts" | "tsx" | "mts" | "cts") => Self::TypeScript,
            Some("go") => Self::Go,
            Some("md" | "markdown") => Self::Markdown,
            Some("toml") => Self::Toml,
            Some("yaml" | "yml") => Self::Yaml,
            Some("json" | "jsonc") => Self::Json,
            Some("sh" | "bash" | "zsh" | "fish") => Self::Shell,
            _ => Self::UnknownText,
        }
    }

    fn is_code(self) -> bool {
        matches!(
            self,
            Self::Rust
                | Self::Python
                | Self::JavaScript
                | Self::TypeScript
                | Self::Go
                | Self::Shell
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AnalyzerField {
    FileName,
    RelativePath,
    Extension,
    Content,
    Identifier,
    Symbol,
    Import,
    Comment,
    StringLiteral,
    Frontmatter,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnalyzerProfile {
    file_type: FileType,
    exact_fields: FieldAnalyzer,
    content: FieldAnalyzer,
    identifier: FieldAnalyzer,
    symbol: FieldAnalyzer,
    import: FieldAnalyzer,
    comment: FieldAnalyzer,
    string_literal: FieldAnalyzer,
    frontmatter: FieldAnalyzer,
}

impl AnalyzerProfile {
    pub fn for_path(path: &Path) -> Self {
        Self::for_file_type(FileType::detect(path))
    }

    pub fn for_file_type(file_type: FileType) -> Self {
        let exact_fields = FieldAnalyzer::exact_identifier();
        let identifier = FieldAnalyzer::identifier();
        let symbol = FieldAnalyzer::identifier();
        let import = FieldAnalyzer::identifier();
        let content = if file_type.is_code() {
            FieldAnalyzer::code_content()
        } else {
            FieldAnalyzer::prose()
        };
        let comment = if file_type == FileType::Markdown {
            FieldAnalyzer::prose()
        } else {
            FieldAnalyzer::comment()
        };
        let string_literal = FieldAnalyzer::string_literal();
        let frontmatter = FieldAnalyzer::prose();

        Self {
            file_type,
            exact_fields,
            content,
            identifier,
            symbol,
            import,
            comment,
            string_literal,
            frontmatter,
        }
    }

    pub fn file_type(self) -> FileType {
        self.file_type
    }

    pub fn analyzer_for(self, field: AnalyzerField) -> FieldAnalyzer {
        match field {
            AnalyzerField::FileName | AnalyzerField::RelativePath | AnalyzerField::Extension => {
                self.exact_fields
            }
            AnalyzerField::Content => self.content,
            AnalyzerField::Identifier => self.identifier,
            AnalyzerField::Symbol => self.symbol,
            AnalyzerField::Import => self.import,
            AnalyzerField::Comment => self.comment,
            AnalyzerField::StringLiteral => self.string_literal,
            AnalyzerField::Frontmatter => self.frontmatter,
        }
    }

    pub fn analyze(self, field: AnalyzerField, text: &str, config: &Config) -> Vec<String> {
        self.analyzer_for(field).analyze(text, config)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldAnalyzer {
    split_identifiers: bool,
    preserve_exact: bool,
    remove_stop_words: bool,
    stem: bool,
}

impl FieldAnalyzer {
    fn exact_identifier() -> Self {
        Self {
            split_identifiers: true,
            preserve_exact: true,
            remove_stop_words: false,
            stem: false,
        }
    }

    fn identifier() -> Self {
        Self {
            split_identifiers: true,
            preserve_exact: true,
            remove_stop_words: false,
            stem: false,
        }
    }

    fn code_content() -> Self {
        Self {
            split_identifiers: true,
            preserve_exact: true,
            remove_stop_words: false,
            stem: false,
        }
    }

    fn prose() -> Self {
        Self {
            split_identifiers: true,
            preserve_exact: false,
            remove_stop_words: true,
            stem: true,
        }
    }

    fn comment() -> Self {
        Self {
            split_identifiers: true,
            preserve_exact: false,
            remove_stop_words: true,
            stem: true,
        }
    }

    fn string_literal() -> Self {
        Self {
            split_identifiers: true,
            preserve_exact: true,
            remove_stop_words: false,
            stem: false,
        }
    }

    pub fn analyze(self, text: &str, config: &Config) -> Vec<String> {
        if self == Self::prose() {
            return content_tokens(text, config);
        }

        text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
            .filter(|lexeme| !lexeme.is_empty())
            .flat_map(|lexeme| self.analyze_lexeme(lexeme, config))
            .collect()
    }

    fn analyze_lexeme(self, lexeme: &str, config: &Config) -> Vec<String> {
        let Some(identifier) = tokenize_identifier(lexeme) else {
            return Vec::new();
        };

        let mut tokens = Vec::new();

        if self.preserve_exact {
            push_unique(&mut tokens, identifier.exact);
            push_unique(&mut tokens, identifier.compound);
        }

        if self.split_identifiers {
            for part in identifier.parts {
                if self.remove_stop_words && config.stop_words.contains(&part) {
                    continue;
                }

                let token = if self.stem {
                    config.stemmer.stem(&part).to_string()
                } else {
                    part
                };
                push_unique(&mut tokens, token);
            }
        }

        tokens
    }
}

fn push_unique(tokens: &mut Vec<String>, token: String) {
    if !tokens.iter().any(|existing| existing == &token) {
        tokens.push(token);
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::Path};

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use super::{AnalyzerField, AnalyzerProfile, FileType};
    use crate::config::Config;

    fn test_config() -> Config {
        Config {
            n_grams: 1,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: stop_words::get(stop_words::LANGUAGE::English)
                .par_iter()
                .map(|word| word.to_string())
                .collect::<HashSet<String>>(),
        }
    }

    #[test]
    fn detects_file_types_from_extensions_and_special_names() {
        assert_eq!(FileType::detect(Path::new("src/lib.rs")), FileType::Rust);
        assert_eq!(FileType::detect(Path::new("tool.py")), FileType::Python);
        assert_eq!(FileType::detect(Path::new("app.jsx")), FileType::JavaScript);
        assert_eq!(FileType::detect(Path::new("app.tsx")), FileType::TypeScript);
        assert_eq!(FileType::detect(Path::new("main.go")), FileType::Go);
        assert_eq!(FileType::detect(Path::new("README.md")), FileType::Markdown);
        assert_eq!(FileType::detect(Path::new("Cargo.toml")), FileType::Toml);
        assert_eq!(
            FileType::detect(Path::new(".github/workflows/ci.yml")),
            FileType::Yaml
        );
        assert_eq!(FileType::detect(Path::new("package.json")), FileType::Json);
        assert_eq!(
            FileType::detect(Path::new("scripts/run.sh")),
            FileType::Shell
        );
        assert_eq!(
            FileType::detect(Path::new("notes.txt")),
            FileType::UnknownText
        );
    }

    #[test]
    fn markdown_content_uses_prose_analysis_but_rust_content_keeps_code_terms() {
        let config = test_config();
        let markdown = AnalyzerProfile::for_file_type(FileType::Markdown).analyze(
            AnalyzerField::Content,
            "the running parser",
            &config,
        );
        let rust = AnalyzerProfile::for_file_type(FileType::Rust).analyze(
            AnalyzerField::Content,
            "the running parser",
            &config,
        );

        assert_eq!(markdown, vec!["run", "parser"]);
        assert_eq!(rust, vec!["the", "running", "parser"]);
    }

    #[test]
    fn exact_code_fields_do_not_stem_or_drop_keywords() {
        let config = test_config();
        let tokens = AnalyzerProfile::for_file_type(FileType::Rust).analyze(
            AnalyzerField::Identifier,
            "matchRunning",
            &config,
        );

        assert_eq!(tokens, vec!["matchrunning", "match", "running"]);
    }

    #[test]
    fn filename_and_path_fields_preserve_exact_forms() {
        let config = test_config();
        let profile = AnalyzerProfile::for_file_type(FileType::Python);

        assert_eq!(
            profile.analyze(AnalyzerField::FileName, "repo_reaper.py", &config),
            vec!["repo_reaper", "reporeaper", "repo", "reaper", "py"]
        );
        assert_eq!(
            profile.analyze(AnalyzerField::RelativePath, "src/query-id.py", &config),
            vec!["src", "query-id", "queryid", "query", "id", "py"]
        );
    }
}
