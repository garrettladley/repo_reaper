use std::fmt;

use crate::tokenizer::AnalyzerField;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub enum DocumentField {
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

impl DocumentField {
    pub const ALL: [Self; 10] = [
        Self::FileName,
        Self::RelativePath,
        Self::Extension,
        Self::Content,
        Self::Identifier,
        Self::Symbol,
        Self::Import,
        Self::Comment,
        Self::StringLiteral,
        Self::Frontmatter,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FileName => "filename",
            Self::RelativePath => "path",
            Self::Extension => "extension",
            Self::Content => "content",
            Self::Identifier => "identifiers",
            Self::Symbol => "symbols",
            Self::Import => "imports",
            Self::Comment => "comments",
            Self::StringLiteral => "string_literals",
            Self::Frontmatter => "frontmatter",
        }
    }

    pub fn analyzer_field(self) -> AnalyzerField {
        match self {
            Self::FileName => AnalyzerField::FileName,
            Self::RelativePath => AnalyzerField::RelativePath,
            Self::Extension => AnalyzerField::Extension,
            Self::Content => AnalyzerField::Content,
            Self::Identifier => AnalyzerField::Identifier,
            Self::Symbol => AnalyzerField::Symbol,
            Self::Import => AnalyzerField::Import,
            Self::Comment => AnalyzerField::Comment,
            Self::StringLiteral => AnalyzerField::StringLiteral,
            Self::Frontmatter => AnalyzerField::Frontmatter,
        }
    }
}

impl fmt::Display for DocumentField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
