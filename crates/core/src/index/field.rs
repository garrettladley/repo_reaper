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
    Comment,
    StringLiteral,
}

impl DocumentField {
    pub const ALL: [Self; 7] = [
        Self::FileName,
        Self::RelativePath,
        Self::Extension,
        Self::Content,
        Self::Identifier,
        Self::Comment,
        Self::StringLiteral,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::FileName => "filename",
            Self::RelativePath => "path",
            Self::Extension => "extension",
            Self::Content => "content",
            Self::Identifier => "identifiers",
            Self::Comment => "comments",
            Self::StringLiteral => "string_literals",
        }
    }

    pub fn analyzer_field(self) -> AnalyzerField {
        match self {
            Self::FileName => AnalyzerField::FileName,
            Self::RelativePath => AnalyzerField::RelativePath,
            Self::Extension => AnalyzerField::Extension,
            Self::Content => AnalyzerField::Content,
            Self::Identifier => AnalyzerField::Identifier,
            Self::Comment => AnalyzerField::Comment,
            Self::StringLiteral => AnalyzerField::StringLiteral,
        }
    }
}

impl fmt::Display for DocumentField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
