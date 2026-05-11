pub mod corpus;
pub mod document_registry;
pub mod field;
pub mod inverted_index;
pub mod term;

pub use corpus::{
    FileSystemIndexCorpus, IndexCorpus, IndexCorpusDocument, IndexCorpusScan, IndexSkipReason,
    SkippedDocument,
};
pub use document_registry::{DocId, DocumentCatalog, DocumentMetadata, DocumentRegistry};
pub use field::DocumentField;
pub use inverted_index::{
    CorpusStats, IndexBuildReport, IndexBuildResult, InvertedIndex, TermDocument,
    TermFrequencySummary,
};
pub use term::Term;
