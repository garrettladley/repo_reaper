pub mod compression;
pub mod corpus;
pub mod document_registry;
pub mod engine;
pub mod event_log;
pub mod field;
pub mod inverted_file;
pub mod inverted_index;
pub mod reader;
pub mod skips;
pub mod snapshot;
pub mod term;

pub use corpus::{
    FileSystemIndexCorpus, IndexCorpus, IndexCorpusDocument, IndexCorpusScan, IndexSkipReason,
    SkippedDocument,
};
pub use document_registry::{DocId, DocumentCatalog, DocumentMetadata, DocumentRegistry};
pub use engine::{SearchEngine, SearchEngineError};
pub use event_log::IndexEvent;
pub use field::DocumentField;
pub use inverted_index::{
    CorpusStats, IndexBuildReport, IndexBuildResult, InvertedIndex, TermDocument,
    TermFrequencySummary,
};
pub use reader::{OwnedPostingList, PostingList, RankedIndexReader};
pub use term::Term;
