pub mod document_registry;
pub mod inverted_index;
pub mod term;

pub use document_registry::{DocId, DocumentCatalog, DocumentMetadata, DocumentRegistry};
pub use inverted_index::{CorpusStats, InvertedIndex, TermDocument, TermFrequencySummary};
pub use term::Term;
