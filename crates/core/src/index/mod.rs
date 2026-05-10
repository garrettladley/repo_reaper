pub mod document_registry;
pub mod inverted_index;
pub mod term;

pub use document_registry::{DocId, DocumentCatalog, DocumentMetadata, DocumentRegistry};
pub use inverted_index::{InvertedIndex, TermDocument};
pub use term::Term;
