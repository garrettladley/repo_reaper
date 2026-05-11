use std::{
    collections::{BTreeMap, HashMap, hash_map::DefaultHasher},
    fs,
    hash::{Hash, Hasher},
    io,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    config::Config,
    index::{
        CorpusStats, DocId, DocumentCatalog, DocumentMetadata, DocumentRegistry, InvertedIndex,
        Term, TermDocument,
    },
};

pub const SNAPSHOT_SCHEMA_VERSION: u32 = 1;
const SNAPSHOT_FILE: &str = "snapshot.json";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SnapshotMetadata {
    pub schema_version: u32,
    pub config_hash: u64,
    pub source_root: PathBuf,
    pub corpus_stats: CorpusStats,
    pub created_unix_secs: u64,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct IndexSnapshot {
    metadata: SnapshotMetadata,
    documents: Vec<DocumentMetadata>,
    postings: Vec<TermSnapshot>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TermSnapshot {
    term: String,
    documents: Vec<TermDocumentSnapshot>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct TermDocumentSnapshot {
    doc_id: u32,
    term_document: TermDocument,
}

#[derive(Debug, thiserror::Error)]
pub enum SnapshotError {
    #[error("snapshot io failed for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("snapshot serialization failed for {path}: {source}")]
    Json {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("snapshot schema version {found} is incompatible with {expected}")]
    Schema { found: u32, expected: u32 },
    #[error("snapshot config hash {found} does not match current config hash {expected}")]
    ConfigHash { found: u64, expected: u64 },
    #[error("snapshot source root {found} does not match current source root {expected}")]
    SourceRoot { found: PathBuf, expected: PathBuf },
}

pub fn snapshot_path(index_dir: &Path) -> PathBuf {
    index_dir.join(SNAPSHOT_FILE)
}

pub fn config_hash(config: &Config) -> u64 {
    let mut hasher = DefaultHasher::new();
    config.n_grams.hash(&mut hasher);
    let mut stop_words = config.stop_words.iter().collect::<Vec<_>>();
    stop_words.sort();
    for word in stop_words {
        word.hash(&mut hasher);
    }
    hasher.finish()
}

pub fn write_snapshot(
    index: &InvertedIndex,
    index_dir: &Path,
    source_root: &Path,
    config: &Config,
) -> Result<SnapshotMetadata, SnapshotError> {
    fs::create_dir_all(index_dir).map_err(|source| SnapshotError::Io {
        path: index_dir.to_path_buf(),
        source,
    })?;

    let metadata = SnapshotMetadata {
        schema_version: SNAPSHOT_SCHEMA_VERSION,
        config_hash: config_hash(config),
        source_root: source_root.to_path_buf(),
        corpus_stats: index.corpus_stats(10),
        created_unix_secs: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs()),
    };
    let snapshot = IndexSnapshot::from_index(index, metadata.clone());
    let path = snapshot_path(index_dir);
    let json = serde_json::to_vec_pretty(&snapshot).map_err(|source| SnapshotError::Json {
        path: path.clone(),
        source,
    })?;
    fs::write(&path, json).map_err(|source| SnapshotError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(metadata)
}

pub fn load_snapshot(
    index_dir: &Path,
    source_root: &Path,
    config: &Config,
) -> Result<InvertedIndex, SnapshotError> {
    let path = snapshot_path(index_dir);
    let json = fs::read(&path).map_err(|source| SnapshotError::Io {
        path: path.clone(),
        source,
    })?;
    let snapshot: IndexSnapshot =
        serde_json::from_slice(&json).map_err(|source| SnapshotError::Json {
            path: path.clone(),
            source,
        })?;
    snapshot.validate(source_root, config)?;
    Ok(snapshot.into_index())
}

impl IndexSnapshot {
    fn from_index(index: &InvertedIndex, metadata: SnapshotMetadata) -> Self {
        let mut documents = index
            .documents_iter()
            .map(|(_, metadata)| metadata.clone())
            .collect::<Vec<_>>();
        documents.sort_by_key(|metadata| metadata.id);

        let mut postings = index
            .postings_iter()
            .map(|(term, documents)| TermSnapshot {
                term: term.0.clone(),
                documents: documents
                    .iter()
                    .map(|(doc_id, term_document)| TermDocumentSnapshot {
                        doc_id: doc_id.as_u32(),
                        term_document: term_document.clone(),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        postings.sort_by_key(|snapshot| snapshot.term.clone());

        Self {
            metadata,
            documents,
            postings,
        }
    }

    fn validate(&self, source_root: &Path, config: &Config) -> Result<(), SnapshotError> {
        if self.metadata.schema_version != SNAPSHOT_SCHEMA_VERSION {
            return Err(SnapshotError::Schema {
                found: self.metadata.schema_version,
                expected: SNAPSHOT_SCHEMA_VERSION,
            });
        }

        let expected_config_hash = config_hash(config);
        if self.metadata.config_hash != expected_config_hash {
            return Err(SnapshotError::ConfigHash {
                found: self.metadata.config_hash,
                expected: expected_config_hash,
            });
        }

        if self.metadata.source_root != source_root {
            return Err(SnapshotError::SourceRoot {
                found: self.metadata.source_root.clone(),
                expected: source_root.to_path_buf(),
            });
        }

        Ok(())
    }

    fn into_index(self) -> InvertedIndex {
        let mut registry = DocumentRegistry::new();
        for metadata in self.documents {
            registry.insert_or_update_with_fields(
                metadata.path,
                metadata.token_length,
                metadata.file_size_bytes,
                metadata.file_type,
                metadata.field_lengths,
            );
        }

        let postings = self
            .postings
            .into_iter()
            .map(|term| {
                let documents = term
                    .documents
                    .into_iter()
                    .map(|document| (DocId::from_u32(document.doc_id), document.term_document))
                    .collect::<BTreeMap<_, _>>();
                (Term(term.term), documents)
            })
            .collect::<HashMap<_, _>>();

        InvertedIndex::from_parts(postings, registry)
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, fs};

    use rust_stemmers::{Algorithm, Stemmer};

    use super::{SnapshotError, load_snapshot, snapshot_path, write_snapshot};
    use crate::{
        config::Config,
        index::InvertedIndex,
        query::AnalyzedQuery,
        ranking::{BM25HyperParams, RankingAlgo},
        tokenizer::n_gram_transform,
    };

    fn config() -> Config {
        Config {
            n_grams: 1,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: HashSet::new(),
        }
    }

    #[test]
    fn snapshot_round_trips_ranked_search_results() {
        let source = tempfile::tempdir().unwrap();
        let index_dir = tempfile::tempdir().unwrap();
        fs::write(source.path().join("a.rs"), "rust rust search").unwrap();
        fs::write(source.path().join("b.rs"), "search").unwrap();
        let config = config();
        let index = InvertedIndex::new(
            source.path(),
            |content| n_gram_transform(content, &config),
            Some(source.path()),
        );
        write_snapshot(&index, index_dir.path(), source.path(), &config).unwrap();

        let loaded = load_snapshot(index_dir.path(), source.path(), &config).unwrap();
        let query = AnalyzedQuery::new("rust search", &config);
        let algo = RankingAlgo::BM25(BM25HyperParams { k1: 1.2, b: 0.75 });

        assert_eq!(
            algo.rank(&loaded, &query, 10),
            algo.rank(&index, &query, 10)
        );
    }

    #[test]
    fn snapshot_rejects_invalid_metadata() {
        let source = tempfile::tempdir().unwrap();
        let index_dir = tempfile::tempdir().unwrap();
        fs::write(source.path().join("a.rs"), "rust").unwrap();
        let config = config();
        let index = InvertedIndex::new(
            source.path(),
            |content| n_gram_transform(content, &config),
            Some(source.path()),
        );
        write_snapshot(&index, index_dir.path(), source.path(), &config).unwrap();

        let mut json = fs::read_to_string(snapshot_path(index_dir.path())).unwrap();
        json = json.replace("\"schema_version\": 1", "\"schema_version\": 99");
        fs::write(snapshot_path(index_dir.path()), json).unwrap();

        assert!(matches!(
            load_snapshot(index_dir.path(), source.path(), &config).unwrap_err(),
            SnapshotError::Schema { .. }
        ));
    }
}
