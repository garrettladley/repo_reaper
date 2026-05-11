use std::{
    fs::{self, OpenOptions},
    io::{self, BufRead, Write},
    path::{Path, PathBuf},
};

use crate::{
    config::Config,
    index::{InvertedIndex, Term},
};

const EVENT_LOG_FILE: &str = "events.jsonl";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum IndexEvent {
    FileAdded { path: PathBuf },
    FileModified { path: PathBuf },
    FileDeleted { path: PathBuf },
}

#[derive(Debug, thiserror::Error)]
pub enum EventLogError {
    #[error("event log io failed for {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("event log json failed for {path} line {line}: {source}")]
    Json {
        path: PathBuf,
        line: usize,
        #[source]
        source: serde_json::Error,
    },
}

pub fn event_log_path(index_dir: &Path) -> PathBuf {
    index_dir.join(EVENT_LOG_FILE)
}

pub fn append_event(index_dir: &Path, event: &IndexEvent) -> Result<(), EventLogError> {
    fs::create_dir_all(index_dir).map_err(|source| EventLogError::Io {
        path: index_dir.to_path_buf(),
        source,
    })?;
    let path = event_log_path(index_dir);
    let mut file = OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
        .map_err(|source| EventLogError::Io {
            path: path.clone(),
            source,
        })?;
    serde_json::to_writer(&mut file, event).map_err(|source| EventLogError::Json {
        path: path.clone(),
        line: 0,
        source,
    })?;
    file.write_all(b"\n")
        .map_err(|source| EventLogError::Io { path, source })
}

pub fn read_events(index_dir: &Path) -> Result<Vec<IndexEvent>, EventLogError> {
    let path = event_log_path(index_dir);
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = fs::File::open(&path).map_err(|source| EventLogError::Io {
        path: path.clone(),
        source,
    })?;
    let mut events = Vec::new();
    for (line_idx, line) in io::BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| EventLogError::Io {
            path: path.clone(),
            source,
        })?;
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str(&line).map_err(|source| EventLogError::Json {
            path: path.clone(),
            line: line_idx + 1,
            source,
        })?;
        events.push(event);
    }
    Ok(events)
}

pub fn clear_events(index_dir: &Path) -> Result<(), EventLogError> {
    let path = event_log_path(index_dir);
    fs::write(&path, b"").map_err(|source| EventLogError::Io { path, source })
}

pub fn replay_events<F>(
    index: &mut InvertedIndex,
    events: &[IndexEvent],
    transform_fn: &F,
    config: &Config,
    fielded: bool,
) where
    F: Fn(&str) -> std::collections::HashMap<Term, u32> + Sync,
{
    for event in events {
        apply_event(index, event, transform_fn, config, fielded);
    }
}

pub fn apply_event<F>(
    index: &mut InvertedIndex,
    event: &IndexEvent,
    transform_fn: &F,
    config: &Config,
    fielded: bool,
) where
    F: Fn(&str) -> std::collections::HashMap<Term, u32> + Sync,
{
    match event {
        IndexEvent::FileAdded { path } | IndexEvent::FileModified { path } => {
            if fielded {
                index.update_fielded(path, config);
            } else {
                index.update(path, transform_fn);
            }
        }
        IndexEvent::FileDeleted { path } => index.remove_document(path),
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use super::{IndexEvent, append_event, clear_events, read_events, replay_events};
    use crate::{
        config::Config,
        index::{InvertedIndex, Term},
    };

    fn transform(content: &str) -> HashMap<Term, u32> {
        content
            .split_whitespace()
            .fold(HashMap::new(), |mut acc, word| {
                *acc.entry(Term(word.to_string())).or_insert(0) += 1;
                acc
            })
    }

    fn config() -> Config {
        Config {
            n_grams: 1,
            stemmer: rust_stemmers::Stemmer::create(rust_stemmers::Algorithm::English),
            stop_words: Default::default(),
        }
    }

    #[test]
    fn event_log_replays_updates_in_order() {
        let source = tempfile::tempdir().unwrap();
        let index_dir = tempfile::tempdir().unwrap();
        let path = source.path().join("a.rs");
        fs::write(&path, "old").unwrap();
        let mut index = InvertedIndex::new(source.path(), transform, None::<&std::path::Path>);
        fs::write(&path, "new").unwrap();

        append_event(index_dir.path(), &IndexEvent::FileModified { path }).unwrap();
        let events = read_events(index_dir.path()).unwrap();
        replay_events(&mut index, &events, &transform, &config(), false);

        assert!(index.get_postings(&Term("old".to_string())).is_none());
        assert!(index.get_postings(&Term("new".to_string())).is_some());
    }

    #[test]
    fn clearing_events_compacts_replay_log() {
        let index_dir = tempfile::tempdir().unwrap();
        append_event(
            index_dir.path(),
            &IndexEvent::FileDeleted {
                path: "a.rs".into(),
            },
        )
        .unwrap();

        clear_events(index_dir.path()).unwrap();

        assert!(read_events(index_dir.path()).unwrap().is_empty());
    }
}
