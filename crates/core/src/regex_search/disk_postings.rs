use std::{
    collections::{BTreeSet, HashMap},
    fs::{self, File},
    hash::{Hash, Hasher},
    io::{self, Write},
    ops::Range,
    path::{Path, PathBuf},
};

use memmap2::{Mmap, MmapOptions};

use super::{
    RegexCandidatePlan, RegexCandidateSelection, RegexPostingsError, Trigram,
    planner::{self, RegexPostingSource},
};
use crate::index::DocId;

const LOOKUP_FILE_NAME: &str = "regex_lookup.bin";
const POSTINGS_FILE_NAME: &str = "regex_postings.bin";
const LOOKUP_MAGIC: &[u8; 8] = b"rrplu001";
const POSTINGS_MAGIC: &[u8; 8] = b"rrpst001";
const HEADER_LEN: usize = 16;
const LOOKUP_ENTRY_LEN: usize = 24;
const DOC_ID_LEN: usize = 4;

#[derive(Debug)]
pub struct MmapRegexPostings {
    lookup: Mmap,
    postings: Mmap,
}

impl MmapRegexPostings {
    pub fn write(
        directory: impl AsRef<Path>,
        postings: &HashMap<Trigram, BTreeSet<DocId>>,
    ) -> Result<(), RegexPostingsError> {
        fs::create_dir_all(directory.as_ref())?;
        let lookup_path = directory.as_ref().join(LOOKUP_FILE_NAME);
        let postings_path = directory.as_ref().join(POSTINGS_FILE_NAME);

        let mut lookup_entries = postings
            .iter()
            .map(|(trigram, doc_ids)| LookupEntry {
                hash: trigram_hash(trigram),
                offset: 0,
                doc_count: doc_ids.len() as u64,
                doc_ids,
            })
            .collect::<Vec<_>>();
        lookup_entries.sort_by(|left, right| {
            left.hash
                .cmp(&right.hash)
                .then_with(|| left.doc_ids.cmp(right.doc_ids))
        });

        let mut postings_file = File::create(postings_path)?;
        postings_file.write_all(POSTINGS_MAGIC)?;
        write_u64(&mut postings_file, lookup_entries.len() as u64)?;

        let mut offset = HEADER_LEN as u64;
        for entry in &mut lookup_entries {
            entry.offset = offset;
            for doc_id in entry.doc_ids {
                write_u32(&mut postings_file, doc_id.as_u32())?;
            }
            offset += entry.doc_count * DOC_ID_LEN as u64;
        }

        let mut lookup_file = File::create(lookup_path)?;
        lookup_file.write_all(LOOKUP_MAGIC)?;
        write_u64(&mut lookup_file, lookup_entries.len() as u64)?;
        for entry in lookup_entries {
            write_u64(&mut lookup_file, entry.hash)?;
            write_u64(&mut lookup_file, entry.offset)?;
            write_u64(&mut lookup_file, entry.doc_count)?;
        }

        Ok(())
    }

    pub fn open(directory: impl AsRef<Path>) -> Result<Self, RegexPostingsError> {
        let lookup = mmap_read_only(directory.as_ref().join(LOOKUP_FILE_NAME))?;
        let postings = mmap_read_only(directory.as_ref().join(POSTINGS_FILE_NAME))?;
        validate_header(&lookup, LOOKUP_MAGIC)?;
        validate_header(&postings, POSTINGS_MAGIC)?;
        validate_lookup_len(&lookup)?;

        Ok(Self { lookup, postings })
    }

    pub fn postings(
        &self,
        trigram: &Trigram,
    ) -> Result<Option<BTreeSet<DocId>>, RegexPostingsError> {
        let hash = trigram_hash(trigram);
        let mut candidates = BTreeSet::new();
        let entry_range = self.lookup_entry_range(hash)?;
        let mut matched = false;

        for index in entry_range {
            let entry = self.lookup_entry(index)?;
            matched = true;
            candidates.extend(self.read_posting_list(entry.offset, entry.doc_count)?);
        }

        Ok(matched.then_some(candidates))
    }

    pub fn planned_candidates_for_regex_plan(
        &self,
        plan: &RegexCandidatePlan,
        all_doc_ids: &[DocId],
    ) -> Result<RegexCandidateSelection, RegexPostingsError> {
        planner::try_plan_candidates(plan, self, all_doc_ids)
    }

    pub fn lookup_entry_count(&self) -> Result<usize, RegexPostingsError> {
        read_count(&self.lookup)
    }

    pub fn mapped_bytes(&self) -> usize {
        self.lookup.len() + self.postings.len()
    }

    fn lookup_entry(&self, index: usize) -> Result<StoredLookupEntry, RegexPostingsError> {
        let start = HEADER_LEN + index * LOOKUP_ENTRY_LEN;
        let end = start + LOOKUP_ENTRY_LEN;
        let bytes = self
            .lookup
            .get(start..end)
            .ok_or(RegexPostingsError::InvalidFormat)?;

        Ok(StoredLookupEntry {
            hash: read_u64(&bytes[0..8])?,
            offset: read_u64(&bytes[8..16])?,
            doc_count: read_u64(&bytes[16..24])?,
        })
    }

    fn posting_len_for_hash(&self, hash: u64) -> Result<Option<usize>, RegexPostingsError> {
        let mut posting_len = 0usize;
        let entry_range = self.lookup_entry_range(hash)?;
        let mut matched = false;

        for index in entry_range {
            let entry = self.lookup_entry(index)?;
            matched = true;
            posting_len = posting_len
                .checked_add(
                    usize::try_from(entry.doc_count)
                        .map_err(|_| RegexPostingsError::FileTooLarge)?,
                )
                .ok_or(RegexPostingsError::FileTooLarge)?;
        }

        Ok(matched.then_some(posting_len))
    }

    fn lookup_entry_range(&self, hash: u64) -> Result<Range<usize>, RegexPostingsError> {
        let entry_count = read_count(&self.lookup)?;
        let lower = self.partition_point(entry_count, |entry_hash| entry_hash < hash)?;
        let upper = self.partition_point(entry_count, |entry_hash| entry_hash <= hash)?;
        Ok(lower..upper)
    }

    fn partition_point(
        &self,
        entry_count: usize,
        predicate: impl Fn(u64) -> bool,
    ) -> Result<usize, RegexPostingsError> {
        let mut left = 0;
        let mut right = entry_count;

        while left < right {
            let mid = left + (right - left) / 2;
            let entry = self.lookup_entry(mid)?;
            if predicate(entry.hash) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        Ok(left)
    }

    fn read_posting_list(
        &self,
        offset: u64,
        doc_count: u64,
    ) -> Result<BTreeSet<DocId>, RegexPostingsError> {
        let start = usize::try_from(offset).map_err(|_| RegexPostingsError::FileTooLarge)?;
        let byte_len = usize::try_from(doc_count)
            .map_err(|_| RegexPostingsError::FileTooLarge)?
            .checked_mul(DOC_ID_LEN)
            .ok_or(RegexPostingsError::FileTooLarge)?;
        let end = start
            .checked_add(byte_len)
            .ok_or(RegexPostingsError::FileTooLarge)?;
        let bytes = self
            .postings
            .get(start..end)
            .ok_or(RegexPostingsError::InvalidFormat)?;

        bytes
            .chunks_exact(DOC_ID_LEN)
            .map(|chunk| read_u32(chunk).map(DocId::from_u32))
            .collect()
    }
}

impl RegexPostingSource for MmapRegexPostings {
    type Error = RegexPostingsError;

    fn posting_len(&self, trigram: &Trigram) -> Result<Option<usize>, Self::Error> {
        self.posting_len_for_hash(trigram_hash(trigram))
    }

    fn postings(&self, trigram: &Trigram) -> Result<Option<BTreeSet<DocId>>, Self::Error> {
        Self::postings(self, trigram)
    }
}

#[derive(Debug)]
struct LookupEntry<'a> {
    hash: u64,
    offset: u64,
    doc_count: u64,
    doc_ids: &'a BTreeSet<DocId>,
}

#[derive(Debug)]
struct StoredLookupEntry {
    hash: u64,
    offset: u64,
    doc_count: u64,
}

fn mmap_read_only(path: PathBuf) -> Result<Mmap, RegexPostingsError> {
    let file = File::open(path)?;
    // SAFETY: Repo Reaper writes immutable index files and opens them read-only here.
    // Callers must replace an index by writing new files instead of mutating mapped bytes.
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(mmap)
}

fn validate_header(bytes: &[u8], magic: &[u8; 8]) -> Result<(), RegexPostingsError> {
    if bytes.len() < HEADER_LEN || &bytes[..8] != magic {
        return Err(RegexPostingsError::InvalidFormat);
    }
    Ok(())
}

fn validate_lookup_len(bytes: &[u8]) -> Result<(), RegexPostingsError> {
    let entry_count = read_count(bytes)?;
    let entries_len = entry_count
        .checked_mul(LOOKUP_ENTRY_LEN)
        .ok_or(RegexPostingsError::FileTooLarge)?;
    let expected_len = HEADER_LEN
        .checked_add(entries_len)
        .ok_or(RegexPostingsError::FileTooLarge)?;

    if bytes.len() != expected_len {
        return Err(RegexPostingsError::InvalidFormat);
    }

    Ok(())
}

fn read_count(bytes: &[u8]) -> Result<usize, RegexPostingsError> {
    usize::try_from(read_u64(&bytes[8..16])?).map_err(|_| RegexPostingsError::FileTooLarge)
}

fn trigram_hash(trigram: &Trigram) -> u64 {
    let mut hasher = StableHasher::default();
    trigram.as_str().hash(&mut hasher);
    hasher.finish()
}

#[derive(Default)]
struct StableHasher(u64);

impl Hasher for StableHasher {
    fn write(&mut self, bytes: &[u8]) {
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        if self.0 == 0 {
            self.0 = 0xcbf2_9ce4_8422_2325;
        }
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(FNV_PRIME);
        }
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_u32(bytes: &[u8]) -> Result<u32, RegexPostingsError> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| RegexPostingsError::InvalidFormat)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64(bytes: &[u8]) -> Result<u64, RegexPostingsError> {
    let bytes: [u8; 8] = bytes
        .try_into()
        .map_err(|_| RegexPostingsError::InvalidFormat)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_collisions_union_posting_lists_as_broader_candidates() {
        let temp = tempfile::tempdir().unwrap();
        let hash = trigram_hash(&Trigram::from("abc"));
        let lookup_path = temp.path().join(LOOKUP_FILE_NAME);
        let postings_path = temp.path().join(POSTINGS_FILE_NAME);

        let mut postings_file = File::create(postings_path).unwrap();
        postings_file.write_all(POSTINGS_MAGIC).unwrap();
        write_u64(&mut postings_file, 2).unwrap();
        write_u32(&mut postings_file, 1).unwrap();
        write_u32(&mut postings_file, 2).unwrap();
        write_u32(&mut postings_file, 3).unwrap();

        let mut lookup_file = File::create(lookup_path).unwrap();
        lookup_file.write_all(LOOKUP_MAGIC).unwrap();
        write_u64(&mut lookup_file, 2).unwrap();
        write_u64(&mut lookup_file, hash).unwrap();
        write_u64(&mut lookup_file, HEADER_LEN as u64).unwrap();
        write_u64(&mut lookup_file, 2).unwrap();
        write_u64(&mut lookup_file, hash).unwrap();
        write_u64(&mut lookup_file, (HEADER_LEN + 2 * DOC_ID_LEN) as u64).unwrap();
        write_u64(&mut lookup_file, 1).unwrap();
        drop(lookup_file);
        drop(postings_file);

        let postings = MmapRegexPostings::open(temp.path()).unwrap();
        let doc_ids = postings.postings(&Trigram::from("abc")).unwrap().unwrap();

        assert_eq!(
            doc_ids,
            [DocId::from_u32(1), DocId::from_u32(2), DocId::from_u32(3)]
                .into_iter()
                .collect()
        );
    }
}
