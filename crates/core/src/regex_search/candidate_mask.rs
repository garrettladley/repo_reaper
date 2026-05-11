use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::{Trigram, trigrams};
use crate::index::DocId;

const POSITION_BUCKETS: u8 = 8;
const NEXT_CHAR_BUCKETS: u32 = 128;

#[derive(Debug, Default)]
pub(crate) struct RegexCandidateMask {
    masks: HashMap<Trigram, BTreeMap<DocId, TrigramDocumentMask>>,
}

impl RegexCandidateMask {
    pub(crate) fn add_document(&mut self, doc_id: DocId, content: &str) {
        let chars = content.chars().collect::<Vec<_>>();
        for (offset, window) in chars.windows(3).enumerate() {
            let trigram = Trigram(window.iter().collect());
            let mask = self
                .masks
                .entry(trigram)
                .or_default()
                .entry(doc_id)
                .or_default();
            mask.add_position(offset);
            if let Some(&next_char) = chars.get(offset + 3) {
                mask.add_next_char(next_char);
            }
        }
    }

    pub(crate) fn remove_document(&mut self, doc_id: DocId) {
        for document_masks in self.masks.values_mut() {
            document_masks.remove(&doc_id);
        }
        self.masks
            .retain(|_, document_masks| !document_masks.is_empty());
    }

    pub(crate) fn filter_literal(
        &self,
        literal: &str,
        candidates: &BTreeSet<DocId>,
    ) -> BTreeSet<DocId> {
        let requirements = LiteralMaskRequirements::new(literal);
        if !requirements.is_selective() {
            return candidates.clone();
        }

        candidates
            .iter()
            .copied()
            .filter(|&doc_id| self.doc_may_match_literal(doc_id, &requirements))
            .collect()
    }

    fn doc_may_match_literal(&self, doc_id: DocId, requirements: &LiteralMaskRequirements) -> bool {
        (0..POSITION_BUCKETS).any(|start_bucket| {
            self.doc_matches_at_start_bucket(doc_id, requirements, start_bucket)
        })
    }

    fn doc_matches_at_start_bucket(
        &self,
        doc_id: DocId,
        requirements: &LiteralMaskRequirements,
        start_bucket: u8,
    ) -> bool {
        requirements.iter().all(|requirement| {
            let Some(mask) = self
                .masks
                .get(&requirement.trigram)
                .and_then(|document_masks| document_masks.get(&doc_id))
            else {
                return true;
            };

            let position_bucket = (start_bucket + requirement.offset_bucket) % POSITION_BUCKETS;
            mask.has_position(position_bucket)
                && requirement
                    .next_char
                    .is_none_or(|next_char| mask.has_next_char(next_char))
        })
    }
}

#[derive(Debug, Default)]
struct TrigramDocumentMask {
    position_buckets: u8,
    next_char_buckets: u128,
}

impl TrigramDocumentMask {
    fn add_position(&mut self, offset: usize) {
        let bucket = (offset as u8) % POSITION_BUCKETS;
        self.position_buckets |= 1_u8 << bucket;
    }

    fn add_next_char(&mut self, char_: char) {
        self.next_char_buckets |= next_char_bit(char_);
    }

    fn has_position(&self, bucket: u8) -> bool {
        self.position_buckets & (1_u8 << bucket) != 0
    }

    fn has_next_char(&self, char_: char) -> bool {
        self.next_char_buckets & next_char_bit(char_) != 0
    }
}

#[derive(Debug)]
struct LiteralMaskRequirements {
    requirements: Vec<LiteralMaskRequirement>,
}

impl LiteralMaskRequirements {
    fn new(literal: &str) -> Self {
        let chars = literal.chars().collect::<Vec<_>>();
        let literal_trigrams = trigrams(literal);
        let requirements = literal_trigrams
            .into_iter()
            .enumerate()
            .map(|(offset, trigram)| LiteralMaskRequirement {
                trigram,
                offset_bucket: (offset as u8) % POSITION_BUCKETS,
                next_char: chars.get(offset + 3).copied(),
            })
            .collect();

        Self { requirements }
    }

    fn is_selective(&self) -> bool {
        self.requirements
            .iter()
            .any(|requirement| requirement.next_char.is_some())
    }

    fn iter(&self) -> impl Iterator<Item = &LiteralMaskRequirement> {
        self.requirements.iter()
    }
}

#[derive(Debug)]
struct LiteralMaskRequirement {
    trigram: Trigram,
    offset_bucket: u8,
    next_char: Option<char>,
}

fn next_char_bit(char_: char) -> u128 {
    let bucket = (char_ as u32) % NEXT_CHAR_BUCKETS;
    1_u128 << bucket
}
