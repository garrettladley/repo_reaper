use crate::index::{
    DocId,
    compression::{CompressionError, decode_u32},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkipPointer {
    pub target_doc_id: DocId,
    pub byte_offset: usize,
    pub ordinal: usize,
    pub previous_doc_id: DocId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkipTable {
    pointers: Vec<SkipPointer>,
    block_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntersectionStats {
    pub inspected_without_skips: usize,
    pub inspected_with_skips: usize,
}

impl SkipTable {
    pub fn build(
        encoded: &[u8],
        doc_count: usize,
        block_size: usize,
    ) -> Result<Self, CompressionError> {
        let mut pointers = Vec::new();
        let mut offset = 0;
        let mut previous_doc = 0;

        for ordinal in 0..doc_count {
            if ordinal > 0 && ordinal % block_size == 0 {
                pointers.push(SkipPointer {
                    target_doc_id: DocId::from_u32(previous_doc),
                    byte_offset: offset,
                    ordinal,
                    previous_doc_id: DocId::from_u32(previous_doc),
                });
            }

            let doc_gap = decode_u32(encoded, &mut offset)?;
            previous_doc += doc_gap;
            let _term_frequency = decode_u32(encoded, &mut offset)?;
            let position_count = decode_u32(encoded, &mut offset)?;
            for _ in 0..position_count {
                let _position_gap = decode_u32(encoded, &mut offset)?;
            }
        }

        Ok(Self {
            pointers,
            block_size,
        })
    }

    pub fn pointers(&self) -> &[SkipPointer] {
        &self.pointers
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }
}

pub fn intersect_doc_ids_with_optional_skips(
    left: &[DocId],
    right: &[DocId],
    use_skips: bool,
) -> (Vec<DocId>, IntersectionStats) {
    if !use_skips {
        let (matches, inspected) = intersect_linear(left, right);
        return (
            matches,
            IntersectionStats {
                inspected_without_skips: inspected,
                inspected_with_skips: inspected,
            },
        );
    }

    let block = (left.len().max(right.len()) as f64).sqrt().max(2.0) as usize;
    let mut matches = Vec::new();
    let mut inspected = 0;
    let mut left_idx = 0;
    let mut right_idx = 0;

    while left_idx < left.len() && right_idx < right.len() {
        inspected += 1;
        let left_doc = left[left_idx];
        let right_doc = right[right_idx];

        if left_doc == right_doc {
            matches.push(left_doc);
            left_idx += 1;
            right_idx += 1;
        } else if left_doc < right_doc {
            left_idx = skip_forward(left, left_idx, right_doc, block);
        } else {
            right_idx = skip_forward(right, right_idx, left_doc, block);
        }
    }

    let (_, baseline) = intersect_linear(left, right);
    (
        matches,
        IntersectionStats {
            inspected_without_skips: baseline,
            inspected_with_skips: inspected,
        },
    )
}

fn skip_forward(list: &[DocId], current: usize, target: DocId, block: usize) -> usize {
    let mut next = current + 1;
    while next + block < list.len() && list[next + block] <= target {
        next += block;
    }
    next
}

fn intersect_linear(left: &[DocId], right: &[DocId]) -> (Vec<DocId>, usize) {
    let mut matches = Vec::new();
    let mut inspected = 0;
    let mut left_idx = 0;
    let mut right_idx = 0;

    while left_idx < left.len() && right_idx < right.len() {
        inspected += 1;
        match left[left_idx].cmp(&right[right_idx]) {
            std::cmp::Ordering::Equal => {
                matches.push(left[left_idx]);
                left_idx += 1;
                right_idx += 1;
            }
            std::cmp::Ordering::Less => left_idx += 1,
            std::cmp::Ordering::Greater => right_idx += 1,
        }
    }

    (matches, inspected)
}

#[cfg(test)]
mod tests {
    use super::{SkipTable, intersect_doc_ids_with_optional_skips};
    use crate::index::{
        DocId,
        compression::{CompressedPostingList, DecodedPosting},
    };

    #[test]
    fn skip_table_records_block_offsets_for_compressed_postings() {
        let postings = (0..8)
            .map(|doc_id| DecodedPosting {
                doc_id: DocId::from_u32(doc_id * 2),
                term_freq: 1,
                positions: Vec::new(),
            })
            .collect::<Vec<_>>();
        let compressed = CompressedPostingList::from_postings(&postings).unwrap();

        let skips = SkipTable::build(compressed.bytes(), compressed.doc_count(), 3).unwrap();

        assert_eq!(skips.pointers().len(), 2);
        assert_eq!(skips.pointers()[0].ordinal, 3);
    }

    #[test]
    fn skipped_intersection_matches_linear_intersection() {
        let left = (0..100).map(DocId::from_u32).collect::<Vec<_>>();
        let right = (50..150).map(DocId::from_u32).collect::<Vec<_>>();

        let (linear, _) = intersect_doc_ids_with_optional_skips(&left, &right, false);
        let (skipped, stats) = intersect_doc_ids_with_optional_skips(&left, &right, true);

        assert_eq!(skipped, linear);
        assert!(stats.inspected_with_skips < stats.inspected_without_skips);
    }
}
