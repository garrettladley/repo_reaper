use crate::index::DocId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompressedPostingList {
    bytes: Vec<u8>,
    doc_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPosting {
    pub doc_id: DocId,
    pub term_freq: u32,
    pub positions: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompressionError {
    #[error("compressed posting list ended unexpectedly")]
    UnexpectedEof,
    #[error("compressed integer is too large")]
    IntegerOverflow,
    #[error("posting list doc ids are not sorted")]
    UnsortedDocIds,
    #[error("posting positions are not sorted")]
    UnsortedPositions,
    #[error("decoded posting gap overflowed")]
    GapOverflow,
}

impl CompressedPostingList {
    pub fn from_postings(postings: &[DecodedPosting]) -> Result<Self, CompressionError> {
        let mut bytes = Vec::new();
        let mut previous_doc = 0;

        for posting in postings {
            let doc = posting.doc_id.as_u32();
            if doc < previous_doc {
                return Err(CompressionError::UnsortedDocIds);
            }

            encode_u32(doc - previous_doc, &mut bytes);
            encode_u32(posting.term_freq, &mut bytes);
            encode_u32(posting.positions.len() as u32, &mut bytes);

            let mut previous_position = 0;
            for &position in &posting.positions {
                if position < previous_position {
                    return Err(CompressionError::UnsortedPositions);
                }
                encode_u32(position - previous_position, &mut bytes);
                previous_position = position;
            }

            previous_doc = doc;
        }

        Ok(Self {
            bytes,
            doc_count: postings.len(),
        })
    }

    pub fn decode(&self) -> Result<Vec<DecodedPosting>, CompressionError> {
        decode_postings(&self.bytes, self.doc_count)
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn doc_count(&self) -> usize {
        self.doc_count
    }
}

pub fn encode_u32(mut value: u32, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push((value as u8 & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

pub fn decode_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, CompressionError> {
    let mut value = 0u32;
    let mut shift = 0;

    loop {
        let byte = *bytes.get(*offset).ok_or(CompressionError::UnexpectedEof)?;
        *offset += 1;

        if shift >= 32 && byte != 0 {
            return Err(CompressionError::IntegerOverflow);
        }

        value |= ((byte & 0x7f) as u32)
            .checked_shl(shift)
            .ok_or(CompressionError::IntegerOverflow)?;

        if byte & 0x80 == 0 {
            return Ok(value);
        }

        shift += 7;
        if shift > 35 {
            return Err(CompressionError::IntegerOverflow);
        }
    }
}

pub fn decode_postings(
    bytes: &[u8],
    doc_count: usize,
) -> Result<Vec<DecodedPosting>, CompressionError> {
    let mut offset = 0;
    let mut previous_doc = 0u32;
    let mut postings = Vec::with_capacity(doc_count);

    for _ in 0..doc_count {
        let doc_gap = decode_u32(bytes, &mut offset)?;
        let doc = previous_doc
            .checked_add(doc_gap)
            .ok_or(CompressionError::GapOverflow)?;
        let term_freq = decode_u32(bytes, &mut offset)?;
        let position_count = decode_u32(bytes, &mut offset)?;
        let mut positions = Vec::with_capacity(position_count as usize);
        let mut previous_position = 0u32;

        for _ in 0..position_count {
            let position_gap = decode_u32(bytes, &mut offset)?;
            let position = previous_position
                .checked_add(position_gap)
                .ok_or(CompressionError::GapOverflow)?;
            positions.push(position);
            previous_position = position;
        }

        postings.push(DecodedPosting {
            doc_id: DocId::from_u32(doc),
            term_freq,
            positions,
        });
        previous_doc = doc;
    }

    Ok(postings)
}

#[cfg(test)]
mod tests {
    use super::{CompressedPostingList, CompressionError, DecodedPosting, decode_u32, encode_u32};
    use crate::index::DocId;

    #[test]
    fn vbyte_round_trips_multi_byte_values() {
        let mut bytes = Vec::new();
        for value in [0, 1, 127, 128, 16_384, u32::MAX] {
            encode_u32(value, &mut bytes);
        }

        let mut offset = 0;
        for value in [0, 1, 127, 128, 16_384, u32::MAX] {
            assert_eq!(decode_u32(&bytes, &mut offset).unwrap(), value);
        }
    }

    #[test]
    fn compressed_postings_round_trip_doc_gaps_and_position_gaps() {
        let postings = vec![
            DecodedPosting {
                doc_id: DocId::from_u32(3),
                term_freq: 2,
                positions: vec![1, 4],
            },
            DecodedPosting {
                doc_id: DocId::from_u32(10),
                term_freq: 1,
                positions: vec![8],
            },
        ];

        let compressed = CompressedPostingList::from_postings(&postings).unwrap();

        assert_eq!(compressed.decode().unwrap(), postings);
    }

    #[test]
    fn malformed_vbyte_fails_without_panic() {
        let mut offset = 0;

        assert_eq!(
            decode_u32(&[0x80], &mut offset).unwrap_err(),
            CompressionError::UnexpectedEof
        );
    }
}
