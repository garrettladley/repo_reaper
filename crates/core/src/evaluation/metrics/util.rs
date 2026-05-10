pub(crate) fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

pub(crate) fn estimate_tokens_from_bytes(bytes: usize) -> usize {
    bytes.div_ceil(4)
}
