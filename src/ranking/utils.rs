pub fn idf(doc_count: usize, doc_freq: usize) -> f64 {
    ((doc_count as f64 + 1.0) / (doc_freq as f64 + 0.5)).ln()
}
