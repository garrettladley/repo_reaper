use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use crate::{
    config::Config,
    index::{
        corpus::{FileSystemIndexCorpus, IndexCorpus, IndexCorpusDocument, SkippedDocument},
        document_registry::{DocId, DocumentCatalog, DocumentMetadata, DocumentRegistry},
        field::DocumentField,
        term::Term,
    },
    ranking::idf,
    tokenizer::{AnalyzerProfile, FileType},
};

#[derive(Debug)]
pub struct TermDocument {
    pub length: usize,
    pub term_freq: usize,
    pub field_frequencies: HashMap<DocumentField, usize>,
    pub field_lengths: HashMap<DocumentField, usize>,
}

impl TermDocument {
    pub fn unfielded(length: usize, term_freq: usize) -> Self {
        Self {
            length,
            term_freq,
            field_frequencies: HashMap::new(),
            field_lengths: HashMap::new(),
        }
    }

    pub fn field_term_freq(&self, field: DocumentField) -> usize {
        self.field_frequencies.get(&field).copied().unwrap_or(0)
    }

    pub fn field_length(&self, field: DocumentField) -> usize {
        self.field_lengths.get(&field).copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusStats {
    pub document_count: usize,
    pub total_token_count: u64,
    pub vocabulary_size: usize,
    pub singleton_term_count: usize,
    pub high_frequency_terms: Vec<TermFrequencySummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermFrequencySummary {
    pub term: String,
    pub collection_frequency: usize,
    pub document_frequency: usize,
}

#[derive(Debug)]
pub struct IndexBuildResult {
    pub index: InvertedIndex,
    pub report: IndexBuildReport,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexBuildReport {
    pub indexed_document_count: usize,
    pub skipped_documents: Vec<SkippedDocument>,
}

impl IndexBuildReport {
    pub fn skipped_document_count(&self) -> usize {
        self.skipped_documents.len()
    }
}

#[derive(Debug)]
pub struct InvertedIndex<R = DocumentRegistry> {
    postings: HashMap<Term, HashMap<DocId, TermDocument>>,
    documents: R,
    document_norms: HashMap<DocId, f64>,
}

#[derive(Debug)]
struct ProcessedDocument {
    path: PathBuf,
    term_frequencies: HashMap<Term, u32>,
    field_term_frequencies: HashMap<DocumentField, HashMap<Term, u32>>,
    field_lengths: HashMap<DocumentField, usize>,
    token_length: usize,
    file_size_bytes: u64,
    file_type: FileType,
}

#[cfg(test)]
type TestDocumentSpec<'a> = (&'a str, &'a [(&'a str, u32)], u64);

impl InvertedIndex<DocumentRegistry> {
    #[cfg(test)]
    pub(crate) fn from_documents(docs: &[(&str, &[(&str, u32)])]) -> Self {
        Self::from_documents_with_sizes(
            &docs
                .iter()
                .map(|(path, terms)| (*path, *terms, 0))
                .collect::<Vec<_>>(),
        )
    }

    #[cfg(test)]
    pub(crate) fn from_documents_with_sizes(docs: &[TestDocumentSpec<'_>]) -> Self {
        let mut documents = DocumentRegistry::new();
        let mut postings: HashMap<Term, HashMap<DocId, TermDocument>> = HashMap::new();

        for &(path, terms, file_size_bytes) in docs {
            let total_len: usize = terms.iter().map(|(_, c)| *c as usize).sum();
            let doc_id =
                documents.insert_or_update(PathBuf::from(path), total_len, file_size_bytes);

            for &(term, freq) in terms {
                postings
                    .entry(Term(term.to_string()))
                    .or_default()
                    .insert(doc_id, TermDocument::unfielded(total_len, freq as usize));
            }
        }

        let document_norms = Self::compute_document_norms(&postings, documents.len());

        InvertedIndex {
            postings,
            documents,
            document_norms,
        }
    }

    pub fn new<P, F>(root: P, transform_fn: F, drop_prefix: Option<P>) -> Self
    where
        P: AsRef<Path>,
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        Self::build(root, transform_fn, drop_prefix).index
    }

    pub fn build<P, F>(root: P, transform_fn: F, drop_prefix: Option<P>) -> IndexBuildResult
    where
        P: AsRef<Path>,
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let corpus = FileSystemIndexCorpus::new(root, drop_prefix);
        Self::from_corpus(&corpus, transform_fn)
    }

    pub fn from_corpus<C, F>(corpus: &C, transform_fn: F) -> IndexBuildResult
    where
        C: IndexCorpus,
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let scan = corpus.scan();
        let mut registry = DocumentRegistry::new();
        let mut postings = HashMap::new();
        for document in &scan.documents {
            let document = Self::analyze_document(document, &transform_fn);
            Self::insert_processed_document(&mut registry, &mut postings, document);
        }

        let document_norms = Self::compute_document_norms(&postings, registry.len());

        let index = Self {
            postings,
            documents: registry,
            document_norms,
        };
        let report = IndexBuildReport {
            indexed_document_count: scan.indexed_document_count(),
            skipped_documents: scan.skipped_documents,
        };

        IndexBuildResult { index, report }
    }

    pub fn new_fielded<P>(root: P, config: &Config, drop_prefix: Option<P>) -> Self
    where
        P: AsRef<Path>,
    {
        Self::build_fielded(root, config, drop_prefix).index
    }

    pub fn build_fielded<P>(root: P, config: &Config, drop_prefix: Option<P>) -> IndexBuildResult
    where
        P: AsRef<Path>,
    {
        let corpus = FileSystemIndexCorpus::new(root, drop_prefix);
        Self::from_corpus_fielded(&corpus, config)
    }

    pub fn from_corpus_fielded<C>(corpus: &C, config: &Config) -> IndexBuildResult
    where
        C: IndexCorpus,
    {
        let scan = corpus.scan();
        let mut registry = DocumentRegistry::new();
        let mut postings = HashMap::new();
        for document in &scan.documents {
            let document = Self::analyze_fielded_document(document, config);
            Self::insert_processed_document(&mut registry, &mut postings, document);
        }

        let document_norms = Self::compute_document_norms(&postings, registry.len());

        let index = Self {
            postings,
            documents: registry,
            document_norms,
        };
        let report = IndexBuildReport {
            indexed_document_count: scan.indexed_document_count(),
            skipped_documents: scan.skipped_documents,
        };

        IndexBuildResult { index, report }
    }
}

impl<R> InvertedIndex<R>
where
    R: DocumentCatalog,
{
    #[cfg(test)]
    fn process_document<F>(
        entry_path: &Path,
        content: &str,
        transform_fn: &F,
    ) -> HashMap<Term, HashMap<DocId, TermDocument>>
    where
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let document = Self::analyze_document(
            &IndexCorpusDocument {
                path: entry_path.to_path_buf(),
                content: content.to_string(),
                file_size_bytes: content.len() as u64,
            },
            transform_fn,
        );
        let mut registry = DocumentRegistry::new();
        let doc_id = registry.insert_or_update_with_fields(
            document.path.clone(),
            document.token_length,
            0,
            document.file_type,
            document.field_lengths.clone(),
        );
        Self::postings_for_document(doc_id, &document)
    }

    fn analyze_document<F>(document: &IndexCorpusDocument, transform_fn: &F) -> ProcessedDocument
    where
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        let term_frequencies = transform_fn(&document.content);
        let token_length: usize = term_frequencies.values().map(|&c| c as usize).sum();

        ProcessedDocument {
            path: document.path.clone(),
            term_frequencies,
            field_term_frequencies: HashMap::new(),
            field_lengths: HashMap::new(),
            token_length,
            file_size_bytes: document.file_size_bytes,
            file_type: FileType::UnknownText,
        }
    }

    fn analyze_fielded_document(
        document: &IndexCorpusDocument,
        config: &Config,
    ) -> ProcessedDocument {
        let file_type = FileType::detect(&document.path);
        let profile = AnalyzerProfile::for_file_type(file_type);
        let raw_fields = raw_document_fields(&document.path, &document.content);
        let mut field_term_frequencies = HashMap::new();
        let mut field_lengths = HashMap::new();
        let mut term_frequencies = HashMap::new();

        for field in DocumentField::ALL {
            let Some(raw_value) = raw_fields.get(&field) else {
                continue;
            };
            let tokens = profile.analyze(field.analyzer_field(), raw_value, config);
            if tokens.is_empty() {
                continue;
            }

            let mut frequencies = HashMap::new();
            for token in tokens {
                *frequencies.entry(Term(token)).or_insert(0) += 1;
            }

            let field_length = frequencies.values().map(|&count| count as usize).sum();
            field_lengths.insert(field, field_length);
            for (term, frequency) in &frequencies {
                *term_frequencies.entry(term.clone()).or_insert(0) += *frequency;
            }
            field_term_frequencies.insert(field, frequencies);
        }

        let token_length = term_frequencies.values().map(|&c| c as usize).sum();

        ProcessedDocument {
            path: document.path.clone(),
            term_frequencies,
            field_term_frequencies,
            field_lengths,
            token_length,
            file_size_bytes: document.file_size_bytes,
            file_type,
        }
    }

    fn postings_for_document(
        doc_id: DocId,
        document: &ProcessedDocument,
    ) -> HashMap<Term, HashMap<DocId, TermDocument>> {
        let mut local_map: HashMap<Term, HashMap<DocId, TermDocument>> = HashMap::new();

        for (term, freq) in &document.term_frequencies {
            let field_frequencies = document
                .field_term_frequencies
                .iter()
                .filter_map(|(field, frequencies)| {
                    frequencies
                        .get(term)
                        .map(|frequency| (*field, *frequency as usize))
                })
                .collect::<HashMap<_, _>>();
            local_map.entry(term.clone()).or_default().insert(
                doc_id,
                TermDocument {
                    length: document.token_length,
                    term_freq: *freq as usize,
                    field_frequencies,
                    field_lengths: document.field_lengths.clone(),
                },
            );
        }

        local_map
    }

    fn insert_processed_document(
        registry: &mut impl DocumentCatalog,
        postings: &mut HashMap<Term, HashMap<DocId, TermDocument>>,
        document: ProcessedDocument,
    ) {
        let doc_id = registry.insert_or_update_with_fields(
            document.path.clone(),
            document.token_length,
            document.file_size_bytes,
            document.file_type,
            document.field_lengths.clone(),
        );

        for (term, doc_map) in Self::postings_for_document(doc_id, &document) {
            postings.entry(term).or_default().extend(doc_map);
        }
    }

    pub fn get_postings(&self, term: &Term) -> Option<&HashMap<DocId, TermDocument>> {
        self.postings.get(term)
    }

    pub fn postings_iter(&self) -> impl Iterator<Item = (&Term, &HashMap<DocId, TermDocument>)> {
        self.postings.iter()
    }

    pub fn document_norm(&self, doc_id: DocId) -> Option<f64> {
        self.document_norms.get(&doc_id).copied()
    }

    pub fn num_docs(&self) -> usize {
        self.documents.len()
    }

    pub fn avg_doc_length(&self) -> f64 {
        self.documents.avg_doc_length()
    }

    pub fn avg_field_length(&self, field: DocumentField) -> f64 {
        self.documents.avg_field_length(field)
    }

    pub fn corpus_stats(&self, high_frequency_limit: usize) -> CorpusStats {
        let mut term_summaries: Vec<TermFrequencySummary> = self
            .postings
            .iter()
            .map(|(term, documents)| TermFrequencySummary {
                term: term.0.clone(),
                collection_frequency: documents.values().map(|doc| doc.term_freq).sum(),
                document_frequency: documents.len(),
            })
            .collect();

        let singleton_term_count = term_summaries
            .iter()
            .filter(|summary| summary.collection_frequency == 1)
            .count();

        term_summaries.sort_by(|left, right| {
            right
                .collection_frequency
                .cmp(&left.collection_frequency)
                .then_with(|| right.document_frequency.cmp(&left.document_frequency))
                .then_with(|| left.term.cmp(&right.term))
        });
        term_summaries.truncate(high_frequency_limit);

        CorpusStats {
            document_count: self.documents.len(),
            total_token_count: self.documents.total_token_length(),
            vocabulary_size: self.postings.len(),
            singleton_term_count,
            high_frequency_terms: term_summaries,
        }
    }

    pub fn doc_id(&self, path: &Path) -> Option<DocId> {
        self.documents.doc_id(path)
    }

    pub fn document(&self, id: DocId) -> Option<&DocumentMetadata> {
        self.documents.get(id)
    }

    pub fn doc_freq(&self, term: &Term) -> usize {
        self.postings
            .get(term)
            .map_or(0, |documents| documents.len())
    }

    pub fn update<F>(&mut self, path: &Path, transform_fn: &F)
    where
        F: Fn(&str) -> HashMap<Term, u32> + Sync,
    {
        if let Some(doc_id) = self.documents.doc_id(path) {
            self.remove_postings_for_doc(doc_id);
        }

        match std::fs::read_to_string(path) {
            Ok(content) => {
                let document = Self::analyze_document(
                    &IndexCorpusDocument {
                        path: path.to_path_buf(),
                        file_size_bytes: content.len() as u64,
                        content,
                    },
                    transform_fn,
                );
                Self::insert_processed_document(&mut self.documents, &mut self.postings, document);
                self.rebuild_document_norms();
            }
            Err(_) => {
                self.documents.remove(path);
                self.rebuild_document_norms();
            }
        }
    }

    pub fn update_fielded(&mut self, path: &Path, config: &Config) {
        if let Some(doc_id) = self.documents.doc_id(path) {
            self.remove_postings_for_doc(doc_id);
        }

        match std::fs::read_to_string(path) {
            Ok(content) => {
                let document = Self::analyze_fielded_document(
                    &IndexCorpusDocument {
                        path: path.to_path_buf(),
                        file_size_bytes: content.len() as u64,
                        content,
                    },
                    config,
                );
                Self::insert_processed_document(&mut self.documents, &mut self.postings, document);
                self.rebuild_document_norms();
            }
            Err(_) => {
                self.documents.remove(path);
                self.rebuild_document_norms();
            }
        }
    }

    pub fn remove_document(&mut self, path: &Path) {
        if let Some(metadata) = self.documents.remove(path) {
            self.remove_postings_for_doc(metadata.id);
            self.rebuild_document_norms();
        }
    }

    fn remove_postings_for_doc(&mut self, doc_id: DocId) {
        self.postings.retain(|_, documents| {
            documents.remove(&doc_id);
            !documents.is_empty()
        });
    }

    fn rebuild_document_norms(&mut self) {
        self.document_norms = Self::compute_document_norms(&self.postings, self.documents.len());
    }

    fn compute_document_norms(
        postings: &HashMap<Term, HashMap<DocId, TermDocument>>,
        num_docs: usize,
    ) -> HashMap<DocId, f64> {
        let mut squared_weights = HashMap::new();

        for doc_map in postings.values() {
            let term_idf = idf(num_docs, doc_map.len());

            for (&doc_id, term_doc) in doc_map {
                let tf = term_doc.term_freq as f64;
                let weight = tf * term_idf;
                *squared_weights.entry(doc_id).or_insert(0.0) += weight * weight;
            }
        }

        squared_weights
            .into_iter()
            .map(|(doc_id, squared_weight)| (doc_id, squared_weight.sqrt()))
            .collect()
    }
}

fn raw_document_fields(path: &Path, content: &str) -> HashMap<DocumentField, String> {
    let mut fields = HashMap::new();

    if let Some(file_name) = path.file_name().and_then(|name| name.to_str()) {
        fields.insert(DocumentField::FileName, file_name.to_string());
    }

    fields.insert(
        DocumentField::RelativePath,
        path.to_string_lossy().into_owned(),
    );

    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        fields.insert(DocumentField::Extension, extension.to_string());
    }

    fields.insert(DocumentField::Content, content.to_string());
    fields.insert(
        DocumentField::Identifier,
        extract_identifier_lexemes(content).join(" "),
    );
    fields.insert(DocumentField::Comment, extract_comments(content).join("\n"));
    fields.insert(
        DocumentField::StringLiteral,
        extract_string_literals(content).join("\n"),
    );

    fields
}

fn extract_identifier_lexemes(content: &str) -> Vec<String> {
    let mut identifiers = Vec::new();
    let mut current = String::new();

    for character in content.chars() {
        if character.is_ascii_alphanumeric() || character == '_' || character == '-' {
            current.push(character);
            continue;
        }

        if contains_identifier_signal(&current) {
            identifiers.push(std::mem::take(&mut current));
        } else {
            current.clear();
        }
    }

    if contains_identifier_signal(&current) {
        identifiers.push(current);
    }

    identifiers
}

fn contains_identifier_signal(candidate: &str) -> bool {
    let mut has_lower = false;
    let mut has_upper = false;
    let mut has_digit = false;

    for character in candidate.chars() {
        has_lower |= character.is_ascii_lowercase();
        has_upper |= character.is_ascii_uppercase();
        has_digit |= character.is_ascii_digit();

        if character == '_' || character == '-' {
            return true;
        }
    }

    (has_lower && has_upper) || (has_digit && (has_lower || has_upper))
}

fn extract_comments(content: &str) -> Vec<String> {
    let mut comments = Vec::new();
    let mut in_block_comment = false;
    let mut block_comment = String::new();

    for line in content.lines() {
        let mut rest = line;

        loop {
            if in_block_comment {
                if let Some(end) = rest.find("*/") {
                    block_comment.push_str(&rest[..end]);
                    comments.push(std::mem::take(&mut block_comment));
                    in_block_comment = false;
                    rest = &rest[end + 2..];
                    continue;
                }

                block_comment.push_str(rest);
                block_comment.push('\n');
                break;
            }

            let line_comment = rest.find("//");
            let hash_comment = rest.find('#');
            let block_start = rest.find("/*");
            let comment_start = [line_comment, hash_comment, block_start]
                .into_iter()
                .flatten()
                .min();

            let Some(start) = comment_start else {
                break;
            };

            if Some(start) == block_start {
                rest = &rest[start + 2..];
                if let Some(end) = rest.find("*/") {
                    comments.push(rest[..end].to_string());
                    rest = &rest[end + 2..];
                    continue;
                }

                block_comment.push_str(rest);
                block_comment.push('\n');
                in_block_comment = true;
                break;
            }

            comments.push(rest[start + 1..].to_string());
            break;
        }
    }

    if !block_comment.is_empty() {
        comments.push(block_comment);
    }

    comments
}

fn extract_string_literals(content: &str) -> Vec<String> {
    let mut literals = Vec::new();
    let mut literal = String::new();
    let mut quote = None;
    let mut escaped = false;

    for character in content.chars() {
        if let Some(active_quote) = quote {
            if escaped {
                literal.push(character);
                escaped = false;
                continue;
            }

            if character == '\\' {
                escaped = true;
                continue;
            }

            if character == active_quote {
                literals.push(std::mem::take(&mut literal));
                quote = None;
                continue;
            }

            literal.push(character);
            continue;
        }

        if matches!(character, '"' | '\'' | '`') {
            quote = Some(character);
        }
    }

    literals
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        path::{Path, PathBuf},
    };

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use super::InvertedIndex;
    use crate::{
        config::Config,
        index::{
            DocId, DocumentField, IndexSkipReason, document_registry::DocumentRegistry, term::Term,
        },
        ranking::idf,
        tokenizer::FileType,
    };

    type TestIndex = InvertedIndex<DocumentRegistry>;

    fn term_freqs(pairs: &[(&str, u32)]) -> HashMap<Term, u32> {
        pairs
            .iter()
            .map(|(s, c)| (Term(s.to_string()), *c))
            .collect()
    }

    fn get_term_doc<'a>(
        result: &'a HashMap<Term, HashMap<DocId, super::TermDocument>>,
        term: &str,
        doc_id: DocId,
    ) -> &'a super::TermDocument {
        result
            .get(&Term(term.to_string()))
            .unwrap()
            .get(&doc_id)
            .unwrap()
    }

    #[test]
    fn term_freq_matches_transform_frequency() {
        let transform_fn =
            |_: &str| -> HashMap<Term, u32> { term_freqs(&[("rust", 3), ("system", 1)]) };

        let result = TestIndex::process_document(Path::new("test.rs"), "", &transform_fn);
        let doc_id = *result
            .get(&Term("rust".to_string()))
            .unwrap()
            .keys()
            .next()
            .unwrap();

        assert_eq!(get_term_doc(&result, "rust", doc_id).term_freq, 3);
        assert_eq!(get_term_doc(&result, "system", doc_id).term_freq, 1);
    }

    #[test]
    fn document_length_is_sum_of_all_frequencies() {
        let transform_fn =
            |_: &str| -> HashMap<Term, u32> { term_freqs(&[("rust", 3), ("system", 1)]) };

        let result = TestIndex::process_document(Path::new("test.rs"), "", &transform_fn);
        let doc_id = *result
            .get(&Term("rust".to_string()))
            .unwrap()
            .keys()
            .next()
            .unwrap();

        assert_eq!(get_term_doc(&result, "rust", doc_id).length, 4);
    }

    #[test]
    fn document_length_consistent_across_all_terms() {
        let transform_fn =
            |_: &str| -> HashMap<Term, u32> { term_freqs(&[("foo", 3), ("bar", 1), ("baz", 1)]) };

        let result = TestIndex::process_document(Path::new("test.rs"), "", &transform_fn);
        let doc_id = *result
            .get(&Term("foo".to_string()))
            .unwrap()
            .keys()
            .next()
            .unwrap();

        let lengths: Vec<usize> = result
            .values()
            .filter_map(|docs| docs.get(&doc_id))
            .map(|td| td.length)
            .collect();

        assert!(lengths.iter().all(|&l| l == 5));
    }

    fn build_index(docs: &[(&str, &[(&str, u32)])]) -> InvertedIndex {
        InvertedIndex::from_documents(docs)
    }

    #[test]
    fn num_docs_returns_unique_document_count() {
        let index = build_index(&[
            ("a.rs", &[("foo", 2), ("bar", 1)]),
            ("b.rs", &[("bar", 1), ("baz", 3)]),
        ]);

        assert_eq!(index.num_docs(), 2);
    }

    #[test]
    fn num_docs_single_document_with_many_terms() {
        let index = build_index(&[("a.rs", &[("a", 1), ("b", 1), ("c", 1), ("d", 1)])]);

        assert_eq!(index.num_docs(), 1);
    }

    #[test]
    fn avg_doc_length_counts_each_document_once() {
        let index = build_index(&[
            ("a.rs", &[("foo", 2), ("bar", 1), ("baz", 2)]),
            ("b.rs", &[("bar", 1), ("qux", 2)]),
        ]);

        assert_eq!(index.avg_doc_length(), 4.0);
    }

    #[test]
    fn avg_doc_length_single_document() {
        let index = build_index(&[("a.rs", &[("x", 3), ("y", 7)])]);

        assert_eq!(index.avg_doc_length(), 10.0);
    }

    #[test]
    fn document_metadata_maps_id_back_to_path() {
        let index = build_index(&[("a.rs", &[("x", 3), ("y", 7)])]);

        let doc_id = index.doc_id(Path::new("a.rs")).unwrap();
        let metadata = index.document(doc_id).unwrap();

        assert_eq!(metadata.path, PathBuf::from("a.rs"));
        assert_eq!(metadata.token_length, 10);
    }

    #[test]
    fn document_norm_is_cached_by_document_id() {
        let index = build_index(&[
            ("a.rs", &[("rare", 1), ("common", 1)]),
            ("b.rs", &[("common", 1)]),
        ]);

        let doc_id = index.doc_id(Path::new("a.rs")).unwrap();
        let expected = (idf(2, 1).powi(2) + idf(2, 2).powi(2)).sqrt();

        assert!((index.document_norm(doc_id).unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn avg_doc_length_empty_index_is_zero() {
        let index = build_index(&[]);

        assert_eq!(index.avg_doc_length(), 0.0);
    }

    #[test]
    fn corpus_stats_reports_document_and_term_totals() {
        let index = build_index(&[
            ("a.rs", &[("shared", 2), ("only_a", 1)]),
            ("b.rs", &[("shared", 1), ("only_b", 4)]),
        ]);

        let stats = index.corpus_stats(10);

        assert_eq!(stats.document_count, 2);
        assert_eq!(stats.total_token_count, 8);
        assert_eq!(stats.vocabulary_size, 3);
        assert_eq!(stats.singleton_term_count, 1);
    }

    #[test]
    fn corpus_stats_reports_high_frequency_terms_in_stable_order() {
        let index = build_index(&[
            ("a.rs", &[("shared", 2), ("alpha", 3), ("zeta", 1)]),
            ("b.rs", &[("shared", 2), ("beta", 3)]),
        ]);

        let stats = index.corpus_stats(3);
        let terms: Vec<_> = stats
            .high_frequency_terms
            .iter()
            .map(|summary| {
                (
                    summary.term.as_str(),
                    summary.collection_frequency,
                    summary.document_frequency,
                )
            })
            .collect();

        assert_eq!(
            terms,
            vec![("shared", 4, 2), ("alpha", 3, 1), ("beta", 3, 1)]
        );
    }

    fn write_temp_file(dir: &std::path::Path, name: &str, content: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    fn identity_transform(content: &str) -> HashMap<Term, u32> {
        content
            .split_whitespace()
            .fold(HashMap::new(), |mut acc, word| {
                *acc.entry(Term(word.to_string())).or_insert(0) += 1;
                acc
            })
    }

    fn test_config() -> Config {
        Config {
            n_grams: 1,
            stemmer: Stemmer::create(Algorithm::English),
            stop_words: stop_words::get(stop_words::LANGUAGE::English)
                .par_iter()
                .map(|word| word.to_string())
                .collect::<HashSet<String>>(),
        }
    }

    #[test]
    fn new_with_drop_prefix_indexes_files_and_strips_paths() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "src/main.rs", "hello world");
        write_temp_file(dir.path(), "src/lib.rs", "hello rust");

        let index = InvertedIndex::new(dir.path(), identity_transform, Some(dir.path()));

        assert_eq!(index.num_docs(), 2);

        let hello_docs = index.get_postings(&Term("hello".to_string())).unwrap();
        assert!(
            index
                .doc_id(Path::new("src/main.rs"))
                .is_some_and(|id| hello_docs.contains_key(&id))
        );
        assert!(
            index
                .doc_id(Path::new("src/lib.rs"))
                .is_some_and(|id| hello_docs.contains_key(&id))
        );
    }

    #[test]
    fn new_with_drop_prefix_stores_relative_paths_not_absolute() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "a.txt", "token");

        let index = InvertedIndex::new(dir.path(), identity_transform, Some(dir.path()));

        let docs = index.get_postings(&Term("token".to_string())).unwrap();
        let paths: Vec<&PathBuf> = docs
            .keys()
            .filter_map(|&doc_id| index.document(doc_id).map(|metadata| &metadata.path))
            .collect();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], &PathBuf::from("a.txt"));
    }

    #[test]
    fn fielded_index_tracks_file_type_and_field_lengths() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(
            dir.path(),
            "src/inverted_index.rs",
            "fn parse2Json() { // running parser\nlet value = \"query-id\"; }",
        );

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let doc_id = index.doc_id(Path::new("src/inverted_index.rs")).unwrap();
        let metadata = index.document(doc_id).unwrap();

        assert_eq!(metadata.file_type, FileType::Rust);
        assert!(metadata.field_length(DocumentField::Content) > 0);
        assert!(metadata.field_length(DocumentField::Identifier) > 0);
        assert!(metadata.field_length(DocumentField::Comment) > 0);
        assert!(metadata.field_length(DocumentField::StringLiteral) > 0);
    }

    #[test]
    fn fielded_index_exposes_per_field_term_frequency() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(
            dir.path(),
            "src/inverted_index.rs",
            "fn unrelated() { let value = \"needle\"; }",
        );

        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let doc_id = index.doc_id(Path::new("src/inverted_index.rs")).unwrap();
        let term_doc = index
            .get_postings(&Term("inverted_index".to_string()))
            .unwrap()
            .get(&doc_id)
            .unwrap();

        assert_eq!(term_doc.field_term_freq(DocumentField::FileName), 1);
        assert_eq!(term_doc.field_term_freq(DocumentField::RelativePath), 1);
        assert_eq!(term_doc.field_term_freq(DocumentField::Content), 0);
    }

    #[test]
    fn new_without_drop_prefix_uses_full_paths() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "a.txt", "token");

        let index = InvertedIndex::new(dir.path(), identity_transform, None::<&std::path::Path>);

        let docs = index.get_postings(&Term("token".to_string())).unwrap();
        let doc_id = docs.keys().next().unwrap();
        let path = &index.document(*doc_id).unwrap().path;

        assert!(path.is_absolute());
    }

    #[test]
    fn build_reports_indexed_and_skipped_documents() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "valid.txt", "token");
        std::fs::write(dir.path().join("binary.bin"), [0xff, 0xfe]).unwrap();

        let result = InvertedIndex::build(dir.path(), identity_transform, Some(dir.path()));

        assert_eq!(result.index.num_docs(), 1);
        assert_eq!(result.report.indexed_document_count, 1);
        assert_eq!(result.report.skipped_document_count(), 1);
        assert_eq!(
            result.report.skipped_documents[0].path,
            dir.path().join("binary.bin")
        );
        assert!(matches!(
            result.report.skipped_documents[0].reason,
            IndexSkipReason::Read { .. }
        ));
    }

    #[test]
    fn update_removes_terms_that_no_longer_exist() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "a.txt", "old shared");
        let path = dir.path().join("a.txt");
        let mut index = InvertedIndex::new(dir.path(), identity_transform, None::<&Path>);
        let original_doc_id = index.doc_id(&path).unwrap();

        write_temp_file(dir.path(), "a.txt", "new shared shared");
        index.update(&path, &identity_transform);
        let updated_doc_id = index.doc_id(&path).unwrap();

        assert!(index.get_postings(&Term("old".to_string())).is_none());

        let new_docs = index.get_postings(&Term("new".to_string())).unwrap();
        assert_eq!(updated_doc_id, original_doc_id);
        assert!(new_docs.contains_key(&updated_doc_id));

        let shared_docs = index.get_postings(&Term("shared".to_string())).unwrap();
        let shared = shared_docs.get(&updated_doc_id).unwrap();
        assert_eq!(shared.length, 3);
        assert_eq!(shared.term_freq, 2);
        assert_eq!(index.num_docs(), 1);
        assert_eq!(index.avg_doc_length(), 3.0);
    }

    #[test]
    fn update_recomputes_cached_document_norm_for_stable_doc_id() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "a.txt", "rust rust code");
        let path = dir.path().join("a.txt");
        let mut index = InvertedIndex::new(dir.path(), identity_transform, None::<&Path>);
        let original_doc_id = index.doc_id(&path).unwrap();
        let original_norm = index.document_norm(original_doc_id).unwrap();

        write_temp_file(dir.path(), "a.txt", "rust");
        index.update(&path, &identity_transform);
        let updated_doc_id = index.doc_id(&path).unwrap();
        let expected = idf(1, 1);

        assert_eq!(updated_doc_id, original_doc_id);
        assert!((index.document_norm(updated_doc_id).unwrap() - expected).abs() < 1e-10);
        assert!(original_norm > expected);
    }

    #[test]
    fn update_missing_file_removes_existing_document() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "a.txt", "token");
        let path = dir.path().join("a.txt");
        let mut index = InvertedIndex::new(dir.path(), identity_transform, None::<&Path>);
        let doc_id = index.doc_id(&path).unwrap();

        std::fs::remove_file(&path).unwrap();
        index.update(&path, &identity_transform);

        assert!(index.get_postings(&Term("token".to_string())).is_none());
        assert!(index.document_norm(doc_id).is_none());
        assert_eq!(index.num_docs(), 0);
        assert_eq!(index.avg_doc_length(), 0.0);
    }

    #[test]
    fn remove_document_removes_only_that_document() {
        let mut index = build_index(&[
            ("a.rs", &[("shared", 1), ("only_a", 1)]),
            ("b.rs", &[("shared", 2), ("only_b", 1)]),
        ]);
        let removed_doc_id = index.doc_id(Path::new("a.rs")).unwrap();

        index.remove_document(Path::new("a.rs"));

        assert!(index.get_postings(&Term("only_a".to_string())).is_none());

        let shared_docs = index.get_postings(&Term("shared".to_string())).unwrap();
        assert!(index.doc_id(Path::new("a.rs")).is_none());
        assert!(index.document_norm(removed_doc_id).is_none());
        assert!(
            index
                .doc_id(Path::new("b.rs"))
                .is_some_and(|id| shared_docs.contains_key(&id))
        );
        assert_eq!(index.num_docs(), 1);
        assert_eq!(index.avg_doc_length(), 3.0);
    }
}
