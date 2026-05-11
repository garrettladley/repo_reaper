use std::collections::{HashMap, HashSet};

use crate::{
    index::{DocId, DocumentField, PostingList, RankedIndexReader, Term, TermDocument},
    query::AnalyzedQuery,
    ranking::Score,
};

pub trait FeedbackTermSource: RankedIndexReader {
    fn feedback_terms(&self) -> Vec<&Term>;
}

impl FeedbackTermSource for crate::index::InvertedIndex {
    fn feedback_terms(&self) -> Vec<&Term> {
        self.postings_iter().map(|(term, _)| term).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeedbackTerm {
    pub term: Term,
    pub weight: f64,
    pub source_documents: Vec<DocId>,
    pub source_fields: Vec<DocumentField>,
}

pub fn expand_query_with_feedback<I>(
    index: &I,
    query: &AnalyzedQuery,
    seed_results: &[Score],
    max_terms: usize,
) -> AnalyzedQuery
where
    I: FeedbackTermSource + Sync,
{
    let mut feedback_terms = collect_feedback_terms(index, query, seed_results, max_terms);
    feedback_terms.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.term.0.cmp(&right.term.0))
    });
    feedback_terms.truncate(max_terms);

    let mut expanded = query.clone();
    expanded.add_feedback_terms(
        feedback_terms
            .into_iter()
            .map(|term| (term.term, term.weight))
            .collect(),
    );
    expanded
}

pub fn collect_feedback_terms<I>(
    index: &I,
    query: &AnalyzedQuery,
    seed_results: &[Score],
    max_terms: usize,
) -> Vec<FeedbackTerm>
where
    I: FeedbackTermSource + Sync,
{
    let original_terms = query
        .terms()
        .map(|(term, _)| term.0.clone())
        .collect::<HashSet<_>>();
    let seed_doc_ids = seed_results
        .iter()
        .filter_map(|score| index.doc_id(&score.doc_path))
        .collect::<Vec<_>>();
    let mut candidates: HashMap<Term, FeedbackAccumulator> = HashMap::new();

    for term in index.feedback_terms() {
        if original_terms.contains(&term.0) {
            continue;
        }
        let Some(documents) = index.postings(term) else {
            continue;
        };

        for doc_id in &seed_doc_ids {
            let Some(term_doc) = documents.get(*doc_id) else {
                continue;
            };
            let field_weight = high_signal_field_weight(term_doc);
            if field_weight == 0.0 {
                continue;
            }

            let accumulator = candidates.entry(term.clone()).or_default();
            accumulator.weight += field_weight * term_doc.term_freq as f64;
            accumulator.source_documents.insert(*doc_id);
            for field in DocumentField::ALL {
                if term_doc.field_term_freq(field) > 0 && high_signal_field(field) > 0.0 {
                    accumulator.source_fields.insert(field);
                }
            }
        }
    }

    let mut terms = candidates
        .into_iter()
        .map(|(term, accumulator)| {
            let mut source_documents = accumulator.source_documents.into_iter().collect::<Vec<_>>();
            source_documents.sort_by_key(|doc_id| doc_id.as_u32());
            let mut source_fields = accumulator.source_fields.into_iter().collect::<Vec<_>>();
            source_fields.sort_unstable();

            FeedbackTerm {
                term,
                weight: (accumulator.weight / seed_doc_ids.len().max(1) as f64).min(0.45),
                source_documents,
                source_fields,
            }
        })
        .collect::<Vec<_>>();

    terms.sort_by(|left, right| {
        right
            .weight
            .total_cmp(&left.weight)
            .then_with(|| left.term.0.cmp(&right.term.0))
    });
    terms.truncate(max_terms);
    terms
}

#[derive(Default)]
struct FeedbackAccumulator {
    weight: f64,
    source_documents: HashSet<DocId>,
    source_fields: HashSet<DocumentField>,
}

fn high_signal_field_weight(term_doc: &TermDocument) -> f64 {
    DocumentField::ALL
        .into_iter()
        .map(|field| high_signal_field(field) * term_doc.field_term_freq(field) as f64)
        .sum()
}

fn high_signal_field(field: DocumentField) -> f64 {
    match field {
        DocumentField::FileName => 0.28,
        DocumentField::RelativePath => 0.20,
        DocumentField::Identifier => 0.22,
        DocumentField::Symbol => 0.24,
        DocumentField::Import => 0.14,
        DocumentField::StringLiteral => 0.12,
        DocumentField::Extension
        | DocumentField::Content
        | DocumentField::Comment
        | DocumentField::Frontmatter => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        path::{Path, PathBuf},
    };

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
    use rust_stemmers::{Algorithm, Stemmer};

    use super::{collect_feedback_terms, expand_query_with_feedback};
    use crate::{
        config::Config,
        index::InvertedIndex,
        query::{AnalyzedQuery, QueryTermProvenance},
        ranking::Score,
    };

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

    fn write_temp_file(dir: &Path, name: &str, content: &str) {
        let path = dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
    }

    #[test]
    fn feedback_terms_use_high_signal_fields() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "src/auth_repository.rs", "fn unrelated() {}");
        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let seed_results = vec![Score {
            doc_path: PathBuf::from("src/auth_repository.rs"),
            score: 1.0,
        }];
        let query = AnalyzedQuery::new_code_search("auth", &test_config());

        let terms = collect_feedback_terms(&index, &query, &seed_results, 4);

        assert!(terms.iter().any(|term| term.term.0 == "repository"));
    }

    #[test]
    fn expanded_query_marks_feedback_provenance() {
        let dir = tempfile::tempdir().unwrap();
        write_temp_file(dir.path(), "src/auth_repository.rs", "fn unrelated() {}");
        let index = InvertedIndex::new_fielded(dir.path(), &test_config(), Some(dir.path()));
        let seed_results = vec![Score {
            doc_path: PathBuf::from("src/auth_repository.rs"),
            score: 1.0,
        }];
        let query = AnalyzedQuery::new_code_search("auth", &test_config());

        let expanded = expand_query_with_feedback(&index, &query, &seed_results, 4);

        assert!(
            expanded
                .terms()
                .any(|(term, query_term)| term.0 == "repository"
                    && query_term.provenance == QueryTermProvenance::Feedback)
        );
    }
}
