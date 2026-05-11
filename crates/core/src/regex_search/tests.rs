use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use regex::Regex;

use super::{
    CorpusDocument, MmapRegexPostings, RegexCandidatePlan, RegexCorpus, RegexSearchEngine, Trigram,
    TrigramIndex,
    sparse_ngram::{sparse_covering_ngrams, sparse_ngrams},
    trigrams,
};

#[derive(Debug, Clone, Default)]
struct TestCorpus {
    documents: BTreeMap<PathBuf, String>,
}

impl TestCorpus {
    fn new(documents: &[(&str, &str)]) -> Self {
        Self {
            documents: documents
                .iter()
                .map(|(path, content)| (PathBuf::from(path), (*content).to_string()))
                .collect(),
        }
    }
}

impl RegexCorpus for TestCorpus {
    fn documents(&self) -> Vec<CorpusDocument> {
        self.documents
            .iter()
            .map(|(path, content)| CorpusDocument {
                path: path.clone(),
                content: content.clone(),
            })
            .collect()
    }

    fn read_document(&self, path: &Path) -> Option<String> {
        self.documents.get(path).cloned()
    }
}

#[test]
fn search_returns_path_byte_range_line_range_and_matched_text() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("src/lib.rs");
    std::fs::create_dir_all(file.parent().unwrap()).unwrap();
    std::fs::write(&file, "alpha\nlet answer = 42;\nomega\n").unwrap();

    let matches = RegexSearchEngine::new(temp.path())
        .search(r"answer = \d+")
        .unwrap();

    assert_eq!(matches.len(), 1);
    let match_ = &matches[0];
    assert_eq!(match_.path, file);
    assert_eq!(match_.byte_range, 10..21);
    assert_eq!(match_.line_range, 2..=2);
    assert_eq!(match_.matched_text, "answer = 42");
}

#[test]
fn search_returns_multiple_matches_in_file_order() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("lib.rs");
    std::fs::write(&file, "todo one\ntodo two\ntodo three\n").unwrap();

    let matches = RegexSearchEngine::new(temp.path())
        .search(r"todo \w+")
        .unwrap();

    let matched_text = matches
        .iter()
        .map(|match_| match_.matched_text.as_str())
        .collect::<Vec<_>>();
    assert_eq!(matched_text, ["todo one", "todo two", "todo three"]);
}

#[test]
fn search_returns_files_in_deterministic_path_order() {
    let temp = tempfile::tempdir().unwrap();
    write_file(temp.path(), "zeta.rs", "needle z").unwrap();
    write_file(temp.path(), "alpha.rs", "needle a").unwrap();
    write_file(temp.path(), "nested/beta.rs", "needle b").unwrap();

    let matches = RegexSearchEngine::new(temp.path())
        .search(r"needle \w")
        .unwrap();

    let names = matches
        .iter()
        .map(|match_| {
            match_
                .path
                .strip_prefix(temp.path())
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect::<Vec<_>>();
    assert_eq!(names, ["alpha.rs", "nested/beta.rs", "zeta.rs"]);
}

#[test]
fn search_returns_error_for_invalid_patterns() {
    let temp = tempfile::tempdir().unwrap();

    let error = RegexSearchEngine::new(temp.path()).search("(").unwrap_err();

    assert!(error.to_string().contains("invalid regex pattern"));
}

#[test]
fn search_uses_regex_line_boundary_flags_explicitly() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("lib.rs");
    std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();

    let matches = RegexSearchEngine::new(temp.path())
        .search(r"(?m)^beta$")
        .unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].path, file);
    assert_eq!(matches[0].byte_range, 6..10);
    assert_eq!(matches[0].line_range, 2..=2);
    assert_eq!(matches[0].matched_text, "beta");
}

#[test]
fn search_reports_multiline_match_line_ranges() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("lib.rs");
    std::fs::write(&file, "alpha\nbeta\ngamma\n").unwrap();

    let matches = RegexSearchEngine::new(temp.path())
        .search(r"(?s)alpha.*gamma")
        .unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].path, file);
    assert_eq!(matches[0].byte_range, 0..16);
    assert_eq!(matches[0].line_range, 1..=3);
    assert_eq!(matches[0].matched_text, "alpha\nbeta\ngamma");
}

#[test]
fn search_skips_non_utf8_files() {
    let temp = tempfile::tempdir().unwrap();
    std::fs::write(temp.path().join("binary.bin"), [0xff, 0xfe, b'n', b'e']).unwrap();
    std::fs::write(temp.path().join("text.txt"), "needle").unwrap();

    let matches = RegexSearchEngine::new(temp.path())
        .search("needle")
        .unwrap();

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].matched_text, "needle");
    assert_eq!(matches[0].path, temp.path().join("text.txt"));
}

#[test]
fn trigrams_are_overlapping_character_windows() {
    let trigrams = trigrams("abcd");

    let values = trigrams.iter().map(Trigram::as_str).collect::<Vec<_>>();
    assert_eq!(values, ["abc", "bcd"]);
}

#[test]
fn trigrams_return_empty_for_short_strings() {
    assert!(trigrams("").is_empty());
    assert!(trigrams("a").is_empty());
    assert!(trigrams("ab").is_empty());
}

#[test]
fn trigrams_include_punctuation_whitespace_and_preserve_case() {
    let trigrams = trigrams("A b!");

    let values = trigrams.iter().map(Trigram::as_str).collect::<Vec<_>>();
    assert_eq!(values, ["A b", " b!"]);
}

#[test]
fn sparse_ngrams_are_deterministic() {
    let first = sparse_ngram_values("foo_bar_baz");
    let second = sparse_ngram_values("foo_bar_baz");

    assert_eq!(first, second);
}

#[test]
fn sparse_ngrams_are_stable_for_code_like_literals() {
    let ngrams = sparse_ngram_values("MAX_FILE_SIZE");

    assert_eq!(
        ngrams,
        [
            "AX", "AX_", "E_", "E_S", "FI", "FIL", "FILE", "FILE_S", "IL", "ILE", "IZ", "IZE",
            "LE", "LE_", "LE_S", "MA", "MAX", "SI", "SIZ", "X_", "X_F", "X_FI", "ZE", "_F", "_FI",
            "_S", "_SI", "_SIZ"
        ]
    );
}

#[test]
fn sparse_covering_ngrams_use_fewer_query_terms_than_classic_trigrams() {
    let covering = sparse_covering_ngram_values("MAX_FILE_SIZE");

    assert_eq!(covering, ["E_S", "ILE", "MAX", "SIZ", "X_FI", "ZE"]);
    assert!(covering.len() < trigrams("MAX_FILE_SIZE").len());
}

#[test]
fn sparse_ngrams_return_empty_for_single_character_inputs() {
    assert!(sparse_ngrams("").is_empty());
    assert!(sparse_ngrams("a").is_empty());
}

#[test]
fn sparse_ngrams_keep_short_bigram_inputs_selective() {
    let ngrams = sparse_ngram_values("ab");

    assert_eq!(ngrams, ["ab"]);
}

#[test]
fn index_maps_every_indexed_doc_id_back_to_path() {
    let first = PathBuf::from("a.rs");
    let second = PathBuf::from("b.rs");
    let index =
        TrigramIndex::with_corpus(TestCorpus::new(&[("a.rs", "abcdef"), ("b.rs", "uvwxyz")]));
    let first_id = index.doc_id(&first).unwrap();
    let second_id = index.doc_id(&second).unwrap();

    assert_eq!(index.num_docs(), 2);
    assert_eq!(index.document(first_id).unwrap().path, first);
    assert_eq!(index.document(second_id).unwrap().path, second);
}

#[test]
fn postings_record_documents_containing_each_trigram() {
    let first = PathBuf::from("a.rs");
    let second = PathBuf::from("b.rs");
    let index =
        TrigramIndex::with_corpus(TestCorpus::new(&[("a.rs", "abcdef"), ("b.rs", "zabczz")]));
    let postings = index.postings(&Trigram::from("abc")).unwrap();

    assert!(postings.contains(&index.doc_id(&first).unwrap()));
    assert!(postings.contains(&index.doc_id(&second).unwrap()));
}

#[test]
fn literal_candidates_are_supersets_of_true_matches() {
    let matching = PathBuf::from("match.rs");
    let false_positive = PathBuf::from("false_positive.rs");
    let missing = PathBuf::from("missing.rs");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "xxabcdefxx"),
        ("false_positive.rs", "abc bcd cde def"),
        ("missing.rs", "abxde"),
    ]));
    let candidates = index.candidates_for_literal("abcdef");

    assert!(candidates.contains(&index.doc_id(&matching).unwrap()));
    assert!(candidates.contains(&index.doc_id(&false_positive).unwrap()));
    assert!(!candidates.contains(&index.doc_id(&missing).unwrap()));
}

#[test]
fn experimental_masks_reduce_phrase_like_literal_false_positives() {
    let matching = PathBuf::from("match.rs");
    let plain_false_positive = PathBuf::from("plain_false_positive.rs");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "xxabcdefxx"),
        ("plain_false_positive.rs", "abc bcd cde def"),
        ("missing.rs", "abxde"),
    ]));

    let plain_candidates = index.candidates_for_literal("abcdef");
    let masked_candidates = index.experimental_masked_candidates_for_literal("abcdef");

    assert_eq!(plain_candidates.len(), 2);
    assert!(plain_candidates.contains(&index.doc_id(&plain_false_positive).unwrap()));
    assert_eq!(masked_candidates.len(), 1);
    assert!(masked_candidates.contains(&index.doc_id(&matching).unwrap()));
}

#[test]
fn experimental_regex_masks_reduce_plain_literal_pattern_false_positives() {
    let matching = PathBuf::from("match.rs");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "xxabcdefxx"),
        ("plain_false_positive.rs", "abc bcd cde def"),
    ]));

    let plain_candidates = index.candidates_for_regex("abcdef");
    let masked_candidates = index.experimental_masked_candidates_for_regex("abcdef");

    assert_eq!(plain_candidates.len(), 2);
    assert_eq!(masked_candidates.len(), 1);
    assert!(masked_candidates.contains(&index.doc_id(&matching).unwrap()));
}

#[test]
fn experimental_regex_masks_fall_back_for_unsupported_patterns() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "foo123bar"),
        ("other.rs", "fooabcbar"),
    ]));

    assert_eq!(
        index.experimental_masked_candidates_for_regex(r"foo\d+bar"),
        index.candidates_for_regex(r"foo\d+bar")
    );
}

#[test]
fn experimental_masks_keep_all_verified_literal_matches() {
    let documents = [
        ("start.rs", "abcdef at start"),
        ("middle.rs", "xx abcdef yy"),
        ("repeated.rs", "abcabc abcdef"),
        ("false_positive.rs", "abc bcd cde def"),
    ];
    let index = TrigramIndex::with_corpus(TestCorpus::new(&documents));
    let masked_candidates = index.experimental_masked_candidates_for_literal("abcdef");

    for (path, content) in documents {
        if content.contains("abcdef") {
            let doc_id = index.doc_id(Path::new(path)).unwrap();
            assert!(masked_candidates.contains(&doc_id));
        }
    }
}

#[test]
fn experimental_masks_fall_back_for_short_literals() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[("a.rs", "ab"), ("b.rs", "zz")]));

    assert_eq!(
        index.experimental_masked_candidates_for_literal("ab"),
        index.candidates_for_literal("ab")
    );
}

#[test]
fn experimental_masks_keep_repeated_overlapping_trigram_matches() {
    let matching = PathBuf::from("match.rs");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "aaaaaa"),
        ("near.rs", "aaa aaa"),
    ]));

    let masked_candidates = index.experimental_masked_candidates_for_literal("aaaaa");

    assert!(masked_candidates.contains(&index.doc_id(&matching).unwrap()));
}

#[test]
fn experimental_sparse_ngram_candidates_are_separate_from_default_candidates() {
    let matching = PathBuf::from("match.rs");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "let value = MAX_FILE_SIZE;"),
        (
            "classic_false_positive.rs",
            "MAX AX_ X_F _FI FIL ILE LE_ E_S _SI SIZ IZE",
        ),
        ("missing.rs", "let value = MIN_FILE_SIZE;"),
    ]));

    let classic_candidates = index.candidates_for_literal("MAX_FILE_SIZE");
    let sparse_candidates = index.experimental_sparse_ngram_candidates_for_literal("MAX_FILE_SIZE");

    assert_eq!(classic_candidates.len(), 2);
    assert_eq!(sparse_candidates.len(), 1);
    assert!(sparse_candidates.contains(&index.doc_id(&matching).unwrap()));
}

#[test]
fn experimental_sparse_ngram_comparison_reports_query_and_index_costs() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "let value = MAX_FILE_SIZE;"),
        (
            "classic_false_positive.rs",
            "MAX AX_ X_F _FI FIL ILE LE_ E_S _SI SIZ IZE",
        ),
        ("missing.rs", "let value = MIN_FILE_SIZE;"),
    ]));

    let comparison = index.experimental_sparse_ngram_comparison_for_literal("MAX_FILE_SIZE");

    assert_eq!(comparison.classic_candidate_count, 2);
    assert_eq!(comparison.sparse_candidate_count, 1);
    assert_eq!(comparison.classic_posting_lookups, 11);
    assert_eq!(comparison.sparse_posting_lookups, 6);
    assert_eq!(comparison.classic_update_token_count, 11);
    assert_eq!(comparison.sparse_update_token_count, 28);
    assert!(comparison.sparse_index_key_count > comparison.classic_index_key_count);
}

#[test]
fn short_literal_candidates_include_all_indexed_documents() {
    let first = PathBuf::from("a.rs");
    let second = PathBuf::from("b.rs");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[("a.rs", "a"), ("b.rs", "b")]));
    let candidates = index.candidates_for_literal("ab");

    assert_eq!(candidates.len(), 2);
    assert!(candidates.contains(&index.doc_id(&first).unwrap()));
    assert!(candidates.contains(&index.doc_id(&second).unwrap()));
}

#[test]
fn literal_search_verifies_candidates_and_reports_candidate_count() {
    let matching = PathBuf::from("match.rs");
    let result = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "xxabcdefxx"),
        ("false_positive.rs", "abc bcd cde def"),
        ("missing.rs", "unrelated"),
    ]))
    .search_literal("abcdef");

    assert_eq!(result.candidate_count, 2);
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].path, matching);
    assert_eq!(result.matches[0].byte_range, 2..8);
    assert_eq!(result.matches[0].line_range, 1..=1);
    assert_eq!(result.matches[0].matched_text, "abcdef");
}

#[test]
fn literal_search_matches_full_scan_results() {
    let result = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("a.rs", "needle one\nneedle two\n"),
        ("nested/b.rs", "no match here"),
        ("z.rs", "needle three"),
    ]))
    .search_literal("needle");

    let matched_paths = result
        .matches
        .iter()
        .map(|match_| match_.path.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert_eq!(matched_paths, ["a.rs", "a.rs", "z.rs"]);
    assert_eq!(result.matches[0].byte_range, 0..6);
    assert_eq!(result.matches[1].byte_range, 11..17);
    assert_eq!(result.matches[2].byte_range, 0..6);
}

#[test]
fn short_literal_search_falls_back_to_all_documents() {
    let first = PathBuf::from("a.rs");
    let result = TrigramIndex::with_corpus(TestCorpus::new(&[("a.rs", "aba"), ("b.rs", "zzz")]))
        .search_literal("ab");

    assert_eq!(result.candidate_count, 2);
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].path, first);
    assert_eq!(result.matches[0].byte_range, 0..2);
}

#[test]
fn selective_literal_search_uses_fewer_candidates_than_corpus() {
    let matching = PathBuf::from("match.rs");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "rare_literal"),
        ("a.rs", "common text"),
        ("b.rs", "more text"),
    ]));
    let result = index.search_literal("rare_literal");

    assert_eq!(index.num_docs(), 3);
    assert_eq!(result.candidate_count, 1);
    assert_eq!(result.matches[0].path, matching);
}

#[test]
fn regex_literal_pattern_decomposes_to_required_trigrams() {
    let trigrams = plan_trigrams(&RegexCandidatePlan::for_pattern("abcdef"));

    assert_eq!(trigrams, ["abc", "bcd", "cde", "def"]);
}

#[test]
fn regex_concatenated_literals_preserve_required_trigrams() {
    let trigrams = plan_trigrams(&RegexCandidatePlan::for_pattern(r"abc\.def"));

    assert_eq!(trigrams, [".de", "abc", "bc.", "c.d", "def"]);
}

#[test]
fn regex_alternation_produces_or_candidate_plan() {
    let plan = RegexCandidatePlan::for_pattern("abcdef|uvwxyz");

    let RegexCandidatePlan::Or(branches) = plan else {
        panic!("expected alternation to produce an OR plan");
    };

    assert_eq!(branches.len(), 2);
    assert_eq!(plan_trigrams(&branches[0]), ["abc", "bcd", "cde", "def"]);
    assert_eq!(plan_trigrams(&branches[1]), ["uvw", "vwx", "wxy", "xyz"]);
}

#[test]
fn regex_wildcards_drop_only_unsafe_local_fragment() {
    let trigrams = plan_trigrams(&RegexCandidatePlan::for_pattern("abcdef.*uvwxyz"));

    assert_eq!(
        trigrams,
        ["abc", "bcd", "cde", "def", "uvw", "vwx", "wxy", "xyz"]
    );
}

#[test]
fn regex_case_insensitive_pattern_falls_back_to_all_candidates() {
    let plan = RegexCandidatePlan::for_pattern("(?i)abcdef");

    assert_eq!(plan, RegexCandidatePlan::All);
}

#[test]
fn regex_character_classes_are_conservative() {
    let singleton = plan_trigrams(&RegexCandidatePlan::for_pattern("ab[c]def"));
    let broad = plan_trigrams(&RegexCandidatePlan::for_pattern("abc[xyz]def"));

    assert_eq!(singleton, ["abc", "bcd", "cde", "def"]);
    assert_eq!(broad, ["abc", "def"]);
}

#[test]
fn regex_candidate_generation_is_superset_of_verified_matches() {
    let documents = [
        ("match_abc.rs", "xxabcdefyy"),
        ("match_uv.rs", "prefix uvwxyzz suffix"),
        ("false_positive.rs", "abc bcd cde def"),
        ("missing.rs", "nothing relevant"),
    ];
    let index = TrigramIndex::with_corpus(TestCorpus::new(&documents));
    let pattern = r"abcdef|uvwxyz";
    let regex = Regex::new(pattern).unwrap();
    let candidates = index.candidates_for_regex(pattern);

    for (path, content) in documents {
        let path = Path::new(path);
        let doc_id = index.doc_id(path).unwrap();
        if regex.is_match(content) {
            assert!(candidates.contains(&doc_id));
        }
    }
}

#[test]
fn regex_planner_selects_rare_trigram_before_common_trigram() {
    let selected = Trigram::from("xyz");
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "abc---xyz"),
        ("common_a.rs", "abc---aaa"),
        ("common_b.rs", "abc---bbb"),
    ]));
    let plan = RegexCandidatePlan::And(vec![Trigram::from("abc"), selected.clone()]);

    let selection = index.planned_candidates_for_regex_plan(&plan);

    assert_eq!(selection.diagnostics.selected_trigrams, [selected]);
    assert_eq!(selection.diagnostics.candidate_count, 1);
    assert!(
        selection
            .candidates
            .contains(&index.doc_id(Path::new("match.rs")).unwrap())
    );
}

#[test]
fn regex_planner_missing_trigram_yields_no_candidates() {
    let index =
        TrigramIndex::with_corpus(TestCorpus::new(&[("a.rs", "abcdef"), ("b.rs", "uvwxyz")]));
    let plan = RegexCandidatePlan::And(vec![Trigram::from("zzz")]);

    let selection = index.planned_candidates_for_regex_plan(&plan);

    assert!(selection.candidates.is_empty());
    assert_eq!(selection.diagnostics.candidate_count, 0);
    assert!(!selection.diagnostics.fell_back_to_full_scan);
}

#[test]
fn regex_planner_common_pattern_falls_back_to_full_scan() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("a.rs", "abc one"),
        ("b.rs", "abc two"),
        ("c.rs", "abc three"),
    ]));
    let plan = RegexCandidatePlan::And(vec![Trigram::from("abc")]);

    let selection = index.planned_candidates_for_regex_plan(&plan);

    assert_eq!(selection.diagnostics.full_corpus_count, 3);
    assert_eq!(selection.diagnostics.candidate_count, 3);
    assert_eq!(selection.diagnostics.selected_trigram_count, 0);
    assert!(selection.diagnostics.fell_back_to_full_scan);
}

#[test]
fn regex_planner_plans_alternation_branches_independently() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("first.rs", "abc---xyz"),
        ("second.rs", "def---uvw"),
        ("common_a.rs", "abc---def"),
        ("common_b.rs", "abc---def"),
    ]));
    let plan = RegexCandidatePlan::Or(vec![
        RegexCandidatePlan::And(vec![Trigram::from("abc"), Trigram::from("xyz")]),
        RegexCandidatePlan::And(vec![Trigram::from("def"), Trigram::from("uvw")]),
    ]);

    let selection = index.planned_candidates_for_regex_plan(&plan);
    let selected = selection
        .diagnostics
        .selected_trigrams
        .iter()
        .map(Trigram::as_str)
        .collect::<Vec<_>>();

    assert_eq!(selected, ["xyz", "uvw"]);
    assert_eq!(selection.diagnostics.candidate_count, 2);
    assert!(
        selection
            .candidates
            .contains(&index.doc_id(Path::new("first.rs")).unwrap())
    );
    assert!(
        selection
            .candidates
            .contains(&index.doc_id(Path::new("second.rs")).unwrap())
    );
}

#[test]
fn regex_planner_selective_pattern_uses_fewer_candidates_than_corpus() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "rare_literal"),
        ("a.rs", "common text"),
        ("b.rs", "more text"),
    ]));

    let selection = index.planned_candidates_for_regex("rare_literal");

    assert_eq!(selection.diagnostics.full_corpus_count, 3);
    assert_eq!(selection.diagnostics.candidate_count, 1);
    assert!(selection.diagnostics.candidate_count < index.num_docs());
}

#[test]
fn refreshing_document_adds_new_file_without_full_reindex() {
    let temp = tempfile::tempdir().unwrap();
    let mut index = TrigramIndex::new(temp.path());
    let file = temp.path().join("new.rs");
    std::fs::write(&file, "fresh_needle").unwrap();

    index.refresh_document(&file);
    let result = index.search_literal("fresh_needle");

    assert_eq!(index.num_docs(), 1);
    assert_eq!(result.candidate_count, 1);
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].path, file);
}

#[test]
fn refreshing_document_removes_stale_postings_for_modified_file() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("lib.rs");
    std::fs::write(&file, "old_needle").unwrap();
    let mut index = TrigramIndex::new(temp.path());
    std::fs::write(&file, "new_needle").unwrap();

    index.refresh_document(&file);

    assert!(index.search_literal("old_needle").matches.is_empty());
    assert_eq!(index.search_literal("new_needle").matches.len(), 1);
}

#[test]
fn removing_document_prevents_deleted_file_from_leaking_candidates() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("lib.rs");
    std::fs::write(&file, "deleted_needle").unwrap();
    let mut index = TrigramIndex::new(temp.path());
    std::fs::remove_file(&file).unwrap();

    let removed = index.remove_document(&file);
    let result = index.search_literal("deleted_needle");

    assert!(removed);
    assert_eq!(index.num_docs(), 0);
    assert_eq!(result.candidate_count, 0);
    assert!(result.matches.is_empty());
}

#[test]
fn mmap_postings_round_trip_lookup_matches_in_memory_postings() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("first.rs", "abcdef"),
        ("second.rs", "zabczz"),
        ("third.rs", "uvwxyz"),
    ]));
    let temp = tempfile::tempdir().unwrap();

    index.write_mmap_postings(temp.path()).unwrap();
    let postings = MmapRegexPostings::open(temp.path()).unwrap();
    let mmap_doc_ids = postings.postings(&Trigram::from("abc")).unwrap().unwrap();

    assert_eq!(
        mmap_doc_ids,
        index.postings(&Trigram::from("abc")).unwrap().clone()
    );
    assert_eq!(postings.lookup_entry_count().unwrap(), 11);
    assert!(postings.mapped_bytes() > 0);
}

#[test]
fn mmap_postings_drive_same_candidate_plan_as_memory_postings() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "xxabcdefyy"),
        ("false_positive.rs", "abc bcd cde def"),
        ("missing.rs", "abxde"),
    ]));
    let temp = tempfile::tempdir().unwrap();
    let all_doc_ids = doc_ids_for_paths(&index, &["false_positive.rs", "match.rs", "missing.rs"]);
    let plan = RegexCandidatePlan::for_pattern("abcdef");
    let memory_candidates = index.candidates_for_regex_plan(&plan);

    index.write_mmap_postings(temp.path()).unwrap();
    let postings = MmapRegexPostings::open(temp.path()).unwrap();
    let mmap_candidates = postings
        .planned_candidates_for_regex_plan(&plan, &all_doc_ids)
        .unwrap()
        .candidates;

    assert_eq!(mmap_candidates, memory_candidates);
}

#[test]
fn mmap_postings_report_missing_trigram_without_full_deserialization() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[("a.rs", "abcdef")]));
    let temp = tempfile::tempdir().unwrap();

    index.write_mmap_postings(temp.path()).unwrap();
    let postings = MmapRegexPostings::open(temp.path()).unwrap();

    assert!(postings.postings(&Trigram::from("zzz")).unwrap().is_none());
}

#[test]
fn unsupported_regex_constructs_fall_back_to_broader_candidates() {
    let index = TrigramIndex::with_corpus(TestCorpus::new(&[
        ("match.rs", "foo123bar"),
        ("other.rs", "unrelated"),
    ]));
    let candidates = index.candidates_for_regex(r"foo\d+bar");

    assert!(candidates.contains(&index.doc_id(Path::new("match.rs")).unwrap()));
}

fn doc_ids_for_paths(index: &TrigramIndex<TestCorpus>, paths: &[&str]) -> Vec<crate::index::DocId> {
    let mut doc_ids = paths
        .iter()
        .map(|path| index.doc_id(Path::new(path)).unwrap())
        .collect::<Vec<_>>();
    doc_ids.sort();
    doc_ids
}

fn write_file(root: &Path, path: &str, content: &str) -> std::io::Result<()> {
    let path = root.join(path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, content)
}

fn sparse_ngram_values(content: &str) -> Vec<String> {
    sparse_ngrams(content)
        .iter()
        .map(|ngram| ngram.0.clone())
        .collect()
}

fn sparse_covering_ngram_values(content: &str) -> Vec<String> {
    sparse_covering_ngrams(content)
        .iter()
        .map(|ngram| ngram.0.clone())
        .collect()
}

fn plan_trigrams(plan: &RegexCandidatePlan) -> Vec<String> {
    let RegexCandidatePlan::And(trigrams) = plan else {
        panic!("expected an AND plan");
    };

    trigrams
        .iter()
        .map(|trigram| trigram.as_str().to_string())
        .collect()
}
