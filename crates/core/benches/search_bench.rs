use std::{collections::HashSet, fs, path::Path};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use repo_reaper_core::{
    config::Config,
    index::InvertedIndex,
    query::AnalyzedQuery,
    ranking::{
        BM25FHyperParams, BM25HyperParams, ProximityConfig, QueryLikelihoodParams, RankingAlgo,
    },
    regex_search::{RegexSearchEngine, TrigramIndex},
    tokenizer::n_gram_transform,
};
use rust_stemmers::{Algorithm, Stemmer};
use tempfile::TempDir;

const CORPUS_SEED: u64 = 0x5eed_cafe;
const DOC_COUNT: usize = 256;
const TOKENS_PER_DOC: usize = 240;
const REGEX_LITERAL: &str = "rare_literal_7";

fn bench_config() -> Config {
    Config {
        n_grams: 1,
        stemmer: Stemmer::create(Algorithm::English),
        stop_words: stop_words::get(stop_words::LANGUAGE::English)
            .iter()
            .map(|word| word.to_string())
            .collect::<HashSet<_>>(),
    }
}

fn bm25() -> RankingAlgo {
    RankingAlgo::BM25(BM25HyperParams { k1: 1.2, b: 0.75 })
}

fn bm25f() -> RankingAlgo {
    RankingAlgo::BM25F(BM25FHyperParams::code_search_defaults())
}

fn bm25_proximity() -> RankingAlgo {
    RankingAlgo::BM25Proximity(
        BM25HyperParams { k1: 1.2, b: 0.75 },
        ProximityConfig::default(),
    )
}

fn query_likelihood() -> RankingAlgo {
    RankingAlgo::QueryLikelihood(QueryLikelihoodParams::dirichlet_defaults())
}

fn deterministic_corpus(doc_count: usize, tokens_per_doc: usize, seed: u64) -> Vec<String> {
    let mut rng = Lcg::new(seed);
    let vocabulary = [
        "repository",
        "index",
        "token",
        "parser",
        "query",
        "ranking",
        "update",
        "delete",
        "watcher",
        "document",
        "metadata",
        "registry",
        "posting",
        "score",
        "metric",
        "precision",
        "recall",
        "cosine",
        "bm25",
        "evaluation",
        "config",
        "error",
        "path",
        "rust",
        "module",
        "search",
        "freshness",
        "latency",
        "corpus",
        "snippet",
        "evidence",
        "identifier",
    ];

    (0..doc_count)
        .map(|doc_id| {
            let mut tokens = Vec::with_capacity(tokens_per_doc + 3);
            tokens.push("repository".to_string());
            tokens.push(format!("module_{doc_id}"));

            for _ in 0..tokens_per_doc {
                let word = vocabulary[rng.next_usize(vocabulary.len())];
                tokens.push(word.to_string());
            }

            tokens.push(format!("identifier_{}", doc_id % 17));
            tokens.push(format!("fn module_{doc_id}_handler() {{"));
            tokens.push(format!("let marker = \"rare_literal_{}\";", doc_id % 17));
            tokens.push("repository_index.lookup(marker);".to_string());
            tokens.push("}".to_string());
            tokens.join(" ")
        })
        .collect()
}

fn write_corpus(root: &Path, docs: &[String]) {
    fs::create_dir_all(root.join("src")).expect("create synthetic corpus directory");

    for (idx, content) in docs.iter().enumerate() {
        fs::write(root.join("src").join(format!("module_{idx}.rs")), content)
            .expect("write synthetic corpus document");
    }
}

fn corpus_fixture() -> TempDir {
    let dir = tempfile::tempdir().expect("create synthetic corpus tempdir");
    let docs = deterministic_corpus(DOC_COUNT, TOKENS_PER_DOC, CORPUS_SEED);
    write_corpus(dir.path(), &docs);
    dir
}

fn build_index(root: &Path, config: &Config) -> InvertedIndex {
    InvertedIndex::new(
        root,
        |content: &str| n_gram_transform(content, config),
        Some(root),
    )
}

fn build_fielded_index(root: &Path, config: &Config) -> InvertedIndex {
    InvertedIndex::new_fielded(root, config, Some(root))
}

fn bench_tokenization(c: &mut Criterion) {
    let config = bench_config();
    let docs = deterministic_corpus(1, TOKENS_PER_DOC * 8, CORPUS_SEED);
    let content = docs.first().expect("synthetic corpus has one document");

    c.bench_function("tokenization/ngram_unigram", |b| {
        b.iter(|| n_gram_transform(black_box(content), black_box(&config)));
    });
}

fn bench_index_build(c: &mut Criterion) {
    let config = bench_config();
    let corpus = corpus_fixture();

    c.bench_with_input(
        BenchmarkId::new("index_build", DOC_COUNT),
        &corpus,
        |b, corpus| {
            b.iter(|| build_index(black_box(corpus.path()), black_box(&config)));
        },
    );
}

fn bench_ranked_search(c: &mut Criterion) {
    let config = bench_config();
    let corpus = corpus_fixture();
    let index = build_index(corpus.path(), &config);
    let fielded_index = build_fielded_index(corpus.path(), &config);
    let query = AnalyzedQuery::new("repository token parser update", &config);
    let fielded_query = AnalyzedQuery::new_code_search("repository token parser update", &config);
    let bm25 = bm25();
    let bm25f = bm25f();
    let bm25_proximity = bm25_proximity();
    let query_likelihood = query_likelihood();

    let mut group = c.benchmark_group("ranked_search");
    group.bench_function("bm25", |b| {
        b.iter(|| bm25.rank(black_box(&index), black_box(&query), black_box(10)));
    });
    group.bench_function("bm25f", |b| {
        b.iter(|| bm25f.rank(black_box(&fielded_index), black_box(&query), black_box(10)));
    });
    group.bench_function("bm25_proximity", |b| {
        b.iter(|| {
            bm25_proximity.rank(
                black_box(&fielded_index),
                black_box(&fielded_query),
                black_box(10),
            )
        });
    });
    group.bench_function("query_likelihood", |b| {
        b.iter(|| query_likelihood.rank(black_box(&index), black_box(&query), black_box(10)));
    });
    group.bench_function("cosine", |b| {
        b.iter(|| {
            RankingAlgo::CosineSimilarity.rank(black_box(&index), black_box(&query), black_box(10))
        });
    });
    group.bench_function("tfidf", |b| {
        b.iter(|| RankingAlgo::TFIDF.rank(black_box(&index), black_box(&query), black_box(10)));
    });
    group.finish();
}

fn bench_search_comparison(c: &mut Criterion) {
    let config = bench_config();
    let corpus = corpus_fixture();
    let ranked_index = build_index(corpus.path(), &config);
    let fielded_index = build_fielded_index(corpus.path(), &config);
    let regex_index = TrigramIndex::new(corpus.path());
    let regex_engine = RegexSearchEngine::new(corpus.path());
    let ranked_query = AnalyzedQuery::new("repository token parser update", &config);
    let fielded_query = AnalyzedQuery::new_code_search("repository token parser update", &config);
    let bm25 = bm25();
    let bm25f = bm25f();
    let bm25_proximity = bm25_proximity();
    let query_likelihood = query_likelihood();

    let mut group = c.benchmark_group("search_comparison");
    group.bench_function("traditional_ir/bm25", |b| {
        b.iter(|| {
            bm25.rank(
                black_box(&ranked_index),
                black_box(&ranked_query),
                black_box(10),
            )
        });
    });
    group.bench_function("traditional_ir/bm25f", |b| {
        b.iter(|| {
            bm25f.rank(
                black_box(&fielded_index),
                black_box(&ranked_query),
                black_box(10),
            )
        });
    });
    group.bench_function("traditional_ir/bm25_proximity", |b| {
        b.iter(|| {
            bm25_proximity.rank(
                black_box(&fielded_index),
                black_box(&fielded_query),
                black_box(10),
            )
        });
    });
    group.bench_function("traditional_ir/query_likelihood", |b| {
        b.iter(|| {
            query_likelihood.rank(
                black_box(&ranked_index),
                black_box(&ranked_query),
                black_box(10),
            )
        });
    });
    group.bench_function("traditional_ir/cosine", |b| {
        b.iter(|| {
            RankingAlgo::CosineSimilarity.rank(
                black_box(&ranked_index),
                black_box(&ranked_query),
                black_box(10),
            )
        });
    });
    group.bench_function("traditional_ir/tfidf", |b| {
        b.iter(|| {
            RankingAlgo::TFIDF.rank(
                black_box(&ranked_index),
                black_box(&ranked_query),
                black_box(10),
            )
        });
    });
    group.bench_function("regex/full_scan_verifier", |b| {
        b.iter(|| {
            regex_engine
                .search(black_box(REGEX_LITERAL))
                .expect("regex benchmark pattern should be valid")
        });
    });
    group.bench_function("regex/classic_trigram_verified_literal", |b| {
        b.iter(|| regex_index.search_literal(black_box(REGEX_LITERAL)));
    });
    group.bench_function("regex/classic_trigram_candidates", |b| {
        b.iter(|| regex_index.candidates_for_literal(black_box(REGEX_LITERAL)));
    });
    group.bench_function("regex/masked_trigram_candidates", |b| {
        b.iter(|| regex_index.experimental_masked_candidates_for_literal(black_box(REGEX_LITERAL)));
    });
    group.bench_function("regex/sparse_ngram_candidates", |b| {
        b.iter(|| {
            regex_index.experimental_sparse_ngram_candidates_for_literal(black_box(REGEX_LITERAL))
        });
    });
    group.finish();
}

fn bench_index_update(c: &mut Criterion) {
    let config = bench_config();

    c.bench_function("index_update/single_document", |b| {
        b.iter_batched(
            || {
                let corpus = corpus_fixture();
                let index = build_index(corpus.path(), &config);
                (corpus, index)
            },
            |(corpus, mut index)| {
                let path = corpus.path().join("src/module_0.rs");
                fs::write(
                    &path,
                    "repository token parser update update update deterministic replacement",
                )
                .expect("rewrite synthetic corpus document");

                index.update(&path, &|content: &str| n_gram_transform(content, &config));
                black_box(index);
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_usize(&mut self, upper_bound: usize) -> usize {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        (self.state % upper_bound as u64) as usize
    }
}

criterion_group!(
    benches,
    bench_tokenization,
    bench_index_build,
    bench_ranked_search,
    bench_search_comparison,
    bench_index_update
);
criterion_main!(benches);
