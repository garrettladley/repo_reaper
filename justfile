# list available recipes
default:
    @just --list

# run the same quality gates as the pull request CI workflow
[group("ci")]
ci: fmt-check lint test deny

# alias for the full local quality gate
[group("ci")]
check: ci

# format with nightly (required for rustfmt.toml options)
[group("dev")]
fmt:
    cargo +nightly fmt --all

# check formatting the same way CI does
[group("ci")]
fmt-check:
    cargo +nightly fmt --all -- --check

# run clippy with all workspace lints as errors
[group("ci")]
lint:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

# run all workspace tests, accepts extra args: just test -- --nocapture
[group("ci")]
test *ARGS:
    cargo test --workspace {{ARGS}}

# run core crate tests
[group("dev")]
test-core *ARGS:
    cargo test -p repo-reaper-core {{ARGS}}

# run cli crate tests
[group("dev")]
test-cli *ARGS:
    cargo test -p repo-reaper-cli {{ARGS}}

# run cargo-deny checks the same way CI does
[group("ci")]
deny:
    cargo deny check

# run the scheduled advisory-only security audit
[group("ci")]
audit:
    cargo deny check advisories

# run the checked-in repo-native evaluation smoke set
[group("dev")]
eval-smoke:
    cargo run -p repo-reaper-cli -- --evaluate --eval-data ./data/eval/repo_reaper.json --eval-format pretty --top-n 10 --fresh

# run the checked-in repo-native evaluation smoke set with field-aware ranking
[group("dev")]
eval-smoke-bm25f:
    cargo run -p repo-reaper-cli -- --evaluate --eval-data ./data/eval/repo_reaper.json --eval-format pretty --top-n 10 --fresh --ranking-algorithm bm25f

# compare the checked-in repo-native evaluation set against a saved JSON baseline
[group("dev")]
eval-compare baseline:
    baseline="{{baseline}}"; cargo run -p repo-reaper-cli -- --evaluate --eval-data ./data/eval/repo_reaper.json --eval-format pretty --top-n 10 --fresh --eval-compare "${baseline#baseline=}"

# compile the workspace
[group("dev")]
build:
    cargo build --workspace

# run the core Criterion benchmark harness
[group("dev")]
bench:
    cargo bench -p repo-reaper-core --bench search_bench

# compare ranked IR search against regex candidate/search paths
[group("dev")]
bench-search-comparison *ARGS:
    cargo bench -p repo-reaper-core --bench search_bench -- search_comparison {{ARGS}}
