# list available recipes
default:
    @just --list

# fmt → lint → test → build
[group("ci")]
check: fmt lint test build

# format with nightly (required for rustfmt.toml options)
[group("dev")]
fmt:
    cargo +nightly fmt

# run clippy with all workspace lints as errors
[group("dev")]
lint:
    cargo clippy --all-targets --all-features -- -D warnings

# run tests, accepts extra args: just test --lib
[group("dev")]
test *ARGS:
    cargo test {{ARGS}}

# run the checked-in repo-native evaluation smoke set
[group("dev")]
eval-smoke:
    cargo run -p repo-reaper-cli -- --evaluate --eval-data ./data/eval/repo_reaper.json --eval-format pretty --top-n 10 --fresh

# compile the workspace
[group("dev")]
build:
    cargo build

# run the core Criterion benchmark harness
[group("dev")]
bench:
    cargo bench -p repo-reaper-core --bench search_bench
