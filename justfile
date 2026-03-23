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

# compile the workspace
[group("dev")]
build:
    cargo build
