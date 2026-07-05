# llm-benchmark-runner — Quality & Development Makefile
# Run with `make` or `make <target>`

.PHONY: check fmt clippy test miri deny all build-release docs

# Quick pre-commit checks
check:
	cargo check --all-targets --all-features

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all-targets

# MIRI — detects undefined behavior (slow, needs nightly toolchain + miri component)
# Tests use mock data — no real network calls, so isolation is safe
miri:
	cargo +nightly miri test

# cargo-deny — license, security, banned crate checks
deny:
	cargo deny check licenses advisories ban

# Full pre-merge check suite
all: check fmt-check clippy test deny

# Build for release (static musl binary)
build-release:
	cargo build --release --target x86_64-unknown-linux-musl

# Build docs
docs:
	cargo doc --open --no-deps

# Run the benchmarks (for local testing)
run:
	cargo run run --config models_config.yaml

# Run without resuming
run-no-resume:
	cargo run run --config models_config.yaml --no-resume

# Generate report from existing results
report:
	cargo run report
