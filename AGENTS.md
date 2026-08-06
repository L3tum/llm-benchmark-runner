# AGENTS.md

Project notes and scoping decisions for code review agents and future maintainers.

## Security Scoping Decisions

### Command Injection via Config (runner.rs) — NOT actioned

The `cmd` string from `models_config.yaml` is passed to `bash -c` in `runner.rs` for starting/stopping model processes. While this could theoretically allow command injection from a malicious config file, **the entire runner executes inside a Docker container** with strong sandboxing defaults (cap_drop_all, read_only_root, network_none, PID limits, memory limits). The container boundary mitigates host-level injection risk.

**Decision:** Accept the risk. If config-level validation is needed in the future, add it as a separate task.

### Shell Command Denylist (docker_runner.rs) — NOT actioned

The `DockerRunner::exec()` method uses a denylist approach (`command.contains("docker")`) to block container escape attempts. While denylists are inherently bypassable, **the exec commands run inside already-sandboxed Docker containers** that have:
- `--cap-drop=ALL` (no Linux capabilities)
- `--network=none` (no network access)
- `--read-only` root filesystem
- PID and memory limits
- No Docker socket mounted by default

**Decision:** Accept the current approach. The container sandboxing provides the primary security boundary. A future allowlist approach could be added if benchmarks require more permissive container configurations.

## Codebase Notes

### Disabled Benchmark Modules

Modules `factbench.rs`, `scifact.rs`, `hdm_bench.rs`, and `stable_toolbench.rs` are commented out of `src/benchmarks/mod.rs` but the `.rs` files remain in the source tree. They are **not compiled** (Rust only compiles modules that are declared in `mod.rs`). They're kept as reference material for potential future re-enablement.

**Decision:** Leave as-is. Moving them to a `disabled/` directory adds organizational overhead without functional benefit.

### Mutex Poisoning Pattern

All benchmark implementations use `.lock().expect(MUTEX_PANIC_MSG)` for interior mutability. If a task panics while holding the lock, the mutex is permanently poisoned. The plan includes replacing this with `unwrap_or_else(|e| e.into_inner())` for graceful recovery (see C7).

### Benchmark Copy-Paste

SWE-Bench (4 variants × ~200 lines) and string transformation benchmarks (4 × ~345 lines) have significant code duplication. See C2 for the deduplication plan.
