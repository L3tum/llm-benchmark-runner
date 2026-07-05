# Plan: Add `--dry-test` / `test-models` subcommand

## Context
Currently the project has a `run` command that starts models, runs benchmarks, and stops them. There is no lightweight way to verify that model configurations work (start, connect, send a prompt, stop) without running full benchmarks. We want a `test-models` subcommand (akin to `--dry-run`) that:
1. Starts each model (`cmd`)
2. Waits for proxy health
3. Sends a simple test chat prompt
4. Stops the model (`cmd_stop` or default SIGTERM/SIGKILL)
5. Reports success/failure for each model

## Approach
Add a new subcommand `test-models` to the CLI. This command will:
- Load the config like `run`
- Loop through each model, reuse `start_model`, `wait_for_health`, and `stop_model` from `runner.rs` (make them public)
- Create a `Client`, send a minimal chat completion (e.g., "Say hello in one word"), and log the result
- Print a summary at the end

## Files to modify
- `src/main.rs` — add `TestModels` subcommand and handler
- `src/runner.rs` — make `start_model`, `wait_for_health`, and `stop_model` public
- `src/client.rs` — no changes needed (already has `chat_completion`)
- `README.md` — document the new `test-models` subcommand with usage example

## Reuse
- `runner.rs::start_model(cmd)` — start model process
- `runner.rs::wait_for_health(client)` — wait for proxy readiness
- `runner.rs::stop_model(cmd_stop, process)` — stop with graceful kill + force kill fallback
- `client.rs::Client::new()` and `Client::chat_completion()` — send test prompt

## Steps
1. Make `start_model`, `wait_for_health`, and `stop_model` public in `runner.rs`
2. Add `TestModels { config: String }` subcommand to `Cli::Commands` in `main.rs`
3. Implement `test_models` function in `main.rs` that:
   - Loads config
   - For each model: start, health check, send "Hello" prompt, log result, stop
   - Collect results and print summary table
4. Wire up in `main()` match arm
5. Update `README.md` with usage instructions for `test-models`

## Verification
- `cargo run -- test-models --config models_config.yaml` (or use config from `test_config.yaml`)
- Manually verify that each model starts, responds to chat completion, and stops
- Check error handling: model fails to start, proxy doesn't respond, prompt returns error
