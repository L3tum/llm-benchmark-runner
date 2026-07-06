# syntax=docker/dockerfile:1

# Build a static musl binary so the runtime can use the tiny Alpine-based
# Docker CLI image. That image gives coding_eval access to `docker run` when
# /var/run/docker.sock is mounted, without shipping a Docker daemon here.
FROM rust:1.96 AS builder

ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup target add x86_64-unknown-linux-musl \
    && apt-get update \
    && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN --mount=type=cache,id=llm-benchmark-runner-cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=llm-benchmark-runner-target,target=/build/target \
    cargo build --target x86_64-unknown-linux-musl --release \
    && cp target/x86_64-unknown-linux-musl/release/llm-benchmark-runner /llm-benchmark-runner

# Runtime includes only the Docker CLI and our static binary. To let
# code/SWE-Bench benchmarks start sandbox/harness containers, run this container with:
#   -v /var/run/docker.sock:/var/run/docker.sock
# and set docker.host_repo_path when the host-visible repo path differs from
# the in-container repo path.
FROM docker:27-cli

COPY --from=builder /llm-benchmark-runner /llm-benchmark-runner
COPY models_config.yaml /models_config.yaml

WORKDIR /
EXPOSE 5050

CMD ["/llm-benchmark-runner"]
