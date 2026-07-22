# syntax=docker/dockerfile:1@sha256:87999aa3d42bdc6bea60565083ee17e86d1f3339802f543c0d03998580f9cb89

# Build a static musl binary so the runtime can use the tiny Alpine-based
# Docker CLI image. That image gives coding_eval access to `docker run` when
# /var/run/docker.sock is mounted, without shipping a Docker daemon here.
FROM rust:1.96@sha256:1f0dbad1df66647807e6952d1db85d0b2bda7606cb2139d82517e4f009967376 AS builder

ENV PATH="/root/.cargo/bin:${PATH}"

RUN rustup target add x86_64-unknown-linux-musl \
    && apt-get update \
    && apt-get install -y --no-install-recommends musl-tools nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN --mount=type=cache,id=llm-benchmark-runner-cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=llm-benchmark-runner-target,target=/build/target \
    cargo build --target x86_64-unknown-linux-musl --release --features renderer-official \
    && cp target/x86_64-unknown-linux-musl/release/llm-benchmark-runner /llm-benchmark-runner

# Runtime includes only the Docker CLI and our static binary. To let
# code/SWE-Bench benchmarks start sandbox/harness containers, run this container with:
#   -v /var/run/docker.sock:/var/run/docker.sock
# and set docker.host_repo_path when the host-visible repo path differs from
# the in-container repo path.
FROM docker:27-cli@sha256:851f91d241214e7c6db86513b270d58776379aacc5eb9c4a87e5b47115e3065c

WORKDIR /app
EXPOSE 5050

COPY --from=builder /llm-benchmark-runner /app/llm-benchmark-runner
COPY models_config.yaml /models_config.yaml

VOLUME /app/benchmark_results
ENTRYPOINT ["/app/llm-benchmark-runner"]
