# syntax=docker/dockerfile:1

# Reusable SWE-Bench harness image for llm-benchmark-runner.
#
# The runner auto-builds this image when docker.build_images=true and the
# configured docker.images.swebench_harness tag is missing. Manual equivalent:
#   docker build -f docker/swebench-harness/Dockerfile \
#     -t llm-benchmark-runner/swebench-harness:latest \
#     docker/swebench-harness
#
# The benchmark runner overrides CMD with:
#   python -m swebench.harness.run_evaluation ...
# and mounts /var/run/docker.sock so the harness can create per-instance
# evaluation containers on the host Docker daemon.

FROM python:3.11-slim-bookworm

ARG SWEBENCH_VERSION=4.1.0

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache

# git is useful for harness utilities and debugging. ca-certificates keeps
# HuggingFace/GitHub downloads working from slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
    && python -m pip install "swebench==${SWEBENCH_VERSION}" \
    && python -m swebench.harness.run_evaluation --help >/tmp/swebench-run-evaluation-help.txt

WORKDIR /work

CMD ["python", "-m", "swebench.harness.run_evaluation", "--help"]
