# Benchmarks image. Built from the repo root (see `make benchmarks-docker`).
# Pinned to the oldest supported interpreter; this overrides the repo's
# .python-version so uv uses the interpreter already in the base image.
FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim
ENV UV_PYTHON=3.10

# git is required by setuptools_scm to derive the version; make drives the benchmarks.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install the exact locked dependency set rather than re-resolving at build time.
# The default groups are included so the `uv run` calls in `make benchmarks`
# find a fully-synced environment and do not re-resolve.
ENV UV_LINK_MODE=copy
RUN uv sync --locked

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

RUN make benchmarks
