# 2. Python with uv, strict typing and enforced quality gates

- **Status:** Accepted
- **Date:** 2026-11-14

## Context

The local speech and AI ecosystem (Whisper variants, Silero VAD, openWakeWord, Kokoro,
the MCP SDK, the Anthropic SDK) is Python-first. Rust or TypeScript would mean
re-implementing or wrapping much of it. Python's main weakness for a long-lived codebase
is the lack of enforced types and reproducible environments.

## Decision

- Python 3.12+ (PEP 695 generics, `typing.override`).
- `uv` for environments and a hash-locked `uv.lock`.
- `pyright` in **strict** mode with zero errors, on both `src/` and `tests/`.
- `ruff` for formatting and a broad lint set, including the bandit-derived `S` rules.
- `pytest` with a 90 % coverage gate; `hypothesis` for property-based tests.
- The same checks run locally (pre-commit) and in CI on macOS and Linux.

## Consequences

Higher upfront effort per change, in exchange for a codebase that is safe to refactor and
easy for contributors to trust. Untyped third-party libraries (e.g. `sounddevice`) are
wrapped in small, typed adapters so the untyped surface stays contained.
