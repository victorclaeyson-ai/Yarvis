# Contributing to Yarvis

Thanks for helping build Yarvis. The bar for code quality is high on purpose: this is an
assistant people trust with their computer.

## Setup

```bash
uv sync
uv run pre-commit install
```

## Before you open a pull request

All of these must pass (CI runs them on macOS and Linux):

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright          # strict mode, zero errors
uv run pytest           # coverage ≥ 90 %
```

## Guidelines

- **Typed everything.** No `Any` without a comment explaining why.
- **Test behaviour, not implementation.** Every bug fix comes with a test that fails without it.
- **No hardware or network in unit tests.** Use the fakes in `tests/fakes.py`.
- **Security-sensitive code** (`src/yarvis/security/`, anything touching tools, secrets,
  audio storage or networking) needs a clear explanation in the PR of what could go wrong.
- **Significant decisions** get an Architecture Decision Record in `docs/adr/`.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
- **Never** commit secrets, audio recordings or model weights.
