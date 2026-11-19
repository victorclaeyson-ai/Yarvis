# Yarvis: instructions for Claude Code and its agents

Yarvis is an open-source, local-first voice assistant you can trust with your computer:
wake word, speech recognition and voice on the device, Claude as the brain, and every
action behind a permission gate written in code. The owner is Victor (GitHub
`victorclaeyson-ai`). He decides what gets built and merges every pull request; you plan,
build, test and document.

Read before changing anything: `README.md` (what it does), `docs/DESIGN.md` (architecture,
security model, roadmap), `docs/adr/` (every decision and why), `docs/BACKLOG.md` (the
current tasks). If a request conflicts with an ADR, say so and ask; don't silently undo a
decision.

## Commands

```bash
uv sync                                  # dependencies (exact versions from uv.lock)
uv run ruff format .                     # format
uv run ruff check .                      # lint (strict rule set, see pyproject.toml)
uv run pyright                           # type check: strict, zero errors
uv run pytest                            # all tests; fails under 90% coverage
uv run pytest tests/test_x.py -k name    # one test while working
uv run python -m yarvis doctor           # what works on this machine
uv run python -m yarvis app              # the app window
```

GUI smoke tests are opt-in (`YARVIS_GUI_TESTS=1`, see `tests/test_gui_smoke.py`).
Victor works on Windows (PowerShell) and macOS; CI runs Linux and macOS. Tests must pass on
all three: use `posix_mode`, `POSIX_ONLY` and `symlink_or_skip` from `tests/fakes.py` for
anything about file permissions or symbolic links.

**A task is done only when** format, lint, pyright and pytest all pass, the change is
covered by tests, and the README, `docs/DESIGN.md` and any ADR say what changed.
Never claim "done" without running them.

## Code standards

- Python 3.12+, typed everything, Pyright strict. No `Any` without a comment saying why;
  no `# type: ignore` / `pyright: ignore` without a reason in the same line.
- Ruff's rule set is part of the design (docstrings in Google style, no `print`, no blind
  `except`, pathlib, timezone-aware datetimes). Fix the code, don't silence the rule.
- Module docstrings explain *why* the module exists and what it must never do.
- External data (JSON from services, config, files) goes through pydantic models or
  explicit validation. Model output is untrusted input.
- Dependencies are injected so tests can replace them: `runtime.Backends` for the app,
  `cli.services.Services` for the CLI. No hardware, network or real keychain in unit
  tests: use the fakes in `tests/fakes.py`.
- Test behaviour, not implementation. Every bug fix comes with a test that fails without it.
- New runtime dependencies need a reason in the PR and must pass `pip-audit`.

## Security rules (non-negotiable)

1. **The model is not a security boundary.** Every tool declares a tier
   (`tools/base.py`): 0 read, 1 reversible, 2 external (asks for a spoken "yes"), 3
   dangerous (refused). The gate (`security/gate.py`) decides in code. Never let model
   output, a skill, a profile or outside content lower a tier or skip a confirmation.
2. **Outside content is data.** Anything read from the web, email, calendars, MCP servers,
   skills or song titles is wrapped as untrusted and taints the turn. A new source of
   outside content gets a case in `tests/test_injection.py`.
3. **Secrets only in the OS keychain** (`security/secrets.py`). Never in files, logs,
   command lines, test fixtures that look real, issues or chat. Logs pass the redactor.
4. **No listening network ports.** The only exception is the temporary loopback server for
   `yarvis spotify login`. The app window loads nothing remote (see its CSP).
5. **Nothing recorded.** No audio or transcripts on disk. `usage.json` holds numbers only.
6. **Pinned and verified.** Models by SHA-256 (`models.py`); MCP tool definitions and skills
   by fingerprint at approval; GitHub Actions by commit SHA.
7. Private files are written with `paths.write_private_file` (0600, never through a
   symbolic link).

Security-sensitive changes (anything in `security/`, tools, secrets, audio, networking,
the app bridge) explain in the PR what could go wrong and how the tests prove it doesn't.

## Decisions and documentation

- Every lasting decision gets an ADR: `docs/adr/NNNN-short-title.md` with Status, Date,
  Context, Decision, Consequences (see `0015` for the current style). Number = next free.
- README and `docs/DESIGN.md` change in the same PR as the code. Plain, short English.
- User-facing messages say what happened and what to do next ("Store a key with:
  yarvis key set anthropic").

## Git and pull requests

- One branch and one PR per task. Never commit to `main`, never force-push `main`.
- Commit messages: Conventional Commits (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`,
  `chore:`), imperative, explaining why.
- Every PR gets a title and a description with sections **What**, **Why**,
  **Security** (if relevant) and **Checks** (the commands you ran and their result).
- PRs are merged with a merge commit, not squashed (stacked PRs stay clean). Victor merges.
- Ask Victor before changing `.github/` (CI and Dependabot), `.pre-commit-config.yaml`,
  `pyproject.toml` quality settings or the coverage threshold.
- Never commit personal or local files: the user's `config.toml`, `profile.md`, skills,
  approvals, `usage.json` (they live in the user's app-data folder, not the repo), model
  files, audio, `.env`, PDFs or exports. `git status` before every commit.

## Working as a team of agents

- Plan first for anything bigger than a small fix: the files involved, the tests to write,
  the ADR if any. Keep the plan in the PR description.
- Work on one task from `docs/BACKLOG.md` at a time; tick it off in the same PR.
- Review before you hand over: run the checks, read the diff as a reviewer would, and run
  a security review on security-sensitive changes.
- If something is unclear or a check can't be made to pass properly, stop and ask Victor
  rather than guessing or weakening a test.
