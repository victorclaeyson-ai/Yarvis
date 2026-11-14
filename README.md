# Yarvis

**A local-first, low-latency voice assistant you can trust with your computer.**

Your own Jarvis — except it lives on your machine, like a yeti lives off the grid.

> **Status: early development (Phase 0 of 5).** The foundations are in place; the voice loop
> arrives in Phase 1. Star or watch the repo to follow along.

## Why Yarvis

- **Feels instant** — target ≤ 800 ms from the end of your sentence to its first word, and
  you can interrupt it mid-sentence.
- **Private by default** — wake word, speech-to-text and text-to-speech run locally. Audio
  never leaves your machine. A fully offline mode is built in.
- **Actually does things** — tools via the Model Context Protocol (MCP), so anyone can plug
  in new abilities.
- **Safe to give power to** — a security model that assumes the AI *will* be manipulated:
  permission tiers, confirmation for consequential actions, and taint tracking for
  untrusted content. Enforced in code, not left to the model.
- **Has a body** — an optional ESP32 desk satellite with mic, speaker and LED ring.

Read the full [design document](docs/DESIGN.md).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations: tooling, CI, config, secrets, event bus, `yarvis doctor` | ✅ done |
| 1 | Voice loop: wake word, streaming STT, Claude, streaming TTS, barge-in | 🚧 next |
| 2 | Tools and the security core: permission gate, MCP | planned |
| 3 | Memory and fully offline mode | planned |
| 4 | ESP32 desk satellite | planned |
| 5 | v1.0 launch | planned |

## Try the foundations

Requires macOS (Apple Silicon) or Linux, and [uv](https://docs.astral.sh/uv/).

```bash
git clone <this repo> && cd yarvis
uv sync
uv run yarvis doctor          # checks Python, audio devices, keychain, config
uv run yarvis config init     # writes a commented config file
uv run yarvis key set anthropic   # stores your API key in the OS keychain
```

## Development

```bash
uv sync                       # install everything, exactly as locked
uv run pre-commit install     # run the checks on every commit
uv run pytest                 # tests (coverage must stay ≥ 90 %)
uv run pyright                # strict type checking
uv run ruff check . && uv run ruff format .
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md) —
please don't open public issues for vulnerabilities.

## License

[Apache 2.0](LICENSE)
