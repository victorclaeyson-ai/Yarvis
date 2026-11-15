# Yarvis

**A local-first, low-latency voice assistant you can trust with your computer.**

Your own Jarvis — except it lives on your machine, like a yeti lives off the grid.

> **Status: early development (Phase 2 of 5).** Talk to Yarvis in its own app window: say
> "Hey Yarvis", ask, interrupt it mid-sentence. It can now set timers and reminders, drive
> your browser (tabs, pages, links, videos), play any song or your own playlists on Spotify,
> type into apps, and control apps and volume on your Mac. Web answers, calendar and memory
> are next. Star or watch the repo to follow along.

## Why Yarvis

- **Feels instant** — Yarvis starts transcribing and asking Claude *during your pauses*,
  speaks from the first comma, and keeps every engine warm
  ([how it works](docs/adr/0005-speculative-turn-taking.md)).
- **Talk like you would to a person** — say "Hey Yarvis" and it greets you; keep talking
  and the conversation stays open; talk over it to interrupt (or say "Hey Yarvis" again on
  speakers); say "thanks" when you're done
  ([how it works](docs/adr/0006-wake-word-and-barge-in.md)).
- **Private by default** — wake word, speech-to-text and text-to-speech run locally (on
  Apple Silicon, speech recognition uses NVIDIA's Parakeet on the GPU, which mishears far
  fewer words than small Whisper models). Audio never leaves your machine and is never
  stored; only the text of your request goes to the language model. A fully offline mode is
  on the roadmap.
- **Actually does things** — timers and reminders that speak up, apps, websites, volume
  and music today; web search and any MCP server (calendar, email, …) next.
- **Safe to give power to** — a security model that assumes the AI *will* be manipulated:
  every action passes a permission gate in code, consequential ones need your spoken "yes"
  (the question is built by code, so you hear exactly what will happen), and content from
  outside can never skip that step
  ([how it works](docs/adr/0007-tools-and-the-permission-gate.md)).
- **Has a body** — an optional ESP32 desk satellite with mic, speaker and LED ring.

Read the full [design document](docs/DESIGN.md).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations: tooling, CI, config, secrets, event bus, `yarvis doctor` | ✅ done |
| 1 | Voice loop: wake word, streaming STT, Claude, streaming TTS, barge-in | ✅ done |
| 2 | Tools and the security core: permission gate, MCP | 🚧 in progress |
| 3 | Memory and fully offline mode | planned |
| 4 | ESP32 desk satellite | planned |
| 5 | v1.0 launch | planned |

## Try it

Requires macOS 14+ on Apple Silicon, or Linux, plus [uv](https://docs.astral.sh/uv/) and an
[Anthropic API key](https://console.anthropic.com).

```bash
git clone https://github.com/victorclaeyson-ai/Yarvis.git && cd Yarvis
uv sync
uv run yarvis key set anthropic   # stores your API key in the OS keychain
uv run yarvis doctor              # checks audio devices, keychain, models
uv run yarvis app                 # opens the Yarvis window: say "Hey Yarvis"
```

The first run downloads the speech models, verified by SHA-256: the voice (350 MB) and, on
Apple Silicon, Parakeet speech recognition (2.5 GB; NVIDIA, CC BY 4.0), or Whisper elsewhere
(500 MB). After that, speech recognition and the voice run fully on your machine; only the
text of the conversation goes to Claude. While you talk to Yarvis, Spotify and browser
videos are turned down so it hears you rather than the song (`duck_music = false` turns that
off), and once Yarvis has started music or a video, say "Hey Yarvis" to talk to it again.

**About the wake word.** "Hey Yarvis" is detected on your computer by
[openWakeWord](https://github.com/dscripka/openWakeWord)'s "hey jarvis" model (it hears
"Yarvis" just as well). While it is on, the microphone stays open — your system's microphone
indicator shows it — but nothing is recorded or sent until you say it. Those pre-trained
models are free for personal, **non-commercial** use (CC BY-NC-SA 4.0), so Yarvis downloads
them separately (4 MB) instead of bundling them. Prefer press-to-talk? Set
`wake_word = false` in the config, or start with `--no-wake-word`.

Curious how fast it is on your machine? `yarvis bench` measures each stage (voice, each
speech recognition engine and what it heard, Claude's first token) and recommends settings.

Prefer the terminal? `yarvis run` does the same in your shell. Both accept `--text` (type
instead of speaking), `--mute` (no voice output) and `--no-wake-word`. `yarvis config init`
writes a commented config file (your name, voice, speed, model, vocabulary for names,
end-of-turn timing, how long a conversation stays open, wake word, interruptions, and which
tools are on).

**Things to try:** "Set a pasta timer for nine minutes", "Remind me at six to call mom",
"Play Feel What You Want by Kristine W", "Spotify volume to forty", "Turn on shuffle",
"Play my workout playlist", "What's playing?", "Open Brave", "Open a new tab", "Go to
YouTube", "Click on the first video", "Pause the video", "Switch to the Gmail tab",
"Search for trains to Ghent", "Go back", "Type 'see you at eight' in Messages", "Quit
Safari" (Yarvis asks first). The first time Yarvis controls an app, macOS asks whether your
terminal may do so.

**Pages in your browser.** To read a page and follow its links (a video, an article),
Yarvis runs a small, fixed script in the current tab. Browsers only allow that once you turn
it on: in Brave or Chrome, View → Developer → Allow JavaScript from Apple Events; in Safari,
Develop → Allow JavaScript from Apple Events. Following a link just opens its https address;
pressing a button always needs your "yes"; private windows are never read.

**Spotify by name.** To play a specific song, album, artist or playlist, Yarvis searches
Spotify's catalogue. Create a developer app at
[developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) (any name,
redirect URI `http://127.0.0.1:8888/callback`, tick "Web API"), then run
`uv run yarvis key set spotify` and paste its client ID and secret; they go to the keychain.
Spotify requires the developer account to have Premium (since February 2026). Without it,
Yarvis opens the search in Spotify instead; volume, shuffle, repeat, skipping and "what's
playing?" work either way. For **your own playlists**, run `uv run yarvis spotify login`
once: Spotify asks you to allow read-only access, and the login stays in the keychain
(`uv run yarvis key delete spotify-login` disconnects it).

**Typing.** Yarvis pastes text into the app in front, so it works with any keyboard layout.
Pressing Enter (which sends, in chat apps) always needs your "yes", and Yarvis never types
into terminals, password managers or System Settings. macOS asks once: System Settings →
Privacy & Security → Accessibility → allow the app you start Yarvis from. Note that this
lets anything started from that app control the keyboard; turn typing off with
`[tools] typing = false` if you'd rather not.

On Linux, the app window needs a GUI backend: `uv pip install 'pywebview[qt]'`.

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
