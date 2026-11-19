# Yarvis

**A local-first, low-latency voice assistant you can trust with your computer.**

Your own Jarvis — except it lives on your machine, like a yeti lives off the grid.

> **Status: early development (Phase 2 of 5).** Talk to Yarvis in its own app window: say
> "Hey Yarvis", ask, interrupt it mid-sentence. It can now read your calendars and email
> (and send, after your "yes"), look things up on the web, tell you the weather, set timers
> and reminders, drive your browser (tabs, pages, links, videos), play any song or your own
> playlists on Spotify, type into apps, control apps and volume on your Mac, and remember
> what you tell it about yourself. An offline mode is next. Star or watch the repo to follow along.

## Why Yarvis

- **Feels instant** — Yarvis starts transcribing and asking Claude *during your pauses*,
  speaks from the first comma, and keeps every engine warm
  ([how it works](docs/adr/0005-speculative-turn-taking.md)). It measures itself: every
  reply's time to the first word, and what it cost, on the dashboard and in `yarvis usage`.
- **Talk like you would to a person** — say "Hey Yarvis" and it greets you; keep talking
  and the conversation stays open; talk over it to interrupt (or say "Hey Yarvis" again on
  speakers); say "thanks" when you're done
  ([how it works](docs/adr/0006-wake-word-and-barge-in.md)).
- **Private by default** — wake word, speech-to-text and text-to-speech run locally (on
  speech recognition is NVIDIA's Parakeet, on the GPU of an Apple Silicon Mac and on the
  processor of any other computer, which mishears far fewer words than small Whisper
  models). Audio never leaves your machine and is never stored; only the text of your
  request goes to the language model. Want the best transcripts money can buy? Add an
  AssemblyAI key and your requests (only those) are transcribed live on its EU servers,
  with your computer as backup
  ([how it works](docs/adr/0012-assemblyai-live-speech-recognition.md)). A fully offline
  mode is on the roadmap.
- **Actually does things** — reads your calendars and email and sends mail, searches the
  web, checks the weather, sets timers and reminders that speak up, drives apps, websites,
  volume and music, remembers facts about you; and anything you add: MCP servers and Agent Skills, each tool approved
  by you ([how it works](docs/adr/0013-mcp-servers-and-agent-skills.md)).
- **Safe to give power to** — a security model that assumes the AI *will* be manipulated:
  every action passes a permission gate in code, consequential ones need your spoken "yes"
  (the question is built by code, so you hear exactly what will happen), and content from
  outside can never skip that step
  ([how it works](docs/adr/0007-tools-and-the-permission-gate.md)). A
  [prompt-injection test suite](docs/adr/0009-web-search-weather-and-injection-tests.md)
  plants the same attack in every page, tab title, song title, place name, search result,
  email, calendar invitation, MCP tool result and skill, and checks that it never gets
  through, nor plants or erases a memory.
- **Has a body** — an optional ESP32 desk satellite with mic, speaker and LED ring.

Read the full [design document](docs/DESIGN.md).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Foundations: tooling, CI, config, secrets, event bus, `yarvis doctor` | ✅ done |
| 1 | Voice loop: wake word, streaming STT, Claude, streaming TTS, barge-in | ✅ done |
| 2 | Tools and the security core: permission gate, MCP, skills | ✅ done |
| 2.5 | Polish and reach: the HUD dashboard with cost and speed (done), memory (done), one-line install, routines, demo | in progress |
| 3 | Fully offline mode | planned |
| 4 | ESP32 desk satellite | planned |
| 5 | v1.0 launch | planned |

## Try it

Requires macOS 14+ on Apple Silicon, Windows 10/11 (early support, see below) or Linux, plus
[uv](https://docs.astral.sh/uv/) and an [Anthropic API key](https://console.anthropic.com).

```bash
git clone https://github.com/victorclaeyson-ai/Yarvis.git && cd Yarvis
uv sync
uv run yarvis key set anthropic   # stores your API key in the OS keychain
uv run yarvis doctor              # checks audio devices, keychain, models
uv run yarvis app                 # opens the Yarvis window: say "Hey Yarvis"
```

The first run downloads the speech models, verified by SHA-256: the voice (350 MB) and
Parakeet speech recognition (2.5 GB; NVIDIA, CC BY 4.0; Whisper, 500 MB, if you decline or
it cannot load). After that, speech recognition (unless you add AssemblyAI, below) and the
voice run fully on your machine; only the text of the conversation goes to Claude. While you
talk to Yarvis, Spotify and browser videos are turned down so it hears you rather than the
song (`duck_music = false` turns that off), and once Yarvis has started music or a video,
say "Hey Yarvis" to talk to it again. Music, email and the calendar are handled in the
background: Spotify and Mail stay behind the app you're in.

**Even better hearing (optional, paid).** Parakeet still mishears some names and short
requests. [AssemblyAI](https://www.assemblyai.com)'s Universal-3.6 Pro hears almost
everything, formats it (names, numbers, punctuation) and answers within a tenth of a second.
Create an account (new accounts get free credit; after that it costs $0.45 per hour of
talking to Yarvis), then:

```bash
uv run yarvis key set assemblyai   # paste the key from your AssemblyAI dashboard
```

From then on, what you say to Yarvis is streamed to AssemblyAI while you say it, and the
speech recognition on your computer is the backup for any turn the cloud can't handle (no
internet, a slow answer). Only your requests are sent: from just before you start talking
until you finish, never while Yarvis waits for "Hey Yarvis" or talks. In the follow-up
window, talk in the room that turns out not to be for Yarvis is transcribed by AssemblyAI
too before Yarvis drops it. By default the EU servers are used, whose data AssemblyAI does
not use for training; for zero retention, email data-opt-out@assemblyai.com from your
account's address. `cloud_stt = false` in the config switches it off; `yarvis key delete
assemblyai` removes the key. On the free plan AssemblyAI allows 5 new sessions (requests) a
minute; adding a payment method raises that to 100.

**About the wake word.** "Hey Yarvis" is detected on your computer by
[openWakeWord](https://github.com/dscripka/openWakeWord)'s "hey jarvis" model (it hears
"Yarvis" just as well). While it is on, the microphone stays open — your system's microphone
indicator shows it — but nothing is recorded or sent until you say it. Those pre-trained
models are free for personal, **non-commercial** use (CC BY-NC-SA 4.0), so Yarvis downloads
them separately (4 MB) instead of bundling them. Prefer press-to-talk? Set
`wake_word = false` in the config, or start with `--no-wake-word`.

**Keep talking.** After Yarvis answers, whatever you say in the next 15 seconds goes to it,
no wake word needed (`follow_up_s`); say "thanks" or "that's all" to end it sooner. After
that, say "Hey Yarvis" again. If you'd like it to stay open longer, `stay_open_s` keeps it
listening, but only for what sounds meant for it (its name, or a command like "open
Spotify"); other talk in the room is dropped on your computer and never reaches Claude. That
filter can also miss a real request that doesn't sound like one, so it's off by default.

**Tell it about yourself.** `yarvis profile edit` opens a short "about me" file: who you
are, what you're working on, how you like to be spoken to, and your own rules ("always read
an email back to me before sending it"). Yarvis knows it at every question, and an edit
counts from your next one ([how it works](docs/adr/0014-about-me-profile.md)). It is sent to
Claude with your requests, so keep it to about a page and never put keys in it;
`yarvis profile show` prints exactly what is sent.

**It remembers.** Tell Yarvis something lasting about yourself ("my sister is called
Lotte") and it keeps it as a short fact, saying "I'll remember that." Ask "what do you
know about me?" or say "forget that" any time. On the command line,
`yarvis memory list|add|forget|export|clear` shows and changes the same facts. Only facts
are kept, never what you said, in a private file; they are sent to Claude with your
requests like the profile. After Yarvis has read a web page or an email, it asks before it
remembers or forgets anything, so outside content can't plant a memory. Set
`[memory] auto = false` to have it remember only when you ask
([how it works](docs/adr/0017-memory.md)).

**The dashboard.** The window is a heads-up display: Yarvis's core in the middle, your
conversation under it, and around it every connection (Claude, speech recognition, Spotify,
your MCP servers…) with what each one uses or what to run to fix it, the microphone, the
permission gate, the weather at home, your timers, and what today cost and how fast Yarvis
answered (time to the first word, the median, a chart of the latest turns; Claude, web
searches and AssemblyAI at list prices). Make the window narrow and it turns into a compact
one, with the connections behind a button. `yarvis usage` shows the last days; only numbers
are kept, never what was said, and `yarvis usage clear` deletes them. Questions asked
within five minutes of each other reuse Claude's prompt cache, so the rules and tools cost
a tenth of the price once they are long enough to cache (they are with tools on) ([why](docs/adr/0016-prompt-caching.md)). With a `location` in
the config, the dashboard asks Open-Meteo for that town's weather twice an hour
([how it works](docs/adr/0015-dashboard-cost-and-speed.md)).

**Add abilities: MCP servers.** Any [MCP](https://modelcontextprotocol.io) server can give
Yarvis new tools, but only the ones you list, each with a tier: 0 runs freely (read only),
1 runs and is shown (reversible), 2 asks you first, 3 never runs. In the config:

```toml
[[mcp.servers]]
name = "files"
command = ["npx", "-y", "@modelcontextprotocol/server-filesystem@<version>", "/Users/me/Notes"]
  [mcp.servers.tools]
  read_text_file = 0
  write_file = 2
```

Then `yarvis mcp approve files` starts it and shows each listed tool exactly as Claude will
read it; approving records a fingerprint, and a tool that changes later is left out until
you approve it again. Servers that need a key get it from the keychain
(`secrets = ["NAME"]`, then `yarvis mcp secret files NAME`); a server Yarvis starts never
sees Yarvis's own keys. Remote servers: `url = "https://..."` (HTTPS only), with an optional
`token`. Whatever a server returns is outside content: after it, every call asks you first.
`yarvis mcp list` shows what is set up.

**Add know-how: Agent Skills.** A skill ([agentskills.io](https://agentskills.io)) is a
folder with a `SKILL.md`: when to use it, and instructions. `yarvis skills add
~/skills/morning-briefing` shows it and copies its text in; Yarvis reads it when a request
calls for it. Scripts in a skill are never run, and a skill can't approve anything: every
action still goes through the gate. `yarvis skills list` and `yarvis skills remove`.

Curious how fast it is on your machine? `yarvis bench` measures each stage (voice, each
speech recognition engine and what it heard, Claude's first token) and recommends settings.

Prefer the terminal? `yarvis run` does the same in your shell. Both accept `--text` (type
instead of speaking), `--mute` (no voice output) and `--no-wake-word`. `yarvis config init`
writes a commented config file (your name and where you live, voice, speed, model,
vocabulary for names, end-of-turn timing, how long a conversation stays open, wake word,
interruptions, and which tools are on).

**Things to try:** "What's on my calendar tomorrow?", "Any new email?", "Read the one from
Thomas", "Reply that I'll be there at seven", "Star it", "Put the newsletter in the trash", "Put dentist on Thursday at three in my
calendar", "What's the weather tomorrow?", "Who won the Belgian Grand Prix?",
"Is the Albert Heijn in Leuven open on Sunday?", "Set a pasta timer for nine minutes", "Remind me at six to call mom",
"Play Feel What You Want by Kristine W", "Spotify volume to forty", "Turn on shuffle",
"Play my workout playlist", "What's playing?", "Open Brave", "Open a new tab", "Go to
YouTube", "Click on the first video", "Pause the video", "Switch to the Gmail tab",
"Search for trains to Ghent", "Go back", "Type 'see you at eight' in Messages", "Quit
Safari" (Yarvis asks first). The first time Yarvis controls an app, macOS asks whether your
terminal may do so.

**Calendar and email.** Yarvis uses the accounts your Mac already has, so it needs no
Google or Microsoft developer setup and never sees your passwords. Add your accounts in
System Settings → Internet Accounts (Google, Microsoft Exchange for Outlook.com and work or
school accounts, iCloud) with Mail, Contacts and Calendars turned on, and let the Mail app
sync once. The first time, macOS asks whether your terminal may see your calendars and
control Mail and Contacts (Mail and Contacts stay hidden in the background). Reading is
instant; sending an email, replying, trashing one or adding an event always needs your
"yes", and Yarvis reads back exactly what it will do ("Should I email Thomas
Peeters (thomas@example.com), from you@gmail.com, …?"). Emails and invitations are written by
other people, so they are treated like web pages: they can't make Yarvis do anything without
your "yes", nor quietly open a link that would carry your data away
([how it works](docs/adr/0010-calendar-and-email-through-the-mac.md)). Like the rest of the
conversation, the text of the emails Yarvis reads goes to Claude. Turn either off with
`[tools] calendar = false` or `mail = false`.

**Web search and the weather.** For anything that may have changed (news, scores, opening
hours), Claude searches the web itself through Anthropic's
[web search tool](https://docs.claude.com/en/docs/agents-and-tools/tool-use/web-search-tool):
no extra account, about 1 cent per search on your Anthropic bill, at most 3 per question
(`[tools] web_search = false` turns it off). Search results are treated like any web page:
they can't make Yarvis do anything without your "yes". The weather comes straight from
[Open-Meteo](https://open-meteo.com) (free, no account; data CC BY 4.0); only the place name
leaves your computer. Set `location = "Leuven, Belgium"` in the config so "the weather" and
"near me" mean where you live.

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
playing?" work either way. Then run `uv run yarvis spotify login` once: Spotify asks you to
let Yarvis read your playlists and control playback (never change your library), and the
login stays in the keychain (`uv run yarvis key delete spotify-login` disconnects it). With
it, **your own playlists** play by name, and music plays through Spotify's own service: in
the background, without the Spotify window jumping in front of what you're doing. If
Spotify isn't open, Yarvis opens it (hidden on a Mac, minimized on Windows) and plays there.

**Typing.** Yarvis pastes text into the app in front, so it works with any keyboard layout.
Pressing Enter (which sends, in chat apps) always needs your "yes", and Yarvis never types
into terminals, password managers or System Settings. macOS asks once: System Settings →
Privacy & Security → Accessibility → allow the app you start Yarvis from. Note that this
lets anything started from that app control the keyboard; turn typing off with
`[tools] typing = false` if you'd rather not.

On Linux, the app window needs a GUI backend: `uv pip install 'pywebview[qt]'`.

**Windows (early support).** Talking, timers, the weather, web search and Spotify work,
and Yarvis can open apps from the Start menu, open websites, set the volume, play, pause or
skip whatever is playing, and quit apps (after your "yes"). Browser control, typing, email
and the calendar are macOS-only for now. Install
[Git](https://git-scm.com/download/win) and uv (`winget install --id astral-sh.uv -e`), then
run the same commands as above in a normal (not administrator) PowerShell. Two Windows
quirks:

- **Smart App Control** (Windows 11) blocks uv's own Python and the speech libraries, with
  errors like `0xc0e90002` or "An Application Control policy has blocked this file": it
  only lets through programs that many other PCs run. Turn it off in Windows Security → App
  & browser control → Smart App Control (Defender keeps running).
- Yarvis uses the microphone Windows has as default: Settings → System → Sound → Input.

Speech recognition on Windows is the same Parakeet model as on a Mac, run on the processor
by [onnx-asr](https://github.com/istupakov/onnx-asr) (no graphics card needed: it is many
times faster than real time on a desktop CPU).

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
