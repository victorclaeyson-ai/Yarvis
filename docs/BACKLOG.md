# Backlog

The work queue for whoever builds Yarvis next (people or agents). One task per branch and
pull request; tick it off here in the same PR. The full roadmap is in `docs/DESIGN.md` §8;
the "why" of earlier work is in `docs/adr/`.

Status on 19 November 2026: phases 0–2 are built, and Phase 2.5 (polish and reach) is
under way. The HUD dashboard with cost and speed (ADR 0015) and prompt caching (ADR 0016)
are done. Sprint 2 now comes before Sprint 1's open tasks, but task 1 (one-line install)
must still land before any public launch.

## Sprint 2: smarter, more tools

Rules for every new tool in this sprint:

- Same tool names and arguments on every platform, with one backend per platform behind a
  Protocol (like `CalendarStore`). Where a tool can't work, it isn't offered, and
  `yarvis doctor` says why.
- Pick the tier by what goes wrong if Claude is tricked into calling it: reading is 0, easy
  to undo is 1, anything sent, deleted or done on the user's behalf outside the computer
  is 2.
- Everything a tool reads is untrusted: mail, files, pages, screen, clipboard, train
  messages. It taints the turn and gets a case in `tests/test_injection.py`.
- Every new dependency: the ADR says why, its licence and that `pip-audit` is clean.
  Pinned in `uv.lock`.
- Tests use fakes only: no network, no real screen, clipboard or accounts.
- The README tool list and the DESIGN.md roadmap change in the same PR. One ADR per task,
  numbered from 0017.

Prices and limits below were checked against the providers' documentation on
19 November 2026.

- [x] **9. Memory.** (ADR 0017)
  Yarvis learns facts about the user and keeps them. They can be seen, edited and forgotten
  by voice ("what do you know about me?", "forget that") and on the command line
  (`yarvis memory list|add|forget|export|clear`). Facts only, in a private file in the
  data folder; no transcripts (CLAUDE.md rule 5). Memories go in the cached stable part of
  the prompt after the profile (ADR 0016), wrapped as data and capped, with a search tool
  for when they outgrow the cap. A memory write or delete in a tainted turn needs a spoken
  yes (DESIGN.md T7); memory text can never lower a tier or skip a confirmation.
  Done when: remembering, listing, forgetting and searching work by voice and CLI; facts
  survive a restart; injection tests show outside content can't change a memory without a
  yes; the ADR proposes whether opt-in short summaries of past conversations are worth it
  (that would change rule 5, so Victor decides).

- [ ] **10. Smart model switching.**
  Haiku 5.5 stays the voice, for a fast first word; hard requests get a bigger model. The
  ADR compares three ways:
  - Anthropic's advisor tool (beta `advisor_20260301`, header `advisor-tool-2026-03-01`):
    Haiku consults a bigger model inside the same request. With Haiku 5.5 as the executor,
    Sonnet 5.5, Opus 5.5 and Opus 4.8 are valid advisors. Advice from the 5.x models comes
    back encrypted (it can be passed on, not read); Opus 4.8's is plain text. The advisor
    is billed at its own rates, and its advice doesn't stream: the reply pauses until it
    arrives.
  - Re-running the turn on Sonnet or Opus after a spoken "let me think about that".
  - The user saying "think hard about…".
  Done when: first-word latency and cost are measured for each; the bigger model's tokens
  are priced in `usage.py` (Sonnet 5.5: $2 input, $10 output, $0.10 cache read per million
  tokens; Opus 5.5: $4, $20, $0.20); a daily dollar cap for it is in the config and
  enforced.

- [ ] **11. Links and exact math.**
  Upgrade web search to the current version (`web_search_20260318`; from `_20260209` on,
  results are filtered with code on Anthropic's side) and add Anthropic's web fetch
  (`web_fetch_20260318`) and code execution (`code_execution_20260521`). All of them run
  on Anthropic's side, never on this computer. Fetched pages are untrusted; the number of
  uses and the content size per turn are capped (`max_uses`, `max_content_tokens`).
  Done when:
  - The ADR covers the exfiltration risk of fetching, and how far Anthropic's rule covers
    it: Claude may only fetch URLs already in the conversation (user messages, our tool
    results, earlier search and fetch results), not URLs it writes itself or ones only in
    the system prompt.
  - `usage.py` prices code execution: free when used with web search or fetch `_20260209`
    or later; otherwise 1,550 free hours a month per organisation, then $0.05 per
    container-hour with a 5-minute minimum. Web fetch costs only its tokens; a search
    costs $10 per 1,000.
  - DESIGN.md's privacy table says code execution (and the result filtering that uses it)
    isn't eligible for zero data retention: Anthropic keeps container data up to 30 days.

- [ ] **12. Belgian trains**, through the iRail API (`api.irail.be`, with `format=json`;
  XML is the default). Next departures with delay and platform (`/liveboard/`),
  connections from A to B (`/connections/`) and disruptions (`/disturbances/`). Tier 0.
  Send a User-Agent with the repo URL (iRail blocks addresses that send none), stay under
  their limit (3 requests a second, a burst of 5) and cache the station list. Understand
  spoken station names ("Brussels South" is Brussel-Zuid) and default to the home station.
  Offered when the home location is in Belgium or `tools.trains` is on.
  Done when: departures, connections and disruptions work with fakes; station matching is
  tested on spoken names; the ADR records iRail's terms (their documentation states no
  price or key; check before shipping).

- [ ] **13. Sees my screen.**
  A tool that takes a screenshot of the active screen, only when asked, downscaled (Claude's
  standard image size is 1,568 pixels on the long edge) and sent with the request. Never
  written to disk, and shown on the dashboard while it happens. Asks first in a tainted
  turn and is never available to routines. Screen text is untrusted. Needs images in tool
  results in `llm/base.py` and `anthropic.py`.
  Done when: tested with a fake screen; `yarvis doctor` checks the macOS Screen Recording
  permission; nothing reaches the disk.

- [ ] **14. Computer control on Windows and macOS.**
  Lock (tier 1); sleep, restart and shut down (2); battery and Wi-Fi status (0); brightness
  (1); snap, maximize, minimize or move a window to the other screen (1); switch to an app
  (1); save a screenshot to Pictures when asked (1).
  Done when: every action works on both systems, tested with fakes; `yarvis doctor` shows
  what a system can't do.

- [ ] **15. Files, documents and the clipboard.**
  Find files by name, kind or date through Windows Search and Spotlight, only inside the
  allowed folders from the config (default: Documents, Desktop, Downloads, OneDrive). Open
  documents; executables, scripts and shortcuts are refused. Read the text of txt, md, pdf,
  docx, pptx and xlsx files, capped and untrusted. Read the clipboard (tier 0, untrusted)
  and write it (tier 1). A hard-coded deny list that the config can't override: `.ssh`,
  keychains, `.env` files, password managers, AppData.
  Done when: tests cover path traversal, symbolic links and the deny list; every document
  parser added is in the ADR.

- [ ] **16. Mail and calendar on the PC**, for Microsoft accounts (Outlook.com, Microsoft
  365, possibly KU Leuven) and Google accounts (Gmail, Google Calendar). The same `mail_*`
  and `calendar_*` tools with new backends, on every OS, plus
  `yarvis account add|list|remove`. Two PRs: Microsoft first, then Google. Same tiers as
  on the Mac (ADR 0010); the narrowest scopes, listed in the ADR; refresh tokens only in
  the keychain.
  - Microsoft: the device-code login through `/common` (personal and work accounts, no
    listening port), with a public client app Victor registers ("Allow public client
    flows" on); the PR gives the steps. Graph scopes `Mail.ReadWrite`, `Mail.Send`,
    `Calendars.ReadWrite`, `offline_access` and `User.Read` (`Mail.Read` can't mark mail
    read, move it to the trash or save drafts). A university tenant may require an
    administrator's consent. Personal accounts have no revoke API, so `account remove`
    deletes the token and points to the account's app-permissions page; work accounts can
    revoke their sessions.
  - Google: its device login doesn't allow Gmail or Calendar scopes, so Google needs a
    temporary loopback login with PKCE, like Spotify's. That changes CLAUDE.md security
    rule 4: ask Victor before building it. Each user brings their own OAuth client, set to
    "In production" (in "Testing", Google ends the login after 7 days). Gmail's read and
    compose scopes are restricted: a shared client would need Google's verification and a
    yearly security assessment. `account remove` revokes the token at Google.
  Done when: both providers pass the mail and calendar tool tests through fakes; tokens
  never leave the keychain; `account remove` is tested.

- [ ] **17. Notes and a to-do list.**
  Add and find notes; add, list and complete to-dos. Stored locally in a private markdown
  file the user can open. Works by voice and with `yarvis todo`. After task 16, optional
  sync with Microsoft To Do and Google Tasks.
  Done when: works by voice and CLI, with tests; the file still reads correctly after the
  user edits it by hand.

## Sprint 1: finish Phase 2.5 (until the January exams)

Exit test for the phase: a stranger installs Yarvis with one command and has a
conversation within five minutes.

- [ ] **1. One-line install and `yarvis init`.**
  Install scripts for macOS/Linux (`install.sh`) and Windows (`install.ps1`) that install
  uv if needed, then Yarvis. `yarvis init` walks through the microphone, speaker, API keys
  (hidden prompts, keychain only), model downloads and a first test sentence, and ends
  with `yarvis doctor`.
  Done when: a fresh Windows and macOS user account can install and talk to Yarvis by
  following the README alone; the scripts download nothing unpinned or unverified; tests
  cover `yarvis init` with fakes; README "Try it" is rewritten around it; an ADR records
  the install approach.

- [x] **2. Prompt caching for Claude.** (ADR 0016)
  Mark the system prompt and tool definitions as cacheable (`llm/anthropic.py`), so
  repeated requests pay the cached rate. `usage.py` already prices cache reads and writes.
  Done when: tests show the cache markers are sent and that a profile edit or tool change
  still reaches Claude; `yarvis usage` shows the lower cost on a test conversation;
  ADR 0015 or a new ADR notes the change.

- [ ] **3. Routines and a morning brief.**
  "Every weekday at 8:00, my morning brief" while Yarvis runs: a routine is a stored,
  user-approved request on a schedule. Ship a morning-brief skill (calendar, unread mail,
  weather, timers today). Consequential steps still ask first: a routine never confirms on
  the user's behalf.
  Done when: routines can be added, listed and removed by voice and CLI; they survive a
  restart (private file); the permission gate treats them like any request; injection
  tests cover routine content; an ADR explains the design.

- [ ] **4. Email drafts on the Mac.**
  "Draft a reply to …" saves to Mail's drafts instead of sending (tier 1: the user sends
  it). Done when: works with every account in Mail; tested with the fake runner; README
  lists it.

- [ ] **5. Dashboard follow-ups.**
  A "Brief me" button (after task 3), the window remembers its size and position, and the
  core label is readable at every window size.
  Done when: covered by the GUI smoke test; nothing in the page loads remotely.

- [ ] **6. AssemblyAI's newer real-time models.**
  AssemblyAI released STT-3.7-Realtime ($0.30/hour) and STT-3.7-Realtime-Agent (successor
  of Universal-3.6 Pro, $0.45/hour). Compare them on accuracy for names and short
  commands, latency and price; switch if better.
  Done when: the choice is in ADR 0012 (or a new ADR), prices in `usage.py` are updated
  with their date, and the streaming tests pass against the new model id.

- [ ] **7. Repository polish.**
  `examples/` (two skills, two MCP set-ups), `CHANGELOG.md`, `CODE_OF_CONDUCT.md`, issue
  and PR templates (ask Victor: `.github/`), a demo GIF at the top of the README.

- [ ] **8. Windows in CI.** (Victor edits `.github/workflows/ci.yml`.)
  Add `windows-latest` to the test matrix; the suite already passes on Windows. Optionally
  limit macOS to Python 3.12 to shorten the queue.

## For Victor (not for agents)

- Merge the open PRs in order: Windows + AssemblyAI (#11), MCP + skills, profile,
  dashboard.
- Sync the Mac (`git stash -u`, `git switch main`, `git pull`, `git stash drop`) and copy
  `profile.md` to it.
- `yarvis spotify login` on both computers; `yarvis key set spotify` on the PC.
- First live test of AssemblyAI; download Parakeet on the PC (`yarvis models download`).
- Optional: zero retention at AssemblyAI (email data-opt-out@assemblyai.com); a payment
  method for 100 sessions a minute.
- Approve (or not) the change to CLAUDE.md security rule 4 for Google's loopback login
  (task 16).
- Register the Microsoft public client app for task 16 (the PR gives the steps).

## Later (Phase 3 and beyond)

Mac and PC as one Yarvis (end-to-end encrypted sync), offline mode with a local model,
Dutch, the ESP32 desk satellite (Phase 4), v1.0 launch with signed releases (Phase 5),
then the cloud version.
