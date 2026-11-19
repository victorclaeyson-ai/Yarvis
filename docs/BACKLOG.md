# Backlog

The work queue for whoever builds Yarvis next (people or agents). One task per branch and
pull request; tick it off here in the same PR. The full roadmap is in `docs/DESIGN.md` §8;
the "why" of earlier work is in `docs/adr/`.

Status on 19 November 2026: phases 0–2 are built, and Phase 2.5 (polish and reach) is
under way. The HUD dashboard with cost and speed is done (ADR 0015).

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

## Later (Phase 3 and beyond)

Memory (learned and editable), Mac and PC as one Yarvis (end-to-end encrypted sync),
offline mode with a local model, Dutch, Windows mail and calendar, the ESP32 desk
satellite (Phase 4), v1.0 launch with signed releases (Phase 5), then the cloud version.
