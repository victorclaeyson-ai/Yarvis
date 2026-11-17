# 15. A dashboard: every connection, what it costs, how fast it answers

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

Yarvis now talks to several services (Claude, Claude's web search, AssemblyAI, Spotify,
MCP servers), and each of them can be on, off, half set up or failing. The window only
showed the model names, and the terminal log was the only place to see that AssemblyAI
had refused a key or an MCP server had not started. Nobody knew what a day of talking
to Yarvis cost, and the speed claim ("it answers fast") had no published number behind
it.

## Decision

1. **A heads-up display.** The window becomes a HUD, designed first as a mockup: the
   core in the middle with the conversation under it, a column on the left for the
   connections, hearing and the permission gate, and one on the right for the weather at
   home, timers, today's cost and the speed. It opens at 1280×800; below 1024 pixels wide
   it turns into the compact window it used to be, with the connections and numbers
   behind a button. It uses the computer's own fonts, so the page still loads nothing.
2. **Connections.** Every connection (Claude, speech recognition, voice, wake word,
   Spotify, calendar and mail, weather, web search, skills, profile, each MCP server)
   has a line with a dot: on, off, a warning or an error, and what it uses or what to
   run to fix it. Claude and AssemblyAI report their health as they go (an error, a turn
   this computer had to take over, back to normal), so the dashboard shows the moment
   something breaks.
3. **Cost and speed.** Under the core: the time to the first word of the last reply,
   the median (p50) of today's spoken turns with a small chart of the latest ones, the
   cost of the last turn and of today. The side panels add the details: tokens,
   searches, minutes of streamed speech and the cost of each, p50 and p90.
4. **Counted where the money goes.** The Claude client counts every request, including
   drafts started at a pause and dropped when the user went on (they are billed too); a
   request cut off mid-stream is estimated from what arrived. AssemblyAI sessions count
   by their length, as AssemblyAI bills them. Prices are list prices in US dollars,
   dated in the code (`usage.PRICES_AS_OF`); a model without a known price is flagged
   rather than guessed.
5. **Only numbers are kept.** One line per day in `usage.json` in the private data
   folder, for a year: tokens, searches, seconds, dollars and the latency of each spoken
   turn. Never what was said. `yarvis usage` shows the last days and the p50/p90;
   `yarvis usage clear` deletes it.
6. **Timers and weather.** Running timers and reminders count down on the dashboard
   (the timer list tells it of every change). With a `location` in the config, the
   dashboard asks Open-Meteo for that town's weather when Yarvis starts and twice an hour
   after: the town's name and coordinates leave the computer, as they already did when
   the user asked about the weather.
7. **Data, not markup.** The page receives every number, name and detail as JSON and
   inserts it with `textContent`, like everything else. Connection details may quote an
   error message, so they pass the secret redactor and are cut to 140 characters first.

## Consequences

- Problems are visible without reading logs, and each one says how to fix it.
- The voice-to-voice median can be published from real use (`yarvis usage`), not from
  a benchmark.
- The cost is an estimate: discounts, price changes and the provider's own rounding
  make the invoice differ. The dashboard and `yarvis usage` say so.
- Prices are code: when a provider changes them, `usage.py` changes with a date. A
  newer model is unpriced (and flagged) until it is added.
- `usage.json` is written at the end of each turn, a few hundred bytes. Two Yarvis
  processes add to each other's numbers instead of overwriting them.
