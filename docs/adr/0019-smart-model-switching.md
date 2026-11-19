# 19. Smart model switching: hard questions go to Opus 5.5

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Haiku 5.5 is Yarvis's voice: it gets the first word out fast and costs little. On hard
questions (a study plan, a comparison, a calculation in several steps) its answers are
weaker. Victor judged them "not that good" next to Opus 5.5. Backlog task 10 asked to
compare three ways of reaching a bigger model on first-word latency, cost and quality,
then build one.

Facts checked on 19 November 2026 (platform.claude.com):
- **Prices** per million tokens (input / output / cache read): Sonnet 5.5 $2 / $10 / $0.10;
  Opus 5.5 $4 / $20 / $0.20. They were already in `usage.py`; `PRICES_AS_OF` now says so.
- **Advisor tool** (`advisor_20260301`, beta header `advisor-tool-2026-03-01`): Haiku 5.5
  can consult Opus 5.5, Sonnet 5.5 or Opus 4.8. The advisor is billed at its own rates,
  its advice doesn't stream, and advice from 5.x models comes back encrypted.

## The measurement

`yarvis bench --switching` (`src/yarvis/model_bench.py`) ran 8 hard and 4 easy prompts
through seven setups, with a hard limit of $3. It read the key from the keychain itself
and kept the answers for review. It ran 76 requests, cost $0.65, and timed them on
Victor's PC. "First word" is when something could be spoken; for B it is the moment code
says a phrase.

| Approach | First word (hard, median) | Full answer (hard, median) | Cost, all its prompts | Bigger model used: hard / easy |
|---|---|---|---|---|
| Haiku 5.5 alone | 2.0 s | 2.5 s | $0.002 | 0/8 · 0/4 |
| A: Haiku + advisor Opus 5.5 | 0.8 s | 10.9 s | $0.221 | 8/8 · 0/4 |
| A: Haiku + advisor Opus 4.8 | 0.9 s | 9.3 s | $0.199 | 8/8 · 0/4 |
| B: re-run on Sonnet 5.5 | 0.9 s | 4.4 s | $0.038 | 6/8 · 0/4 |
| B: re-run on Opus 5.5 | 0.9 s | 5.2 s | $0.066 | 6/8 · 0/4 |
| C: think hard → Sonnet 5.5 | 2.7 s | 4.1 s | $0.042 | 8/8 · — |
| C: think hard → Opus 5.5 | 3.1 s | 4.6 s | $0.077 | 8/8 · — |

- **Correctness:** all seven got the three questions with one right answer right:
  savings, the bat and the ball, and weekdays.
- **Victor's judgement:** reading the open answers side by side, he rated **Opus 5.5 the
  best** and Haiku alone weak.
- **The advisor (A)** was the slowest and the most expensive. Its "Let me think about
  that." before every hard answer was also unwelcome.
- **The hand-over (B)** worked: Haiku asked for help on 6 of 8 hard prompts and on none
  of the easy ones.
- **Formatting:** Opus 5.5 writes markdown bold despite the voice prompt. Speech already
  strips markdown (`speech/sentences.to_speakable`), so it is never read out.

## Decision

1. **Opus 5.5 answers hard questions** (`llm.big_model = "claude-opus-5-5"`). Haiku 5.5
   stays the voice for everything else.
2. **Two ways in** (`llm.switching = "auto"`, the default):
   - **Haiku hands over:** it is offered a `think_harder` tool, and when it calls it,
     code re-runs the round on Opus with the same messages, without Haiku's call. Opus
     isn't offered `think_harder`.
   - **The user asks:** a request that starts with "think hard", "think carefully",
     "think it through" or "take your time" goes to Opus from the start. Code recognises
     the phrase, and only at the start of the user's own request, after a name followed
     by a comma ("Yarvis, think hard …").

   `"asked"` keeps only the second way; `"off"` turns both off.
3. **A short, varied phrase** fills the pause: "One moment.", "Let me see.", "Hmm, let me
   work that out.", "Give me a second." or "Good question, one moment.". Code picks it,
   never the same one twice in a row, the moment Opus takes over. Victor chose this over
   a sound or silence.
4. **A daily cap, enforced in code:** `llm.big_model_daily_usd`, default $0.50.
   - `usage.json` keeps one more number per day, `claude_big_usd`: the part of Claude's
     cost spent by the bigger model's client. It is counted by which client made the
     request, never by the model name in the answer, so an unusual id can't slip past.
     The file stays numbers only.
   - If the bigger model has no known price, its cap closes for the day (fail closed)
     and the log says why.
   - Before every switch, code checks today's spend. Over the cap, a hand-over is
     declined and Haiku answers itself, silently.
   - The check comes before the call and the cost is known after it, so the last call
     of the day can take the total a few cents over: the cap is "about $0.50".
   - A "think hard" over the cap gets one spoken note a day ("I've used today's budget
     for thinking harder, so here's my quick answer.").
   - `think_harder` stays offered over the cap, so the tools, and Claude's prompt cache,
     don't change.
5. **No money on guesses.** Yarvis starts answering during a pause in speech, as a guess
   that may be thrown away. Opus is only asked once the request is the user's real one.
6. **`think_harder` is handled by code, not by the toolbox or the gate.** It isn't an
   action: it only chooses which model answers, and the budget check is in code. It
   counts only when it is Haiku's only call; mixed with other calls, those run through
   the gate as usual and Haiku is told to hand over on its own. If Haiku already said
   something, no phrase is added. Opus reads the same system prompt but isn't offered the
   tool; if it calls it anyway, it is told to answer itself.
7. **If Opus fails before saying anything** (overloaded, rate-limited), Haiku answers the
   question in the same turn and no hand-over is offered again in that turn. A failure
   after Opus started speaking is reported like any other.
8. **The bigger model runs inside the same turn:**
   - the same system prompt and the same toolbox;
   - the same permission gate and the same taint (ADR 0018);
   - 4,096 output tokens instead of the voice's 1,024: this is `max_tokens`, which
     includes Opus 5.5's thinking. It is not a thinking budget; Opus 5.5 always thinks
     adaptively and rejects a fixed budget;
   - its own client, warmed up at start, with its own prompt cache.

   When Opus thinks and then calls a tool, the API needs that assistant turn's thinking
   blocks back, signed and unchanged, with the tool results; the same holds for a
   `pause_turn` continuation. The conversation keeps only text and tool calls, but the
   Claude client keeps the raw blocks of any turn that holds more than that (thinking,
   web search) and sends them back exactly (`AnthropicProvider._keep`, built for web
   search, ADR 0009). Tests use a fake API that rejects a request missing them. The last
   tool round's `tool_choice: none` is allowed with thinking; only forcing a tool (`any`,
   `tool`) is refused on Opus 5.5.

## Consequences

- A hard question gets a spoken phrase within about a second and Opus's answer after
  about five, instead of Haiku's weaker answer after two and a half.
- **What one Opus answer costs** (prices of 19 November 2026; Opus 5.5 per million
  tokens: $4 input, $5 cache write, $0.20 cache read, $20 output):
  - **Thinking and the answer:** 383 output tokens in the bench's median (at most 559),
    about $0.008 (at most $0.011).
  - **The prefix:** the tool definitions and the cached part of the system prompt. On a
    Mac with every tool on it is about 6,000 tokens (37 tool definitions, about 5,100
    tokens; rules and abilities, about 850; plus the profile and memory), counted at four
    characters per token. Opus has its own prompt cache:
    - the first Opus answer in five minutes **writes** it, about 6,000 × $5 = $0.030;
    - later ones within five minutes **read** it, about $0.001.
  - **The rest of the request:** the conversation, the question and the clock, a few
    hundred to a thousand tokens at $4, $0.001–0.004. Haiku's hand-over call costs a
    fraction of a cent.
  - **Total:** about **$0.04** for a hard question on its own, and about **$0.011** for
    one asked within five minutes of the last.
- **What the cap allows:** with $0.50 a day, about **12** hard questions spread over the
  day, and up to about **45** asked in bursts. On the PC, with fewer tools, the prefix is
  smaller and the numbers are a little higher. The bench's own cost ($0.010 per answer)
  sent no tools and no cache, so it left the prefix out.
- The advisor tool isn't used, so nothing depends on its beta API.
- Haiku decides what counts as hard; a miss either way is cheap (Haiku answers, or Opus
  spends a few cents). "Think hard" always works when the user wants Opus, until the cap.
- The cap is also what limits outside content: a web page or an email can try to make
  Haiku call `think_harder` again and again. That only costs money, never acts (the gate
  and the taint are unchanged), and the cap ends it for the day.
- The bench stays in the repo: `yarvis bench --switching --budget 3` repeats the
  comparison when models or prices change.
