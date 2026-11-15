# 5. Speculative turn-taking

- **Status:** Accepted
- **Date:** 2026-11-14

## Context

The first measurements on a MacBook Air showed 2.4–3.7 s from the end of the user's
sentence to Yarvis's first word. Every stage ran strictly after the previous one:

```
end-of-turn wait (0.70) → speech-to-text (0.42) → Claude first token (0.78)
  → wait for a full sentence (~0.5) → voice (0.3)
```

The end-of-turn wait is necessary (otherwise Yarvis interrupts people who pause to think),
but nothing useful happened during it.

## Decision

1. **Speculate during pauses.** The endpointer reports a `PAUSE` after 224 ms of silence and
   a `RESUME` if speech continues. At a pause, a `Speculation` transcribes the audio so far
   and starts a Claude **draft**. At end of turn the speculation is used as-is; on `RESUME`
   it is discarded.
2. **Drafts are not history.** A `Draft` generates immediately but is only remembered once
   consumed. Discarded drafts never enter the conversation, so the model never sees a
   question the user did not finish. Interrupted replies remember what was actually heard.
3. **Speak from the first clause.** The first chunk sent to the voice ends at the first
   comma/colon/dash once it is at least 24 characters long (12 since
   [ADR 0006](0006-wake-word-and-barge-in.md)).
4. **Warm everything.** Each engine runs once at startup; the HTTPS connection to Claude is
   kept alive for 5 minutes (the client default was 5 seconds) and refreshed when the user
   starts speaking.
5. **Wait longer when the sentence sounds unfinished.** The speculative transcript is ready
   a few hundred milliseconds into the pause. If it ends in a way sentences rarely end
   ("and", "the", "um", a trailing comma or "…"), the endpointer waits up to
   `thinking_pause_ms` (2 s) instead of `end_of_turn_ms` (0.9 s). The heuristic is
   deliberately conservative: when in doubt, answer.
6. **Follow-up listening.** After a spoken reply, Yarvis listens for `follow_up_s` (5 s;
   15 s since [ADR 0006](0006-wake-word-and-barge-in.md)) without a key press, so a
   conversation flows; silence ends it quietly.
7. **Measure, don't guess.** `yarvis bench` measures every stage on the user's machine,
   including Whisper on the Apple GPU (`stt_device = "mps"`, experimental, with automatic
   fallback to the CPU).

## Consequences

- With speculation, the end-of-turn wait is mostly hidden: latency becomes roughly
  `pause + speech-to-text + first token + first clause + voice`. That is why the base wait
  could be raised from 0.7 s to 0.9 s at almost no cost: the first version cut people off
  mid-thought.
- A speculation that is discarded costs a few input tokens and a little CPU. Users can turn
  it off with `voice.speculative = false`.
- Timing fields changed meaning: `stt_ms` and `llm_ms` are now measured from when each stage
  actually started, which may be before the end of the turn.
