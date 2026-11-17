# 12. AssemblyAI live speech recognition, with this computer as backup

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

Mishearing is the most visible failure of a voice assistant: a wrong name or a missing word
turns into a wrong answer, and every correction costs a whole turn. Parakeet (ADR 0008) is
the most accurate model that runs comfortably on a laptop, yet in daily use it still
mishears names, places and short requests ("Kristine", "Leuven", "next one"), on the Mac
and on the PC. The first user asked for transcripts that are as good as money can buy.

Hosted streaming models are now clearly ahead. AssemblyAI's Universal-3.6 Pro (October
2026) leads public streaming benchmarks on word error rate (about 2% on Coval's
September–October 2026 comparison) with a final transcript about 0.1 s after it is asked
for. It costs $0.45 per hour of open session; a personal assistant uses minutes a day.

## Decision

1. **AssemblyAI is optional, and on as soon as its key is stored.** `yarvis key set
   assemblyai` stores the key in the OS keychain (ADR 0003). `voice.cloud_stt = false`
   turns it off without deleting the key. Without a key nothing changes: speech is
   recognized on this computer, as before.
2. **Yarvis keeps deciding when you start and stop talking.** The endpointer, speculation
   at pauses (ADR 0005) and barge-in are unchanged. A new hook, `listen.AudioTap`,
   receives the utterance while it is spoken: `begin` (when speech starts, with its
   pre-roll), `feed` (every later frame), `end` (complete, or abandoned). The server's own
   turn detection is switched off (`min_turn_silence`/`max_turn_silence` of 10 s): two
   judges of "are you done?" would disagree. When Yarvis needs the words, at a pause or at
   the end of the turn, it sends `ForceEndpoint` and receives the formatted final
   transcript. A request covering audio that an earlier one already covers is not sent
   again (the end of a turn after a silent pause reuses the pause's transcript).
3. **One WebSocket session per utterance**, opened when speech starts (connecting happens
   while you talk) and closed right after the final transcript (sessions are billed by
   the second). The audio recorded before the session opened (the pre-roll and, after the
   wake word, the phrase itself) is sent at most 1 s at once and then at twice real time,
   because AssemblyAI refuses audio sent much faster than it is spoken. If it says so
   anyway, later sessions use a slower pace.
4. **This computer is always the backup.** The local engine (Parakeet, or Whisper) is
   loaded at startup as before. Any failure (no internet, the connection or the request
   taking too long, a server error, the free plan's limit of 5 new sessions a minute)
   makes *that* turn fall back to it, transcribing the same audio, with the reason in the
   log. A refused key or account turns AssemblyAI off until Yarvis restarts. Audio that
   was never streamed (the warm-up, benchmarks) always goes to the local engine.
5. **Accuracy aids**: the model is pinned (`universal-3-6-pro`) and the server's echo of
   it is checked; the API version is pinned; the session is English-only for now (Dutch
   arrives with Phase 3); key terms boost the words Yarvis must spell right: the
   vocabulary setting, app names, the user's name and location (at most 100, as
   AssemblyAI allows).

## Privacy and security

- **What is sent:** the audio of each request, from just before the endpointer hears you
  start talking until the end of your turn. Never the wake word wait, never anything
  while Yarvis talks, and never Yarvis's replies or the conversation. In the follow-up
  window, speech in the room that turns out not to be for Yarvis has been transcribed by
  AssemblyAI before it is dropped (with `stay_open_s`, too).
- **Where:** EU servers by default (`voice.cloud_region = "eu"`). AssemblyAI states it does
  not use data sent to its European servers for model training. Zero retention of
  streaming audio and transcripts requires opting out of training
  (data-opt-out@assemblyai.com, from the account's address); `"global"` uses the nearest
  servers instead.
- **The key** is only ever in the OS keychain and in the `Authorization` header of a TLS
  connection with a verified certificate, never in a URL, a file or a log line (it is
  registered with the redactor when read).
- **Trust:** the transcript is treated exactly like the local one: it is the user's
  request, so the permission gate (ADR 0007) applies unchanged. A compromised speech
  service could put words in the user's mouth, including the "yes" to a confirmation;
  this is the same trust one gives the microphone driver, and the reason this is opt-in.
- **Untrusted input:** server messages are parsed with a strict model (unknown fields and
  messages are ignored); error texts are only logged, truncated.

## Consequences

- Transcripts are much better: names, numbers and punctuation come out formatted, which
  also helps Claude. Latency stays the same or improves (the final transcript arrives
  ~0.1 s after the request, versus a full local transcription).
- Yarvis needs the internet for its best hearing, but never stops hearing without it.
- A new dependency, `websockets` (BSD-3-Clause), maintained and widely used.
- `yarvis doctor` shows which engine hears you; the log says when a turn fell back.
- Not done yet: keeping one session open for a whole conversation (fewer sessions on the
  free plan, no pre-roll to catch up), and sending Yarvis's question as context
  (`agent_context`) to help with short answers. Both are easy to add if needed.
