# 20. Cartesia's voice, with Kokoro as backup

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Yarvis has spoken with Kokoro (82M parameters, ONNX, Apache-2.0) on the user's own
computer. Victor prefers a voice from Cartesia, a cloud text-to-speech service.

Checked against Cartesia's documentation, pricing page and privacy policy on 19 November
2026:

- **Model:** `sonic-3.6` is the current one: 44 languages, and Cartesia claims about 90 ms
  to the first audio. The older sonic-2 and sonic-turbo models were retired on 20 October
  2026.
- **Connection:**
  - a WebSocket at `wss://api.cartesia.ai/tts/websocket`, with headers
    `Authorization: Bearer <key>` and `Cartesia-Version: 2026-08-14`;
  - one `context_id` per utterance;
  - raw `pcm_f32le` at 24 kHz, the same format as Kokoro.
- **Sending whole sentences:** `max_buffer_delay_ms: 0` must be set; the default of 3,000
  ms adds up to 3 s of delay.
- **Cancelling** only stops audio that hasn't started yet.
- **Idle connections** are closed after 5 minutes.
- **Price:** about one credit per character. The Free plan has 20,000 credits a month and
  2 requests at a time; Pro is $5 a month for 100,000, then $65 per million.
- **Privacy:**
  - Cartesia may use the text it receives to train its models unless the user opts out;
    the opt-out applies from then on.
  - Its service is "designed for users in the United States".
  - No-retention and EU servers are Enterprise only.
  - All Sonic 3.6 audio carries an inaudible AI watermark.

## Decision

1. **Cartesia speaks when chosen** (`voice.tts = "cartesia"`, default stays `"kokoro"`).
   The key lives only in the keychain (`yarvis key set cartesia`), is sent only in the
   `Authorization` header (never in the address), and is removed from logs (`sk_car_…`).
2. **Default voice: Archie** (`ef191366-f52f-447a-a398-ed8c0f2943a1`), British English
   male, one of Cartesia's recommended voices for Sonic 3.6 (Victor's choice). Any voice id
   can be set in `voice.cartesia_voice`. The model is `voice.cartesia_model`
   (`"sonic-3.6"`), and `voice.speech_rate` becomes Cartesia's speed, limited to 0.6–1.5.
3. **Sentence by sentence, over one connection.**
   - Each sentence Claude finishes goes out at once as its own context, with
     `max_buffer_delay_ms: 0`, and comes back as 24 kHz audio. The speaker and the
     pipeline don't change.
   - The connection stays open between sentences. When Cartesia has closed it after five
     idle minutes, it is opened once more.
   - When the user interrupts, the sentence is cancelled and the connection closed, so no
     late audio can reach the next sentence. The next sentence opens a new one.
4. **Kokoro takes over, sentence by sentence** (Victor's choice), so Yarvis never goes
   silent:
   - no connection, a lost connection, an error, an unreadable answer, incomplete audio,
     no audio within 2 seconds, a sentence not finished within 20 seconds, or more than
     16 MB of audio: Kokoro speaks that sentence, and the connection is closed;
   - a refused key (HTTP 401/402/403) or used-up credits (`quota_exceeded`), even in an
     error for the whole connection: Cartesia is switched off until Yarvis restarts.

   The dashboard's Voice line shows "Cartesia · Kokoro as backup", and a warning or error
   with the reason when Kokoro has taken over. Kokoro is loaded at start, so taking over
   costs no time.
5. **Only the sentence leaves the computer.** Never the profile, memory, history or what
   the user said. The text is never logged, and audio is never written to disk.
   Cartesia's messages are validated (pydantic) before use.
6. **Cost:**
   - `usage.json` keeps two more numbers per day: `voice_chars` and `voice_usd`.
   - `voice.cartesia_usd_per_million_chars` is the plan's price: 0 for the Free plan (the
     default, Victor's plan), 65 on Pro.
7. **Privacy steps:** `yarvis key set cartesia` and `yarvis doctor` both point to
   Cartesia's training opt-out (play.cartesia.ai/privacy/data-controls). Victor has
   submitted it. DESIGN.md's privacy table states what is sent and kept.
8. **No new dependency:** the WebSocket uses `websockets`, already used for AssemblyAI;
   Cartesia's SDK isn't needed.
9. **Straight to Cartesia:** the connection ignores any system proxy (`proxy=None`), so
   the key and the text go only to `api.cartesia.ai` over TLS.

## Consequences

- Yarvis sounds the way Victor prefers. Cartesia is reached only while Yarvis speaks, one
  sentence at a time.
- **What changes for privacy:**
  - What Yarvis says, which can include calendar, email and memory content, now leaves the
    computer, to a US service.
  - Cartesia may keep it, and train on it unless the user opted out. Kokoro stays the
    private default.
- **Cost:** a typical spoken answer is 100–300 characters, so the Free plan's 20,000
  credits a month last roughly 70–200 answers. After that Cartesia answers
  `quota_exceeded` and Kokoro takes over for the rest of the session.
- **Possible double billing:** when Cartesia's audio arrives too late or incomplete, Kokoro
  speaks the sentence, but Cartesia may still count its characters. Yarvis counts only
  sentences Cartesia spoke, so `yarvis usage` can be slightly below Cartesia's own count.
- **Behind a proxy:** a network that only allows traffic through a proxy can't reach
  Cartesia; Kokoro speaks instead.
- **Latency:** Cartesia's first audio is a network round trip plus about 90 ms (Cartesia's
  claim; not yet measured here). Kokoro on this computer starts in about 100 ms.
- **Not verified:**
  - whether Cartesia also accepts the key in an `X-API-Key` header (the Bearer header is
    its documented convention and what its SDK sends);
  - whether stock voice ids never change.
