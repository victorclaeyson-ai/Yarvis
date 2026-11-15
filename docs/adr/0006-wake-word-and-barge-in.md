# 6. Wake word, barge-in and a faster first word

- **Status:** Accepted
- **Date:** 2026-11-15

## Context

Pressing Space before every question is not how anyone imagines talking to Jarvis. Yarvis
also could not be interrupted except with the stop button, and on the user's MacBook the
voice still took 0.4–0.8 s to synthesize the first sentence.

Constraints: everything stays on the device, no heavy new dependencies, and the security
and privacy story must remain simple to explain. PortAudio offers no echo cancellation, so
on speakers the microphone hears Yarvis itself.

## Decision

1. **openWakeWord's "hey jarvis" model, run directly on ONNX Runtime.** Yarvis feeds
   80 ms chunks through the three upstream models (mel spectrogram → speech embedding →
   wake word) in about 150 lines (`speech/wakeword.py`), instead of depending on the
   `openwakeword` package, which pulls in scikit-learn, SciPy and TensorFlow Lite. ONNX
   Runtime is already a dependency. Measured on synthetic speech (four Kokoro voices):

   | Phrase | Score |
   |---|---|
   | "Hey Yarvis" / "Hey Yarvis, what time is it?" / "Okay Yarvis" | 0.99–1.00 |
   | "I'm Yarvis, your assistant." / "Yarvis here." | ≤ 0.02 |
   | "I was talking to Travis yesterday." / "Hey, are you there?" | ≤ 0.01 |

   Yarvis introducing itself does not wake it. Cost: ~0.7 ms of one CPU core per 32 ms
   frame (~2 %).
2. **Licence.** The openWakeWord code is Apache-2.0, but its pre-trained models are
   CC BY-NC-SA 4.0 (non-commercial). Yarvis does not redistribute them: they are
   downloaded at runtime from the upstream release, pinned by URL, size and SHA-256 like
   every other model, and the licence is stated before the download. This fits a free,
   open-source project. A commercial edition must ship its own model (openWakeWord's
   training pipeline makes a custom "Hey Yarvis" model feasible).
3. **One shared microphone.** With the wake word on, a single input stream stays open for
   the session, so the operating system's microphone indicator is on and truthful. Three
   readers take turns on it: the wake-word detector (idle), the endpointer (listening) and
   the barge-in monitor (replying). Hand-overs pass audio along instead of dropping it:
   - Detection lands 160–500 ms *after* the phrase (measured), so a request said in one
     breath would lose its first words. The listener therefore receives the last 2 s, and
     "Hey Yarvis" is cut from the transcript before it reaches Claude.
   - If the user only said "Hey Yarvis", Yarvis keeps listening without discarding what
     was said meanwhile; during the pause it waits up to `thinking_pause_ms`.
   - Audio recorded while nobody was reading (Yarvis's own voice) is flushed before a
     fresh start.

   Reading must be safe to cancel, because readers hand over by cancelling each other.
4. **Barge-in adapts to the hardware.** `barge_in = "auto"` (default) chooses:
   - **voice** on headphones: just start talking. Sustained speech (VAD ≥ 0.7 for 320 ms,
     so a cough or "mm-hm" is not enough) stops the reply, and the words that interrupted
     are kept as the start of the next request;
   - **wake word** on speakers: say "Hey Yarvis". The detector ignores Yarvis's own
     sentences, so it works without echo cancellation.

   Headphones are recognised by the output device's name ("AirPods", "Headphones", …).
   Unknown devices count as speakers: the cautious answer, because voice barge-in on
   speakers would make Yarvis interrupt itself. An interrupted reply is remembered up to
   where it was cut off.
5. **A shorter first chunk.** Kokoro's synthesis time is proportional to the length of the
   audio it produces, so the first chunk now ends at a comma from 12 characters (was 24,
   ADR 0005), and the system prompt asks Claude to lead with a short first sentence. On
   six typical replies the first chunk dropped from 3.9 s to 2.8 s of audio on average
   (−29 %), and to a third where a comma comes early ("Good question,"). Kokoro's int8
   model was measured too and rejected: three times slower on CPU.

6. **Conversations stay open, and Yarvis says hello.** (Added after the first real use.)
   - After a reply, the conversation stays open for `follow_up_s` = 15 s (was 5 s, which
     felt too short): no wake word needed. Noise or a cough does not close it early; only
     the time left counts. Saying "thanks", "that's all", "never mind" or "bye" closes it
     right after Yarvis answers.
   - A bare "Hey Yarvis" is answered with a short greeting ("Good evening, Victor.", later
     "Yes, Victor?") before listening. Greetings are fixed phrases, synthesized in the
     background at startup, so they play instantly, cost nothing, and never reach Claude
     or the conversation history. The user's name comes from `assistant.user_name`.
     Talking over the greeting skips it.

## Consequences

- Privacy: the microphone is open for the whole session when the wake word is on, but
  audio stays in memory for a few seconds at most, is never written to disk and never
  sent anywhere. The app shows a "HEY YARVIS" indicator; `wake_word = false` (or
  `--no-wake-word`) restores press-to-talk with the microphone open only while listening.
- False wake-ups are tuned with `wake_word_threshold` (0.5). A short chime confirms each
  wake-up (`wake_chime`).
- Future work: echo cancellation (the macOS voice-processing audio unit) would allow voice
  barge-in on speakers, and a custom-trained "Hey Yarvis" model would remove the
  non-commercial licence restriction.
