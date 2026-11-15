# 8. Parakeet for speech recognition on Apple Silicon

- **Status:** Accepted
- **Date:** 2026-11-15

## Context

The first real user reported that Yarvis misunderstood "nearly half" of what he said:
"play We Go On on Spotify" became "spot five" and the wrong song. Yarvis used Whisper
`base.en` on the CPU, chosen for speed (about 0.3 s per request on his Mac). It is small,
it struggles with accents (he is Belgian), with AirPods' Bluetooth microphone audio, and
with names.

Whisper models accurate enough to fix this (`small.en`, `large-v3-turbo`) are slow on a
CPU because Whisper pads every utterance to 30 seconds: a short command costs as much as
a long one. Turbo on the Apple GPU (MLX) takes about 0.7 s on an M1 Max.

## Decision

1. **NVIDIA Parakeet TDT 0.6B v2, on the Apple GPU through MLX, is the default on Apple
   Silicon** (`stt = "auto"`). On the Hugging Face Open ASR leaderboard it gets 6.05 % of
   words wrong on average, against 7.83 % for Whisper `large-v3-turbo` (and more for the
   small Whisper models). Its cost grows with the length of the audio instead of a fixed
   30 s window, so short commands take a fraction of a second, comparable to before.
2. **Downloaded and verified by Yarvis**, like the voice and wake-word models: from URLs
   pinned to one Hugging Face commit, weights checked by SHA-256, the small
   `config.json` checked against its git object id (all Hugging Face publishes for files
   kept outside LFS). The weights are safetensors, never pickle. Loading never touches
   the network.
3. **One thread for MLX.** MLX arrays belong to the thread whose stream created them, so
   loading and every transcription run on a single dedicated worker thread.
4. **Whisper stays** for Linux, Intel Macs, as the fallback if Parakeet fails to load
   (Yarvis logs why and carries on with Whisper rather than going deaf), and by choice
   (`stt = "faster-whisper"`). Its default becomes `small.en` with beam search, and its
   prompt always includes the names of what Yarvis controls (Spotify, Brave, …).
5. **The model is told it hears through speech recognition**: read requests by how they
   sound, act on the most likely meaning, and say exactly what was done so a mistake is
   heard and can be corrected.

## Consequences

- A 2.5 GB one-time download on Apple Silicon (asked first; declining keeps Whisper).
- Parakeet takes no vocabulary hints. "Jarvis" for "Yarvis" is already handled by the
  wake-phrase stripping; the `vocabulary` setting now only applies to Whisper.
- `parakeet-mlx` and MLX are installed only on Apple Silicon Macs (environment markers),
  so Linux installs are unchanged. MLX needs macOS 14, which Yarvis already requires.
- Parakeet v3 speaks 25 European languages, including Dutch: the natural path for the
  multilingual phase.
- `yarvis bench` measures Parakeet next to Whisper on the CPU and GPU, with what each
  heard, so the choice can be checked on any Mac.
