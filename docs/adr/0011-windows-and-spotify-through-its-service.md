# 11. Windows, and Spotify through its own service

- **Status:** Accepted
- **Date:** 2026-11-16

## Context

The first user's most powerful computer is a Windows PC (with an NVIDIA RTX 4070), and many
people who might try Yarvis use Windows. Yarvis was built for macOS: everything that acts on
the computer goes through AppleScript.

On the Mac, a related problem: telling the Spotify app to play a track (AppleScript `play
track`) makes Spotify bring itself to the front, even without `activate`. Giving the focus
back afterwards (ADR 0007 §9) is a workaround that depends on timing and on System Events.

## Decision

1. **The core runs on Windows as it is.** Speech (Whisper on the processor), the voice, the
   wake word, the app window, timers, the weather and web search needed no changes. `yarvis
   doctor` reports Windows as early support instead of a failure. Commands run with a
   scrubbed environment (no API keys) built per platform, and without a console window.
2. **Windows tools**, with the same rules as on the Mac:
   - apps are found among the Start menu's apps (`Get-StartApps`) and opened by the ID
     Windows gave them, so a name from the model can never be a path or a command;
   - PowerShell only runs fixed commands; the one value put into a command is a process ID
     Windows itself listed, formatted as a number;
   - quitting an app asks its main window to close, as clicking ✕ would, after a "yes";
   - volume and play/pause/next/previous use the volume and media keys, which reach
     whichever app is playing (2% per volume key, so the volume is set to the nearest even
     percent);
   - websites must be plain `https://` URLs.
   Browser control, typing, email and the calendar stay macOS-only for now.
3. **Spotify plays through its Web API** where the user allowed it, on both platforms.
   `yarvis spotify login` now also asks for `user-read-playback-state` and
   `user-modify-playback-state` (never the scopes that change the library). Yarvis plays on
   this computer's Spotify app, found among the user's devices by the computer's name (or
   the only computer listed), and starts it if it is not open: hidden on a Mac (`open -g
   -j`), minimized on Windows (`Spotify.exe --minimized`; the Store version cannot be).
   Nothing touches a window, so Spotify stays behind the user's app. Now playing, volume,
   shuffle, repeat and seeking use the same service.
4. **Falling back.** Without that permission (an older login), or when Spotify's service
   cannot be reached, a Mac tells the Spotify app as before; Windows explains how to give
   the permission. The model never sees tokens or device IDs.

## Consequences

- Users log in to Spotify once more to give the playback permission.
- Spotify's service needs the developer app's owner to have Premium (as searching already
  does since February 2026) and an internet connection; playing on another device (a phone,
  a speaker) is now a small step.
- Windows Smart App Control blocks uv's Python and the speech libraries; the README says how
  to proceed. CI does not run on Windows yet, so Windows code is kept thin and its logic is
  tested through fakes on Linux and macOS.
- Speech recognition: Whisper `small.en` on the processor misheard too much in real use
  ("Hey generous", "15W" for Kristine W). Windows and Linux now run the same Parakeet model
  as a Mac, through ONNX (see ADR 0008), with Whisper as the fallback.
