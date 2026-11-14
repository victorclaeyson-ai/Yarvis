# 4. Desktop app with pywebview, without a network port

- **Status:** Accepted
- **Date:** 2026-11-14

## Context

Talking to Yarvis through a terminal prompt works but is awkward, and it does not show
what the assistant is doing (listening, thinking, speaking). The options considered:

1. **A local web server + browser tab.** Easy to make attractive, but it opens a listening
   port. Any web page the user visits could then try to talk to it (CSRF, DNS rebinding),
   which needs tokens, origin and host checks — and contradicts design principle 4
   ("no listening sockets").
2. **A menu-bar app** (e.g. `rumps`). Always available, but macOS-only and visually minimal.
3. **A native window that renders HTML** (pywebview: WebKit on macOS, GTK/Qt on Linux).

## Decision

Use pywebview (option 3).

- The page is passed to the window as an HTML **string**, so pywebview starts no HTTP
  server: the app opens **no network port**.
- JavaScript reaches Python only through a three-method API (`hello`, `talk`, `send`)
  with validated arguments. Private attributes are not exposed.
- Python sends the page **JSON data only**, and the page inserts text with `textContent`,
  never as HTML, because model output is untrusted.
- A strict Content-Security-Policy (`default-src 'none'`, `connect-src 'none'`) blocks
  any remote resource or network request from the page. It must allow `'unsafe-eval'`,
  because pywebview generates its JavaScript API at runtime with `new Function()`; the
  first version omitted it and the talk button silently did nothing. The page itself never
  evaluates strings, which a static test checks.
- `private_mode` is on (no persistent web storage) and developer tools are off.

## Consequences

- One codebase for the UI on macOS and Linux; the terminal mode (`yarvis run`) remains.
- Linux users need a GUI backend (GTK or Qt) for `yarvis app`.
- Tests check the security properties of the page statically (CSP present, no
  `innerHTML`/`eval`, no remote URLs) and the bridge behaviour with fakes. An opt-in
  end-to-end test (`tests/test_gui_smoke.py`) drives the real window under Xvfb.
- Packaging as a double-clickable `.app` is a separate step before launch.
