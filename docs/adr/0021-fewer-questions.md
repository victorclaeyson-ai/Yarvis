# 21. Fewer questions: new memories shown with Undo, links on the user's say

- **Status:** Accepted (changes point 5 of ADR 0017 for remembering, and the link rule of
  ADR 0010 when the user turns it off)
- **Date:** 2026-11-19

## Context

Victor found the questions "really annoying". The gate log since 16 November shows 12
questions in all:
- 8 were `memory_remember` in a marked conversation;
- 3 were opening a website after a search;
- 1 was quitting an app.

Since ADR 0018, outside content marks the whole conversation. Weather answers, Spotify song
titles and web searches all count as outside content, so almost every conversation was
marked. In a marked conversation, these asked:
- **every new memory** (ADR 0017, threat T7): Yarvis also keeps facts unasked
  (`memory.auto`), so "Should I remember that…?" came up all the time;
- **every web address opened** (ADR 0010), because a link chosen after reading a page or
  email can carry what was read to a stranger's site (`evil.example/?notes=…`).

What a planted memory can do is limited:
- Memory is wrapped as data, capped (ADR 0017), and can't lower a tier or skip a
  confirmation: `test_a_remembered_instruction_lowers_no_tier` proves it.
- What remains is a misleading "fact", which the user can see and remove.

A link opened without a question is a real way out for data. Victor decided he accepts that
risk for himself.

## Decision

1. **Remembering never asks.** `memory_remember` no longer sets `asks_when_tainted`. It
   stays tier 1.
2. **Every kept fact is shown, with Undo.**
   - The tool calls `on_kept(fact)`, the dashboard receives `remembered(id, text)`, and the
     window shows "remembered: “…” UNDO".
   - Undo calls `AppApi.forget_fact(id)` from the user's click; the model is not involved.
     Code checks that the id is a whole number from 1 to 1,000,000.
   - The answer is "forgotten", "gone" (already forgotten) or "failed" (the file was busy
     or couldn't be saved). After "failed" the button stays and the window says to try
     again or use `yarvis memory forget N`, so the user never believes a fact is gone when
     it isn't.
   - If the window can't show a fact, the tool result says so. The model then doesn't
     think the write failed and try again.
   - In the terminal, the activity line shows the fact; "forget that" or
     `yarvis memory forget` undo it.
3. **Forgetting still asks after outside content.** Otherwise a planted instruction could
   erase facts the user cares about ("is allergic to peanuts") without anyone noticing.
4. **`security.confirm_links`**, on by default. When the user turns it off, opening a web
   address doesn't ask after outside content:
   - It covers only tools that set `opens_links`: `open_website` (macOS, Windows, Linux)
     and the browser's new-tab tool. The address opens in the user's browser, and the
     window's activity line shows its site, path and query, cut to the first 60
     characters.
   - Everything else that sends data out still asks after outside content: MCP calls,
     because a server could send the data anywhere, and every external (tier 2) action,
     such as mail or calendar events.
   - Only the user's own config can change it. The default stays on, for anyone else who
     installs Yarvis.
5. **Victor's own config:** `confirm_tier2 = false` and `confirm_links = false`. External
   actions still ask after outside content.

## Consequences

- Most questions are gone. Victor's log would have asked 1 of the 12 times; that one
  question was quitting an app, which no longer asks with `confirm_tier2 = false`.
- A page, email or song title can now add a misleading fact without a question. It shows
  at once with Undo and can never unlock a tool or skip a confirmation. The injection
  suite checks both for every source of outside content
  (`test_a_memory_planted_by_outside_content_is_shown_and_unlocks_nothing`).
- With `confirm_links = false`, outside content can make Yarvis open an address that
  carries what it has read, without a question. It is visible in the window (the first
  60 characters, so a long payload is not shown in full) and in the browser, but only
  afterwards.
  - `test_without_link_questions_an_address_opens_but_mail_still_asks` shows the limit:
    sending mail still asks.
  - `test_the_link_setting_reaches_the_gate` proves the setting is what decides.
- **Not changed, and not covered by this setting:** `browser_search` puts the model's
  words into a Google or Brave search and has never asked after outside content, whatever
  the setting. A poisoned page could make Yarvis search for something it read. The risk is
  low, because the data goes to a search engine rather than a site the attacker owns, but
  the question should be added in a later change.
- The window gains one method callable from the page, `forget_fact`. It only forgets; it
  cannot add or change a fact. The API-surface test lists it.
- A failed memory read is now tried again next time, instead of hiding memory until the
  file changes. Undo from the window makes two processes writing the file a normal case.
