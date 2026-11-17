# 14. An "about me" profile, known at every request

- **Status:** Accepted
- **Date:** 2026-11-17

## Context

Yarvis knew the user's name and town (`assistant.user_name`, `location`) and nothing
else, so every answer was written for a stranger. Assistants assembled around Claude Code
solve this with a `CLAUDE.md`: a file the user writes about their role, projects,
priorities, tone and rules, read at every request (Cindy Zhu's "Build your own Jarvis
with Claude" guide is a good example). Real memory, which Yarvis learns by itself, is
Phase 3; a file the user writes is the simple, predictable first step.

## Decision

1. **`profile.md`, next to `config.toml`,** in plain Markdown. `yarvis profile edit`
   creates it from a short template (who I am, what I'm working on, how to talk to me, my
   rules) and opens it in the computer's text editor; `yarvis profile show` prints exactly
   what is sent; `yarvis doctor` checks it.
2. **Read before every request**, again only when the file changed: an edit counts from
   the next question, without a restart.
3. **Only what was written is sent.** `<!-- comments -->`, the template's empty bullet
   points and headings with nothing under them are left out.
4. **Trusted, but not powerful.** The user wrote it, so it goes into the system prompt as
   their own words, not as outside content. The prompt says it never switches off the
   confirmations and permissions, and it could not anyway: those are decided by the
   permission gate in code (ADR 0007), whatever the model is told.
5. **Guard rails.** It is sent to the language model with every request, so a profile
   that looks like it holds an API key or token is not used at all, and neither is one
   over 6,000 characters (about 1,500 tokens). Either way Yarvis keeps working without it
   and says why in the log, in `yarvis profile show` and in `yarvis doctor`.

## Consequences

- Answers fit the user: their projects, their priorities, their tone, their rules.
- The profile is sent to Anthropic with every request (as the config's name and town
  already were). The privacy section of the design doc says so.
- It costs input tokens on every request; the size limit keeps that small.
- Phase 3 memory can build on it: facts Yarvis learns can be shown and kept in the same
  way, under the user's control.
