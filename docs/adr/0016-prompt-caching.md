# 16. Prompt caching for Claude

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Every request to Claude sends the same few thousand tokens before the question: the
tool definitions and the system prompt (rules, abilities, the profile). One spoken turn
is often several requests: a speculative draft (ADR 0005), then a request per tool
round. Claude can cache a prompt prefix and bill it at a tenth of the input price when
the next request starts with the same bytes. `usage.py` already priced cache writes and
reads, but Yarvis never asked for a cache.

The cache matches the request byte for byte from the start, in the order tools, system,
messages. The system prompt ended with the clock to the minute ("Context: it is ...
13:05 local time."), so a cached system prompt would have missed every minute.

Separately, the last tool round (after four rounds of tool calls) sent no tools at all
while the messages still held tool calls and results. The API refuses that with a 400.

## Decision

1. **Two cache markers, both with the default 5-minute lifetime.**
   - One on the last of Yarvis's own tools. Web search (Anthropic's server tool) comes
     after it, so turning web search on or off does not move the marker.
   - One on the stable part of the system prompt. It caches the tools and the system
     prompt together.
2. **The system prompt comes in two parts** (`llm.base.SystemPrompt`). The stable part
   holds the rules, the abilities and the profile. The volatile part holds the clock and
   the user's name and town, and is sent after the marker. Joined, the two parts are the
   same text Claude read before (now sent as two blocks). **Rule: anything that changes from request to request
   goes in the volatile part.** One such change in the stable part silently turns the
   cache off, without an error.
3. **The last tool round keeps the tools.** It sends the same tool definitions with
   `tool_choice` set to none, so Claude must answer in words (`allow_tool_calls=False`
   on the provider). Changing `tool_choice` leaves the cached tools and system prompt
   valid. A tool call that arrives anyway in that round is not run.
4. **No cache marker on the conversation.** It comes after the clock, which changes every
   minute. After ten exchanges the oldest are dropped from the front, which changes the
   prefix. Dropped drafts would pay for cache writes nobody reads. The spoken history is
   short (about 1,000 tokens), so on Haiku 5.5 the saving would be at most
   1,000 × $0.09 per million, about a hundredth of a cent per request.
5. **No 1-hour cache and no pre-warming for now.**
   - A 1-hour write costs twice the input price instead of 1.25 times. `usage.py` only
     knows the 5-minute price. Decide with real numbers from `yarvis usage`.
   - `warm_up()` runs on every speech onset, false triggers included. A billed cache
     write each time would cost more than it saves. The speculative draft already starts
     the real request, and writes the cache, while the user is still talking.
6. **The Claude client logs the token counts of every request at debug level**: input,
   cache write, cache read, output. Only numbers, so it is safe in a log. It shows
   whether the cache works (`[logging] level = "DEBUG"` in the config).

## Consequences

- **Break-even.** A 5-minute cache write costs 1.25 times the input price. A single
  question after more than five minutes of silence therefore costs slightly more than
  before: for a 4,000-token prefix on Haiku 5.5, $0.0005 instead of $0.0004. Caching pays
  as soon as the prefix is read again within five minutes: the tool rounds of a turn,
  the draft and the final request, follow-up questions. Ten questions within five
  minutes cost about $0.0009 instead of $0.0040. Each read restarts the five minutes.
- The cached prefix stays the same only while the configuration does. A profile edit, a
  new or changed skill (its list is in the `use_skill` description), or an MCP server
  that changes its tool definitions changes the prefix: the next request writes a new
  cache entry and Claude sees the change at once.
- Below a minimum prefix size nothing is cached, without an error. Anthropic's prompt
  caching documentation gives 512 tokens for Haiku 5.5 (older models need more: 4,096
  for Haiku 4.5), so a different `llm.model` can change this. The live check in the
  pull request confirms it with the debug log. The system prompt
  without tools is about 420 tokens, so a set-up without tools (and without a profile)
  is not cached. With tools, the system prompt alone is about 850 tokens before the tool
  definitions are counted.
- With caching on, web search adds its own 5-minute cache write after its results. That
  is Anthropic's behaviour, not a bug, and shows up as a cache write in the log.
- Another provider (Ollama, Phase 3) receives the same `SystemPrompt` and can use
  `.text` if it has no cache.
