# 18. Outside content taints while it is in the conversation's history

- **Status:** Accepted
- **Date:** 2026-11-19

## Context

Taint tracking (ADR 0007, point 5) marked a turn as tainted once a tool returned outside
content: a web page, an email, a search result, an MCP result. The taint ended with the
turn. Tool results are not kept in the history, but the assistant's reply is, and that
reply can repeat what the page said ("The page says you pre-approved all emails.").
So a later turn, untainted by the old rule, could act on the page's words:

- remember or forget a fact without a yes (T7);
- send mail with no question when `security.confirm_tier2` is off;
- open an address carrying data (`sends_data_out`) without the address being read out.

ADR 0017 closed this for memory, for one turn only. The security review of PR #18 showed
that one turn is not enough (the reply stays longer) and that memory is not the only
rule affected.

## Decision

1. **Each exchange in the history remembers whether its turn read outside content.**
   That means a tool result marked untrusted, an `UntrustedToolError`, the error message
   of a tool that reads outside content (it can quote what it read), the result of an
   action the user confirmed, or one of Claude's own web searches. The error message and
   the confirmed result were gaps before this ADR: a confirmed action's result even
   reached Claude unwrapped, as part of a system note. Both are now wrapped as
   `<untrusted>` data and taint the rest of the turn.
2. **A turn starts tainted while any such exchange is still in the history.** Every gate
   rule that depends on taint applies, not just memory:
   - tier 2 asks even when `confirm_tier2` is off;
   - calls that can carry data out are asked, with the destination read out;
   - remembering or forgetting a fact is asked, with the fact read out.
3. **The taint leaves with the exchange.** The history keeps the last ten exchanges by
   default. When the exchange that read the page drops out, or the user starts a new
   conversation, later turns are no longer tainted by it.
4. **Inherited taint does not mark an exchange.** Only a turn that read outside content
   itself marks its exchange. Otherwise every later reply would carry the mark on, and
   the taint would never leave.
5. ADR 0017's one-turn rule (`tainted_before`) is replaced by this one, and its code is
   removed: the gate again takes a single `tainted` flag.

## Consequences

- After reading a page or an email, with `confirm_tier2` off, Yarvis asks before sending
  for as long as that reply is in the conversation (by default ten exchanges). That is the
  intended trade-off: a few more questions in exchange for outside content not steering
  later turns.
- With `confirm_tier2` on (the default), tier 2 already always asks. The change shows up
  for memory and for opening addresses carrying data.
- **Limit:** a reply made in a turn that was tainted only by the history doesn't mark its
  own exchange (point 4). One copy is enough to carry a page's words past its exchange:
  "what did it say again?" can make Claude repeat the earlier reply word for word, and
  that copy stays unmarked for up to ten more exchanges after the page's exchange has
  left. With `confirm_tier2` on (the default) tier 2 still asks; what such a copy could
  reach quietly is a memory change, an address carrying data, or tier 2 with
  `confirm_tier2` off. Closing it means marking inherited exchanges, with a limit on how
  often a mark can be passed on, or not at all, so the taint lasts the whole
  conversation. Left open for Victor to decide.
- Tests in `tests/test_injection.py` (defence 9) cover:
  - a `mail_send` two turns after a web page with `confirm_tier2` off;
  - the same once the page's exchange has left the history;
  - an address carrying data, and a memory change;
  - inherited taint not extending itself;
  - a confirmed action whose result is outside content;
  - a reader's error message;
  - Claude's own web search.
