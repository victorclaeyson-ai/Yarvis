# 18. Outside content taints the rest of the conversation

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

1. **Each exchange in the history remembers whether its turn held outside content.**
   That means any of these:
   - a tool result marked untrusted, or an `UntrustedToolError`;
   - the error message of a tool that reads outside content (it can quote what it read);
   - the result of an action the user confirmed;
   - one of Claude's own web searches;
   - outside content carried forward from earlier in the conversation.

   The error message and the confirmed result were gaps before this ADR: a confirmed
   action's result even reached Claude unwrapped, as part of a system note. Both are now
   wrapped as `<untrusted>` data and taint the rest of the turn.
2. **A turn starts tainted while any such exchange is in the history.** Every gate rule
   that depends on taint applies, not just memory:
   - tier 2 asks even when `confirm_tier2` is off;
   - calls that can carry data out are asked, with the destination read out;
   - remembering or forgetting a fact is asked, with the fact read out.
3. **The mark carries forward for the whole conversation (Victor's decision).** A turn
   that started tainted marks its own exchange too. So a word-for-word repeat ("what did
   it say again?") can't carry a page's words past the page's own exchange unmarked. The
   taint lasts until the conversation ends.
4. **In return, conversations end.** The history, and with it the taint, is cleared:
   - after `conversation.idle_reset_minutes` of silence (default 10), at the next
     question;
   - when the user says "new conversation" ("Yarvis, new conversation", "let's start a
     new conversation"). This is recognised by code, not by the model, so outside content
     can't end a conversation, and only for a request the user actually made: a guess
     started at a pause in speech never ends one. Yarvis answers "Okay, a new
     conversation." without asking Claude;
   - with the **NEW** button above the conversation on the dashboard. It sends the same
     command (`AppApi.new_conversation`, the fifth method of the window's API).
5. ADR 0017's one-turn rule (`tainted_before`) is replaced by this one, and its code is
   removed: the gate again takes a single `tainted` flag.

## Consequences

- After reading a page or an email, with `confirm_tier2` off, Yarvis asks before sending
  and before changing memory for the rest of that conversation. The way out is ten quiet
  minutes or "new conversation". That is the intended trade-off: a few more questions in
  exchange for outside content never steering later turns.
- With `confirm_tier2` on (the default), tier 2 already always asks. The change shows up
  for memory and for opening addresses carrying data.
- A conversation is also forgotten after ten quiet minutes, so a question the next morning
  no longer has yesterday's context. Before, the last ten exchanges stayed until Yarvis
  was restarted. Memory (ADR 0017) is what carries facts across conversations.
- Tests in `tests/test_injection.py` (defence 9) cover:
  - a `mail_send` two turns after a web page with `confirm_tier2` off, asked;
  - the taint lasting after the page's own exchange has left the history, until "new
    conversation";
  - a word-for-word repeat keeping the mark;
  - an address carrying data, and a memory change;
  - a confirmed action whose result is outside content;
  - a reader's error message;
  - Claude's own web search.

  `tests/test_conversation.py` covers the ten quiet minutes, the spoken command (and
  sentences that only contain the words), and a guess at a pause never forgetting
  anything. `tests/test_app.py` covers the button.
