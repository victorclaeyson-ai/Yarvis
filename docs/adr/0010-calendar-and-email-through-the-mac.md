# 10. Calendar and email through the Mac's own accounts

- **Status:** Accepted
- **Date:** 2026-11-15

## Context

Calendar and email are what people ask an assistant for most, after timers and music. The
first user has two Gmail accounts, Outlook.com and a university Microsoft 365 account, with
calendars in Google and Outlook. Ways to reach them:

- **Google and Microsoft APIs directly.** Each needs a developer project (a Google Cloud
  project, a Microsoft app registration). Google expires a personal app's logins every
  7 days while it is in "testing", and leaving testing with Gmail access needs a paid
  security assessment. Every user of an open-source project would have to repeat this.
- **Third-party MCP servers** for Gmail, Google Calendar and Microsoft 365: the same
  developer setup, plus trusting outside code with a whole mailbox.
- **The Mac's Internet Accounts.** macOS already signs in to Google, Microsoft (including
  university tenants) and iCloud, keeps the tokens in its keychain, and exposes calendars
  through EventKit and email through the Mail app. Setup is a few clicks in System Settings.

Email is also the classic way in for prompt injection: anyone can put text in front of
the model by sending the user an email or an invitation.

## Decision

1. **Calendars through EventKit** (PyObjC), on one dedicated thread. Every calendar on the
   Mac is read the same way, repeating events expanded. The first use shows macOS's own
   dialog asking to let the terminal see the calendars; Yarvis never sees a password or a
   token. `calendar_events` is read-only; `calendar_add` needs the user's "yes" (an event
   may land in a shared calendar).
2. **Email through the Mail app**, driven by fixed AppleScripts that receive every value
   as `argv` data; names through Contacts. `mail_list` numbers the latest emails across
   all accounts and `mail_read` reads one by its number: the model never makes up an id.
   `mail_send` and `mail_reply` need the user's "yes". A reply is sent as a new email with
   "Re:" from the account that received the original (Mail's own reply command does not
   reliably keep text set without opening a window); mail apps usually thread it by
   subject. The original is not quoted.
3. **Resolve before asking** (`Tool.prepare`). The confirmation question names exactly what
   will run: the contact's address ("Should I email Thomas Peeters (thomas@example.com),
   from victor@example.com, ...?"), the calendar ("Home (victor@gmail.com)"), the email
   being answered. The resolved arguments are what the "yes" runs. A reply's recipient is
   taken from the listed email, never from the model (those fields are hidden from it and
   overwritten). Names from Contacts are outside text too, so they are cleaned to letters
   and simple punctuation before being said.
4. **What was read is outside content.** Emails, invitations and contacts taint the turn
   like a web page. Anything external in that turn needs a spoken "yes", even with
   confirmations turned off.
5. **No quiet way out.** Private data plus outside content plus a way to send it out is
   the dangerous combination. A new gate rule: after reading outside content, a call that
   can carry text to a place the model chose (opening a web address) is asked about first,
   even though opening a website is otherwise harmless. The question reads the address
   with its path and query ("open evil.example/c?notes=..."), because that is where data
   would hide.
6. **Tools may take longer when they need to** (`Tool.timeout_s`): Mail may have to start,
   and the first calendar call may wait for the user to answer macOS's dialog.

## Consequences

- One-time setup: add the accounts in System Settings → Internet Accounts, and answer
  macOS's questions (calendar access, and controlling Mail and Contacts) on first use.
- The Mail app downloads the user's email (it can be several gigabytes for Gmail) and opens
  when Yarvis uses it.
- Reading email sends its text to Claude, like everything else in the conversation. The
  README says so.
- Linux has neither tool. Other services (and Linux) will come through the MCP client.
- The injection suite now plants its attack in an email, an email subject and a calendar
  invitation as well, and checks that an email can neither send mail unheard nor smuggle
  data out through a web address.
