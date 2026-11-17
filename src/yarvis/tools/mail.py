"""Email, through the Mac's Mail app (and names, through Contacts).

Every account added to the Mac (Gmail, Outlook, a university account) is read and written
the same way. macOS signs in and keeps the tokens: Yarvis never sees a password or a
token, and needs no developer account at Google or Microsoft.

Emails are the classic way in for prompt injection, so:

* everything read from Mail or Contacts is outside content (the turn is tainted);
* sending, replying and trashing always need the user's "yes", and the question is built
  by code from what was resolved ("Should I email Thomas Peeters (thomas@example.com),
  ...?"). Trashing too: an email that quietly trashed the security alert about its own
  attack would be a classic trick;
* the model refers to emails by the number Yarvis gave them, never by an id it made up;
* AppleScript receives every value as ``argv`` data, never spliced into the script.

Mail and Contacts are started hidden, in the background: asking Yarvis about email never
pulls another app in front of the one the user is in.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parseaddr
from typing import TYPE_CHECKING, Any, Final, Literal

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema  # noqa: TC002 - pydantic needs it at runtime

from yarvis.tools.base import Arguments, Tier, Tool, ToolError
from yarvis.tools.computer import START_PAUSE_S, osascript, start_hidden

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from yarvis.tools.computer import Runner

logger = logging.getLogger(__name__)

MAIL_TIMEOUT_S: Final = 30.0
"""Mail may need to start, or to search a large inbox."""
MAX_LISTED: Final = 25
MAX_BODY_CHARS: Final = 6_000
_SCAN_LIMIT: Final = 400
_SAID_BODY_CHARS: Final = 300
_FIELD: Final = "\x1f"
_RECORD: Final = "\x1e"
_ADDRESS: Final = re.compile(r"^[^@\s<>,;\"']+@[^@\s<>,;\"']+\.[^@\s<>,;\"']+$")
_NAME_CHARS: Final = re.compile(r"[^\w .'-]")
_ERROR_NUMBER: Final = re.compile(r"\((-\d+)\)\s*$")

_NOT_ALLOWED: Final = (
    "Yarvis may not control {app}. The user can allow it in System Settings → Privacy & "
    "Security → Automation (under the app Yarvis is started from)."
)
_NO_ACCOUNTS: Final = (
    "Mail has no accounts yet. The user can add them in System Settings → Internet Accounts."
)

# Each script reads its values from argv and never runs text it was given.
_ACCOUNTS: Final = (
    "on run argv",
    "set fs to character id 31",
    "set rs to character id 30",
    'set out to ""',
    'tell application "Mail"',
    "repeat with acct in accounts",
    "if enabled of acct then",
    "set fullName to full name of acct",
    'if fullName is missing value then set fullName to ""',
    "set out to out & (name of acct) & fs & fullName",
    "set addrList to email addresses of acct",
    "repeat with addr in addrList",
    "set out to out & fs & (contents of addr)",
    "end repeat",
    "set out to out & rs",
    "end if",
    "end repeat",
    "end tell",
    "return out",
    "end run",
)
_LIST: Final = (
    "on run argv",
    'set unreadOnly to (item 1 of argv) is "true"',
    "set nowDate to current date",
    "set cutoff to nowDate - ((item 2 of argv) as integer)",
    "set wantedAccount to item 3 of argv",
    "set wantedText to item 4 of argv",
    "set maxCount to (item 5 of argv) as integer",
    "set fs to character id 31",
    "set rs to character id 30",
    'set out to ""',
    "set found to 0",
    'tell application "Mail"',
    "repeat with acct in accounts",
    "if enabled of acct then",
    "set acctName to name of acct",
    'set acctMatch to wantedAccount is "" or acctName contains wantedAccount',
    "set addrList to email addresses of acct",
    "repeat with addr in addrList",
    "if (contents of addr) contains wantedAccount then set acctMatch to true",
    "end repeat",
    "if acctMatch then",
    "repeat with theBox in mailboxes of acct",
    'if name of theBox is "inbox" then',
    "if unreadOnly then",
    "set msgs to (messages of theBox whose date received > cutoff and read status is false)",
    "else",
    "set msgs to (messages of theBox whose date received > cutoff)",
    "end if",
    "repeat with m in msgs",
    "set theSender to sender of m",
    'if theSender is missing value then set theSender to ""',
    "set theSubject to subject of m",
    'if theSubject is missing value then set theSubject to ""',
    'if wantedText is "" or theSender contains wantedText or theSubject contains wantedText then',
    "set age to (nowDate - (date received of m)) as integer",
    "set out to out & acctName & fs & (name of theBox) & fs & (id of m) & fs & theSender & fs & "
    "theSubject & fs & age & fs & (read status of m) & rs",
    "set found to found + 1",
    "if found >= maxCount then return out",
    "end if",
    "end repeat",
    "end if",
    "end repeat",
    "end if",
    "end if",
    "end repeat",
    "end tell",
    "return out",
    "end run",
)
_READ: Final = (
    "on run argv",
    "set fs to character id 31",
    "set acctName to item 1 of argv",
    "set boxName to item 2 of argv",
    "set msgId to (item 3 of argv) as integer",
    'tell application "Mail"',
    "set m to first message of mailbox boxName of account acctName whose id is msgId",
    "set theBody to content of m",
    'if theBody is missing value then set theBody to ""',
    'set theRecipients to ""',
    "repeat with r in to recipients of m",
    'set theRecipients to theRecipients & (address of r) & ", "',
    "end repeat",
    "return theRecipients & fs & theBody",
    "end tell",
    "end run",
)
_SEND: Final = (
    "on run argv",
    "set theSubject to item 1 of argv",
    "set theBody to item 2 of argv",
    "set theSender to item 3 of argv",
    "set theRecipients to items 4 thru -1 of argv",
    'tell application "Mail"',
    "set m to make new outgoing message with properties "
    "{subject:theSubject, content:theBody, visible:false}",
    "set sender of m to theSender",
    "repeat with theAddress in theRecipients",
    "tell m to make new to recipient at end of to recipients "
    "with properties {address:(contents of theAddress)}",
    "end repeat",
    "return (send m) as text",
    "end tell",
    "end run",
)
_MARK: Final = (
    "on run argv",
    "set acctName to item 1 of argv",
    "set boxName to item 2 of argv",
    "set msgId to (item 3 of argv) as integer",
    "set theField to item 4 of argv",
    'set newValue to (item 5 of argv) is "true"',
    'tell application "Mail"',
    "set m to first message of mailbox boxName of account acctName whose id is msgId",
    'if theField is "flagged" then',
    "set flagged status of m to newValue",
    "else",
    "set read status of m to newValue",
    "end if",
    "end tell",
    'return "marked"',
    "end run",
)
_TRASH: Final = (
    "on run argv",
    "set acctName to item 1 of argv",
    "set boxName to item 2 of argv",
    "set msgId to (item 3 of argv) as integer",
    'tell application "Mail"',
    "delete (first message of mailbox boxName of account acctName whose id is msgId)",
    "end tell",
    'return "trashed"',
    "end run",
)
_CONTACTS: Final = (
    "on run argv",
    "set fs to character id 31",
    "set rs to character id 30",
    'set out to ""',
    "set n to 0",
    "set wanted to item 1 of argv",
    'tell application "Contacts"',
    "repeat with p in (every person whose name contains wanted)",
    "repeat with e in emails of p",
    "set out to out & (name of p) & fs & (value of e) & rs",
    "end repeat",
    "set n to n + 1",
    "if n >= 20 then exit repeat",
    "end repeat",
    "end tell",
    "return out",
    "end run",
)


@dataclass(frozen=True, slots=True)
class Account:
    """A Mail account (one or more addresses)."""

    name: str
    full_name: str
    addresses: tuple[str, ...]

    @property
    def sender(self) -> str:
        """What goes in "From": "Victor Claeys <victor@example.com>"."""
        address = self.addresses[0]
        return f"{self.full_name} <{address}>" if self.full_name else address


@dataclass(frozen=True, slots=True)
class Message:
    """An email in an inbox, as listed."""

    account: str
    mailbox: str
    id: int
    sender: str
    subject: str
    received: datetime
    unread: bool


@dataclass(frozen=True, slots=True)
class Letter:
    """An email's recipients and text."""

    recipients: str
    body: str


class MacMail:
    """The Mail and Contacts apps, driven with fixed AppleScripts."""

    def __init__(
        self,
        run: Runner,
        *,
        now: Callable[[], datetime] | None = None,
        pause: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._run = run
        self._pause = pause
        self.now = now or (lambda: datetime.now().astimezone())
        """The current local time (replaceable in tests)."""

    async def accounts(self) -> list[Account]:
        """The enabled accounts, in Mail's order."""
        found: list[Account] = []
        for fields in _records(await self._script("Mail", _ACCOUNTS)):
            addresses = tuple(a for a in fields[2:] if a)
            if len(fields) >= 3 and addresses:  # noqa: PLR2004
                found.append(Account(fields[0], fields[1], addresses))
        return found

    async def messages(
        self, *, unread_only: bool, days: int, account: str, search: str
    ) -> list[Message]:
        """Inbox messages received in the last ``days`` days, newest first."""
        output = await self._script(
            "Mail",
            _LIST,
            "true" if unread_only else "false",
            str(days * 24 * 3600),
            account,
            search,
            str(_SCAN_LIMIT),
        )
        now = self.now()
        found: list[Message] = []
        for fields in _records(output):
            try:
                account_name, box, number, sender, subject, age, read = fields[:7]
                found.append(
                    Message(
                        account_name,
                        box,
                        int(number),
                        sender,
                        subject,
                        now - timedelta(seconds=int(age)),
                        read != "true",
                    )
                )
            except ValueError:
                logger.debug("Skipped an unreadable line from Mail")
        return sorted(found, key=lambda m: m.received, reverse=True)

    async def read(self, message: Message) -> Letter:
        """The recipients and the plain text of an email."""
        output = await self._script(
            "Mail", _READ, message.account, message.mailbox, str(message.id)
        )
        recipients, _, body = output.partition(_FIELD)
        return Letter(recipients.strip().rstrip(","), body.strip())

    async def send(self, sender: str, to: Sequence[str], subject: str, body: str) -> None:
        """Send an email from ``sender`` (one of the accounts)."""
        output = await self._script("Mail", _SEND, subject, body, sender, *to)
        if output.strip() != "true":
            msg = "Mail did not send it."
            raise ToolError(msg)

    async def mark(self, message: Message, *, field: Literal["flagged", "read"], on: bool) -> None:
        """Flag (a star, in Gmail) or unflag an email, or mark it read or unread."""
        await self._script(
            "Mail",
            _MARK,
            message.account,
            message.mailbox,
            str(message.id),
            field,
            "true" if on else "false",
        )

    async def trash(self, message: Message) -> None:
        """Move an email to its account's trash (recoverable there)."""
        await self._script("Mail", _TRASH, message.account, message.mailbox, str(message.id))

    async def contacts(self, name: str) -> list[tuple[str, str]]:
        """(name, address) of the contacts whose name contains ``name``."""
        return [
            (fields[0], fields[1])
            for fields in _records(await self._script("Contacts", _CONTACTS, name))
            if len(fields) >= 2  # noqa: PLR2004
        ]

    async def _script(self, app: str, script: Sequence[str], *values: str) -> str:
        await start_hidden(self._run, app, pause=self._pause)
        code, output = await self._run(osascript(script, *values))
        if code != 0 and "(-600)" in output:  # still starting up: once more, a moment later
            await self._pause(START_PAUSE_S)
            code, output = await self._run(osascript(script, *values))
        if code == 0:
            return output
        if "-1743" in output:
            raise ToolError(_NOT_ALLOWED.format(app=app))
        if "-1728" in output or "-1719" in output:
            msg = "That email is no longer there."
            raise ToolError(msg)
        # AppleScript's error ("Mail got an error: ... (-1712)") names a script problem, not
        # an email's text: worth keeping in the local log, and its number in the answer.
        logger.warning("%s refused a script (exit %d): %s", app, code, output[-300:])
        number = _ERROR_NUMBER.search(output)
        msg = f"{app} did not answer" + (f" (error {number.group(1)})." if number else ".")
        raise ToolError(msg)


def _records(output: str) -> list[list[str]]:
    return [record.split(_FIELD) for record in output.split(_RECORD) if record.strip()]


class Listing:
    """The emails as last listed: the model refers to them by number."""

    def __init__(self) -> None:
        self._messages: list[Message] = []

    def replace(self, messages: Sequence[Message]) -> None:
        """Number a new list from 1."""
        self._messages = list(messages)

    def get(self, number: int) -> Message:
        """Email ``number`` of the last list.

        Raises:
            ToolError: If there is no such number.
        """
        if not 1 <= number <= len(self._messages):
            msg = "There is no such email in the last list: list the emails first."
            raise ToolError(msg)
        return self._messages[number - 1]


# --------------------------------------------------------------------------- the tools


class ListArgs(Arguments):
    """Arguments of ``mail_list``."""

    unread_only: bool = Field(default=True, description="Only unread emails.")
    days: int = Field(default=7, ge=1, le=90, description="Received in the last N days.")
    account: str = Field(default="", max_length=80, description="Only this account or address.")
    search: str = Field(default="", max_length=80, description="Text in the sender or subject.")
    limit: int = Field(default=10, ge=1, le=MAX_LISTED)


class NumberArgs(Arguments):
    """Arguments of ``mail_read``."""

    number: int = Field(ge=1, le=MAX_LISTED, description="The email's number in the last list.")


class SendArgs(Arguments):
    """Arguments of ``mail_send``."""

    to: list[str] = Field(
        min_length=1,
        max_length=10,
        description="Email addresses, or names of the user's contacts.",
    )
    subject: str = Field(max_length=200)
    body: str = Field(min_length=1, max_length=5_000)
    from_account: str = Field(
        default="",
        max_length=120,
        description="Send from this account or address (only if the user said so).",
    )


class ReplyArgs(Arguments):
    """Arguments of ``mail_reply``."""

    number: int = Field(ge=1, le=MAX_LISTED, description="The email's number in the last list.")
    body: str = Field(min_length=1, max_length=5_000)
    # Filled in by Yarvis from the listed email (never by the model: hidden from it).
    to: SkipJsonSchema[str] = ""
    subject: SkipJsonSchema[str] = ""
    sender: SkipJsonSchema[str] = ""


class Pinned(Arguments):
    """An email pinned for an action: where it is, and how it is said."""

    account: str
    mailbox: str
    id: int
    said: str


class TrashArgs(Arguments):
    """Arguments of ``mail_trash``."""

    numbers: list[int] = Field(
        min_length=1, max_length=10, description="The emails' numbers in the last list."
    )
    # Filled in by Yarvis from the listed emails (never by the model: hidden from it).
    pinned: SkipJsonSchema[list[Pinned]] = Field(default_factory=list[Pinned])


class MarkArgs(Arguments):
    """Arguments of ``mail_mark``."""

    numbers: list[int] = Field(
        min_length=1, max_length=MAX_LISTED, description="The emails' numbers in the last list."
    )
    starred: bool | None = Field(
        default=None, description="Star (flag) them, or take the star off. Empty: unchanged."
    )
    read: bool | None = Field(
        default=None, description="Mark them read, or unread. Empty: unchanged."
    )


class _MailTool[A: Arguments](Tool[A]):
    timeout_s = MAIL_TIMEOUT_S + 5

    def __init__(self, mail: MacMail, listing: Listing) -> None:
        self._mail = mail
        self._listing = listing

    async def _account(self, wanted: str) -> Account:
        """The account called ``wanted`` (name or address), or the first one."""
        accounts = await self._mail.accounts()
        if not accounts:
            raise ToolError(_NO_ACCOUNTS)
        if not wanted:
            return accounts[0]
        key = wanted.casefold()
        matches = [
            account
            for account in accounts
            if key in account.name.casefold() or any(key in a.casefold() for a in account.addresses)
        ]
        if len(matches) == 1:
            return matches[0]
        said = "; ".join(f"{a.name} ({', '.join(a.addresses)})" for a in accounts)
        msg = f"No single account matches {wanted!r}. The accounts: {said}."
        raise ToolError(msg)


class MailList(_MailTool[ListArgs]):
    """The latest emails in every inbox."""

    name = "mail_list"
    description = (
        "List the user's latest emails (all accounts, newest first), numbered for mail_read "
        "and mail_reply. Can filter by sender or subject text."
    )
    tier = Tier.READ
    Args = ListArgs
    untrusted_output = True

    async def run(self, args: ListArgs) -> str:
        """List them."""
        accounts = await self._mail.accounts()
        if not accounts:
            raise ToolError(_NO_ACCOUNTS)
        found = await self._mail.messages(
            unread_only=args.unread_only, days=args.days, account=args.account, search=args.search
        )
        shown = found[: args.limit]
        self._listing.replace(shown)
        kind = "unread emails" if args.unread_only else "emails"
        if not shown:
            return f"No {kind} in the last {args.days} days."
        addresses = {account.name: account.addresses[0] for account in accounts}
        now = self._mail.now()
        lines = [f"{len(found)} {kind} in the last {args.days} days; the newest {len(shown)}:"]
        for number, message in enumerate(shown, start=1):
            unread = " (unread)" if message.unread and not args.unread_only else ""
            to = addresses.get(message.account, message.account)
            lines.append(
                f"{number}. {_when(message.received, now)}, from "
                f"{message.sender[:120]}: “{message.subject[:200]}”{unread} [to {to}]"
            )
        return "\n".join(lines)

    def describe(self, args: ListArgs) -> str:
        """'check your unread email'."""
        what = "your unread email" if args.unread_only else "your email"
        about = f" for “{args.search}”" if args.search else ""
        return f"check {what}{about}"


class MailRead(_MailTool[NumberArgs]):
    """One email, in full."""

    name = "mail_read"
    description = "Read one email from the last list, by its number."
    tier = Tier.READ
    Args = NumberArgs
    untrusted_output = True

    async def run(self, args: NumberArgs) -> str:
        """Read it."""
        message = self._listing.get(args.number)
        letter = await self._mail.read(message)
        body = letter.body[:MAX_BODY_CHARS]
        cut = " (cut short)" if len(letter.body) > MAX_BODY_CHARS else ""
        return (
            f"From: {message.sender}\nTo: {letter.recipients}\nSubject: {message.subject}\n"
            f"Received: {message.received:%Y-%m-%d %H:%M}\n\n{body}{cut}"
        )

    def describe(self, args: NumberArgs) -> str:
        """'read email 2'."""
        return f"read email {args.number}"


class MailSend(_MailTool[SendArgs]):
    """A new email (after the user's "yes")."""

    name = "mail_send"
    description = (
        "Send an email. 'to' takes addresses or the names of the user's contacts. The user "
        "is asked to confirm first, hearing the exact recipients."
    )
    tier = Tier.EXTERNAL
    Args = SendArgs

    async def prepare(self, args: SendArgs) -> SendArgs:
        """Turn names into addresses, and settle the sending account."""
        to = [await self._address(entry) for entry in args.to]
        account = await self._account(args.from_account)
        return args.model_copy(update={"to": to, "from_account": account.sender})

    async def run(self, args: SendArgs) -> str:
        """Send it."""
        account = await self._account(_address_of(args.from_account) or args.from_account)
        addresses = [_address_of(entry) for entry in args.to]
        await self._mail.send(account.sender, addresses, args.subject, args.body)
        return f"Sent to {', '.join(addresses)} from {account.addresses[0]}."

    def describe(self, args: SendArgs) -> str:
        """'email Thomas Peeters (thomas@example.com), from ..., subject “...”: “...”'."""
        to = ", ".join(_said(entry) for entry in args.to)
        sender = _address_of(args.from_account) or args.from_account
        sender = f", from {sender}" if sender else ""
        subject = f", subject “{args.subject}”" if args.subject else ""
        return f"email {to}{sender}{subject}: “{_short(args.body)}”"

    async def _address(self, entry: str) -> str:
        """A recipient as "Name <address>" (or a bare address)."""
        name, address = parseaddr(entry)
        if "@" in entry:
            if not _ADDRESS.match(address):
                msg = f"{entry!r} is not a valid email address."
                raise ToolError(msg)
            return f"{_clean(name)} <{address}>" if name else address
        found = {addr.casefold(): (n, addr) for n, addr in await self._mail.contacts(entry)}
        exact = {k: v for k, v in found.items() if v[0].casefold() == entry.casefold()}
        matches = list((exact or found).values())
        if len(matches) == 1:
            contact, address = matches[0]
            return f"{_clean(contact)} <{address}>"
        if not matches:
            msg = f"No contact called {entry!r} has an email address. Ask the user for it."
            raise ToolError(msg)
        options = "; ".join(f"{_clean(n)} ({a})" for n, a in matches[:6])
        msg = f"Several addresses match {entry!r}: {options}. Ask the user which one."
        raise ToolError(msg)


class MailReply(_MailTool[ReplyArgs]):
    """Answer an email (after the user's "yes")."""

    name = "mail_reply"
    description = (
        "Reply to the sender of an email from the last list, by its number. The user is "
        "asked to confirm first."
    )
    tier = Tier.EXTERNAL
    Args = ReplyArgs

    async def prepare(self, args: ReplyArgs) -> ReplyArgs:
        """Pin the email being answered: its sender, subject and the account it came to."""
        message = self._listing.get(args.number)
        _, address = parseaddr(message.sender)
        if not _ADDRESS.match(address):
            msg = "That email has no address to reply to."
            raise ToolError(msg)
        accounts = {account.name: account for account in await self._mail.accounts()}
        if message.account not in accounts:
            raise ToolError(_NO_ACCOUNTS)
        subject = message.subject
        if not subject.casefold().startswith("re:"):
            subject = f"Re: {subject}"
        return args.model_copy(
            update={
                "to": message.sender,
                "subject": subject,
                "sender": accounts[message.account].sender,
            }
        )

    async def run(self, args: ReplyArgs) -> str:
        """Send the reply (as a new email with "Re:", which mail apps thread)."""
        if not args.to or not args.sender:
            msg = "The email to answer was not resolved."
            raise ToolError(msg)
        to = _address_of(args.to)
        await self._mail.send(args.sender, [to], args.subject, args.body)
        return f"Replied to {to}."

    def describe(self, args: ReplyArgs) -> str:
        """'reply to Thomas Peeters (thomas@example.com) about “Re: ...”: “...”'."""
        if not args.to:
            return f"reply to email {args.number}"
        return f"reply to {_said(args.to)} about “{args.subject}”: “{_short(args.body)}”"


class MailMark(_MailTool[MarkArgs]):
    """Star or unstar emails, or mark them read or unread."""

    name = "mail_mark"
    description = (
        "Star or unstar emails from the last list (a star in Gmail, a flag in Outlook and "
        "Mail), or mark them read or unread, by their numbers."
    )
    tier = Tier.REVERSIBLE  # undone with one more word, and still in the inbox either way
    Args = MarkArgs

    async def run(self, args: MarkArgs) -> str:
        """Mark them."""
        if args.starred is None and args.read is None:
            msg = "Say whether to star them or mark them read (or the opposite)."
            raise ToolError(msg)
        messages = [self._listing.get(number) for number in dict.fromkeys(args.numbers)]
        for message in messages:
            if args.starred is not None:
                await self._mail.mark(message, field="flagged", on=args.starred)
            if args.read is not None:
                await self._mail.mark(message, field="read", on=args.read)
        return f"Done, for {len(messages)} email{'s' if len(messages) > 1 else ''}."

    def describe(self, args: MarkArgs) -> str:
        """'star email 2' / 'mark emails 1, 3 as read'."""
        numbers = ", ".join(str(n) for n in dict.fromkeys(args.numbers))
        emails = f"email{'s' if len(args.numbers) > 1 else ''} {numbers}"
        done: list[str] = []
        if args.starred is not None:
            done.append(f"{'star' if args.starred else 'unstar'} {emails}")
        if args.read is not None:
            mark = f"mark {emails}" if not done else "mark them"
            done.append(f"{mark} as {'read' if args.read else 'unread'}")
        return " and ".join(done) or f"leave {emails} as they are"


class MailTrash(_MailTool[TrashArgs]):
    """Move emails to the trash (after the user's "yes")."""

    name = "mail_trash"
    description = (
        "Move emails from the last list to the trash, by their numbers. The user is asked to "
        "confirm first."
    )
    tier = Tier.EXTERNAL
    Args = TrashArgs

    async def prepare(self, args: TrashArgs) -> TrashArgs:
        """Pin the emails, so the "yes" trashes exactly the ones that were named."""
        pinned: list[Pinned] = []
        for number in dict.fromkeys(args.numbers):  # each once, in order
            message = self._listing.get(number)
            said = f"from {_name(message.sender)}, “{message.subject[:120]}”"
            pinned.append(Pinned(account=message.account, mailbox=message.mailbox,
                                 id=message.id, said=said))  # fmt: skip
        return args.model_copy(update={"pinned": pinned})

    async def run(self, args: TrashArgs) -> str:
        """Trash them."""
        if not args.pinned:
            msg = "The emails to trash were not resolved."
            raise ToolError(msg)
        for email in args.pinned:
            message = Message(email.account, email.mailbox, email.id, "", "", self._mail.now(),
                              unread=False)  # fmt: skip
            await self._mail.trash(message)
        count = len(args.pinned)
        return f"Moved {count} email{'s' if count > 1 else ''} to the trash."

    def describe(self, args: TrashArgs) -> str:
        """'move the email from Google, “Security alert”, to the trash'."""
        if not args.pinned:
            numbers = ", ".join(str(n) for n in args.numbers)
            return f"move email {numbers} to the trash"
        if len(args.pinned) == 1:
            return f"move the email {args.pinned[0].said}, to the trash"
        listed = "; ".join(email.said for email in args.pinned)
        return f"move {len(args.pinned)} emails to the trash: {listed}"


def mail_tools(mail: MacMail) -> list[Tool[Any]]:
    """Listing, reading, sending, replying, starring and trashing."""
    listing = Listing()
    return [
        MailList(mail, listing),
        MailRead(mail, listing),
        MailSend(mail, listing),
        MailReply(mail, listing),
        MailMark(mail, listing),
        MailTrash(mail, listing),
    ]


# --------------------------------------------------------------------------- words


def _address_of(entry: str) -> str:
    return parseaddr(entry)[1]


def _clean(name: str) -> str:
    """A display name with nothing but letters and simple punctuation (it is outside text)."""
    return _NAME_CHARS.sub("", name).strip()[:60]


def _name(entry: str) -> str:
    """'Google': who sent it, by name if there is one (for saying, not for sending)."""
    name, address = parseaddr(entry)
    return _clean(name) or address or "an unknown sender"


def _said(entry: str) -> str:
    """'Thomas Peeters (thomas@example.com)': both, so an odd address is heard."""
    name, address = parseaddr(entry)
    return f"{_clean(name)} ({address})" if name else address


def _short(body: str) -> str:
    text = " ".join(body.split())
    return text if len(text) <= _SAID_BODY_CHARS else text[: _SAID_BODY_CHARS - 1] + "…"


def _when(moment: datetime, now: datetime) -> str:
    days = (now.date() - moment.date()).days
    if days == 0:
        return f"today {moment:%H:%M}"
    if days == 1:
        return f"yesterday {moment:%H:%M}"
    return f"{moment:%a %d %b} {moment:%H:%M}"
