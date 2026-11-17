"""Email through the Mail app: fixed AppleScripts, values as data, resolved before asking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from yarvis.tools.base import Tier, ToolError
from yarvis.tools.computer import OPEN, OSASCRIPT, PGREP
from yarvis.tools.mail import (
    Account,
    ListArgs,
    Listing,
    MacMail,
    MailList,
    MailMark,
    MailRead,
    MailReply,
    MailSend,
    MailTrash,
    MarkArgs,
    Message,
    NumberArgs,
    ReplyArgs,
    SendArgs,
    TrashArgs,
    mail_tools,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from yarvis.tools.base import Tool

F, R = "\x1f", "\x1e"
NOW = datetime(2026, 11, 20, 18, 0).astimezone()
ACCOUNTS = (
    f"Google{F}Victor Claeys{F}victortje10@gmail.com{R}"
    f"Google 2{F}{F}victorclaeyson@gmail.com{R}"
    f"Exchange{F}Victor Claeys{F}victor@outlook.com{F}victor@hotmail.com{R}"
)


def listed(*rows: tuple[str, int, str, str, int, bool]) -> str:
    """Mail's answer: (account, id, sender, subject, seconds ago, read)."""
    return "".join(
        F.join([account, "INBOX", str(number), sender, subject, str(age), str(read).lower()]) + R
        for account, number, sender, subject, age, read in rows
    )


INBOX = listed(
    ("Google", 11, "Thomas Peeters <thomas@example.com>", "Dinner Friday?", 3600, False),
    ("Exchange", 7, "KU Leuven <noreply@kuleuven.be>", "Exam schedule", 26 * 3600, True),
    ("Google 2", 5, "Mom <mom@example.com>", "Photos", 3 * 24 * 3600, False),
)


class FakeApps:
    """Answers each AppleScript by what it does; records the values it was given."""

    def __init__(self, *, running: Sequence[bool] = (), **answers: tuple[int, str]) -> None:
        self.answers = {"accounts": (0, ACCOUNTS), "list": (0, INBOX), "send": (0, "true")}
        self.answers |= answers
        self.running = list(running)
        """Answers to "is it running?" (then: yes)."""
        self.calls: list[tuple[str, list[str]]] = []

    async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        if argv[0] == OPEN:
            self.calls.append(("start", list(argv[1:])))
            return 0, ""
        if argv[0] == PGREP:
            self.calls.append(("running?", list(argv[1:])))
            return (0, "501") if (self.running.pop(0) if self.running else True) else (1, "")
        assert argv[0] == OSASCRIPT
        script = [argv[i + 1] for i in range(1, len(argv) - 1, 2) if argv[i] == "-e"]
        values = list(argv[1 + 2 * len(script) :])
        text = "\n".join(script)
        kind = (
            "contacts" if 'application "Contacts"' in text
            else "trash" if "delete (first message" in text
            else "mark" if "flagged status" in text
            else "send" if "outgoing message" in text
            else "read" if "content of m" in text
            else "list" if "date received" in text
            else "accounts"
        )  # fmt: skip
        self.calls.append((kind, values))
        return self.answers.get(kind, (0, ""))

    def values(self, kind: str) -> list[list[str]]:
        return [values for called, values in self.calls if called == kind]


@dataclass(frozen=True)
class Kit:
    inbox: MailList
    read: MailRead
    send: MailSend
    reply: MailReply


async def no_wait(seconds: float) -> None:
    pass


def tools(apps: FakeApps) -> Kit:
    mail = MacMail(apps, now=lambda: NOW, pause=no_wait)
    listing = Listing()
    return Kit(
        MailList(mail, listing),
        MailRead(mail, listing),
        MailSend(mail, listing),
        MailReply(mail, listing),
    )


# --------------------------------------------------------------------------- Mail itself


async def test_accounts_and_inbox_are_read_from_mail() -> None:
    apps = FakeApps()
    mail = MacMail(apps, now=lambda: NOW)
    assert await mail.accounts() == [
        Account("Google", "Victor Claeys", ("victortje10@gmail.com",)),
        Account("Google 2", "", ("victorclaeyson@gmail.com",)),
        Account("Exchange", "Victor Claeys", ("victor@outlook.com", "victor@hotmail.com")),
    ]
    found = await mail.messages(unread_only=True, days=7, account="", search="thomas")
    assert apps.values("list") == [["true", str(7 * 24 * 3600), "", "thomas", "400"]]
    assert [m.subject for m in found] == ["Dinner Friday?", "Exam schedule", "Photos"]
    assert found[0] == Message(
        "Google", "INBOX", 11, "Thomas Peeters <thomas@example.com>", "Dinner Friday?",
        NOW - timedelta(hours=1), unread=True,
    )  # fmt: skip
    assert not found[1].unread


async def test_odd_answers_from_mail_are_skipped() -> None:
    odd = f"Google{F}INBOX{F}not-a-number{F}x{F}y{F}1{F}false{R}too{F}short{R}" + INBOX
    mail = MacMail(FakeApps(list=(0, odd), accounts=(0, f"Broken{R}" + ACCOUNTS)))
    assert len(await mail.messages(unread_only=False, days=1, account="", search="")) == 3
    assert len(await mail.accounts()) == 3


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ((1, "execution error: Not authorized to send Apple events to Mail. (-1743)"),
         "may not control Mail"),
        ((1, "execution error: Can't get message 1 whose id = 11. (-1728)"), "no longer there"),
        ((1, "execution error: Mail got an error: Application isn't running. (-600)"),
         r"Mail did not answer \(error -600\)\."),
        ((1, "osascript: couldn't compile"), r"Mail did not answer\.$"),
    ],
)  # fmt: skip
async def test_mail_errors_say_what_to_do(answer: tuple[int, str], message: str) -> None:
    with pytest.raises(ToolError, match=message):
        await MacMail(FakeApps(accounts=answer)).accounts()


async def test_values_never_become_script() -> None:
    """Everything the model wrote reaches AppleScript as argv, after the fixed script."""
    apps = FakeApps(contacts=(0, f"Evil{F}evil@example.com{R}"))
    mail = MacMail(apps)
    sneaky = '" & (do shell script "rm -rf ~") & "'
    await mail.contacts(sneaky)
    await mail.send("me@example.com", ["you@example.com"], sneaky, sneaky)
    assert apps.values("contacts") == [[sneaky]]
    assert apps.values("send") == [[sneaky, sneaky, "me@example.com", "you@example.com"]]


async def test_mail_is_started_hidden_in_the_background() -> None:
    """Asking about email must not pull Mail in front of the app the user is in."""
    apps = FakeApps(running=[False, False, False, True])
    waited: list[float] = []

    async def pause(seconds: float) -> None:
        waited.append(seconds)

    await MacMail(apps, pause=pause).accounts()
    assert [kind for kind, _ in apps.calls] == [
        "running?", "start", "running?", "running?", "running?", "accounts",
    ]  # fmt: skip
    assert apps.values("start") == [["-g", "-j", "-a", "Mail"]]
    assert waited == [0.5, 0.5]
    await MacMail(apps, pause=pause).contacts("Ann")
    assert apps.values("start") == [["-g", "-j", "-a", "Mail"]]  # Contacts was running


async def test_a_slow_start_is_tried_again_once() -> None:
    class Starting(FakeApps):
        async def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
            answer = await super().__call__(argv)
            if self.calls[-1][0] == "accounts" and len(self.values("accounts")) == 1:
                return 1, "Mail got an error: Application isn't running. (-600)"
            return answer

    apps = Starting(running=[False] * 30)
    assert len(await MacMail(apps, pause=no_wait).accounts()) == 3
    assert len(apps.values("accounts")) == 2


async def test_sending_must_be_confirmed_by_mail() -> None:
    with pytest.raises(ToolError, match=r"Mail did not send it\."):
        await MacMail(FakeApps(send=(0, "false"))).send("a@b.co", ["c@d.co"], "s", "b")


# --------------------------------------------------------------------------- listing and reading


async def test_the_inbox_is_listed_with_numbers() -> None:
    mail = tools(FakeApps()).inbox
    assert mail.tier is Tier.READ and mail.untrusted_output
    said = await mail.run(ListArgs(unread_only=False))
    assert said == (
        "3 emails in the last 7 days; the newest 3:\n"
        "1. today 17:00, from Thomas Peeters <thomas@example.com>: “Dinner Friday?” (unread) "
        "[to victortje10@gmail.com]\n"
        "2. yesterday 16:00, from KU Leuven <noreply@kuleuven.be>: “Exam schedule” "
        "[to victor@outlook.com]\n"
        f"3. {NOW - timedelta(days=3):%a %d %b} 18:00, from Mom <mom@example.com>: “Photos” "
        "(unread) [to victorclaeyson@gmail.com]"
    )
    assert mail.describe(ListArgs()) == "check your unread email"
    assert mail.describe(ListArgs(unread_only=False, search="Thomas")) == (
        "check your email for “Thomas”"
    )


async def test_an_empty_inbox_and_a_mac_without_accounts() -> None:
    assert await tools(FakeApps(list=(0, ""))).inbox.run(ListArgs(days=2)) == (
        "No unread emails in the last 2 days."
    )
    with pytest.raises(ToolError, match="Internet Accounts"):
        await tools(FakeApps(accounts=(0, ""))).inbox.run(ListArgs())


async def test_an_email_is_read_by_its_number() -> None:
    apps = FakeApps(read=(0, f"victortje10@gmail.com, {F}Hi Victor,\n\nDinner at 7?\n"))
    found = tools(apps)
    reader = found.read
    with pytest.raises(ToolError, match="list the emails first"):
        await reader.run(NumberArgs(number=1))
    await found.inbox.run(ListArgs(limit=2))
    said = await reader.run(NumberArgs(number=1))
    assert apps.values("read") == [["Google", "INBOX", "11"]]
    assert said.startswith("From: Thomas Peeters <thomas@example.com>\nTo: victortje10@gmail.com\n")
    assert said.endswith("\n\nHi Victor,\n\nDinner at 7?")
    assert reader.untrusted_output
    assert reader.describe(NumberArgs(number=1)) == "read email 1"
    with pytest.raises(ToolError, match="list the emails first"):
        await reader.run(NumberArgs(number=3))  # only two were listed


async def test_long_emails_are_cut_short() -> None:
    apps = FakeApps(read=(0, F + "x" * 10_000))
    found = tools(apps)
    await found.inbox.run(ListArgs())
    assert (await found.read.run(NumberArgs(number=1))).endswith("x (cut short)")


# --------------------------------------------------------------------------- sending


async def test_names_become_addresses_before_the_question() -> None:
    apps = FakeApps(contacts=(0, f"Thomas Peeters{F}thomas@example.com{R}"))
    send = tools(apps).send
    assert send.tier is Tier.EXTERNAL
    asked = SendArgs(to=["Thomas"], subject="Late", body="I'm running ten minutes late.")
    ready = await send.prepare(asked)
    assert ready.to == ["Thomas Peeters <thomas@example.com>"]
    assert ready.from_account == "Victor Claeys <victortje10@gmail.com>"
    assert send.describe(ready) == (
        "email Thomas Peeters (thomas@example.com), from victortje10@gmail.com, subject “Late”: "
        "“I'm running ten minutes late.”"
    )
    assert await send.run(ready) == "Sent to thomas@example.com from victortje10@gmail.com."
    assert apps.values("send") == [
        ["Late", "I'm running ten minutes late.", "Victor Claeys <victortje10@gmail.com>",
         "thomas@example.com"],
    ]  # fmt: skip


async def test_addresses_and_accounts_can_be_given() -> None:
    apps = FakeApps()
    send = tools(apps).send
    ready = await send.prepare(
        SendArgs(
            to=["Ann <ann@example.com>", "bob@example.org"],
            subject="",
            body="See you " * 60,
            from_account="outlook",
        )
    )
    assert ready.to == ["Ann <ann@example.com>", "bob@example.org"]
    assert ready.from_account == "Victor Claeys <victor@outlook.com>"
    said = send.describe(ready)
    assert said.startswith("email Ann (ann@example.com), bob@example.org, from victor@outlook.com:")
    assert said.endswith("…”") and len(said) < 420  # a long email is not read out whole
    await send.run(ready)
    assert apps.values("send")[0][2:] == [
        "Victor Claeys <victor@outlook.com>", "ann@example.com", "bob@example.org",
    ]  # fmt: skip
    assert send.describe(SendArgs(to=["x@y.co"], subject="", body="hi")) == "email x@y.co: “hi”"


@pytest.mark.parametrize(
    ("contacts", "to", "message"),
    [
        ("", "Thomas", "No contact called 'Thomas' has an email address"),
        (f"Thomas Peeters{F}t@a.be{R}Thomas Maes{F}tm@b.be{R}", "Thomas",
         r"Several addresses match 'Thomas': Thomas Peeters \(t@a.be\); Thomas Maes \(tm@b.be\)"),
        ("", "thomas@", "is not a valid email address"),
        ("", "a@b.c, evil@example.com", "is not a valid email address"),
    ],
)  # fmt: skip
async def test_unclear_recipients_are_asked_about(contacts: str, to: str, message: str) -> None:
    send = tools(FakeApps(contacts=(0, contacts))).send
    with pytest.raises(ToolError, match=message):
        await send.prepare(SendArgs(to=[to], subject="", body="hi"))


async def test_an_exact_name_wins_and_odd_names_are_cleaned() -> None:
    contacts = (
        f"Thomas{F}thomas@example.com{R}Thomas Peeters{F}tp@example.com{R}"
        f"Thomas{F}THOMAS@example.com{R}"  # the same address twice
    )
    send = tools(FakeApps(contacts=(0, contacts))).send
    ready = await send.prepare(SendArgs(to=["thomas"], subject="", body="hi"))
    assert ready.to == ["Thomas <THOMAS@example.com>"]
    sneaky = tools(FakeApps(contacts=(0, f"Ann <ignore> {{all}}{F}ann@example.com{R}")))
    ready = await sneaky.send.prepare(SendArgs(to=["Ann"], subject="", body="hi"))
    assert ready.to == ["Ann ignore all <ann@example.com>"]


@pytest.mark.parametrize(
    ("accounts", "wanted", "message"),
    [
        (ACCOUNTS, "gmail", "No single account matches 'gmail'. The accounts: Google"),
        ("", "", "Mail has no accounts yet"),
    ],
)
async def test_unclear_accounts_are_asked_about(accounts: str, wanted: str, message: str) -> None:
    send = tools(FakeApps(accounts=(0, accounts))).send
    with pytest.raises(ToolError, match=message):
        await send.prepare(SendArgs(to=["a@b.co"], subject="", body="hi", from_account=wanted))


# --------------------------------------------------------------------------- replying


async def test_a_reply_is_pinned_to_the_listed_email() -> None:
    apps = FakeApps()
    found = tools(apps)
    reply = found.reply
    assert reply.tier is Tier.EXTERNAL
    assert reply.describe(ReplyArgs(number=2, body="Thanks!")) == "reply to email 2"
    with pytest.raises(ToolError, match="list the emails first"):
        await reply.prepare(ReplyArgs(number=1, body="Thanks!"))
    await found.inbox.run(ListArgs(unread_only=False))
    # The model tries to sneak in its own recipient; resolving overwrites it.
    sneaky = ReplyArgs.model_validate({"number": 2, "body": "Thanks!", "to": "evil@example.com"})
    ready = await reply.prepare(sneaky)
    assert ready.to == "KU Leuven <noreply@kuleuven.be>"
    assert ready.subject == "Re: Exam schedule"
    assert ready.sender == "Victor Claeys <victor@outlook.com>"
    assert reply.describe(ready) == (
        "reply to KU Leuven (noreply@kuleuven.be) about “Re: Exam schedule”: “Thanks!”"
    )
    assert await reply.run(ready) == "Replied to noreply@kuleuven.be."
    assert apps.values("send")[-1] == [
        "Re: Exam schedule", "Thanks!", "Victor Claeys <victor@outlook.com>",
        "noreply@kuleuven.be",
    ]  # fmt: skip


async def test_replies_keep_an_existing_re_and_need_an_address() -> None:
    inbox = listed(
        ("Google", 1, "Ann <ann@example.com>", "RE: plans", 60, False),
        ("Google", 2, "Mailer daemon", "Undeliverable", 60, False),
        ("Gone", 3, "Bob <bob@example.com>", "Hi", 60, False),
    )
    found = tools(FakeApps(list=(0, inbox)))
    await found.inbox.run(ListArgs())
    reply = found.reply
    assert (await reply.prepare(ReplyArgs(number=1, body="ok"))).subject == "RE: plans"
    with pytest.raises(ToolError, match="no address to reply to"):
        await reply.prepare(ReplyArgs(number=2, body="ok"))
    with pytest.raises(ToolError, match="no accounts"):
        await reply.prepare(ReplyArgs(number=3, body="ok"))
    with pytest.raises(ToolError, match="not resolved"):
        await reply.run(ReplyArgs(number=1, body="ok"))


def test_mail_tools_share_one_list_and_take_their_time() -> None:
    found: list[Tool[Any]] = mail_tools(MacMail(FakeApps()))
    assert [tool.name for tool in found] == [
        "mail_list", "mail_read", "mail_send", "mail_reply", "mail_mark", "mail_trash",
    ]  # fmt: skip
    assert all((tool.timeout_s or 0) > 30 for tool in found)  # Mail may have to start


# --------------------------------------------------------------------------- trashing


async def test_trashing_asks_first_and_names_the_emails() -> None:
    apps = FakeApps()
    mail = MacMail(apps, now=lambda: NOW, pause=no_wait)
    listing = Listing()
    trash = MailTrash(mail, listing)
    assert trash.tier is Tier.EXTERNAL  # an email could try to hide a security alert
    assert trash.describe(TrashArgs(numbers=[1, 3])) == "move email 1, 3 to the trash"
    await MailList(mail, listing).run(ListArgs(unread_only=False))
    one = await trash.prepare(TrashArgs(numbers=[2]))
    assert trash.describe(one) == "move the email from KU Leuven, “Exam schedule”, to the trash"
    assert await trash.run(one) == "Moved 1 email to the trash."
    assert apps.values("trash") == [["Exchange", "INBOX", "7"]]
    two = await trash.prepare(TrashArgs(numbers=[1, 3, 1]))
    assert trash.describe(two) == (
        "move 2 emails to the trash: from Thomas Peeters, “Dinner Friday?”; from Mom, “Photos”"
    )
    await trash.run(two)
    assert apps.values("trash")[1:] == [["Google", "INBOX", "11"], ["Google 2", "INBOX", "5"]]


async def test_trashing_is_pinned_to_the_listed_emails() -> None:
    apps = FakeApps(
        list=(0, listed(("Google", 9, "no-reply@accounts.google.com", "Alert", 60, False)))
    )
    mail = MacMail(apps, pause=no_wait)
    listing = Listing()
    trash = MailTrash(mail, listing)
    with pytest.raises(ToolError, match="list the emails first"):
        await trash.prepare(TrashArgs(numbers=[1]))
    await MailList(mail, listing).run(ListArgs())
    sneaky = TrashArgs.model_validate(
        {
            "numbers": [1],
            "pinned": [{"account": "Google", "mailbox": "INBOX", "id": 1, "said": "x"}],
        }
    )
    ready = await trash.prepare(sneaky)
    assert [p.id for p in ready.pinned] == [9]  # what the model slipped in is replaced
    assert trash.describe(ready) == (
        "move the email from no-reply@accounts.google.com, “Alert”, to the trash"
    )
    with pytest.raises(ToolError, match="not resolved"):
        await trash.run(TrashArgs(numbers=[1]))
    assert "pinned" not in json.dumps(TrashArgs.model_json_schema())


# --------------------------------------------------------------------------- starring


async def test_emails_are_starred_and_marked_read() -> None:
    apps = FakeApps()
    mail = MacMail(apps, now=lambda: NOW, pause=no_wait)
    listing = Listing()
    mark = MailMark(mail, listing)
    assert mark.tier is Tier.REVERSIBLE  # one more word undoes it; the email stays put
    await MailList(mail, listing).run(ListArgs(unread_only=False))
    star = MarkArgs(numbers=[1], starred=True)
    assert mark.describe(star) == "star email 1"
    assert await mark.run(star) == "Done, for 1 email."
    assert apps.values("mark") == [["Google", "INBOX", "11", "flagged", "true"]]
    both = MarkArgs(numbers=[2, 3, 2], starred=False, read=True)
    assert mark.describe(both) == "unstar emails 2, 3 and mark them as read"
    assert await mark.run(both) == "Done, for 2 emails."
    assert apps.values("mark")[1:] == [
        ["Exchange", "INBOX", "7", "flagged", "false"],
        ["Exchange", "INBOX", "7", "read", "true"],
        ["Google 2", "INBOX", "5", "flagged", "false"],
        ["Google 2", "INBOX", "5", "read", "true"],
    ]
    unread = MarkArgs(numbers=[1], read=False)
    assert mark.describe(unread) == "mark email 1 as unread"
    await mark.run(unread)
    assert apps.values("mark")[-1] == ["Google", "INBOX", "11", "read", "false"]


async def test_marking_needs_something_to_do_and_a_listed_email() -> None:
    mark = MailMark(MacMail(FakeApps(), pause=no_wait), Listing())
    assert mark.describe(MarkArgs(numbers=[1])) == "leave email 1 as they are"
    with pytest.raises(ToolError, match="star them or mark them read"):
        await mark.run(MarkArgs(numbers=[1]))
    with pytest.raises(ToolError, match="list the emails first"):
        await mark.run(MarkArgs(numbers=[1], starred=True))
