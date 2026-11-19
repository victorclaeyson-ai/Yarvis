"""The permission gate: every tool call passes here, and there is no other way to run one.

The model is not a security boundary (design doc §5.1). Whatever it was told or tricked
into, the decision to run a tool is made here, by code, from the tool's declared tier and
whether the turn has read untrusted content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from yarvis.tools.base import Tier

if TYPE_CHECKING:
    from typing import Any

    from yarvis.tools.base import Tool

logger = logging.getLogger(__name__)


class Verdict(StrEnum):
    """What may happen with a tool call."""

    ALLOW = "allow"
    CONFIRM = "confirm"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class Decision:
    """The gate's answer, with a reason the model can relay."""

    verdict: Verdict
    reason: str


class Gate:
    """Decides, per call, whether a tool runs, needs the user's "yes", or is refused."""

    def __init__(self, *, confirm_external: bool = True, allow_dangerous: bool = False) -> None:
        self._confirm_external = confirm_external
        self._allow_dangerous = allow_dangerous

    def check(self, tool: Tool[Any], *, tainted: bool, sends_out: bool = False) -> Decision:
        """Decide for one call.

        ``tainted``: the model's context holds untrusted content: read in this turn, or in
        a reply still in the conversation's history (ADR 0018). ``sends_out``: the call can
        carry text to a destination the model chose (``Tool.sends_data_out``).
        """
        asks = tool.asks_when_tainted and tainted
        decision = self._decide(tool.tier, tainted=tainted, sends_out=sends_out, asks=asks)
        logger.info(
            "Gate: %s (tier %d%s%s) -> %s",
            tool.name,
            tool.tier,
            ", tainted" if tainted else "",
            ", sends data out" if sends_out else "",
            decision.verdict.value,
        )
        return decision

    def _decide(self, tier: Tier, *, tainted: bool, sends_out: bool, asks: bool) -> Decision:
        if tier >= Tier.DANGEROUS:
            if self._allow_dangerous:
                # Design doc §5.3: dangerous actions need an on-screen confirmation per
                # call, never a spoken one. Until that exists, they stay refused.
                return Decision(Verdict.DENY, "This needs on-screen confirmation, not built yet.")
            return Decision(Verdict.DENY, "Dangerous actions are disabled.")
        if tier == Tier.EXTERNAL and (self._confirm_external or tainted):
            return Decision(Verdict.CONFIRM, "Needs the user's confirmation.")
        if tainted and sends_out:
            # Outside content (an email, a page) may be steering the model into sending what
            # it has read somewhere: the user hears where it would go and decides.
            return Decision(Verdict.CONFIRM, "Sends data out after reading outside content.")
        if asks:
            # Memory poisoning (T7): a page or email must not make Yarvis "remember" a false
            # instruction, or forget something, without the user hearing it first.
            return Decision(
                Verdict.CONFIRM, "Changes what Yarvis remembers after reading outside content."
            )
        return Decision(Verdict.ALLOW, "Allowed.")
