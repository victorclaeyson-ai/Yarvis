"""The shape of a tool: a name, a description, typed arguments, a permission tier."""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar, cast

from pydantic import BaseModel, ConfigDict

from yarvis.llm.base import ToolSpec

if TYPE_CHECKING:
    from collections.abc import Mapping


class Tier(IntEnum):
    """How much harm a tool can do (design doc §5.3)."""

    READ = 0
    """Read-only and local: runs automatically."""
    REVERSIBLE = 1
    """A side effect that is easy to undo: runs automatically, and the user sees it."""
    EXTERNAL = 2
    """External or hard to undo: runs only after the user says "yes"."""
    DANGEROUS = 3
    """Shell commands, deleting files, payments: disabled."""


class Arguments(BaseModel):
    """Base for tool arguments: strict, immutable, no unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ToolError(Exception):
    """A tool could not do its job. The message is shown to the model (and maybe the user)."""


class Tool[A: Arguments](ABC):
    """Something Yarvis can do. Subclasses set the class attributes and implement ``run``."""

    name: ClassVar[str]
    description: ClassVar[str]
    tier: ClassVar[Tier]
    Args: ClassVar[type[Arguments]]
    untrusted_output: ClassVar[bool] = False
    """True if results contain outside content (web pages, messages): the turn is tainted."""
    timeout_s: ClassVar[float | None] = None
    """How long one call may take; None: the toolbox's default."""

    def spec(self) -> ToolSpec:
        """The tool as the model sees it."""
        return ToolSpec(self.name, self.description, self.Args.model_json_schema())

    def parse(self, raw: Mapping[str, Any]) -> A:
        """Validate the model's arguments (raises ``pydantic.ValidationError``)."""
        return cast("A", self.Args.model_validate(dict(raw)))

    async def prepare(self, args: A) -> A:
        """Resolve what the call refers to, before the gate and the confirmation question.

        "Email Thomas" becomes the address in the user's contacts, so the question read back
        ("Should I email Thomas Peeters, thomas@example.com, ...?") and what runs after the
        user's "yes" are exactly the same. Must not change anything.

        Raises:
            ToolError: If it cannot be resolved (no such contact, several matches).
        """
        return args

    def sends_data_out(self, args: A) -> bool:
        """Whether this call can carry text to a destination the model chose (a URL).

        After the turn has read outside content, such calls need the user's "yes" even when
        harmless otherwise: an email could ask to open ``https://evil.example/?notes=...``.
        """
        return False

    @abstractmethod
    async def run(self, args: A) -> str:
        """Do it; return a short, factual result for the model.

        Raises:
            ToolError: With a message the model can relay.
        """

    def starts_playback(self, args: A) -> bool:
        """Whether this call starts music or a video.

        Sound starting mid-conversation would otherwise reach the microphone: Yarvis stops
        listening for a plain voice interruption and ends the conversation after the reply.
        """
        return False

    @abstractmethod
    def describe(self, args: A) -> str:
        """What this call does, as a verb phrase ("set a 10-minute timer").

        Shown to the user for every action and read back for confirmations, so it must be
        built from the validated arguments by code, never from model prose.
        """
