"""Scrub secrets out of text before it is logged or displayed.

Two layers:

1. **Known values** — every secret Yarvis loads is registered, so it is redacted verbatim
   wherever it appears.
2. **Patterns** — common credential formats are redacted even if Yarvis never loaded them
   (for example a token echoed back in an error from a third-party service).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Final, override

REDACTED: Final = "[REDACTED]"

# Short values would cause false positives everywhere; real secrets are much longer.
MIN_KNOWN_SECRET_LENGTH: Final = 8

DEFAULT_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),  # Anthropic
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),  # OpenAI-style
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),  # GitHub
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),  # Slack
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),  # AWS access key id
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]{16,}=*"),  # HTTP bearer tokens
)


class Redactor:
    """Thread-safe redaction of registered secrets and known credential patterns."""

    def __init__(self, patterns: tuple[re.Pattern[str], ...] = DEFAULT_PATTERNS) -> None:
        self._patterns = patterns
        self._known: set[str] = set()
        self._lock = threading.Lock()

    def register(self, secret: str) -> None:
        """Redact ``secret`` verbatim from now on. Very short values are ignored."""
        if len(secret) < MIN_KNOWN_SECRET_LENGTH:
            return
        with self._lock:
            self._known.add(secret)

    def redact(self, text: str) -> str:
        """Return ``text`` with every secret replaced by ``[REDACTED]``."""
        with self._lock:
            # Longest first, so a secret that contains another is removed whole.
            known = sorted(self._known, key=len, reverse=True)
        for secret in known:
            text = text.replace(secret, REDACTED)
        for pattern in self._patterns:
            text = pattern.sub(REDACTED, text)
        return text

    def contains_secret(self, text: str) -> bool:
        """Return True if ``text`` contains anything that would be redacted."""
        return self.redact(text) != text


class RedactingFilter(logging.Filter):
    """Logging filter that redacts the message, arguments and traceback of every record.

    Attach it to *handlers* (not loggers): logger-level filters do not run for records
    propagated from child loggers.
    """

    def __init__(self, redactor: Redactor) -> None:
        super().__init__()
        self._redactor = redactor

    @override
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redactor.redact(record.getMessage())
        record.args = None
        if record.exc_info:
            text = logging.Formatter().formatException(record.exc_info)
            record.exc_text = self._redactor.redact(text)
            record.exc_info = None
        if record.stack_info:
            record.stack_info = self._redactor.redact(record.stack_info)
        return True


# Process-wide redactor used by logging and the CLI.
default_redactor: Final = Redactor()
