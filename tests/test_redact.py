from __future__ import annotations

import io
import logging

import pytest
from hypothesis import given
from hypothesis import strategies as st

from yarvis.security.redact import REDACTED, RedactingFilter, Redactor

SECRET_ALPHABET = st.characters(categories=["Ll", "Nd"])


def test_registered_secret_is_redacted(redactor: Redactor) -> None:
    redactor.register("hunter2-but-longer")  # gitleaks:allow
    assert redactor.redact("token=hunter2-but-longer;") == f"token={REDACTED};"  # gitleaks:allow


def test_short_values_are_not_registered(redactor: Redactor) -> None:
    redactor.register("abc")
    assert redactor.redact("abc") == "abc"


def test_longest_secret_is_removed_whole(redactor: Redactor) -> None:
    redactor.register("abcdefgh")
    redactor.register("abcdefgh-ijklmnop")
    assert redactor.redact("abcdefgh-ijklmnop") == REDACTED


@pytest.mark.parametrize(
    "secret",
    [
        "sk-ant-api03-AbCdEf123456",  # gitleaks:allow
        "sk-proj-abcdefghijklmnopqrstuv",  # gitleaks:allow
        "ghp_abcdefghijklmnopqrstuvwxyz0123",  # gitleaks:allow
        "xoxb-1234567890-abcdef",  # gitleaks:allow
        "AKIAIOSFODNN7EXAMPLE",  # gitleaks:allow
        "Bearer abcdefghijklmnop.qrstuv",  # gitleaks:allow
    ],
)
def test_known_credential_formats_are_redacted(redactor: Redactor, secret: str) -> None:
    redacted = redactor.redact(f"error from server: {secret} rejected")
    assert secret not in redacted
    assert REDACTED in redacted


def test_ordinary_text_is_untouched(redactor: Redactor) -> None:
    text = "Set a timer for 10 minutes and ask Thomas about the risk-assessment doc."
    assert redactor.redact(text) == text
    assert not redactor.contains_secret(text)


@given(secret=st.text(SECRET_ALPHABET, min_size=8, max_size=40), prefix=st.text(), suffix=st.text())
def test_registered_secret_never_survives(secret: str, prefix: str, suffix: str) -> None:
    redactor = Redactor()
    redactor.register(secret)
    assert secret not in redactor.redact(prefix + secret + suffix)


def _capture(redactor: Redactor) -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter(redactor))
    logger = logging.getLogger(f"test-redact-{id(stream)}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_filter_redacts_format_arguments(redactor: Redactor) -> None:
    redactor.register("super-secret-value")
    logger, stream = _capture(redactor)
    logger.info("key is %s", "super-secret-value")
    assert stream.getvalue().strip() == f"key is {REDACTED}"


def test_filter_redacts_tracebacks(redactor: Redactor) -> None:
    redactor.register("super-secret-value")
    logger, stream = _capture(redactor)
    try:
        raise ValueError("bad key super-secret-value")
    except ValueError:
        logger.exception("request failed")
    output = stream.getvalue()
    assert "super-secret-value" not in output
    assert "ValueError" in output


def test_filter_redacts_stack_info(redactor: Redactor) -> None:
    redactor.register("super-secret-value")
    logger, stream = _capture(redactor)
    logger.info("hello super-secret-value", stack_info=True)
    assert "super-secret-value" not in stream.getvalue()
