"""``yarvis spotify login``: the consent flow and its loopback listener."""

from __future__ import annotations

import socket
import threading
from typing import TYPE_CHECKING
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from pydantic import SecretStr

from yarvis.security.secrets import KEYRING_SERVICE
from yarvis.spotify_login import (
    REDIRECT_URI,
    TOKEN_URL,
    LoginError,
    authorize_url,
    code_from_callback,
    exchange_code,
    login,
    new_state,
    wait_for_callback,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fakes import FakeKeyring, Invoke
    from yarvis.cli.app import Services

CLIENT_ID = "0123456789abcdef0123456789abcdef"
SECRET = SecretStr("fedcba9876543210fedcba9876543210")  # gitleaks:allow - a test value
STATE = "state-1234"


def test_the_consent_page_asks_for_playlists_and_the_player_only() -> None:
    url = urlsplit(authorize_url(CLIENT_ID, STATE))
    assert (url.scheme, url.netloc, url.path) == ("https", "accounts.spotify.com", "/authorize")
    query = parse_qs(url.query)
    assert query["client_id"] == [CLIENT_ID]
    assert query["redirect_uri"] == [REDIRECT_URI] == ["http://127.0.0.1:8888/callback"]
    assert query["scope"] == [
        "playlist-read-private playlist-read-collaborative user-read-playback-state "
        "user-modify-playback-state"
    ]
    assert not any("modify" in s and "playback" not in s for s in query["scope"][0].split())
    assert query["response_type"] == ["code"]
    assert query["state"] == [STATE]
    assert new_state() != new_state() and len(new_state()) >= 32


def test_the_answer_must_be_for_this_login() -> None:
    assert code_from_callback(f"/callback?code=abc&state={STATE}", STATE) == "abc"
    for target, message in [
        ("/callback?code=abc&state=forged", "did not match"),
        ("/callback?code=abc", "did not match"),
        ("/callback?code=abc&state=%C3%A9t%C3%A9", "did not match"),  # not ASCII
        (f"/callback?error=access_denied&state={STATE}", "did not give access"),
        (f"/callback?error=server_error&state={STATE}", "Spotify said: server_error"),
        (f"/callback?state={STATE}", "no code"),
    ]:
        with pytest.raises(LoginError, match=message):
            code_from_callback(target, STATE)


def token_endpoint(status: int, body: object) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == TOKEN_URL
        assert request.headers["authorization"].startswith("Basic ")
        form = parse_qs(request.content.decode())
        assert form == {
            "grant_type": ["authorization_code"],
            "code": ["abc"],
            "redirect_uri": [REDIRECT_URI],
        }
        return httpx2.Response(status, json=body)

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_the_code_is_exchanged_for_a_refresh_token() -> None:
    http = token_endpoint(200, {"access_token": "a", "refresh_token": "refresh-me"})
    assert exchange_code(CLIENT_ID, SECRET, "abc", http) == "refresh-me"
    with pytest.raises(LoginError, match="redirect URI is exactly"):
        exchange_code(CLIENT_ID, SECRET, "abc", token_endpoint(400, {"error": "invalid_grant"}))
    with pytest.raises(LoginError, match="did not hand out"):
        exchange_code(CLIENT_ID, SECRET, "abc", token_endpoint(200, {"access_token": "a"}))
    with pytest.raises(LoginError, match="did not hand out"):
        exchange_code(CLIENT_ID, SECRET, "abc", token_endpoint(200, ["not", "a", "dict"]))

    def offline(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("offline")

    with pytest.raises(LoginError, match="could not be reached"):
        exchange_code(
            CLIENT_ID, SECRET, "abc", httpx2.Client(transport=httpx2.MockTransport(offline))
        )


def test_the_listener_waits_for_spotify_on_loopback_only() -> None:
    answers: list[int] = []

    def visit(port: int) -> None:
        def run() -> None:
            with httpx2.Client(base_url=f"http://127.0.0.1:{port}") as browser:
                answers.append(browser.get("/favicon.ico").status_code)
                page = browser.get(f"/callback?code=abc&state={STATE}")
                answers.append(page.status_code)
                assert b"close this tab" in page.content

        threading.Thread(target=run, daemon=True).start()

    target = wait_for_callback(port=0, timeout_s=10, on_listening=visit)
    assert target == f"/callback?code=abc&state={STATE}"
    assert answers[0] == 404  # anything else is turned away


def test_the_listener_gives_up() -> None:
    with pytest.raises(LoginError, match="in time"):
        wait_for_callback(port=0, timeout_s=0.2)
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        with pytest.raises(LoginError, match="in use"):
            wait_for_callback(port=taken.getsockname()[1], timeout_s=0.2)


def test_the_whole_flow_opens_the_browser_only_once_listening() -> None:
    events: list[str] = []

    def receive(ready: Callable[[int], None]) -> str:
        events.append("listening")
        ready(8888)
        return f"/callback?code=abc&state={STATE}"

    token = login(
        CLIENT_ID,
        SECRET,
        open_url=lambda url: events.append(f"opened {urlsplit(url).netloc}"),
        receive=receive,
        http=token_endpoint(200, {"refresh_token": "refresh-me"}),
        state=STATE,
    )
    assert token == "refresh-me"  # noqa: S105 - a test value
    assert events == ["listening", "opened accounts.spotify.com"]


# --------------------------------------------------------------------------- the command


def test_login_command(invoke: Invoke, services: Services, fake_keyring: FakeKeyring) -> None:
    assert "yarvis key set spotify" in invoke("spotify", "login").output  # no app keys yet
    fake_keyring.store[(KEYRING_SERVICE, "spotify")] = f"{CLIENT_ID}:{SECRET.get_secret_value()}"
    calls: list[tuple[str, str]] = []

    def fake_login(client_id: str, secret: SecretStr) -> str:
        calls.append((client_id, secret.get_secret_value()))
        return "refresh-token-from-spotify-0123456789"

    services.spotify_login = fake_login
    result = invoke("spotify", "login")
    assert result.exit_code == 0, result.output
    assert "Connected" in result.output
    assert calls == [(CLIENT_ID, SECRET.get_secret_value())]
    assert fake_keyring.store[(KEYRING_SERVICE, "spotify-login")] == (
        "refresh-token-from-spotify-0123456789"
    )

    def refused(client_id: str, secret: SecretStr) -> str:
        raise LoginError("You did not give access.")

    services.spotify_login = refused
    result = invoke("spotify", "login")
    assert result.exit_code == 1
    assert "Spotify login failed: You did not give access." in result.output


def test_the_login_token_cannot_be_typed_in(invoke: Invoke) -> None:
    result = invoke("key", "set", "spotify-login")
    assert result.exit_code == 1
    assert "yarvis spotify login" in result.output
