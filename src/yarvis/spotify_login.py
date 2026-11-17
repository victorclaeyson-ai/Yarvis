"""``yarvis spotify login``: your playlists and your player (with your consent, once).

Spotify's authorization code flow, run once from the terminal:

1. Your browser opens Spotify's consent page: Yarvis asks to read your playlists and to
   control playback (play, pause, skip, volume). It can never change your playlists.
2. Spotify sends the browser back to ``http://127.0.0.1:8888/callback`` (the redirect URI
   registered in your Spotify developer app). For those few moments Yarvis listens on that
   loopback address, only for this one answer, and checks it carries the random ``state``
   it sent, so no other page can slip in a code of its own.
3. The code is exchanged for a refresh token, which goes into the OS keychain.

Yarvis keeps no server running: the listener exists only while you log in.
"""

from __future__ import annotations

import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Final, cast
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx2

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import SecretStr

AUTHORIZE_URL: Final = "https://accounts.spotify.com/authorize"
TOKEN_URL: Final = "https://accounts.spotify.com/api/token"  # noqa: S105 - a URL
CALLBACK_HOST: Final = "127.0.0.1"
CALLBACK_PORT: Final = 8888
REDIRECT_URI: Final = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/callback"
SCOPES: Final = (
    "playlist-read-private",
    "playlist-read-collaborative",
    "user-read-playback-state",  # which devices there are, what is playing
    "user-modify-playback-state",  # play, pause, skip, volume: in the background, any device
)
"""Reading playlists and controlling playback; never changing the user's library."""
LOGIN_TIMEOUT_S: Final = 180.0

_DONE_PAGE: Final = (
    b"<!doctype html><meta charset=utf-8><title>Yarvis</title>"
    b"<body style='font:16px system-ui;margin:3em'><h1>Done.</h1>"
    b"<p>You can close this tab and go back to the terminal.</p>"
)


class LoginError(Exception):
    """The login did not complete."""


def new_state() -> str:
    """A random value that ties Spotify's answer to this login attempt."""
    return secrets.token_urlsafe(24)


def authorize_url(client_id: str, state: str) -> str:
    """Spotify's consent page for this app, these scopes and this state."""
    query = urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": " ".join(SCOPES),
            "state": state,
        }
    )
    return f"{AUTHORIZE_URL}?{query}"


def code_from_callback(target: str, state: str) -> str:
    """The authorization code in Spotify's redirect, after checking the state.

    Raises:
        LoginError: If the user declined, or the answer is not for this login.
    """
    query = parse_qs(urlsplit(target).query)
    if not secrets.compare_digest(query.get("state", [""])[0].encode(), state.encode()):
        msg = "Spotify's answer did not match this login. Try again."
        raise LoginError(msg)
    if error := query.get("error", [""])[0]:
        msg = "You did not give access." if error == "access_denied" else f"Spotify said: {error}"
        raise LoginError(msg)
    code = query.get("code", [""])[0]
    if not code:
        msg = "Spotify's answer had no code. Try again."
        raise LoginError(msg)
    return code


def exchange_code(client_id: str, client_secret: SecretStr, code: str, http: httpx2.Client) -> str:
    """Trade the code for a refresh token.

    Raises:
        LoginError: If Spotify refuses.
    """
    try:
        response = http.post(
            TOKEN_URL,
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI},
            auth=(client_id, client_secret.get_secret_value()),
        )
    except httpx2.HTTPError as exc:
        msg = "Spotify could not be reached. Check the internet connection."
        raise LoginError(msg) from exc
    if response.status_code != 200:  # noqa: PLR2004
        msg = (
            f"Spotify refused the login (HTTP {response.status_code}). Check that the app's "
            f"redirect URI is exactly {REDIRECT_URI}."
        )
        raise LoginError(msg)
    payload: object = response.json()
    token = cast("dict[str, object]", payload).get("refresh_token") if _is_dict(payload) else None
    if not isinstance(token, str) or not token:
        msg = "Spotify did not hand out a refresh token."
        raise LoginError(msg)
    return token


def _is_dict(value: object) -> bool:
    return isinstance(value, dict)


class _CallbackServer(HTTPServer):
    target: str | None = None


class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/callback":
            cast("_CallbackServer", self.server).target = self.path
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(_DONE_PAGE)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence: the URL holds the code, which must not end up in any log."""


def wait_for_callback(
    *,
    timeout_s: float = LOGIN_TIMEOUT_S,
    port: int = CALLBACK_PORT,
    on_listening: Callable[[int], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> str:
    """Listen on the loopback address until Spotify's redirect arrives; return its path.

    Raises:
        LoginError: If the port is taken or nothing arrives in time.
    """
    try:
        server = _CallbackServer((CALLBACK_HOST, port), _CallbackHandler)
    except OSError as exc:
        msg = f"Port {port} is in use by another program; close it and try again."
        raise LoginError(msg) from exc
    with server:
        server.timeout = 0.5
        if on_listening is not None:
            on_listening(server.server_address[1])
        deadline = clock() + timeout_s
        while clock() < deadline:
            server.handle_request()
            if server.target is not None:
                return server.target
    msg = "No answer from Spotify in time. Try again."
    raise LoginError(msg)


def _listen(on_listening: Callable[[int], None]) -> str:
    return wait_for_callback(on_listening=on_listening)


def login(
    client_id: str,
    client_secret: SecretStr,
    *,
    open_url: Callable[[str], object],
    receive: Callable[[Callable[[int], None]], str] = _listen,
    http: httpx2.Client | None = None,
    state: str | None = None,
) -> str:
    """Run the whole flow; return the refresh token.

    The browser is only sent to Spotify once the listener is ready, so even an instant
    redirect (consent given earlier) finds it.

    Raises:
        LoginError: If any step fails.
    """
    state = state or new_state()
    url = authorize_url(client_id, state)

    def ready(_port: int) -> None:
        open_url(url)

    target = receive(ready)
    code = code_from_callback(target, state)
    with http or httpx2.Client(timeout=15.0, follow_redirects=False) as client:
        return exchange_code(client_id, client_secret, code, client)
