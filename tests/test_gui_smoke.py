"""End-to-end smoke test of the real app window (opt-in: it opens a window).

Run on Linux with:  YARVIS_GUI_TESTS=1 xvfb-run -a uv run pytest tests/test_gui_smoke.py
(needs the GTK WebKit bindings: apt install python3-gi gir1.2-webkit2-4.1, and a venv
created with --system-site-packages so PyGObject is importable).

This test exists because a too-strict Content-Security-Policy once silently broke the
talk button: every unit test passed, but the real window could not call into Python.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pydantic import SecretStr

from fakes import FakeLLM, FakeWakeWord
from test_runtime_and_run import NO_WAKE_WORD, FakeBackends, FakeMic
from test_speech_engines import frames, speech_clip

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from yarvis.speech.base import AudioArray

pytestmark = pytest.mark.skipif(
    os.environ.get("YARVIS_GUI_TESTS") != "1", reason="opens a real window; set YARVIS_GUI_TESTS=1"
)


def test_real_window_end_to_end(tmp_path: Path) -> None:  # pragma: no cover - opt-in
    webview = pytest.importorskip("webview")
    import yarvis.app.window as appwin

    captured: dict[str, Any] = {}
    real_create = webview.create_window

    def capture(*args: Any, **kwargs: Any) -> Any:
        window = real_create(*args, **kwargs)
        captured["window"] = window
        return window

    webview.create_window = capture  # restored below

    class SlowNoiseMic(FakeMic):
        """Real-time paced background noise: never contains speech."""

        async def __anext__(self) -> AudioArray:
            await asyncio.sleep(0.032)
            return np.random.default_rng().normal(0, 0.002, 512).astype(np.float32)

    results: list[str] = []

    def driver() -> None:
        while "window" not in captured:
            time.sleep(0.05)
        w = captured["window"]
        w.events.loaded.wait(15)

        def js(code: str) -> Any:
            return w.evaluate_js(code)

        def pill() -> str:
            return str(js("document.getElementById('tag').textContent"))

        def log() -> str:
            return js(
                "Array.from(document.querySelectorAll('#log .line .txt')).map(e => e.textContent).join(' | ')"
            )

        def wait_for(predicate: Callable[[], bool], timeout: float, label: str) -> bool:
            end = time.time() + timeout
            while time.time() < end:
                if predicate():
                    results.append(f"OK   {label}")
                    return True
                time.sleep(0.1)
            results.append(f"FAIL {label}  (pill={pill()!r}, log={log()!r})")
            return False

        wait_for(lambda: pill() == "ONLINE", 20, "app becomes Ready")
        models = js("document.getElementById('models').textContent")
        ok = "haiku" in models and "base.en" in models
        results.append(("OK   " if ok else "FAIL ") + f"models shown: {models!r}")
        api_ok = js(
            "typeof window.pywebview.api.talk === 'function' && typeof window.pywebview.api.stop === 'function'"
        )
        results.append(
            ("OK   " if api_ok else "FAIL ") + "pywebview API functions exist in the page"
        )

        # 1. Click talk: full voice turn.
        js("document.getElementById('talk').click()")
        wait_for(lambda: "what time is it" in log(), 10, "voice turn: transcript appears")
        wait_for(lambda: "It is half past two." in log(), 10, "voice turn: reply appears")
        wait_for(lambda: pill() == "ONLINE", 10, "voice turn: back to Ready")
        timing = js("document.getElementById('latency').textContent")
        results.append(
            ("OK   " if "TO FIRST WORD" in timing else "FAIL ") + f"timing shown: {timing!r}"
        )

        # 2. Space key, then stop with Space while listening.
        fakes.mic = SlowNoiseMic()
        js("document.dispatchEvent(new KeyboardEvent('keydown', {code: 'Space', key: ' '}))")
        wait_for(lambda: pill() == "LISTENING", 5, "Space starts listening")
        is_stop = js("document.getElementById('talk').classList.contains('stop')")
        results.append(("OK   " if is_stop else "FAIL ") + "button turns into a stop button")
        time.sleep(1.0)
        js("document.dispatchEvent(new KeyboardEvent('keydown', {code: 'Space', key: ' '}))")
        wait_for(lambda: "// stopped" in log(), 5, "Space stops listening")
        wait_for(lambda: pill() == "ONLINE", 5, "back to Ready after stop")

        # 3. Typed message.
        js(
            "const t = document.getElementById('text'); t.value = 'hello'; t.dispatchEvent(new Event('input'));"
        )
        js(
            "document.getElementById('text').dispatchEvent(new KeyboardEvent('keydown', {key: 'Enter'}))"
        )
        wait_for(lambda: "Good to hear from you." in log(), 10, "typed message gets a reply")
        wait_for(lambda: pill() == "ONLINE", 10, "back to Ready after typed message")
        w.destroy()

    fakes = FakeBackends(FakeLLM(["It is half past two."], ["Good to hear ", "from you."]))
    threading.Thread(target=driver, daemon=True).start()
    try:
        # This walk-through swaps microphones between steps, so it runs without the wake
        # word (which keeps one stream open for the whole session).
        appwin.run_app(
            NO_WAKE_WORD, api_key=SecretStr("k"), cache_dir=tmp_path, backends=fakes.build()
        )
    finally:
        webview.create_window = real_create
    failures = [r for r in results if r.startswith("FAIL")]
    assert results, "the driver never ran"
    assert not failures, "\n".join(results)


class WakeMic(FakeMic):
    """Real-time paced noise; says "Hey Yarvis, what time is it..." once ``speak`` is set."""

    def __init__(self) -> None:
        super().__init__()
        self.speak = threading.Event()
        self._said = False

    async def __anext__(self) -> AudioArray:
        await asyncio.sleep(0.032)
        if self.speak.is_set() and not self._said:
            self._said = True
            trigger = np.full(512, 0.01, np.float32)
            trigger[0] = FakeWakeWord.TRIGGER
            self._pending = [trigger, *frames(speech_clip())]
        if self._pending:
            return self._pending.pop(0)
        return np.random.default_rng().normal(0, 0.002, 512).astype(np.float32)


def test_real_window_wake_word(tmp_path: Path) -> None:  # pragma: no cover - opt-in
    webview = pytest.importorskip("webview")
    import yarvis.app.window as appwin
    from yarvis.config import Config

    captured: dict[str, Any] = {}
    real_create = webview.create_window

    def capture(*args: Any, **kwargs: Any) -> Any:
        window = real_create(*args, **kwargs)
        captured["window"] = window
        return window

    webview.create_window = capture
    fakes = FakeBackends(FakeLLM(["It is half past two."]))
    mic = WakeMic()
    fakes.mic = mic
    results: list[str] = []

    def driver() -> None:
        while "window" not in captured:
            time.sleep(0.05)
        w = captured["window"]
        w.events.loaded.wait(15)

        def js(code: str) -> Any:
            return w.evaluate_js(code)

        def log() -> str:
            return js(
                "Array.from(document.querySelectorAll('#log .line .txt'))"
                ".map(e => e.textContent).join(' | ')"
            )

        def wait_for(predicate: Callable[[], bool], timeout: float, label: str) -> None:
            end = time.time() + timeout
            while time.time() < end:
                if predicate():
                    results.append(f"OK   {label}")
                    return
                time.sleep(0.1)
            results.append(f"FAIL {label}  (log={log()!r})")

        pill = "document.getElementById('tag').textContent"
        wait_for(lambda: js(pill) == "ONLINE", 20, "app becomes Ready")
        wait_for(lambda: not js("document.getElementById('wake').hidden"), 5, "wake badge shown")
        wait_for(lambda: "HEY YARVIS" in js("document.getElementById('hint').textContent"), 5,
                 "hint mentions the wake word")  # fmt: skip
        mic.speak.set()
        wait_for(lambda: "hey yarvis" in log(), 10, "wake word shows in the log")
        wait_for(lambda: "what time is it" in log(), 15, "request transcribed")
        wait_for(lambda: "It is half past two." in log(), 10, "reply appears")
        wait_for(lambda: js(pill) == "ONLINE", 15, "back to Ready")
        w.destroy()

    threading.Thread(target=driver, daemon=True).start()
    try:
        short = Config.model_validate({"voice": {"follow_up_s": 3}})  # quick back to Ready
        appwin.run_app(short, api_key=SecretStr("k"), cache_dir=tmp_path, backends=fakes.build())
    finally:
        webview.create_window = real_create
    failures = [r for r in results if r.startswith("FAIL")]
    assert results, "the driver never ran"
    assert not failures, "\n".join(results)
