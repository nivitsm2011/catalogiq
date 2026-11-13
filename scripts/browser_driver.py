# ruff: noqa: E501  (JavaScript snippets and long docstrings)
"""A tiny Chrome DevTools driver for Microsoft Edge (headless) used to click through and screenshot the app.

No new dependency: it speaks the DevTools protocol over the websocket client that ships with Streamlit
(tornado). Used by ``take_screenshots.py`` and for the manual click-through checks of Phase 8.
"""

from __future__ import annotations

import asyncio
import base64
import json
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import requests
from tornado.websocket import websocket_connect

EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


class Browser:
    """Headless Edge with one tab. Use as ``async with Browser() as b``."""

    def __init__(self, width: int = 1440, height: int = 1000) -> None:
        self.width, self.height = width, height
        with (
            socket.socket() as sock
        ):  # a free port per browser, so a slow-closing Edge cannot collide
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
        self.proc: subprocess.Popen[bytes] | None = None
        self.ws: Any = None
        self._id = 0
        self._tmp = tempfile.mkdtemp(prefix="edge_")
        self.console: list[str] = []

    async def __aenter__(self) -> Browser:
        self.proc = subprocess.Popen(
            [
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--hide-scrollbars",
                f"--remote-debugging-port={self.port}",
                f"--user-data-dir={self._tmp}",
                "--no-first-run",
                "--use-fake-ui-for-media-stream",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(60):
            try:
                pages = requests.get(f"http://127.0.0.1:{self.port}/json/list", timeout=1).json()
                target = next(p for p in pages if p["type"] == "page")
                break
            except (requests.RequestException, StopIteration):
                await asyncio.sleep(0.5)
        else:
            raise RuntimeError("Edge did not start")
        self.ws = await websocket_connect(
            target["webSocketDebuggerUrl"], max_message_size=200 * 1024 * 1024
        )
        await self.send("Page.enable")
        await self.send("Runtime.enable")
        await self.resize(self.width, self.height)
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self.ws is not None:
            self.ws.close()
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        shutil.rmtree(self._tmp, ignore_errors=True)

    async def send(self, method: str, **params: Any) -> dict[str, Any]:
        """Send one DevTools command and wait for its answer."""
        self._id += 1
        await self.ws.write_message(
            json.dumps({"id": self._id, "method": method, "params": params})
        )
        while True:
            msg = json.loads(await self.ws.read_message())
            if msg.get("method") == "Runtime.consoleAPICalled" and msg["params"]["type"] in (
                "error",
                "warning",
            ):
                self.console.append(
                    " ".join(
                        str(a.get("value", a.get("description", ""))) for a in msg["params"]["args"]
                    )
                )
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def resize(self, width: int, height: int) -> None:
        """Set the viewport size."""
        await self.send(
            "Emulation.setDeviceMetricsOverride",
            width=width,
            height=height,
            deviceScaleFactor=1,
            mobile=False,
        )

    async def goto(self, url: str, wait: float = 6.0) -> None:
        """Open a URL and wait (real time) for the page to render."""
        await self.send("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def js(self, expression: str) -> Any:
        """Evaluate JavaScript in the page and return the value."""
        res = await self.send(
            "Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True
        )
        return res.get("result", {}).get("value")

    async def text(self) -> str:
        """Visible text of the page."""
        return str(await self.js("document.body.innerText"))

    async def click_text(
        self, label: str, tag: str = "button", nth: int = 0, wait: float = 20.0
    ) -> bool:
        """Click the nth visible element of ``tag`` whose text contains ``label`` (polls until it appears)."""
        script = f"""(() => {{ const els = [...document.querySelectorAll({json.dumps(tag)})].filter(e => e.innerText.includes({json.dumps(label)}) && e.offsetParent !== null);
        if (els.length <= {nth}) return false; els[{nth}].scrollIntoView({{block:'center'}}); els[{nth}].click(); return true; }})()"""
        end = time.time() + wait
        while True:
            if await self.js(script):
                return True
            if time.time() > end:
                return False
            await asyncio.sleep(0.7)

    async def wait_for_text(self, needles: list[str], timeout: float = 45.0) -> list[str]:
        """Wait until every needle is on the page; returns the ones still missing after the timeout."""
        end = time.time() + timeout
        while True:
            text = (await self.text()).lower()
            missing = [n for n in needles if n.lower() not in text]
            if not missing or time.time() > end:
                return missing
            await asyncio.sleep(1.0)

    async def upload(self, path: Path) -> bool:
        """Put a local file into the first visible file input (Streamlit's uploader)."""
        doc = await self.send("DOM.getDocument")
        found = await self.send(
            "DOM.querySelectorAll", nodeId=doc["root"]["nodeId"], selector='input[type="file"]'
        )
        if not found["nodeIds"]:
            return False
        await self.send("DOM.setFileInputFiles", files=[str(path)], nodeId=found["nodeIds"][0])
        return True

    async def mouse_click(self, x: float, y: float) -> None:
        """A real left click at page coordinates."""
        for kind in ("mousePressed", "mouseReleased"):
            await self.send(
                "Input.dispatchMouseEvent", type=kind, x=x, y=y, button="left", clickCount=1
            )

    async def select_option(self, label: str, option_text: str) -> bool:
        """Open the Streamlit selectbox named ``label`` and pick ``option_text``."""
        js_rect = f"""(() => {{ const box = [...document.querySelectorAll('[data-testid="stSelectbox"]')].find(e => (e.querySelector('input') || {{}}).ariaLabel === {json.dumps(label)});
        if (!box) return null; const el = box.querySelector('button') || box.querySelector('input'); el.scrollIntoView({{block:'center'}}); const r = el.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }})()"""
        for _ in range(20):  # the dropdown may not be rendered yet
            if await self.js(js_rect):
                break
            await asyncio.sleep(0.75)
        else:
            return False
        await asyncio.sleep(0.4)
        await self.mouse_click(*await self.js(js_rect))
        await asyncio.sleep(1.0)
        return bool(
            await self.js(
                f"""(() => {{ const o = [...document.querySelectorAll('[role="option"]')].find(e => e.innerText.trim() === {json.dumps(option_text)}); if (!o) return false; o.click(); return true; }})()"""
            )
        )

    async def type_chat(self, text: str) -> bool:
        """Type into Streamlit's chat input and press Enter."""
        ok = await self.js(
            """(() => { const t = document.querySelector('textarea[data-testid="stChatInputTextArea"]'); if (!t) return false; t.focus(); return true; })()"""
        )
        if not ok:
            return False
        await self.send("Input.insertText", text=text)
        await self.send(
            "Input.dispatchKeyEvent",
            type="keyDown",
            key="Enter",
            code="Enter",
            windowsVirtualKeyCode=13,
        )
        await self.send(
            "Input.dispatchKeyEvent",
            type="keyUp",
            key="Enter",
            code="Enter",
            windowsVirtualKeyCode=13,
        )
        return True

    async def errors(self) -> list[str]:
        """Streamlit exception boxes currently shown on the page."""
        return list(
            await self.js(
                '[...document.querySelectorAll(\'[data-testid="stException"], [data-testid="stAlertContainer"]\')].map(e => e.innerText).filter(t => /Error|Traceback|exception/i.test(t))'
            )
            or []
        )

    async def wait_idle(self, timeout: float = 90.0) -> None:
        """Wait until Streamlit is no longer running its script."""
        end = time.time() + timeout
        await asyncio.sleep(1.0)
        while time.time() < end:
            running = await self.js("!!document.querySelector('[data-testid=\"stStatusWidget\"]')")
            if not running:
                await asyncio.sleep(1.0)
                if not await self.js(
                    "!!document.querySelector('[data-testid=\"stStatusWidget\"]')"
                ):
                    return
            await asyncio.sleep(1.0)

    async def screenshot(self, path: Path, full_page: bool = True, max_height: int = 6000) -> Path:
        """Save a PNG; with ``full_page`` the viewport is stretched to the page's scroll height first."""
        if full_page:
            height = await self.js(
                "(document.querySelector('[data-testid=\"stMain\"]') || document.documentElement).scrollHeight"
            )
            for _ in range(3):  # content can grow as images load, so measure again after resizing
                await self.resize(self.width, min(int(height or self.height), max_height))
                await asyncio.sleep(1.5)
                height = await self.js(
                    "(document.querySelector('[data-testid=\"stMain\"]') || document.documentElement).scrollHeight"
                )
        res = await self.send("Page.captureScreenshot", format="png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(res["data"]))
        await self.resize(self.width, self.height)
        return path
