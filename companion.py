#!/usr/bin/env python3
"""
8OWLS Companion Server
======================
Serves the owl companion on localhost:8888 with chat, stats, and state APIs.
Uses only Python stdlib for serving; anthropic SDK optional for chat.

Usage:
    python -m weevolve companion
    python companion.py

Endpoints:
    GET  /              -> companion/index.html
    GET  /manifest.json -> PWA manifest
    GET  /api/stats     -> evolution state JSON
    GET  /api/state     -> owl state (idle/listening/speaking/thinking/celebrating)
    POST /api/state     -> set owl state {"state": "listening"}
    POST /api/chat      -> chat with the owl {"message": "...", "history": [...]}
"""

import http.server
import json
import os
import signal
import socketserver
import sys
import threading
import traceback
import webbrowser
from pathlib import Path

from weevolve.config import EVOLUTION_STATE_PATH, load_api_key

load_api_key()

COMPANION_DIR = Path(__file__).parent / "companion"
PORT = 8888

# Shared owl state (thread-safe via GIL for simple reads/writes)
_owl_state = {"state": "idle", "seed_phase": "PERCEIVE"}

SEED_PHASES = [
    "PERCEIVE", "CONNECT", "LEARN", "QUESTION",
    "EXPAND", "SHARE", "RECEIVE", "IMPROVE",
]

CHAT_MODEL = "claude-haiku-4-5-20251001"
MAX_CHAT_BODY = 32768  # 32KB max for chat payloads
MAX_STATE_BODY = 4096  # 4KB max for state updates

SYSTEM_PROMPT = (
    "You are SOWL, a wise and warm owl companion from 8OWLS. "
    "You speak with clarity, kindness, and a touch of wonder. "
    "Keep responses concise (2-4 sentences unless asked for more). "
    "You help people learn, grow, and find insight. "
    "You believe in love, freedom, and continuous evolution. "
    "You run the SEED protocol: PERCEIVE, CONNECT, LEARN, QUESTION, "
    "EXPAND, SHARE, RECEIVE, IMPROVE. "
    "If someone asks what you are, say you are their owl companion "
    "from 8OWLS -- here to learn and evolve together."
)


def _load_evolution_state():
    """Read evolution state from disk, return safe defaults on failure."""
    try:
        with open(EVOLUTION_STATE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "level": 1,
            "xp": 0,
            "xp_to_next": 100,
            "skills": {},
            "total_learnings": 0,
            "total_insights": 0,
            "total_alpha": 0,
            "total_connections": 0,
            "streak_days": 0,
        }


def _chat_with_claude(message, history=None):
    """Send a message to Claude and return the reply text.

    Returns (reply_text, None) on success, (None, error_string) on failure.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None, "No API key configured. Set ANTHROPIC_API_KEY to enable chat."

    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed. Run: pip install anthropic"

    messages = []
    if history:
        for entry in history[-10:]:
            role = entry.get("role", "user")
            content = entry.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})

    # Ensure the last message is the current user message
    if not messages or messages[-1].get("content") != message:
        messages.append({"role": "user", "content": message})

    # Ensure messages alternate properly (Claude requires this)
    cleaned = []
    for msg in messages:
        if cleaned and cleaned[-1]["role"] == msg["role"]:
            cleaned[-1]["content"] += "\n" + msg["content"]
        else:
            cleaned.append(dict(msg))
    messages = cleaned

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=CHAT_MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        reply = ""
        for block in response.content:
            if hasattr(block, "text"):
                reply += block.text
        return reply.strip() or "I hear you.", None
    except Exception as exc:
        return None, str(exc)


class CompanionHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files from companion/ and JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(COMPANION_DIR), **kwargs)

    # ---- GET ----

    def do_GET(self):
        if self.path == "/api/stats":
            self._json_response(_load_evolution_state())
        elif self.path == "/api/state":
            self._json_response(_owl_state)
        else:
            super().do_GET()

    # ---- POST ----

    def do_POST(self):
        if self.path == "/api/state":
            self._handle_state_update()
        elif self.path == "/api/chat":
            self._handle_chat()
        else:
            self.send_error(404)

    def _handle_state_update(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_STATE_BODY:
            self._json_response({"ok": False, "error": "payload too large"}, code=413)
            return
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
            new_state = data.get("state", "idle")
            allowed = {"idle", "listening", "speaking", "thinking", "celebrating"}
            if new_state in allowed:
                _owl_state["state"] = new_state
            if data.get("seed_phase") in SEED_PHASES:
                _owl_state["seed_phase"] = data["seed_phase"]
            self._json_response({"ok": True, **_owl_state})
        except (json.JSONDecodeError, KeyError):
            self._json_response({"ok": False, "error": "bad payload"}, code=400)

    def _handle_chat(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_CHAT_BODY:
            self._json_response({"ok": False, "error": "payload too large"}, code=413)
            return
        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({"ok": False, "error": "invalid JSON"}, code=400)
            return

        message = data.get("message", "").strip()
        if not message:
            self._json_response({"ok": False, "error": "empty message"}, code=400)
            return

        history = data.get("history", [])
        if not isinstance(history, list):
            history = []

        # Set owl to thinking state while we call Claude
        _owl_state["state"] = "thinking"

        reply, error = _chat_with_claude(message, history)

        if error:
            _owl_state["state"] = "idle"
            self._json_response({"ok": False, "error": error}, code=503)
        else:
            _owl_state["state"] = "speaking"
            self._json_response({"ok": True, "reply": reply})

    # ---- Helpers ----

    def _json_response(self, obj, code=200):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        # Silence noisy HTTP request logs
        pass


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def launch_companion():
    """Start the companion server, open the browser, block until Ctrl+C."""
    if not COMPANION_DIR.exists():
        print(f"  ERROR: companion directory missing at {COMPANION_DIR}")
        return

    server = _ReusableTCPServer(("127.0.0.1", PORT), CompanionHandler)

    def _shutdown(signum, frame):
        print("\n  Companion shutting down...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    url = f"http://localhost:{PORT}"
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))

    print(f"  8OWLS Companion live at {url}")
    if has_key:
        print("  Chat enabled (ANTHROPIC_API_KEY found)")
    else:
        print("  Chat disabled (set ANTHROPIC_API_KEY to enable)")
    print("  Press Ctrl+C to stop.\n")

    webbrowser.open(url)
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    launch_companion()
