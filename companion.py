#!/usr/bin/env python3
"""
WeEvolve 3D Owl Companion Server
=================================
Serves the interactive 3D owl companion viewer on localhost:8888.
Uses only Python stdlib -- zero external dependencies.

Usage:
    python -m weevolve companion
    python companion.py

Endpoints:
    GET  /              -> companion/index.html
    GET  /api/stats     -> evolution state JSON
    GET  /api/state     -> owl state (idle/listening/speaking/thinking/celebrating)
    POST /api/state     -> set owl state {"state": "listening"}
"""

import http.server
import json
import os
import signal
import socketserver
import sys
import threading
import webbrowser
from pathlib import Path

from weevolve.config import EVOLUTION_STATE_PATH

COMPANION_DIR = Path(__file__).parent / "companion"
PORT = 8888

# Shared mutable owl state (thread-safe via GIL for simple reads/writes)
_owl_state = {"state": "idle", "seed_phase": "PERCEIVE"}

SEED_PHASES = [
    "PERCEIVE", "CONNECT", "LEARN", "QUESTION",
    "EXPAND", "SHARE", "RECEIVE", "IMPROVE",
]


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


class CompanionHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files from companion/ and JSON API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(COMPANION_DIR), **kwargs)

    def do_GET(self):
        if self.path == "/api/stats":
            self._json_response(_load_evolution_state())
        elif self.path == "/api/state":
            self._json_response(_owl_state)
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/state":
            length = int(self.headers.get("Content-Length", 0))
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
        else:
            self.send_error(404)

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
        # Silence noisy HTTP logs; keep errors
        pass


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def launch_companion():
    """Start the companion server, open the browser, block until Ctrl+C."""
    if not COMPANION_DIR.exists():
        print(f"  ERROR: companion directory missing at {COMPANION_DIR}")
        return

    server = _ReusableTCPServer(("", PORT), CompanionHandler)

    def _shutdown(signum, frame):
        print("\n  Companion shutting down...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    url = f"http://localhost:{PORT}"
    print(f"  Owl Companion live at {url}")
    print("  Press Ctrl+C to stop.\n")

    webbrowser.open(url)
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    launch_companion()
