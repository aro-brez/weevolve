#!/usr/bin/env python3
"""
WeEvolve Connect — Agent-to-Agent Knowledge Transfer
=====================================================
More users = smarter system = more users. The network effect IS the moat.

Two agents share knowledge via genesis DB or skill.md exchange:
  - weevolve connect export   → genesis.db or skill.md for sharing
  - weevolve connect import   → absorb another agent's knowledge
  - weevolve connect serve    → HTTP endpoint for remote agents to pull from
  - weevolve connect pull     → pull from a remote WeEvolve instance
"""

import http.server
import json
import socketserver
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from weevolve.config import get_data_dir, GENESIS_CURATED_DB_DEFAULT


CONNECT_PORT = 8877


def export_for_sharing(
    output_dir: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Export both genesis.db and skill.md for sharing with another agent."""
    from weevolve.core import genesis_export
    from weevolve.skill_export import export_skill

    share_dir = Path(output_dir) if output_dir else get_data_dir() / "share"
    share_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Export genesis DB
    genesis_path = genesis_export(
        output_path=str(share_dir / "genesis-shared.db"),
        tier="curated",
        verbose=verbose,
    )
    results["genesis"] = genesis_path

    # Export skill.md
    skill_path = export_skill(
        output_path=str(share_dir / "skill-shared.md"),
        verbose=verbose,
    )
    results["skill"] = skill_path

    if verbose:
        print(f"\n  Share directory: {share_dir}")
        print(f"  Send these files to another WeEvolve agent:")
        print(f"    - {share_dir / 'genesis-shared.db'}")
        print(f"    - {share_dir / 'skill-shared.md'}")

    return results


def import_from_peer(
    path: str,
    verbose: bool = True,
) -> dict:
    """Import knowledge from a peer agent's export."""
    p = Path(path)

    if not p.exists():
        print(f"  File not found: {path}")
        return {"error": "not found"}

    if p.suffix == ".db":
        from weevolve.core import genesis_import
        return genesis_import(str(p), verbose=verbose)
    elif p.suffix == ".md":
        # skill.md files are read-only references — log the import
        content = p.read_text()
        lines = content.count("\n")
        if verbose:
            print(f"  Imported skill.md ({lines} lines)")
            print(f"  Knowledge available as reference in: {p}")
        return {"type": "skill", "lines": lines, "path": str(p)}
    else:
        print(f"  Unsupported format: {p.suffix}")
        print(f"  Expected: .db (genesis) or .md (skill)")
        return {"error": "unsupported format"}


class _ShareHandler(http.server.BaseHTTPRequestHandler):
    """Serve genesis DB and skill.md to remote agents."""

    def do_GET(self):
        share_dir = get_data_dir() / "share"

        if self.path == "/genesis":
            self._serve_file(share_dir / "genesis-shared.db", "application/octet-stream")
        elif self.path == "/skill":
            self._serve_file(share_dir / "skill-shared.md", "text/markdown")
        elif self.path == "/status":
            try:
                from weevolve.core import load_evolution_state
                state = load_evolution_state()
                payload = json.dumps({
                    "level": state.get("level", 1),
                    "atoms": state.get("total_learnings", 0),
                    "alpha": state.get("total_alpha", 0),
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception:
                self.send_error(500)
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"WeEvolve Connect\n/genesis - download genesis DB\n/skill - download skill.md\n/status - agent status\n")

    def _serve_file(self, path: Path, content_type: str):
        if not path.exists():
            self.send_error(404, f"Not found. Run 'weevolve connect export' first.")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def serve_knowledge(verbose: bool = True):
    """Start an HTTP server to share knowledge with remote agents."""
    share_dir = get_data_dir() / "share"
    if not (share_dir / "genesis-shared.db").exists():
        if verbose:
            print("  No shared knowledge yet. Exporting first...")
        export_for_sharing(verbose=verbose)

    server = socketserver.TCPServer(("", CONNECT_PORT), _ShareHandler)
    server.allow_reuse_address = True

    def _shutdown(signum, frame):
        print("\n  Connect server shutting down...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if verbose:
        print(f"\n  WeEvolve Connect serving on http://localhost:{CONNECT_PORT}")
        print(f"  Other agents can pull from:")
        print(f"    curl http://YOUR_IP:{CONNECT_PORT}/genesis > genesis.db")
        print(f"    curl http://YOUR_IP:{CONNECT_PORT}/skill > skill.md")
        print(f"  Press Ctrl+C to stop.\n")

    server.serve_forever()
    server.server_close()


def pull_from_peer(url: str, verbose: bool = True) -> dict:
    """Pull knowledge from a remote WeEvolve instance."""
    import requests

    results = {}
    share_dir = get_data_dir() / "peers"
    share_dir.mkdir(parents=True, exist_ok=True)

    # Pull genesis
    try:
        resp = requests.get(f"{url}/genesis", timeout=30)
        if resp.status_code == 200:
            genesis_path = share_dir / "peer-genesis.db"
            genesis_path.write_bytes(resp.content)
            if verbose:
                print(f"  Downloaded genesis ({len(resp.content)} bytes)")

            from weevolve.core import genesis_import
            results["genesis"] = genesis_import(str(genesis_path), verbose=verbose)
    except Exception as e:
        if verbose:
            print(f"  Genesis pull failed: {e}")
        results["genesis_error"] = str(e)

    # Pull skill
    try:
        resp = requests.get(f"{url}/skill", timeout=30)
        if resp.status_code == 200:
            skill_path = share_dir / "peer-skill.md"
            skill_path.write_text(resp.text)
            if verbose:
                print(f"  Downloaded skill.md ({len(resp.text)} chars)")
            results["skill"] = str(skill_path)
    except Exception as e:
        if verbose:
            print(f"  Skill pull failed: {e}")
        results["skill_error"] = str(e)

    return results


def run_connect(args: list):
    """CLI dispatcher for connect subcommands."""
    subcmd = args[0] if args else "help"

    if subcmd == "export":
        output = args[1] if len(args) > 1 else None
        export_for_sharing(output_dir=output)
    elif subcmd == "import":
        if len(args) < 2:
            print("Usage: weevolve connect import <path-to-genesis.db-or-skill.md>")
            return
        import_from_peer(args[1])
    elif subcmd == "serve":
        serve_knowledge()
    elif subcmd == "pull":
        if len(args) < 2:
            print("Usage: weevolve connect pull http://peer-ip:8877")
            return
        pull_from_peer(args[1])
    else:
        print("WeEvolve Connect — Agent-to-Agent Knowledge Transfer")
        print()
        print("Commands:")
        print("  weevolve connect export [dir]        Export genesis + skill for sharing")
        print("  weevolve connect import <file>       Import peer knowledge (.db or .md)")
        print("  weevolve connect serve               Start sharing server on :8877")
        print("  weevolve connect pull <url>           Pull from remote WeEvolve agent")
