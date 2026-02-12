"""
WeEvolve Content Watcher - Auto-learn from new files
=====================================================
Monitors ~/.weevolve/watch/ for new files and processes them through SEED.

Supported formats: .txt, .md, .json, .url
For .url files: reads the URL from the file content and learns from it.

Usage:
  weevolve watch              # Start watching (poll every 10s)
  weevolve watch --interval 5 # Custom poll interval in seconds

(C) LIVE FREE = LIVE FOREVER
"""

import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Set

from weevolve.config import DATA_DIR

# Watcher paths
WATCH_DIR = DATA_DIR / "watch"
PROCESSED_PATH = DATA_DIR / "watch_processed.json"

SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".url"}

# ANSI colors (match core.py)
CYAN = "\033[36m"
MAGENTA = "\033[35m"
GREEN_C = "\033[32m"
YELLOW_C = "\033[33m"
BLUE_C = "\033[34m"
LIME_C = "\033[38;5;190m"
DIM_C = "\033[2m"
RED_C = "\033[31m"
BOLD_C = "\033[1m"
RESET_C = "\033[0m"

SEED_OWL_PHASES = [
    ("LYRA", "PERCEIVE", CYAN),
    ("PRISM", "CONNECT", MAGENTA),
    ("SAGE", "LEARN", GREEN_C),
    ("QUEST", "QUESTION", YELLOW_C),
    ("NOVA", "EXPAND", BLUE_C),
    ("ECHO", "SHARE", LIME_C),
    ("LUNA", "RECEIVE", DIM_C),
    ("SOWL", "IMPROVE", RED_C),
]


def _seed_phase(idx: int, detail: str = "") -> None:
    """Print a colored SEED phase indicator."""
    if 0 <= idx < len(SEED_OWL_PHASES):
        owl, phase, color = SEED_OWL_PHASES[idx]
        print(f"  {color}{owl}{RESET_C} {DIM_C}{phase}{RESET_C} {detail}")


def _load_processed() -> Dict:
    """Load the set of already-processed file paths and their timestamps."""
    if PROCESSED_PATH.exists():
        try:
            with open(PROCESSED_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_processed(processed: Dict) -> None:
    """Persist the processed-files registry."""
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_PATH, "w") as f:
        json.dump(processed, f, indent=2)


def _read_url_file(path: Path) -> str:
    """Extract the URL from a .url file. Supports plain text and INI-style."""
    content = path.read_text(errors="replace").strip()
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("URL="):
            return line[4:]
        if line.startswith("http"):
            return line
    return content


def _process_file(path: Path) -> bool:
    """
    Process a single file through the WeEvolve learn loop.
    Returns True if learning succeeded.
    """
    # Lazy import to avoid circular deps
    from weevolve.core import learn, load_evolution_state

    ext = path.suffix.lower()
    name = path.name

    _seed_phase(0, f"perceiving {name}...")

    if ext == ".url":
        url = _read_url_file(path)
        if not url or not url.startswith("http"):
            print(f"  {RED_C}[SKIP]{RESET_C} No valid URL in {name}")
            return False
        _seed_phase(1, f"connecting to {url[:60]}...")
        delta = learn(url, source_type="url", verbose=True)
    elif ext == ".json":
        try:
            raw = path.read_text(errors="replace")
            data = json.loads(raw)
            content = json.dumps(data, indent=2)[:8000]
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  {RED_C}[ERROR]{RESET_C} Bad JSON in {name}: {exc}")
            return False
        _seed_phase(1, f"connecting JSON content ({len(content)} chars)...")
        delta = learn(content, source_type="text", verbose=True)
    else:
        try:
            content = path.read_text(errors="replace")
        except OSError as exc:
            print(f"  {RED_C}[ERROR]{RESET_C} Cannot read {name}: {exc}")
            return False
        _seed_phase(1, f"connecting text content ({len(content)} chars)...")
        delta = learn(content, source_type="text", verbose=True)

    if delta:
        state = load_evolution_state()
        xp_gained = delta.get("xp_earned", 0)
        atoms = state.get("total_learnings", 0)
        level = state.get("level", 1)
        xp = state.get("xp", 0)
        xp_next = state.get("xp_to_next", 100)
        _mini_dashboard(name, xp_gained, atoms, level, xp, xp_next)
        return True

    return False


def _mini_dashboard(
    filename: str,
    xp_gained: int,
    total_atoms: int,
    level: int,
    xp: int,
    xp_next: int,
) -> None:
    """Print a compact post-learn dashboard."""
    bar_len = int(20 * xp / max(1, xp_next))
    bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
    print(f"\n  {BOLD_C}--- WATCHER RESULT ---{RESET_C}")
    print(f"  File:  {filename}")
    print(f"  XP:    +{xp_gained}")
    print(f"  Atoms: {total_atoms}")
    print(f"  Level: {level}  [{LIME_C}{bar}{RESET_C}] {xp}/{xp_next}")
    print(f"  {BOLD_C}----------------------{RESET_C}\n")


def run_watcher(interval: int = 10) -> None:
    """
    Main watcher loop. Polls WATCH_DIR every `interval` seconds for new files.
    Graceful shutdown on Ctrl+C.
    """
    WATCH_DIR.mkdir(parents=True, exist_ok=True)

    shutdown = False

    def _handle_sigint(_sig, _frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _handle_sigint)

    print(f"\n{'=' * 60}")
    print(f"  (*) WeEvolve WATCHER - Auto-Learn From New Files")
    print(f"{'=' * 60}")
    print(f"  Watch dir:   {WATCH_DIR}")
    print(f"  Interval:    {interval}s")
    print(f"  Extensions:  {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    print(f"  Ctrl+C to stop")
    print(f"{'=' * 60}\n")

    _seed_phase(7, "watcher loop starting...")

    processed = _load_processed()
    cycle = 0

    while not shutdown:
        cycle += 1
        try:
            candidates = [
                p
                for p in WATCH_DIR.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            new_files = [
                p for p in candidates if str(p) not in processed
            ]

            if new_files:
                print(
                    f"  [{datetime.now().strftime('%H:%M:%S')}] "
                    f"Found {len(new_files)} new file(s)"
                )

            for file_path in sorted(new_files, key=lambda p: p.stat().st_mtime):
                if shutdown:
                    break

                print(f"\n  {BOLD_C}>>> Processing: {file_path.name}{RESET_C}")
                success = _process_file(file_path)

                processed[str(file_path)] = {
                    "processed_at": datetime.now(timezone.utc).isoformat(),
                    "success": success,
                }
                _save_processed(processed)

            if not shutdown:
                time.sleep(interval)

        except Exception as exc:
            print(f"  {RED_C}[ERROR]{RESET_C} Watcher cycle {cycle} failed: {exc}")
            if not shutdown:
                time.sleep(interval)

    # Graceful shutdown
    print(f"\n  (*) Watcher stopping gracefully.")
    _save_processed(processed)

    from weevolve.core import load_evolution_state

    state = load_evolution_state()
    total_processed = sum(1 for v in processed.values() if v.get("success"))
    print(f"  Files processed this run: {total_processed}")
    print(f"  Level {state.get('level', 1)} | "
          f"Atoms: {state.get('total_learnings', 0)}")
    print(f"  Knowledge persisted. SEED continues.\n")
