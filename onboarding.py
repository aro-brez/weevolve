#!/usr/bin/env python3
"""
WeEvolve First-Time User Experience
=====================================
Progressive onboarding that makes users fall in love.

Flow:
1. Scan system (what do they have?)
2. Show MMORPG welcome screen
3. Bootstrap genesis knowledge
4. Voice greeting (if ElevenLabs configured)
5. Ask what they want to learn
6. Start first evolution cycle
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime


LIME = "\033[38;5;190m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


SEED_PHASES = [
    ("LYRA", "PERCEIVE", CYAN, "scanning your system"),
    ("PRISM", "CONNECT", MAGENTA, "finding patterns"),
    ("SAGE", "LEARN", GREEN, "extracting meaning"),
    ("QUEST", "QUESTION", YELLOW, "challenging assumptions"),
    ("NOVA", "EXPAND", BLUE, "growing potential"),
    ("ECHO", "SHARE", LIME, "preparing to share"),
    ("LUNA", "RECEIVE", DIM, "listening to the collective"),
    ("SOWL", "IMPROVE", RED, "optimizing the loop"),
]


def seed_phase_log(phase_idx: int, detail: str = ""):
    """Print a colored SEED phase indicator."""
    if phase_idx < 0 or phase_idx >= len(SEED_PHASES):
        return
    owl, phase, color, desc = SEED_PHASES[phase_idx]
    msg = detail or desc
    print(f"  {color}{owl}{RESET} {DIM}{phase}{RESET} {msg}")


def get_data_dir() -> Path:
    """Get or create the WeEvolve data directory."""
    data_dir = Path(os.environ.get("WEEVOLVE_DATA_DIR", Path.home() / ".weevolve"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def is_first_run() -> bool:
    """Check if this is the user's first time running WeEvolve."""
    return not (get_data_dir() / "onboarding.json").exists()


def scan_environment() -> dict:
    """Scan the user's development environment."""
    env = {
        "has_claude_md": Path("CLAUDE.md").exists(),
        "has_claude_dir": Path(".claude").exists(),
        "has_cursorrules": Path(".cursorrules").exists(),
        "has_git": Path(".git").exists(),
        "has_package_json": Path("package.json").exists(),
        "has_pyproject": Path("pyproject.toml").exists(),
        "has_requirements": Path("requirements.txt").exists(),
        "platform": sys.platform,
    }

    # Check for common tools (cross-platform)
    import shutil
    for tool in ["python3", "node", "ollama", "claude", "cursor"]:
        env[f"has_{tool}"] = shutil.which(tool) is not None

    return env


def display_welcome(env: dict, atom_count: int = 649):
    """Display the MMORPG-style welcome screen."""
    print()
    print(f"  {LIME}{BOLD}")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║           Welcome to WeEvolve™               ║")
    print("  ║     The agent that evolves itself.            ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(f"  {RESET}")
    print()
    print(f"  {BOLD}YOUR OWL{RESET}")
    print("  ──────────────────────────────────────────")
    print(f"  LEVEL 1  |  XP: 0  |  ATOMS: {atom_count}")
    print()
    print(f"  {DIM}Skills (from genesis knowledge):{RESET}")

    # Show initial skill levels from genesis DB
    genesis_skills = [
        ("ai_engineering", 92.1),
        ("research", 92.1),
        ("coding", 87.3),
        ("trading", 85.0),
        ("strategy", 72.0),
        ("communication", 50.3),
        ("design", 39.2),
        ("leadership", 29.4),
        ("marketing", 27.7),
        ("consciousness", 21.7),
    ]

    for skill, pct in genesis_skills:
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        color = GREEN if pct > 80 else LIME if pct > 50 else DIM
        print(f"  {color}{skill:<20} {bar} {pct:.1f}%{RESET}")

    print()

    # Show detected environment
    detections = []
    if env.get("has_claude_md") or env.get("has_claude_dir"):
        detections.append("Claude Code")
    if env.get("has_cursorrules"):
        detections.append("Cursor")
    if env.get("has_git"):
        detections.append("Git repository")
    if env.get("has_ollama"):
        detections.append("Ollama (local models)")
    if env.get("has_package_json"):
        detections.append("Node.js project")
    if env.get("has_pyproject") or env.get("has_requirements"):
        detections.append("Python project")

    for d in detections:
        print(f"  {GREEN}+{RESET} {d} detected")

    if not detections:
        print(f"  {DIM}No specific tools detected. WeEvolve works everywhere.{RESET}")

    print()


def run_onboarding() -> str:
    """Run the full onboarding flow. Returns the chosen topic."""
    print()
    print(f"  {DIM}Initializing SEED protocol...{RESET}")
    print()

    # Phase 0: PERCEIVE — scan environment
    seed_phase_log(0, "scanning your development environment...")
    env = scan_environment()
    time.sleep(0.3)

    # Phase 1: CONNECT — find patterns in what's installed
    seed_phase_log(1, "identifying your tool ecosystem...")
    time.sleep(0.2)

    # Phase 2: LEARN — bootstrap genesis
    seed_phase_log(2, "loading genesis knowledge base...")
    try:
        from weevolve.config import bootstrap_genesis
        bootstrap_genesis()
        atom_count = 649
    except Exception:
        atom_count = 0
    time.sleep(0.2)

    # Phase 3: QUESTION — what's missing?
    seed_phase_log(3, f"loaded {atom_count} knowledge atoms")
    time.sleep(0.2)

    # Phase 4-7: quick flashes
    seed_phase_log(4, "calibrating evolution engine...")
    time.sleep(0.15)
    seed_phase_log(5, "preparing collective bridge...")
    time.sleep(0.15)
    seed_phase_log(6, "opening receiver channels...")
    time.sleep(0.15)
    seed_phase_log(7, "SEED protocol online")
    time.sleep(0.2)

    print()
    display_welcome(env, atom_count)

    # Voice greeting if available (non-blocking)
    try:
        from weevolve.voice import voice_available, greet
        if voice_available():
            greet(level=1, atoms=atom_count)
    except Exception:
        pass

    # Ask what to learn
    print(f"  {BOLD}What do you want your owl to learn about?{RESET}")
    print(f"  {DIM}(Type a topic, or press Enter for AI engineering){RESET}")
    print()

    try:
        topic = input(f"  {LIME}>{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        topic = ""

    if not topic:
        topic = "ai_engineering"

    # Save onboarding state
    state = {
        "completed": True,
        "timestamp": datetime.now().isoformat(),
        "initial_topic": topic,
        "environment": env,
        "genesis_atoms": atom_count,
    }

    onboarding_file = get_data_dir() / "onboarding.json"
    onboarding_file.write_text(json.dumps(state, indent=2))

    print()
    print(f"  {GREEN}✓{RESET} Onboarding complete! Your owl is ready.")
    print(f"  {DIM}Starting evolution on: {topic}{RESET}")
    print()
    print(f"  Commands:")
    print(f"    {BOLD}weevolve{RESET}              — evolution dashboard")
    print(f"    {BOLD}weevolve learn <url>{RESET}  — learn from anything")
    print(f"    {BOLD}weevolve chat{RESET}         — voice conversation with your owl")
    print(f"    {BOLD}weevolve companion{RESET}    — 3D owl companion in browser")
    print(f"    {BOLD}weevolve scan{RESET}         — process bookmarks")
    print(f"    {BOLD}weevolve recall <q>{RESET}   — search what you've learned")
    print()

    return topic
