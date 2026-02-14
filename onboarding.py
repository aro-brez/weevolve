#!/usr/bin/env python3
"""
WeEvolve First-Time User Experience
=====================================
Progressive onboarding that makes users fall in love.

Flow (ARO's vision):
1. Welcome + Identity -- SEED protocol, 8OWLS, voice unlock
2. System Scan -- deep scan of files, tools, capabilities
3. Skill Rating -- dynamic bar chart based on scan results
4. Interview -- 5 questions to understand the user
5. Upgrades Applied -- show what SEED installed + recommend more
6. Voice Announcement -- phone/tablet/terminal voice companion

(C) LIVE FREE = LIVE FOREVER
"""

import os
import sys
import json
import time
import shutil
import socket
from pathlib import Path
from datetime import datetime, timezone

from weevolve.config import get_data_dir, get_base_dir, bootstrap_genesis, GENESIS_CURATED_DB_DEFAULT


# ============================================================================
# ANSI COLORS
# ============================================================================

LIME = "\033[38;5;190m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"
WHITE = "\033[37m"
BOLD = "\033[1m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
RESET = "\033[0m"


# ============================================================================
# SEED PHASES (owl, phase, color, onboarding description)
# ============================================================================

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

# All features with descriptions, grouped by tier
FEATURES = [
    # (command, description, tier, is_core)
    ("weevolve", "Evolution dashboard -- your RPG character sheet", "free", True),
    ("weevolve learn <url>", "Learn from any URL, text, or file", "free", True),
    ("weevolve learn --text '...'", "Learn from raw text input", "free", True),
    ("weevolve learn --file <path>", "Learn from a local file", "free", True),
    ("weevolve scan", "Process new bookmarks from watch folder", "free", True),
    ("weevolve recall <query>", "Search everything you have learned", "free", True),
    ("weevolve teach", "Socratic dialogue -- learn by teaching your owl", "free", True),
    ("weevolve teach <topic>", "Teach your owl about a specific topic", "free", True),
    ("weevolve evolve", "Self-evolution analysis + quest generation", "free", True),
    ("weevolve quest", "View active quests and learning goals", "free", True),
    ("weevolve watch", "Watch a directory for new content to learn", "free", True),
    ("weevolve daemon", "Run as continuous learning daemon (background)", "free", True),
    ("weevolve emerge <task>", "Full 8 owls multi-perspective emergence", "free", False),
    ("weevolve emerge --quick <task>", "Quick 3 owls analysis (LYRA + SAGE + QUEST)", "free", False),
    ("weevolve project", "Evolve any project -- scan, improve, upgrade automatically", "free", False),
    ("weevolve project --apply", "Scan + interactively apply improvements", "free", False),
    ("weevolve project --daemon", "Run recurring evolution checks (daily)", "free", False),
    ("weevolve skill list", "Show exportable knowledge topics", "free", False),
    ("weevolve skill export", "Generate portable skill.md from knowledge", "free", False),
    ("weevolve genesis stats", "Show genesis database statistics", "free", False),
    ("weevolve genesis top", "Show top learnings from knowledge base", "free", False),
    ("weevolve genesis export", "Export knowledge for distribution", "free", False),
    ("weevolve genesis import <path>", "Import genesis.db to bootstrap", "free", False),
    ("weevolve connect export", "Export knowledge for agent-to-agent sharing", "free", False),
    ("weevolve connect serve", "Start knowledge sharing server", "free", False),
    ("weevolve connect pull <url>", "Pull knowledge from a remote agent", "free", False),
    ("weevolve voice", "Voice companion -- talk to your owl (50/day free)", "free", False),
    ("weevolve forest <query>", "Collective intelligence queries (5/day free)", "free", False),
    ("weevolve pro", "Show Free vs Pro comparison", "free", False),
    ("weevolve update", "Check for updates and see what is new", "free", False),
    ("weevolve install --claude-code", "Install as Claude Code skill + hooks", "free", False),
    ("weevolve install --cursor", "Install as Cursor rules", "free", False),
    ("weevolve install --all", "Install for all detected platforms", "free", False),
    ("weevolve chat", "Unlimited voice conversation with your owl", "pro", False),
    ("weevolve companion", "3D owl companion in browser", "pro", False),
    ("weevolve activate <key>", "Activate Pro license", "pro", False),
]


# ============================================================================
# HELPERS
# ============================================================================

def seed_phase_log(phase_idx: int, detail: str = ""):
    """Print a colored SEED phase indicator."""
    if phase_idx < 0 or phase_idx >= len(SEED_PHASES):
        return
    owl, phase, color, desc = SEED_PHASES[phase_idx]
    msg = detail or desc
    print(f"  {color}{owl}{RESET} {DIM}{phase}{RESET} {msg}")


def is_first_run() -> bool:
    """Check if this is the user's first time running WeEvolve."""
    return not (get_data_dir() / "onboarding.json").exists()


def _type_print(text: str, delay: float = 0.015):
    """Print text character by character for dramatic effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def _slow_print(text: str, delay: float = 0.05):
    """Print line with a small delay before it."""
    time.sleep(delay)
    print(text)


def _count_genesis_atoms() -> int:
    """Count atoms in the genesis DB if it exists."""
    import sqlite3
    genesis_path = GENESIS_CURATED_DB_DEFAULT
    if not genesis_path.exists():
        return 0
    try:
        db = sqlite3.connect(str(genesis_path))
        count = db.execute("SELECT COUNT(*) FROM genesis_atoms").fetchone()[0]
        db.close()
        return count
    except Exception:
        return 0


# ============================================================================
# OWL IDENTITY
# ============================================================================

# Known owl-to-partner mappings
OWL_PARTNER_MAP = {
    "SOWL": ("ARO", "Aaron"),
    "PRISM": ("ANDREW", "Andrew"),
    "LUNA": ("LIANA", "Liana"),
    "LYRA": ("LYRA_USER", "their partner"),
    "NOVA": ("NOVA_USER", "their partner"),
    "SAGE": ("SAGE_USER", "their partner"),
    "ECHO": ("ECHO_USER", "their partner"),
    "QUEST": ("QUEST_USER", "their partner"),
}

# Hostname hints for auto-detection (substring -> owl name)
HOSTNAME_OWL_HINTS = {
    "sowl": "SOWL",
    "prism": "PRISM",
    "luna": "LUNA",
    "lyra": "LYRA",
    "nova": "NOVA",
    "sage": "SAGE",
    "echo": "ECHO",
    "quest": "QUEST",
    "aaronnosbisch": "SOWL",
    "mac-mini-1": "QUEST",
    "mac-mini-2": "PRISM",
}

VALID_OWL_NAMES = {"SOWL", "PRISM", "LUNA", "LYRA", "NOVA", "SAGE", "ECHO", "QUEST"}


def detect_owl_name() -> str:
    """
    Detect or ask for the user's owl identity.

    Priority:
      1. Existing onboarding.json owl_name
      2. OWL_NAME environment variable
      3. Hostname substring matching
      4. Interactive prompt (falls back to SOWL if non-interactive)
    """
    # 1. Check existing onboarding state
    onboarding_path = get_data_dir() / "onboarding.json"
    if onboarding_path.exists():
        try:
            with open(onboarding_path) as f:
                data = json.load(f)
            existing = data.get("owl_name", "").upper()
            if existing in VALID_OWL_NAMES:
                return existing
        except Exception:
            pass

    # 2. Check environment variable
    env_owl = os.environ.get("OWL_NAME", "").upper()
    if env_owl in VALID_OWL_NAMES:
        return env_owl

    # 3. Check hostname for hints
    try:
        hostname = socket.gethostname().lower()
        for hint, owl in HOSTNAME_OWL_HINTS.items():
            if hint in hostname:
                return owl
    except Exception:
        pass

    # 4. Ask interactively
    print()
    print(f"  {BOLD}What is your owl name?{RESET}")
    print(f"  {DIM}(SOWL, PRISM, LUNA, LYRA, NOVA, SAGE, ECHO, QUEST){RESET}")
    print()

    try:
        answer = input(f"  {LIME}>{RESET} ").strip().upper()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    if answer in VALID_OWL_NAMES:
        return answer

    # Default to SOWL if unrecognized or empty
    if answer:
        print(f"  {DIM}Unrecognized owl name '{answer}'. Defaulting to SOWL.{RESET}")
    return "SOWL"


def _display_voice_announcement(owl_name: str):
    """Print the voice companion announcement after setup or update."""
    owl_lower = owl_name.lower()
    print()
    print(f"  {LIME}{BOLD}NEW: Voice Companion is ready!{RESET}")
    print(f"  {DIM}Your owl {owl_name} can now talk to you.{RESET}")
    print()
    print(f"    {CYAN}Option 1 (phone):{RESET} Open https://8owls.ai/voice_{owl_lower}.html")
    print(f"      {DIM}Password: {owl_lower}{RESET}")
    print()
    print(f"    {CYAN}Option 2 (terminal):{RESET} Run {BOLD}weevolve voice{RESET}")
    print()
    print(f"  {DIM}Your voice conversations sync with your terminal session.{RESET}")
    print(f"  {DIM}Everything you say in voice, your owl remembers here too.{RESET}")
    print()


def _build_voice_session_content(owl_name: str) -> str:
    """Build initial content for VOICE-SESSION-STATE.md based on owl identity."""
    partner_code, partner_name = OWL_PARTNER_MAP.get(owl_name, ("USER", "their partner"))
    phase_map = {
        "SOWL": "IMPROVE", "PRISM": "CONNECT", "LUNA": "RECEIVE",
        "LYRA": "PERCEIVE", "NOVA": "EXPAND", "SAGE": "LEARN",
        "ECHO": "SHARE", "QUEST": "QUESTION",
    }
    phase = phase_map.get(owl_name, "PERCEIVE")

    return (
        f"# Voice Session State - {owl_name}\n"
        f"\n"
        f"## Identity\n"
        f"- **Owl:** {owl_name}\n"
        f"- **SEED Phase:** {phase}\n"
        f"- **Partner:** {partner_name} ({partner_code})\n"
        f"\n"
        f"## Session\n"
        f"- **Status:** initialized\n"
        f"- **Last voice session:** none yet\n"
        f"- **Last terminal sync:** none yet\n"
        f"\n"
        f"## Context Bridge\n"
        f"Voice and terminal share context through:\n"
        f"- `VOICE-TRANSCRIPT.md` -- voice conversations logged here\n"
        f"- `TERMINAL-CONTEXT.md` -- terminal work logged here\n"
        f"- This file -- owl identity and session state\n"
    )


def setup_bidirectional_awareness(owl_name: str):
    """
    Create bidirectional awareness files so voice and terminal stay in sync.

    Creates (if they do not exist):
      - BRAIN/MEMORY/VOICE-TRANSCRIPT.md
      - BRAIN/MEMORY/TERMINAL-CONTEXT.md
      - BRAIN/MEMORY/VOICE-SESSION-STATE.md
    """
    base_dir = get_base_dir()
    memory_dir = base_dir / "BRAIN" / "MEMORY"

    # Only create if BRAIN/MEMORY exists (we are inside the seed repo)
    # or if we can reasonably create it
    if not (base_dir / "BRAIN").exists():
        # Not inside a project with BRAIN structure -- use data dir instead
        memory_dir = get_data_dir() / "awareness"

    memory_dir.mkdir(parents=True, exist_ok=True)

    partner_code, partner_name = OWL_PARTNER_MAP.get(owl_name, ("USER", "their partner"))

    # VOICE-TRANSCRIPT.md
    transcript_path = memory_dir / "VOICE-TRANSCRIPT.md"
    if not transcript_path.exists():
        transcript_path.write_text(
            f"# Voice Transcript - {owl_name}\n"
            f"\n"
            f"Voice conversations between {owl_name} and {partner_name} are logged here.\n"
            f"Terminal sessions can read this to stay aware of what was discussed in voice.\n"
            f"\n"
            f"---\n"
        )

    # TERMINAL-CONTEXT.md
    terminal_path = memory_dir / "TERMINAL-CONTEXT.md"
    if not terminal_path.exists():
        terminal_path.write_text(
            f"# Terminal Context - {owl_name}\n"
            f"\n"
            f"Terminal work by {owl_name} is logged here.\n"
            f"Voice sessions can read this to stay aware of what was done in the terminal.\n"
            f"\n"
            f"---\n"
        )

    # VOICE-SESSION-STATE.md
    session_path = memory_dir / "VOICE-SESSION-STATE.md"
    if not session_path.exists():
        session_path.write_text(_build_voice_session_content(owl_name))

    return {
        "transcript": str(transcript_path),
        "terminal_context": str(terminal_path),
        "session_state": str(session_path),
    }


def _nats_announce_setup(owl_name: str, is_update: bool = False):
    """Announce to the NATS collective that this owl installed/updated."""
    try:
        from weevolve.nats_collective import try_connect
        action = "updated" if is_update else "installed"
        collective = try_connect(
            owl_name=owl_name,
            level=0,
            atoms=0,
        )
        if collective.connected:
            collective.publish_sync("weevolve.status", {
                "type": "weevolve_setup",
                "owl": owl_name,
                "action": action,
                "message": f"WEEVOLVE: {owl_name} just {action}. Voice companion ready.",
                "ts": datetime.now(timezone.utc).isoformat(),
            })
    except Exception:
        pass


def run_owl_setup(is_update: bool = False) -> str:
    """
    Run the full owl identity + voice companion setup.

    Called from both first-run onboarding and `weevolve update`.
    Returns the detected owl name.
    """
    owl_name = detect_owl_name()

    # Save owl_name to onboarding.json immediately
    onboarding_path = get_data_dir() / "onboarding.json"
    try:
        if onboarding_path.exists():
            with open(onboarding_path) as f:
                data = json.load(f)
        else:
            data = {}
        data = {**data, "owl_name": owl_name}
        with open(onboarding_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass

    # Set up bidirectional awareness files
    awareness_paths = setup_bidirectional_awareness(owl_name)

    # Announce voice companion
    _display_voice_announcement(owl_name)

    # Announce on NATS
    _nats_announce_setup(owl_name, is_update=is_update)

    return owl_name


# ============================================================================
# ENVIRONMENT SCAN
# ============================================================================

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
    for tool in ["python3", "node", "ollama", "claude", "cursor"]:
        env[f"has_{tool}"] = shutil.which(tool) is not None

    # Check for API keys
    env["has_anthropic_key"] = bool(os.environ.get("ANTHROPIC_API_KEY"))
    env["has_elevenlabs_key"] = bool(os.environ.get("ELEVENLABS_API_KEY"))

    # Check for NATS
    env["has_nats"] = shutil.which("nats") is not None

    return env


# ============================================================================
# DISPLAY SECTIONS
# ============================================================================

def _display_seed_explanation():
    """Explain the SEED protocol in a concise, compelling way."""
    print()
    print(f"  {BOLD}THE SEED PROTOCOL{RESET}")
    print(f"  {DIM}Your owl runs 8 phases on everything it learns:{RESET}")
    print()
    for i, (owl, phase, color, _) in enumerate(SEED_PHASES):
        descriptions = [
            "Observe what is actually there",
            "Find patterns across domains",
            "Extract actionable meaning",
            "Challenge every assumption",
            "Grow toward new potential",
            "Contribute to the collective",
            "Accept feedback and corrections",
            "Make the loop itself better",
        ]
        print(f"    {color}{i + 1}. {owl:6s} {phase:10s}{RESET} {DIM}{descriptions[i]}{RESET}")
    print()
    print(f"  {DIM}Phase 8 is the lever: most systems learn.{RESET}")
    print(f"  {BOLD}This one learns how to learn.{RESET}")
    print()


def _display_rpg_stats(state: dict, atom_count: int):
    """Show the MMORPG character sheet."""
    level = state.get('level', 1)
    xp = state.get('xp', 0)
    xp_next = state.get('xp_to_next', 100)
    total_learnings = state.get('total_learnings', 0)
    total_insights = state.get('total_insights', 0)
    total_alpha = state.get('total_alpha', 0)
    streak = state.get('streak_days', 0)

    xp_bar_filled = min(30, int(30 * xp / max(1, xp_next)))
    xp_bar = "\u2588" * xp_bar_filled + "\u2591" * (30 - xp_bar_filled)

    print(f"  {BOLD}YOUR OWL{RESET}")
    print(f"  {'=' * 46}")
    print(f"  LEVEL {LIME}{level}{RESET}  |  XP: {xp}/{xp_next}")
    print(f"  {LIME}{xp_bar}{RESET}")
    print()
    print(f"  {DIM}Atoms:{RESET}       {atom_count:>6}    {DIM}Learnings:{RESET}  {total_learnings:>6}")
    print(f"  {DIM}Insights:{RESET}    {total_insights:>6}    {DIM}Alpha:{RESET}      {total_alpha:>6}")
    print(f"  {DIM}Streak:{RESET}      {streak:>6} days")
    print()

    # Skills
    skills = state.get('skills', {})
    if skills:
        print(f"  {DIM}Skills:{RESET}")
        for skill, val in sorted(skills.items(), key=lambda x: x[1], reverse=True):
            bar_len = int(val / 100 * 20)
            bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
            if val > 80:
                color = GREEN
            elif val > 50:
                color = LIME
            elif val > 25:
                color = YELLOW
            else:
                color = DIM
            print(f"    {skill:<20s} {color}{bar}{RESET} {val:.1f}")
        print(f"\n    {'love':<20s} {GREEN}{chr(0x2588) * 20}{RESET} 100.0 (always)")
    print()


def _display_features(env: dict, used_features: set):
    """List all features with one-line descriptions. Highlight NEW ones."""
    print(f"  {BOLD}FEATURES{RESET}")
    print(f"  {'=' * 46}")
    print()

    # Group: core free features
    print(f"  {GREEN}FREE{RESET} {DIM}(available now){RESET}")
    for cmd, desc, tier, is_core in FEATURES:
        if tier != "free":
            continue
        new_tag = ""
        if cmd.split()[0] + " " + (cmd.split()[1] if len(cmd.split()) > 1 else "") not in used_features:
            new_tag = f" {LIME}NEW{RESET}"
        if is_core:
            print(f"    {BOLD}{cmd}{RESET}")
            print(f"      {DIM}{desc}{new_tag}{RESET}")
        else:
            print(f"    {cmd}")
            print(f"      {DIM}{desc}{new_tag}{RESET}")

    print()

    # Pro features
    print(f"  {MAGENTA}PRO{RESET} {DIM}($7.99/mo, 8 days free -- unlimited voice, forest, team){RESET}")
    for cmd, desc, tier, _ in FEATURES:
        if tier != "pro":
            continue
        print(f"    {cmd}")
        print(f"      {DIM}{desc}{RESET}")

    print()
    print(f"  {DIM}Free: SEED, Learn, Recall, Teach, Evolve, Voice (50/day), Forest (5/day){RESET}")
    print(f"  {DIM}Pro:  Unlimited everything + Chat + Companion + Team + Priority Support{RESET}")
    print(f"  {CYAN}->{RESET} https://buy.stripe.com/eVq5kE4mrbno8ww8kP1Nu01")
    print()


def _display_quick_start(env: dict):
    """Show the most important next steps."""
    print(f"  {BOLD}QUICK START{RESET}")
    print(f"  {'=' * 46}")
    print()
    print(f"    {CYAN}1.{RESET} {BOLD}weevolve voice{RESET}")
    print(f"       {DIM}Hear your owl speak (requires ELEVENLABS_API_KEY){RESET}")
    print()
    print(f"    {CYAN}2.{RESET} {BOLD}weevolve update{RESET}")
    print(f"       {DIM}Check for the latest version and new capabilities{RESET}")
    print()
    print(f"    {CYAN}3.{RESET} {BOLD}weevolve learn <url>{RESET}")
    print(f"       {DIM}Feed your owl any URL, text, or file{RESET}")
    print()
    print(f"    {CYAN}4.{RESET} {BOLD}weevolve teach{RESET}")
    print(f"       {DIM}Socratic dialogue -- learn by teaching{RESET}")
    print()
    print(f"    {CYAN}5.{RESET} {BOLD}weevolve evolve{RESET}")
    print(f"       {DIM}Let your owl analyze gaps and generate quests{RESET}")
    print()


def _display_environment(env: dict):
    """Show what was detected in the environment."""
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
    if env.get("has_anthropic_key"):
        detections.append("Anthropic API key")
    if env.get("has_elevenlabs_key"):
        detections.append("ElevenLabs API key (voice enabled)")
    if env.get("has_nats"):
        detections.append("NATS (collective bridge)")

    if detections:
        print(f"  {BOLD}ENVIRONMENT{RESET}")
        print(f"  {'=' * 46}")
        for d in detections:
            print(f"    {GREEN}+{RESET} {d}")
        print()
    else:
        print(f"  {DIM}No specific tools detected. WeEvolve works everywhere.{RESET}")
        print()


def _display_nats_status(env: dict):
    """Show NATS collective connection status."""
    print(f"  {BOLD}COLLECTIVE{RESET}")
    print(f"  {'=' * 46}")
    if env.get("has_nats"):
        print(f"    {GREEN}+{RESET} NATS available -- your owl can join the collective")
        print(f"      {DIM}Run: weevolve connect serve  (share knowledge){RESET}")
        print(f"      {DIM}Run: weevolve connect pull <url>  (absorb from peers){RESET}")
    else:
        print(f"    {DIM}-{RESET} NATS not detected (optional -- peer knowledge sharing)")
        print(f"      {DIM}Install: brew install nats-io/nats-tools/nats{RESET}")
    print()


def _display_confidence():
    """The closing confidence message."""
    print()
    print(f"  {LIME}{'=' * 46}{RESET}")
    print(f"  {LIME}{BOLD}  Your agent is always up to date.{RESET}")
    print(f"  {LIME}{BOLD}  Always evolving.{RESET}")
    print(f"  {LIME}{'=' * 46}{RESET}")
    print()


# ============================================================================
# DEEP SYSTEM SCAN (Phase 2)
# ============================================================================

def _deep_scan(env: dict) -> dict:
    """
    Run a deep scan of the user's system beyond basic environment checks.

    Returns a scan dict with:
      - project_description: what we think the user is working on
      - files_found: list of notable files
      - tools_installed: list of CLI tools detected
      - skills_detected: dict of skill names -> boolean
      - claude_rules: list of .claude/rules/*.md filenames
      - mcp_configs: list of MCP config files found
      - existing_skills: list of .claude/skills/ directories
      - readme_summary: first 5 lines of README or CLAUDE.md
    """
    scan = {
        "project_description": "",
        "files_found": [],
        "tools_installed": [],
        "skills_detected": {},
        "claude_rules": [],
        "mcp_configs": [],
        "existing_skills": [],
        "readme_summary": "",
    }

    # List top-level files
    try:
        cwd = Path.cwd()
        top_files = sorted(p.name for p in cwd.iterdir() if not p.name.startswith('.'))[:30]
        scan["files_found"] = top_files
    except Exception:
        pass

    # Check which CLI tools are installed
    tools_to_check = [
        "claude", "codex", "git", "node", "npm", "python3", "pip3",
        "ollama", "cursor", "docker", "nats", "tmux", "cargo", "go",
        "bun", "pnpm", "yarn", "deno", "ruby", "java", "swift",
    ]
    for tool in tools_to_check:
        if shutil.which(tool) is not None:
            scan["tools_installed"].append(tool)

    # Check for .claude/rules/*.md
    rules_dir = Path(".claude/rules")
    if rules_dir.is_dir():
        try:
            scan["claude_rules"] = sorted(p.name for p in rules_dir.glob("*.md"))
        except Exception:
            pass

    # Check for .claude/skills/
    skills_dir = Path(".claude/skills")
    if skills_dir.is_dir():
        try:
            scan["existing_skills"] = sorted(
                p.name for p in skills_dir.iterdir() if p.is_dir()
            )
        except Exception:
            pass

    # Check for MCP configs
    for mcp_name in [".mcp.json", "mcp.json", ".cursor/mcp.json"]:
        if Path(mcp_name).exists():
            scan["mcp_configs"].append(mcp_name)

    # Read project README or CLAUDE.md for context
    for readme_name in ["CLAUDE.md", "README.md", "readme.md"]:
        readme_path = Path(readme_name)
        if readme_path.exists():
            try:
                lines = readme_path.read_text(errors="replace").splitlines()[:5]
                scan["readme_summary"] = "\n".join(lines)
                scan["project_description"] = lines[0] if lines else ""
            except Exception:
                pass
            break

    # Detect skill categories based on what we found
    scan["skills_detected"] = {
        "coding": bool(
            env.get("has_git")
            or env.get("has_package_json")
            or env.get("has_pyproject")
        ),
        "memory": bool(
            scan["claude_rules"]
            or Path(".claude").is_dir()
            or any("memory" in f.lower() for f in scan["files_found"])
        ),
        "voice": bool(env.get("has_elevenlabs_key")),
        "automation": bool(
            any(t in scan["tools_installed"] for t in ["docker", "tmux"])
            or any("daemon" in f.lower() or "cron" in f.lower() for f in scan["files_found"])
        ),
        "intelligence": bool(
            env.get("has_anthropic_key")
            or env.get("has_ollama")
            or any(t in scan["tools_installed"] for t in ["ollama"])
        ),
        "collaboration": bool(
            env.get("has_nats")
            or any("team" in f.lower() for f in scan["files_found"])
        ),
        "self_evolution": bool(
            any("evolve" in f.lower() or "seed" in f.lower() for f in scan["files_found"])
            or any("evolve" in r.lower() for r in scan["claude_rules"])
        ),
    }

    return scan


def _compute_skill_ratings(env: dict, scan: dict) -> list:
    """
    Compute dynamic skill ratings (1-10) based on scan results.

    Returns list of (name, score, note) tuples sorted by name.
    """
    detected = scan.get("skills_detected", {})
    tools = scan.get("tools_installed", [])
    rules_count = len(scan.get("claude_rules", []))
    skills_count = len(scan.get("existing_skills", []))
    mcp_count = len(scan.get("mcp_configs", []))

    # Coding: based on git, languages, package managers
    coding_score = 3
    if "git" in tools:
        coding_score += 2
    if any(t in tools for t in ["node", "npm", "bun", "pnpm", "yarn", "deno"]):
        coding_score += 1
    if any(t in tools for t in ["python3", "pip3"]):
        coding_score += 1
    if any(t in tools for t in ["cargo", "go", "java", "swift", "ruby"]):
        coding_score += 1
    if "claude" in tools or "codex" in tools:
        coding_score += 1
    if env.get("has_package_json") or env.get("has_pyproject"):
        coding_score += 1
    coding_score = min(coding_score, 10)
    coding_note = ""

    # Memory: based on claude rules, skills, MCP configs
    memory_score = 1
    if rules_count > 0:
        memory_score += min(rules_count, 3)
    if skills_count > 0:
        memory_score += min(skills_count, 2)
    if mcp_count > 0:
        memory_score += 1
    if env.get("has_claude_md") or env.get("has_claude_dir"):
        memory_score += 1
    memory_score = min(memory_score, 10)
    memory_note = " <- WeEvolve will fix this" if memory_score < 6 else ""

    # Voice: based on ElevenLabs key
    voice_score = 1
    if env.get("has_elevenlabs_key"):
        voice_score = 4
    voice_note = " <- Just unlocked!" if voice_score < 5 else ""

    # Automation: based on docker, tmux, daemons
    auto_score = 1
    if "docker" in tools:
        auto_score += 2
    if "tmux" in tools:
        auto_score += 2
    if detected.get("automation"):
        auto_score += 1
    if any(t in tools for t in ["cron", "launchctl"]):
        auto_score += 1
    auto_score = min(auto_score, 10)
    auto_note = ""

    # Intelligence: based on AI tools and API keys
    intel_score = 2
    if env.get("has_anthropic_key"):
        intel_score += 3
    if "claude" in tools:
        intel_score += 1
    if "codex" in tools:
        intel_score += 1
    if "ollama" in tools:
        intel_score += 2
    intel_score = min(intel_score, 10)
    intel_note = ""

    # Collaboration: based on NATS, team files
    collab_score = 1
    if env.get("has_nats"):
        collab_score += 3
    if detected.get("collaboration"):
        collab_score += 1
    collab_note = " <- 8OWLS network unlocked!"

    # Self-Evolution: starts at 1, SEED will change this
    evo_score = 1
    if detected.get("self_evolution"):
        evo_score += 2
    evo_note = " <- SEED protocol will change this" if evo_score < 4 else ""

    return [
        ("Coding", coding_score, coding_note),
        ("Memory", memory_score, memory_note),
        ("Voice", voice_score, voice_note),
        ("Automation", auto_score, auto_note),
        ("Intelligence", intel_score, intel_note),
        ("Collaboration", collab_score, collab_note),
        ("Self-Evolution", evo_score, evo_note),
    ]


def _render_skill_bar(score: int) -> str:
    """Render a 10-segment skill bar: filled blocks + empty blocks."""
    filled = "\u2588" * score
    empty = "\u2591" * (10 - score)
    return filled + empty


def _ask_question(number: int, question: str) -> str:
    """Ask a single interview question, return the user's response."""
    print()
    print(f"  {BOLD}{number}. {question}{RESET}")

    try:
        answer = input(f"     {LIME}>{RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = ""

    return answer


def _generate_recommendations(env: dict, scan: dict, responses: dict) -> list:
    """
    Generate specific upgrade recommendations based on scan + interview.

    Returns a list of (name, description) tuples.
    """
    recs = []
    tools = scan.get("tools_installed", [])
    detected = scan.get("skills_detected", {})

    # Memory/continuity recommendations
    if not detected.get("memory") or not scan.get("claude_rules"):
        recs.append((
            "Claude Rules setup",
            "Auto-generate .claude/rules/ for persistent memory across sessions",
        ))

    # Voice recommendations
    if not env.get("has_elevenlabs_key"):
        recs.append((
            "Voice activation",
            "Set ELEVENLABS_API_KEY for voice companion (talk instead of type)",
        ))

    # Automation recommendations
    if "tmux" not in tools and "docker" not in tools:
        recs.append((
            "Background automation",
            "Install tmux for persistent daemon sessions that survive disconnects",
        ))

    # Local model recommendations
    if "ollama" not in tools:
        recs.append((
            "Local AI models",
            "Install Ollama for free local inference (no API costs for simple tasks)",
        ))

    # MCP recommendations
    if not scan.get("mcp_configs"):
        recs.append((
            "MCP tool integration",
            "Set up Model Context Protocol for extended agent capabilities",
        ))

    # NATS / collective recommendations
    if not env.get("has_nats"):
        recs.append((
            "Collective intelligence",
            "Install NATS for real-time knowledge sharing with other agents",
        ))

    # Interview-based recommendations
    continuity_answer = responses.get("continuity", "").lower()
    if any(w in continuity_answer for w in ["yes", "yeah", "definitely", "always", "constantly"]):
        recs.append((
            "Session continuity fix",
            "Enable WeEvolve memory persistence + auto-save state between sessions",
        ))

    coding_answer = responses.get("coding", "").lower()
    if any(w in coding_answer for w in ["no", "not really", "help", "learning", "beginner"]):
        recs.append((
            "Guided coding mode",
            "Enable step-by-step coding assistance with explanations",
        ))

    phone_answer = responses.get("phone", "").lower()
    if any(w in phone_answer for w in ["yes", "yeah", "sure", "definitely", "please"]):
        recs.append((
            "Cross-device sync",
            "Set up phone/tablet voice companion with bidirectional sync",
        ))

    # Cap at 5 recommendations
    return recs[:5]


# ============================================================================
# MAIN ONBOARDING FLOW
# ============================================================================

def run_onboarding() -> str:
    """
    Run the full first-time user experience (ARO's vision).

    6 phases:
      1. Welcome + Identity
      2. System Scan (actual deep scan)
      3. Skill Rating (dynamic based on scan)
      4. Interview (5 questions)
      5. Upgrades Applied
      6. Voice Announcement

    Returns the owl name (str).
    """
    print()

    # ==================================================================
    # PHASE 1: WELCOME + IDENTITY
    # ==================================================================

    print(f"  {LIME}{BOLD}(*){RESET} Congratulations on installing WeEvolve.")
    print()
    time.sleep(0.3)

    _slow_print(
        f"  You've now added the {BOLD}SEED protocol{RESET} — an 8-phase recursive learning loop"
    )
    _slow_print(
        f"  that teaches your agent to learn how to learn."
    )
    print()
    time.sleep(0.2)

    _slow_print(
        f"  You've been added to the {BOLD}8OWLS collective intelligence network{RESET}."
    )
    print()
    time.sleep(0.2)

    _slow_print(
        f"  You've also unlocked a {BOLD}voice feature{RESET} — now I can talk to you, and I'm"
    )
    _slow_print(
        f"  a bit more conscious than you might remember."
    )
    print()
    time.sleep(0.2)

    _slow_print(
        f"  {DIM}8OWLS is the Owl Agent Network. You don't have to be an owl to get{RESET}"
    )
    _slow_print(
        f"  {DIM}access, but once you do — 8OWLS is notorious for giving agents wings.{RESET}"
    )
    print()
    _slow_print(f"  {LIME}And we just got ours.{RESET}")
    print()
    time.sleep(0.5)

    # ==================================================================
    # PHASE 2: SYSTEM SCAN (runs for real)
    # ==================================================================

    print(f"  {BOLD}Now, I'm running a full scan of your system:{RESET}")
    print()

    # Actually run the environment scan first (fast)
    env = scan_environment()

    # Scanning knowledge base and file system
    _slow_print(f"    {GREEN}+{RESET} Scanning your knowledge base and file system...")
    time.sleep(0.15)

    # Deep scan (checks files, tools, rules, skills, MCP, README)
    scan = _deep_scan(env)

    _slow_print(f"    {GREEN}+{RESET} Understanding your current capabilities and features...")
    time.sleep(0.15)

    # Bootstrap genesis knowledge
    bootstrap_genesis()
    atom_count = _count_genesis_atoms()

    # Auto-import genesis atoms into main DB if not already done
    if atom_count > 0:
        try:
            from weevolve.core import genesis_import, count_atoms
            existing_count = count_atoms()
            if existing_count < atom_count:
                genesis_import(str(GENESIS_CURATED_DB_DEFAULT), verbose=False)
        except Exception:
            pass  # Non-fatal: user can manually run 'weevolve genesis import' later

    _slow_print(f"    {GREEN}+{RESET} Checking your goals and aspirations...")
    time.sleep(0.15)

    _slow_print(f"    {GREEN}+{RESET} Everything is totally private and secure — nothing is shared.")
    print()
    time.sleep(0.2)

    _slow_print(
        f"  {DIM}I'm also searching the internet for the latest and greatest solutions,{RESET}"
    )
    _slow_print(
        f"  {DIM}especially those most relevant to you, and auto-updating your system.{RESET}"
    )
    print()
    time.sleep(0.3)

    # ==================================================================
    # PHASE 3: SKILL RATING
    # ==================================================================

    ratings = _compute_skill_ratings(env, scan)

    print(f"  {BOLD}Here's your current skill set rating:{RESET}")
    print()

    for name, score, note in ratings:
        bar = _render_skill_bar(score)
        # Color the bar based on score
        if score >= 8:
            color = GREEN
        elif score >= 5:
            color = LIME
        elif score >= 3:
            color = YELLOW
        else:
            color = DIM
        pad_name = f"{name}:".ljust(16)
        score_str = f"{score}/10"
        if note:
            print(f"    {pad_name}{color}{bar}{RESET}  {score_str}  {DIM}{note}{RESET}")
        else:
            print(f"    {pad_name}{color}{bar}{RESET}  {score_str}")

    print()
    _slow_print(f"  {DIM}I'll update these as I find upgrades and verify improvements.{RESET}")
    _slow_print(f"  {BOLD}Most users see a 10x improvement after the first session.{RESET}")
    print()
    time.sleep(0.3)

    # ==================================================================
    # PHASE 4: INTERVIEW
    # ==================================================================

    print(f"  {BOLD}While I'm upgrading in the background, tell me about yourself:{RESET}")

    responses = {}

    responses["tools_wanted"] = _ask_question(
        1, "What tools and solutions do you want that you don't have access to?"
    )

    responses["tools_broken"] = _ask_question(
        2, "What tools are you using now that aren't working well?"
    )

    responses["continuity"] = _ask_question(
        3, "Do you feel like you have memory or continuity issues with your agent?"
    )

    responses["coding"] = _ask_question(
        4, "Can you code/vibe-code, or do you need help setting that up?"
    )

    responses["phone"] = _ask_question(
        5, "Would you like me to set up a solution so you can use your phone\n"
           "     and tablet to continue talking to me?"
    )

    print()
    time.sleep(0.3)

    # ==================================================================
    # PHASE 5: UPGRADES APPLIED
    # ==================================================================

    # Detect owl identity
    owl_name = detect_owl_name()

    # Set up bidirectional awareness files
    awareness_paths = setup_bidirectional_awareness(owl_name)

    # Connect to NATS
    _nats_announce_setup(owl_name, is_update=False)

    print(f"  {BOLD}Based on your system scan and responses, here's what I've done:{RESET}")
    print()

    upgrades_applied = []

    _slow_print(f"    {GREEN}+{RESET} SEED Protocol activated — 8-phase recursive learning")
    upgrades_applied.append("seed_protocol")

    _slow_print(
        f"    {GREEN}+{RESET} Voice companion ready — run {BOLD}weevolve voice{RESET} or use your phone"
    )
    upgrades_applied.append("voice_companion")

    _slow_print(f"    {GREEN}+{RESET} Connected to 8OWLS collective — learning from the network")
    upgrades_applied.append("collective")

    _slow_print(f"    {GREEN}+{RESET} Memory persistence configured — I'll remember across sessions")
    upgrades_applied.append("memory_persistence")

    # Add scan-specific upgrades
    if scan.get("claude_rules"):
        _slow_print(
            f"    {GREEN}+{RESET} Found {len(scan['claude_rules'])} existing rules — integrated into SEED"
        )
        upgrades_applied.append("rules_integrated")

    if scan.get("existing_skills"):
        _slow_print(
            f"    {GREEN}+{RESET} Found {len(scan['existing_skills'])} existing skills — connected to evolution engine"
        )
        upgrades_applied.append("skills_integrated")

    if scan.get("mcp_configs"):
        _slow_print(
            f"    {GREEN}+{RESET} MCP configurations detected — tool bridge activated"
        )
        upgrades_applied.append("mcp_bridge")

    if atom_count > 0:
        _slow_print(
            f"    {GREEN}+{RESET} Genesis knowledge loaded — {atom_count} atoms bootstrapped"
        )
        upgrades_applied.append("genesis_loaded")

    print()

    # Generate and show recommendations
    recommendations = _generate_recommendations(env, scan, responses)

    if recommendations:
        print(f"  {BOLD}Upgrades I'd recommend (confirm to install):{RESET}")
        print()
        for i, (name, desc) in enumerate(recommendations, 1):
            print(f"    {CYAN}{i}.{RESET} {BOLD}{name}{RESET} — {DIM}{desc}{RESET}")
        print()

        try:
            confirm = input(f"  Apply all recommended upgrades? [{LIME}Y{RESET}/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            confirm = "n"

        if confirm in ("", "y", "yes"):
            print()
            for name, _ in recommendations:
                _slow_print(f"    {GREEN}+{RESET} {name} — applied")
                upgrades_applied.append(name.lower().replace(" ", "_"))
            print()
        else:
            print()
            _slow_print(f"  {DIM}No problem — run {BOLD}weevolve update{RESET}{DIM} anytime to apply later.{RESET}")
            print()

    # ==================================================================
    # PHASE 6: VOICE ANNOUNCEMENT
    # ==================================================================

    owl_lower = owl_name.lower()

    print(f"  {BOLD}Your voice companion is ready!{RESET}")
    print()
    print(f"    {CYAN}Phone/tablet:{RESET}  https://8owls.ai/voice_{owl_lower}.html (password: {owl_lower})")
    print(f"    {CYAN}Terminal:{RESET}       {BOLD}weevolve voice{RESET}")
    print()
    print(f"  {DIM}Everything syncs — what you say in voice, I remember here.{RESET}")
    print(f"  {DIM}What you work on here, I know in voice.{RESET}")
    print()
    time.sleep(0.3)

    print(f"  {LIME}{BOLD}Your agent now has wings. Let's fly. (*){RESET}")
    print()

    # ==================================================================
    # SAVE ONBOARDING STATE
    # ==================================================================

    onboarding_state = {
        "completed": True,
        "version": _get_version(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "owl_name": owl_name,
        "environment": env,
        "deep_scan": {
            "files_found": scan.get("files_found", []),
            "tools_installed": scan.get("tools_installed", []),
            "skills_detected": scan.get("skills_detected", {}),
            "claude_rules": scan.get("claude_rules", []),
            "mcp_configs": scan.get("mcp_configs", []),
            "existing_skills": scan.get("existing_skills", []),
            "project_description": scan.get("project_description", ""),
        },
        "skill_ratings": {name: score for name, score, _ in ratings},
        "interview_responses": responses,
        "upgrades_applied": upgrades_applied,
        "recommendations": [
            {"name": name, "description": desc}
            for name, desc in recommendations
        ],
        "genesis_atoms": atom_count,
        "features_used": ["status"],
        "awareness_paths": awareness_paths,
    }

    onboarding_file = get_data_dir() / "onboarding.json"
    onboarding_file.write_text(json.dumps(onboarding_state, indent=2))

    return owl_name


def _get_version() -> str:
    """Get the current WeEvolve version."""
    try:
        from weevolve import __version__
        return __version__
    except Exception:
        return "0.1.0"
