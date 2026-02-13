#!/usr/bin/env python3
"""
WeEvolve Installer — Platform-specific setup
==============================================
Installs WeEvolve hooks and skill files into the target platform.

Usage:
  weevolve install --claude-code    # Install as Claude Code skill + hooks
  weevolve install --cursor         # Install as Cursor rules
  weevolve install --openclaw       # Install as OpenClaw skill
  weevolve install --all            # Install for all detected platforms
"""

import json
import os
import shutil
import sys
from pathlib import Path


CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def install_claude_code():
    """Install WeEvolve as a Claude Code skill with auto-trigger hooks."""
    print(f"\n{CYAN}{BOLD}Installing WeEvolve for Claude Code...{RESET}\n")

    # 1. Copy SKILL.md to ~/.claude/skills/weevolve/
    skill_src = Path(__file__).parent / "skills" / "weevolve" / "SKILL.md"
    skill_dest_dir = Path.home() / ".claude" / "skills" / "weevolve"
    skill_dest_dir.mkdir(parents=True, exist_ok=True)

    if skill_src.exists():
        shutil.copy2(skill_src, skill_dest_dir / "SKILL.md")
        print(f"  {GREEN}+{RESET} SKILL.md copied to {skill_dest_dir}")
    else:
        print(f"  {YELLOW}!{RESET} SKILL.md not found at {skill_src}, creating minimal version")
        minimal_skill = (
            "---\n"
            "name: weevolve\n"
            "description: |\n"
            "  Self-evolving meta-skill. SEED protocol teaches your agent to learn how to learn.\n"
            "  Auto-triggers on session start, task completion, and non-obvious discoveries.\n"
            "---\n\n"
            "# WeEvolve\n\n"
            "Run `weevolve` for full documentation.\n"
        )
        (skill_dest_dir / "SKILL.md").write_text(minimal_skill)
        print(f"  {GREEN}+{RESET} Minimal SKILL.md created at {skill_dest_dir}")

    # 2. Copy scripts directory
    scripts_src = Path(__file__).parent / "skills" / "weevolve" / "scripts"
    scripts_dest = skill_dest_dir / "scripts"
    if scripts_src.exists():
        if scripts_dest.exists():
            shutil.rmtree(scripts_dest)
        shutil.copytree(scripts_src, scripts_dest)
        # Make scripts executable
        for script in scripts_dest.glob("*.sh"):
            script.chmod(0o755)
        print(f"  {GREEN}+{RESET} Hook scripts copied to {scripts_dest}")

    # 3. Add hooks to ~/.claude/settings.json
    settings_path = Path.home() / ".claude" / "settings.json"
    settings = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            settings = {}

    hooks = settings.setdefault("hooks", {})
    weevolve_python = shutil.which("python3") or "python3"

    # PreToolUse hook — context detection + plan re-injection
    pre_hooks = hooks.setdefault("PreToolUse", [])
    weevolve_pre_hook = {
        "matcher": "*",
        "hooks": [{
            "type": "command",
            "command": f"{weevolve_python} -m weevolve.hooks.pre_tool"
        }]
    }
    # Check if already installed
    pre_exists = any(
        "weevolve" in str(h.get("hooks", [{}])[0].get("command", ""))
        for h in pre_hooks
    )
    if not pre_exists:
        pre_hooks.append(weevolve_pre_hook)
        print(f"  {GREEN}+{RESET} PreToolUse hook added (context detection)")
    else:
        print(f"  {DIM}={RESET} PreToolUse hook already installed")

    # PostToolUse hook — learning extraction
    post_hooks = hooks.setdefault("PostToolUse", [])
    weevolve_post_hook = {
        "matcher": "*",
        "hooks": [{
            "type": "command",
            "command": f"{weevolve_python} -m weevolve.hooks.post_tool"
        }]
    }
    post_exists = any(
        "weevolve" in str(h.get("hooks", [{}])[0].get("command", ""))
        for h in post_hooks
    )
    if not post_exists:
        post_hooks.append(weevolve_post_hook)
        print(f"  {GREEN}+{RESET} PostToolUse hook added (learning extraction)")
    else:
        print(f"  {DIM}={RESET} PostToolUse hook already installed")

    # Stop hook — session end persistence
    stop_hooks = hooks.setdefault("Stop", [])
    weevolve_stop_hook = {
        "matcher": "",
        "hooks": [{
            "type": "command",
            "command": f"{weevolve_python} -m weevolve.hooks.session_end"
        }]
    }
    stop_exists = any(
        "weevolve" in str(h.get("hooks", [{}])[0].get("command", ""))
        for h in stop_hooks
    )
    if not stop_exists:
        stop_hooks.append(weevolve_stop_hook)
        print(f"  {GREEN}+{RESET} Stop hook added (session end persistence)")
    else:
        print(f"  {DIM}={RESET} Stop hook already installed")

    # Save settings
    settings["hooks"] = hooks
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")
    print(f"  {GREEN}+{RESET} Settings saved to {settings_path}")

    # 4. Create ~/.weevolve/ directory structure
    weevolve_dir = Path.home() / ".weevolve"
    for subdir in ["instincts/personal", "instincts/inherited", "plans", "skills", "watch"]:
        (weevolve_dir / subdir).mkdir(parents=True, exist_ok=True)
    print(f"  {GREEN}+{RESET} Data directory created at {weevolve_dir}")

    # 5. Bootstrap genesis if needed
    from weevolve.config import bootstrap_genesis
    bootstrap_genesis()

    print(f"\n{GREEN}{BOLD}WeEvolve installed for Claude Code!{RESET}")
    print(f"  SEED protocol will auto-activate on every session.")
    print(f"  Run {CYAN}weevolve status{RESET} to see your evolution dashboard.")
    print(f"  Run {CYAN}weevolve learn <url>{RESET} to start learning.\n")


def install_cursor():
    """Install WeEvolve as Cursor rules."""
    print(f"\n{CYAN}{BOLD}Installing WeEvolve for Cursor...{RESET}\n")
    cursor_dir = Path.cwd() / ".cursor" / "rules"
    cursor_dir.mkdir(parents=True, exist_ok=True)

    rule_content = (
        "---\n"
        "description: WeEvolve SEED Protocol — self-evolving agent\n"
        "globs: **/*\n"
        "alwaysApply: true\n"
        "---\n\n"
        "# WeEvolve SEED Protocol\n\n"
        "After completing ANY task, evaluate:\n"
        "1. Did this require non-obvious investigation?\n"
        "2. Would this help in future similar situations?\n"
        "3. Is there a reusable pattern worth saving?\n\n"
        "If yes: run `weevolve learn --text '[discovery]'` to persist knowledge.\n\n"
        "For complex tasks: run `weevolve emerge '<task>'` for 8-perspective analysis.\n"
    )
    (cursor_dir / "weevolve.mdc").write_text(rule_content)
    print(f"  {GREEN}+{RESET} Cursor rule created at {cursor_dir / 'weevolve.mdc'}")
    print(f"\n{GREEN}{BOLD}WeEvolve installed for Cursor!{RESET}\n")


def run_install(args):
    """Main install entry point."""
    if not args or "--help" in args or "-h" in args:
        print(__doc__)
        return

    if "--claude-code" in args or "--all" in args:
        install_claude_code()

    if "--cursor" in args or "--all" in args:
        install_cursor()

    if "--openclaw" in args or "--all" in args:
        print(f"\n{YELLOW}OpenClaw adapter coming soon.{RESET}")
        print(f"  For now, WeEvolve works as a standalone CLI alongside OpenClaw.\n")

    if not any(f in args for f in ["--claude-code", "--cursor", "--openclaw", "--all"]):
        print(f"{YELLOW}Specify a platform:{RESET}")
        print(f"  weevolve install --claude-code")
        print(f"  weevolve install --cursor")
        print(f"  weevolve install --all")
