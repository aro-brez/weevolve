#!/usr/bin/env python3
"""
WeEvolve Hooks Installer - Wire Hooks into Claude Code Settings
================================================================
Reads ~/.claude/settings.json, adds WeEvolve hooks, preserves existing hooks.

Usage:
  python install_hooks.py           # Install hooks
  python install_hooks.py --check   # Check if hooks are installed
  python install_hooks.py --remove  # Remove WeEvolve hooks (keep others)

Creates the ~/.weevolve/ directory structure:
  ~/.weevolve/
    observations.jsonl      - Tool use observations
    observations.archive/   - Rotated observation archives
    instincts/              - Learned patterns by task type
    plans/                  - Active and archived task plans
    error_flags.jsonl       - Errors flagged for instinct extraction
    session_log.jsonl       - Session summaries
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


# ============================================================================
# PATHS
# ============================================================================

_CLAUDE_SETTINGS = Path.home() / ".claude" / "settings.json"
_WEEVOLVE_DIR = Path.home() / ".weevolve"
_HOOKS_DIR = Path(__file__).parent.resolve()

# Hook definitions: the actual commands Claude Code will execute
_HOOKS: dict[str, dict[str, Any]] = {
    "PreToolUse": {
        "matcher": "*",
        "hooks": [
            {
                "type": "command",
                "command": f"python3 {_HOOKS_DIR / 'pre_tool.py'}",
            }
        ],
    },
    "PostToolUse": {
        "matcher": "*",
        "hooks": [
            {
                "type": "command",
                "command": f"python3 {_HOOKS_DIR / 'post_tool.py'}",
            }
        ],
    },
    "Stop": {
        "matcher": "*",
        "hooks": [
            {
                "type": "command",
                "command": f"python3 {_HOOKS_DIR / 'session_end.py'}",
            }
        ],
    },
}

# Marker to identify WeEvolve hooks in the settings
_MARKER = "weevolve"


# ============================================================================
# DIRECTORY SETUP
# ============================================================================


def setup_directories() -> list[str]:
    """Create the ~/.weevolve/ directory structure. Returns created dirs."""
    dirs_to_create = [
        _WEEVOLVE_DIR,
        _WEEVOLVE_DIR / "instincts",
        _WEEVOLVE_DIR / "plans",
        _WEEVOLVE_DIR / "observations.archive",
    ]
    created: list[str] = []
    for d in dirs_to_create:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))
    return created


# ============================================================================
# SETTINGS I/O
# ============================================================================


def _load_settings() -> dict:
    """Load ~/.claude/settings.json or return an empty dict."""
    if not _CLAUDE_SETTINGS.exists():
        return {}
    try:
        return json.loads(_CLAUDE_SETTINGS.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_settings(settings: dict) -> None:
    """Write settings back to ~/.claude/settings.json."""
    _CLAUDE_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    _CLAUDE_SETTINGS.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n"
    )


# ============================================================================
# HOOK DETECTION
# ============================================================================


def _is_weevolve_hook(hook_entry: dict) -> bool:
    """Check if a hook entry belongs to WeEvolve."""
    hooks_list = hook_entry.get("hooks", [])
    for h in hooks_list:
        cmd = h.get("command", "")
        if _MARKER in cmd or "weevolve" in cmd:
            return True
    return False


def _hook_already_installed(settings: dict) -> bool:
    """Check if WeEvolve hooks are already in the settings."""
    hooks_section = settings.get("hooks", {})
    for hook_type in ("PreToolUse", "PostToolUse", "Stop"):
        entries = hooks_section.get(hook_type, [])
        if isinstance(entries, list):
            for entry in entries:
                if _is_weevolve_hook(entry):
                    return True
    return False


# ============================================================================
# INSTALL
# ============================================================================


def install() -> dict[str, Any]:
    """
    Install WeEvolve hooks into Claude Code settings.
    Preserves all existing hooks. Idempotent.

    Returns a summary dict.
    """
    # Setup directories
    created_dirs = setup_directories()

    # Load settings
    settings = _load_settings()

    # Check if already installed
    if _hook_already_installed(settings):
        return {
            "status": "already_installed",
            "message": "WeEvolve hooks are already installed.",
            "dirs_created": created_dirs,
        }

    # Ensure hooks section exists
    if "hooks" not in settings:
        settings["hooks"] = {}

    hooks_section = settings["hooks"]
    installed_types: list[str] = []

    for hook_type, hook_config in _HOOKS.items():
        existing = hooks_section.get(hook_type, [])
        if not isinstance(existing, list):
            existing = []

        # Append our hook (do not overwrite existing hooks)
        existing.append(hook_config)
        hooks_section[hook_type] = existing
        installed_types.append(hook_type)

    settings["hooks"] = hooks_section
    _save_settings(settings)

    return {
        "status": "installed",
        "message": f"WeEvolve hooks installed for: {', '.join(installed_types)}",
        "hook_types": installed_types,
        "dirs_created": created_dirs,
        "settings_path": str(_CLAUDE_SETTINGS),
    }


# ============================================================================
# REMOVE
# ============================================================================


def remove() -> dict[str, Any]:
    """
    Remove WeEvolve hooks from Claude Code settings.
    Preserves all other hooks.
    """
    settings = _load_settings()
    if not _hook_already_installed(settings):
        return {
            "status": "not_installed",
            "message": "WeEvolve hooks are not installed.",
        }

    hooks_section = settings.get("hooks", {})
    removed_types: list[str] = []

    for hook_type in ("PreToolUse", "PostToolUse", "Stop"):
        entries = hooks_section.get(hook_type, [])
        if not isinstance(entries, list):
            continue
        filtered = [e for e in entries if not _is_weevolve_hook(e)]
        if len(filtered) < len(entries):
            removed_types.append(hook_type)
        if filtered:
            hooks_section[hook_type] = filtered
        else:
            hooks_section.pop(hook_type, None)

    settings["hooks"] = hooks_section
    _save_settings(settings)

    return {
        "status": "removed",
        "message": f"WeEvolve hooks removed from: {', '.join(removed_types)}",
        "removed_types": removed_types,
    }


# ============================================================================
# CHECK
# ============================================================================


def check() -> dict[str, Any]:
    """Check installation status."""
    settings = _load_settings()
    installed = _hook_already_installed(settings)

    # Check directories
    dirs_exist = {
        "weevolve_dir": _WEEVOLVE_DIR.exists(),
        "instincts_dir": (_WEEVOLVE_DIR / "instincts").exists(),
        "plans_dir": (_WEEVOLVE_DIR / "plans").exists(),
    }

    # Check hook scripts exist
    scripts_exist = {
        "pre_tool.py": (_HOOKS_DIR / "pre_tool.py").exists(),
        "post_tool.py": (_HOOKS_DIR / "post_tool.py").exists(),
        "session_end.py": (_HOOKS_DIR / "session_end.py").exists(),
        "context_detector.py": (_HOOKS_DIR / "context_detector.py").exists(),
    }

    return {
        "hooks_installed": installed,
        "directories": dirs_exist,
        "scripts": scripts_exist,
        "settings_path": str(_CLAUDE_SETTINGS),
        "hooks_dir": str(_HOOKS_DIR),
    }


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]

    if "--check" in args:
        result = check()
        if result["hooks_installed"]:
            print("WeEvolve hooks: INSTALLED")
        else:
            print("WeEvolve hooks: NOT INSTALLED")
        print(f"  Settings: {result['settings_path']}")
        print(f"  Hooks dir: {result['hooks_dir']}")
        for name, exists in result["directories"].items():
            status = "OK" if exists else "MISSING"
            print(f"  {name}: {status}")
        for name, exists in result["scripts"].items():
            status = "OK" if exists else "MISSING"
            print(f"  {name}: {status}")
        sys.exit(0 if result["hooks_installed"] else 1)

    elif "--remove" in args:
        result = remove()
        print(result["message"])
        sys.exit(0)

    else:
        result = install()
        print(result["message"])
        if result.get("dirs_created"):
            for d in result["dirs_created"]:
                print(f"  Created: {d}")
        if result.get("hook_types"):
            for ht in result["hook_types"]:
                print(f"  Installed: {ht}")
        sys.exit(0)


if __name__ == "__main__":
    main()
