#!/usr/bin/env python3
"""
WeEvolve PreToolUse Hook - Runs BEFORE Every Tool Invocation
=============================================================
External process. Under 100ms. Zero API calls.

What it does:
  1. Reads tool info from stdin (JSON from Claude Code)
  2. Classifies task type and complexity via context_detector
  3. Loads relevant instincts from ~/.weevolve/instincts/
  4. Re-injects task_plan.md first 30 lines if active
  5. Appends observation to ~/.weevolve/observations.jsonl
  6. Outputs additionalContext to stdout (injected into agent context)

Claude Code hook protocol:
  - stdin: JSON with tool_name, tool_input, session_id, hook_event_name
  - stdout: JSON with optional additionalContext string
  - exit 0: allow tool execution
  - exit 2: block tool execution (we never block)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import context_detector from same package (works whether installed or not)
_HOOKS_DIR = Path(__file__).parent

# Inline-safe import: if weevolve is installed, use it; otherwise, direct import
try:
    from weevolve.hooks.context_detector import classify
except ImportError:
    # Running as standalone script — import from same directory
    sys.path.insert(0, str(_HOOKS_DIR))
    from context_detector import classify  # type: ignore[import-not-found]


# ============================================================================
# PATHS
# ============================================================================

_WEEVOLVE_DIR = Path.home() / ".weevolve"
_OBSERVATIONS_FILE = _WEEVOLVE_DIR / "observations.jsonl"
_INSTINCTS_DIR = _WEEVOLVE_DIR / "instincts"
_PLANS_DIR = _WEEVOLVE_DIR / "plans"
_ACTIVE_PLAN = _PLANS_DIR / "task_plan.md"
_MAX_OBS_SIZE_MB = 10


# ============================================================================
# OBSERVATION LOGGING
# ============================================================================


def _ensure_dirs() -> None:
    """Create WeEvolve directories if they do not exist."""
    _WEEVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    _INSTINCTS_DIR.mkdir(parents=True, exist_ok=True)
    _PLANS_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_observations_if_needed() -> None:
    """Archive observations file if it exceeds the size limit."""
    if not _OBSERVATIONS_FILE.exists():
        return
    try:
        size_mb = _OBSERVATIONS_FILE.stat().st_size / (1024 * 1024)
        if size_mb >= _MAX_OBS_SIZE_MB:
            archive_dir = _WEEVOLVE_DIR / "observations.archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            _OBSERVATIONS_FILE.rename(archive_dir / f"observations-{ts}.jsonl")
    except OSError:
        pass


def _append_observation(observation: dict) -> None:
    """Append a single observation to the JSONL file."""
    _rotate_observations_if_needed()
    try:
        with _OBSERVATIONS_FILE.open("a") as f:
            f.write(json.dumps(observation, separators=(",", ":")) + "\n")
    except OSError:
        pass


# ============================================================================
# INSTINCT LOADING
# ============================================================================


def _load_instincts(task_type: str) -> list[str]:
    """
    Load relevant instincts for the given task type.

    Instinct files live at ~/.weevolve/instincts/<task_type>.jsonl
    Each line is a JSON object with at least a "pattern" key.
    Returns up to 5 most recent instinct patterns.
    """
    instinct_file = _INSTINCTS_DIR / f"{task_type}.jsonl"
    if not instinct_file.exists():
        return []

    instincts: list[str] = []
    try:
        lines = instinct_file.read_text().strip().split("\n")
        # Take last 5 (most recent)
        for line in lines[-5:]:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                pattern = entry.get("pattern", "")
                if pattern:
                    instincts.append(pattern)
            except (json.JSONDecodeError, AttributeError):
                continue
    except OSError:
        pass

    return instincts


# ============================================================================
# PLAN INJECTION
# ============================================================================


def _load_active_plan() -> str | None:
    """
    Load the first 30 lines of the active task plan if it exists.

    This implements the "planning-with-files" pattern from Superpowers:
    re-inject the plan at the start of each tool use so the agent
    stays on track even as context compacts.
    """
    if not _ACTIVE_PLAN.exists():
        return None

    try:
        lines = _ACTIVE_PLAN.read_text().split("\n")[:30]
        content = "\n".join(lines).strip()
        if content:
            return content
    except OSError:
        pass

    return None


# ============================================================================
# CONTEXT BUILDING
# ============================================================================


def _build_additional_context(
    classification: dict,
    instincts: list[str],
    plan: str | None,
) -> str:
    """
    Build the additionalContext string injected into the agent's prompt.

    Keep it minimal: the goal is to prime the agent without bloating context.
    Only outputs when there is actionable context to inject.
    """
    parts: list[str] = []

    # Plan reminder (highest priority — keeps agent on track)
    if plan:
        parts.append(f"[Active Plan]\n{plan}")

    # Instincts (learned patterns from past interactions)
    if instincts:
        formatted = "\n".join(f"  - {i}" for i in instincts)
        parts.append(f"[WeEvolve Instincts for {classification['task_type']}]\n{formatted}")

    # High complexity warning
    if classification["complexity"] >= 7:
        parts.append(
            f"[Complexity: {classification['complexity']}/10] "
            f"{classification['recommended_action']}"
        )

    if not parts:
        return ""

    return "\n\n".join(parts)


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """PreToolUse hook entry point."""
    _ensure_dirs()

    # Read hook data from stdin
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        sys.exit(0)

    tool_name = data.get("tool_name", data.get("tool", "unknown"))
    tool_input = data.get("tool_input", {})
    session_id = data.get("session_id", "unknown")

    # Classify
    classification = classify(tool_name, tool_input)

    # Load instincts
    instincts = _load_instincts(classification["task_type"])

    # Load active plan
    plan = _load_active_plan()

    # Record observation
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    input_summary = {}
    if isinstance(tool_input, dict):
        input_summary = {k: str(v)[:200] for k, v in tool_input.items()}

    observation = {
        "ts": now,
        "event": "pre_tool",
        "tool": tool_name,
        "session": session_id,
        "task_type": classification["task_type"],
        "complexity": classification["complexity"],
        "input_keys": list(input_summary.keys())[:10],
    }
    _append_observation(observation)

    # Build and output additional context
    additional_context = _build_additional_context(classification, instincts, plan)

    if additional_context:
        output = {"additionalContext": additional_context}
        sys.stdout.write(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
