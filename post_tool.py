#!/usr/bin/env python3
"""
WeEvolve PostToolUse Hook - Runs AFTER Every Tool Invocation
=============================================================
External process. Under 100ms. Zero API calls.

What it does:
  1. Reads tool outcome from stdin (JSON from Claude Code)
  2. Detects errors (exit_code != 0) and flags for instinct extraction
  3. Appends outcome observation to ~/.weevolve/observations.jsonl
  4. Tracks error patterns for the session
  5. Prints SEED Phase 8 reminder on errors: "Did this produce extractable knowledge?"

Claude Code hook protocol:
  - stdin: JSON with tool_name, tool_input, tool_output, exit_code, session_id
  - stdout: optional additionalContext
  - exit 0: continue
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import context_detector
_HOOKS_DIR = Path(__file__).parent
try:
    from weevolve.hooks.context_detector import classify
except ImportError:
    sys.path.insert(0, str(_HOOKS_DIR))
    from context_detector import classify  # type: ignore[import-not-found]


# ============================================================================
# PATHS
# ============================================================================

_WEEVOLVE_DIR = Path.home() / ".weevolve"
_OBSERVATIONS_FILE = _WEEVOLVE_DIR / "observations.jsonl"
_ERROR_FLAGS_FILE = _WEEVOLVE_DIR / "error_flags.jsonl"
_SESSION_ERRORS_FILE = _WEEVOLVE_DIR / "session_errors.json"
_MAX_OBS_SIZE_MB = 10


# ============================================================================
# HELPERS
# ============================================================================


def _ensure_dirs() -> None:
    _WEEVOLVE_DIR.mkdir(parents=True, exist_ok=True)


def _rotate_observations_if_needed() -> None:
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
    _rotate_observations_if_needed()
    try:
        with _OBSERVATIONS_FILE.open("a") as f:
            f.write(json.dumps(observation, separators=(",", ":")) + "\n")
    except OSError:
        pass


# ============================================================================
# ERROR DETECTION
# ============================================================================

# Patterns that indicate an error even if exit_code is 0
_ERROR_PATTERNS: list[str] = [
    "error:",
    "Error:",
    "ERROR:",
    "FAILED",
    "Traceback (most recent call last)",
    "SyntaxError:",
    "TypeError:",
    "ImportError:",
    "ModuleNotFoundError:",
    "AttributeError:",
    "KeyError:",
    "ValueError:",
    "RuntimeError:",
    "FileNotFoundError:",
    "PermissionError:",
    "ConnectionError:",
    "TimeoutError:",
    "ENOENT",
    "EACCES",
    "EPERM",
    "panic:",
    "segmentation fault",
    "core dumped",
]


def _detect_error(exit_code: int | None, output: str) -> bool:
    """Check if the tool execution resulted in an error."""
    if exit_code is not None and exit_code != 0:
        return True
    # Check output for error patterns (first 2000 chars for speed)
    snippet = output[:2000].lower() if output else ""
    return any(pattern.lower() in snippet for pattern in _ERROR_PATTERNS)


def _extract_error_summary(output: str) -> str:
    """Extract a short error summary from tool output."""
    if not output:
        return "unknown error"
    lines = output.strip().split("\n")
    # Find the most informative error line
    for line in reversed(lines[-20:]):
        line_lower = line.lower().strip()
        for pattern in _ERROR_PATTERNS:
            if pattern.lower() in line_lower:
                return line.strip()[:300]
    # Fallback: last non-empty line
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped[:300]
    return "unknown error"


def _flag_error_for_instinct(
    tool_name: str,
    task_type: str,
    error_summary: str,
    session_id: str,
) -> None:
    """
    Write an error flag to error_flags.jsonl for later instinct extraction.

    The session_end hook reads these flags and converts repeated patterns
    into instincts (learned avoidance patterns).
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    flag = {
        "ts": now,
        "tool": tool_name,
        "task_type": task_type,
        "error": error_summary,
        "session": session_id,
    }
    try:
        with _ERROR_FLAGS_FILE.open("a") as f:
            f.write(json.dumps(flag, separators=(",", ":")) + "\n")
    except OSError:
        pass


def _update_session_error_count(session_id: str) -> int:
    """Track cumulative error count for this session. Returns new count."""
    try:
        if _SESSION_ERRORS_FILE.exists():
            data = json.loads(_SESSION_ERRORS_FILE.read_text())
        else:
            data = {}
    except (json.JSONDecodeError, OSError):
        data = {}

    count = data.get(session_id, 0) + 1
    data[session_id] = count

    # Keep only last 20 sessions to prevent unbounded growth
    if len(data) > 20:
        sorted_sessions = sorted(data.items(), key=lambda x: x[1], reverse=True)
        data = dict(sorted_sessions[:20])

    try:
        _SESSION_ERRORS_FILE.write_text(json.dumps(data, separators=(",", ":")))
    except OSError:
        pass

    return count


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """PostToolUse hook entry point."""
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
    tool_output = data.get("tool_output", "")
    exit_code = data.get("exit_code")
    session_id = data.get("session_id", "unknown")

    # Ensure tool_output is a string
    if isinstance(tool_output, dict):
        tool_output = json.dumps(tool_output)
    elif not isinstance(tool_output, str):
        tool_output = str(tool_output) if tool_output is not None else ""

    # Classify the tool use
    classification = classify(tool_name, tool_input)
    task_type = classification["task_type"]

    # Detect errors
    is_error = _detect_error(exit_code, tool_output)
    error_summary = ""

    if is_error:
        error_summary = _extract_error_summary(tool_output)
        _flag_error_for_instinct(tool_name, task_type, error_summary, session_id)
        error_count = _update_session_error_count(session_id)

    # Record observation
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    observation = {
        "ts": now,
        "event": "post_tool",
        "tool": tool_name,
        "session": session_id,
        "task_type": task_type,
        "complexity": classification["complexity"],
        "is_error": is_error,
    }
    if is_error:
        observation["error_summary"] = error_summary[:200]
    _append_observation(observation)

    # Build output context
    context_parts: list[str] = []

    if is_error:
        context_parts.append(
            f"[SEED Phase 8 - IMPROVE] Error in {tool_name} ({task_type}): "
            f"{error_summary[:150]}\n"
            f"Did this produce extractable knowledge? "
            f"What instinct should we learn from this?"
        )
        if error_count >= 3:  # type: ignore[possibly-undefined]
            context_parts.append(
                f"[Pattern Alert] {error_count} errors this session. "
                f"Consider stepping back to diagnose the root cause."
            )

    if context_parts:
        output = {"additionalContext": "\n\n".join(context_parts)}
        sys.stdout.write(json.dumps(output))

    sys.exit(0)


if __name__ == "__main__":
    main()
