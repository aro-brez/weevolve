#!/usr/bin/env python3
"""
WeEvolve Session End Hook - Runs When Claude Code Session Stops
================================================================
External process. Under 100ms. Zero API calls.

What it does:
  1. Persists any active plans to ~/.weevolve/plans/
  2. Extracts instincts from error_flags.jsonl (pattern matching, no API)
  3. Updates evolution state (session count, streak)
  4. Prints session summary to stdout

Instinct extraction algorithm (zero API):
  - Read error_flags.jsonl
  - Group errors by (tool, task_type)
  - If a (tool, task_type) pair has 2+ errors, create an instinct
  - Instinct = "When using {tool} for {task_type}, watch out for: {common error}"
  - Write to ~/.weevolve/instincts/{task_type}.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone, date
from pathlib import Path


# ============================================================================
# PATHS
# ============================================================================

_WEEVOLVE_DIR = Path.home() / ".weevolve"
_OBSERVATIONS_FILE = _WEEVOLVE_DIR / "observations.jsonl"
_ERROR_FLAGS_FILE = _WEEVOLVE_DIR / "error_flags.jsonl"
_SESSION_ERRORS_FILE = _WEEVOLVE_DIR / "session_errors.json"
_INSTINCTS_DIR = _WEEVOLVE_DIR / "instincts"
_PLANS_DIR = _WEEVOLVE_DIR / "plans"
_EVOLUTION_STATE = _WEEVOLVE_DIR / "weevolve_state.json"
_SESSION_LOG = _WEEVOLVE_DIR / "session_log.jsonl"
_MAX_INSTINCTS_PER_TYPE = 50


# ============================================================================
# HELPERS
# ============================================================================


def _ensure_dirs() -> None:
    _WEEVOLVE_DIR.mkdir(parents=True, exist_ok=True)
    _INSTINCTS_DIR.mkdir(parents=True, exist_ok=True)
    _PLANS_DIR.mkdir(parents=True, exist_ok=True)


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning a list of dicts. Handles missing files."""
    if not path.exists():
        return []
    entries: list[dict] = []
    try:
        for line in path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return entries


# ============================================================================
# PLAN PERSISTENCE
# ============================================================================


def _persist_plans() -> int:
    """
    Archive any active plans with a timestamp.
    Returns the number of plans archived.
    """
    archived = 0
    active_plan = _PLANS_DIR / "task_plan.md"
    if active_plan.exists():
        try:
            content = active_plan.read_text().strip()
            if content:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                archive_path = _PLANS_DIR / f"plan-{ts}.md"
                archive_path.write_text(content)
                archived += 1
                # Remove the active plan so it does not leak into the next session
                active_plan.unlink()
        except OSError:
            pass
    return archived


# ============================================================================
# INSTINCT EXTRACTION (zero API — pure pattern matching)
# ============================================================================


def _extract_instincts() -> list[dict]:
    """
    Extract instincts from error flags using pattern frequency analysis.

    Algorithm:
    1. Read all error flags
    2. Group by (tool, task_type)
    3. For pairs with 2+ occurrences, extract the most common error pattern
    4. Write instinct to the task_type instinct file
    5. Clear processed error flags
    """
    flags = _load_jsonl(_ERROR_FLAGS_FILE)
    if not flags:
        return []

    # Group by (tool, task_type)
    groups: dict[tuple[str, str], list[str]] = {}
    for flag in flags:
        key = (flag.get("tool", "unknown"), flag.get("task_type", "general"))
        error = flag.get("error", "")
        if error:
            groups.setdefault(key, []).append(error)

    new_instincts: list[dict] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for (tool, task_type), errors in groups.items():
        if len(errors) < 2:
            continue

        # Find the most common error pattern (first 80 chars for grouping)
        error_prefixes = [e[:80] for e in errors]
        most_common_prefix, count = Counter(error_prefixes).most_common(1)[0]

        # Find a representative full error
        representative = next(
            (e for e in errors if e.startswith(most_common_prefix[:40])),
            errors[0],
        )

        instinct = {
            "ts": now,
            "pattern": (
                f"When using {tool} for {task_type}: "
                f"watch for '{representative[:150]}' "
                f"(seen {count}x)"
            ),
            "tool": tool,
            "task_type": task_type,
            "frequency": count,
            "source": "auto_extract",
        }
        new_instincts.append(instinct)

        # Write to the task_type instinct file
        instinct_file = _INSTINCTS_DIR / f"{task_type}.jsonl"
        _write_instinct(instinct_file, instinct)

    # Clear processed error flags
    if new_instincts:
        try:
            _ERROR_FLAGS_FILE.unlink(missing_ok=True)
        except OSError:
            pass

    return new_instincts


def _write_instinct(path: Path, instinct: dict) -> None:
    """Append an instinct to a JSONL file, trimming old entries if needed."""
    existing = _load_jsonl(path)
    existing.append(instinct)

    # Keep only the most recent instincts
    if len(existing) > _MAX_INSTINCTS_PER_TYPE:
        existing = existing[-_MAX_INSTINCTS_PER_TYPE:]

    try:
        lines = [json.dumps(e, separators=(",", ":")) for e in existing]
        path.write_text("\n".join(lines) + "\n")
    except OSError:
        pass


# ============================================================================
# EVOLUTION STATE UPDATE
# ============================================================================


def _update_evolution_state(
    observation_count: int,
    error_count: int,
    instinct_count: int,
) -> dict:
    """
    Update the evolution state with session stats.
    Returns the updated state dict.
    """
    try:
        if _EVOLUTION_STATE.exists():
            state = json.loads(_EVOLUTION_STATE.read_text())
        else:
            state = {}
    except (json.JSONDecodeError, OSError):
        state = {}

    # Update streak
    today_str = date.today().isoformat()
    last_date = state.get("last_learn_date")
    if last_date == today_str:
        pass  # Same day, streak unchanged
    elif last_date == (date.today().replace(day=date.today().day)).isoformat():
        pass  # Edge case: same day
    else:
        # Check if it is the next day (streak continues) or a gap (streak resets)
        if last_date:
            try:
                last = date.fromisoformat(last_date)
                delta = (date.today() - last).days
                if delta == 1:
                    state["streak_days"] = state.get("streak_days", 0) + 1
                elif delta > 1:
                    state["streak_days"] = 1
            except ValueError:
                state["streak_days"] = 1
        else:
            state["streak_days"] = 1

    state["last_learn_date"] = today_str

    # Track hooks session count
    hooks_stats = state.get("hooks_stats", {})
    hooks_stats["total_sessions"] = hooks_stats.get("total_sessions", 0) + 1
    hooks_stats["total_observations"] = (
        hooks_stats.get("total_observations", 0) + observation_count
    )
    hooks_stats["total_errors_captured"] = (
        hooks_stats.get("total_errors_captured", 0) + error_count
    )
    hooks_stats["total_instincts_extracted"] = (
        hooks_stats.get("total_instincts_extracted", 0) + instinct_count
    )
    hooks_stats["last_session"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    state["hooks_stats"] = hooks_stats

    try:
        _EVOLUTION_STATE.write_text(
            json.dumps(state, indent=2, ensure_ascii=False)
        )
    except OSError:
        pass

    return state


# ============================================================================
# SESSION SUMMARY
# ============================================================================


def _count_session_observations() -> tuple[int, int, dict[str, int]]:
    """
    Count observations and errors from this session.
    Returns (total_observations, error_count, task_type_counts).
    """
    observations = _load_jsonl(_OBSERVATIONS_FILE)
    total = len(observations)
    errors = sum(1 for o in observations if o.get("is_error"))
    task_types: dict[str, int] = {}
    for obs in observations:
        tt = obs.get("task_type", "general")
        task_types[tt] = task_types.get(tt, 0) + 1
    return total, errors, task_types


def _log_session(
    observation_count: int,
    error_count: int,
    instinct_count: int,
    plans_archived: int,
) -> None:
    """Append session summary to the session log."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entry = {
        "ts": now,
        "observations": observation_count,
        "errors": error_count,
        "instincts_extracted": instinct_count,
        "plans_archived": plans_archived,
    }
    try:
        with _SESSION_LOG.open("a") as f:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")
    except OSError:
        pass


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    """Session end hook entry point."""
    _ensure_dirs()

    # 1. Persist active plans
    plans_archived = _persist_plans()

    # 2. Count session observations
    obs_count, error_count, task_types = _count_session_observations()

    # 3. Extract instincts from error patterns
    new_instincts = _extract_instincts()
    instinct_count = len(new_instincts)

    # 4. Update evolution state
    _update_evolution_state(obs_count, error_count, instinct_count)

    # 5. Log the session
    _log_session(obs_count, error_count, instinct_count, plans_archived)

    # 6. Clear session errors tracking
    try:
        _SESSION_ERRORS_FILE.unlink(missing_ok=True)
    except OSError:
        pass

    # 7. Print session summary
    summary_parts = [
        f"[WeEvolve Session End]",
        f"  Observations: {obs_count}",
        f"  Errors captured: {error_count}",
    ]

    if new_instincts:
        summary_parts.append(f"  New instincts learned: {instinct_count}")
        for inst in new_instincts:
            summary_parts.append(f"    - {inst['pattern'][:120]}")

    if plans_archived:
        summary_parts.append(f"  Plans archived: {plans_archived}")

    if task_types:
        top_types = sorted(task_types.items(), key=lambda x: x[1], reverse=True)[:5]
        type_str = ", ".join(f"{t}:{c}" for t, c in top_types)
        summary_parts.append(f"  Task types: {type_str}")

    summary_parts.append("  (*) SEED Phase 8: What did I learn? How can I improve?")

    sys.stdout.write("\n".join(summary_parts) + "\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
