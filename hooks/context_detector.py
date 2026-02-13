#!/usr/bin/env python3
"""
WeEvolve Context Detector - Pure Python Task Classifier
========================================================
Zero API calls. Zero dependencies. Under 5ms execution.

Classifies tool_name + parameters into task types and scores complexity.
Used by pre_tool.py and post_tool.py to annotate observations.
"""

from __future__ import annotations

from typing import Any

# ============================================================================
# TASK TYPE CLASSIFICATION
# ============================================================================

# Tool name -> task type mapping (exact match first, then prefix/pattern)
_TOOL_TYPE_MAP: dict[str, str] = {
    # Coding tools
    "Edit": "coding",
    "Write": "coding",
    "NotebookEdit": "coding",
    # Reading / research
    "Read": "research",
    "WebFetch": "research",
    "WebSearch": "research",
    "Glob": "research",
    "Grep": "research",
    # Testing
    "Bash": "general",  # refined below by command content
    # Git
    "Task": "agent_orchestration",
    "Skill": "agent_orchestration",
}

# Bash command prefix -> task type (checked in order, first match wins)
_BASH_PATTERNS: list[tuple[str, str]] = [
    # Testing
    ("pytest", "testing"),
    ("vitest", "testing"),
    ("jest", "testing"),
    ("npm test", "testing"),
    ("npm run test", "testing"),
    ("cargo test", "testing"),
    ("go test", "testing"),
    ("python -m pytest", "testing"),
    ("python -m unittest", "testing"),
    # Deployment
    ("docker", "deployment"),
    ("kubectl", "deployment"),
    ("helm", "deployment"),
    ("terraform", "deployment"),
    ("vercel", "deployment"),
    ("fly deploy", "deployment"),
    ("npm run build", "deployment"),
    ("npm run deploy", "deployment"),
    # Git
    ("git commit", "git"),
    ("git push", "git"),
    ("git pull", "git"),
    ("git merge", "git"),
    ("git rebase", "git"),
    ("git checkout", "git"),
    ("git branch", "git"),
    ("git stash", "git"),
    ("git diff", "git"),
    ("git log", "git"),
    ("git status", "git"),
    ("gh pr", "git"),
    ("gh issue", "git"),
    # Debugging
    ("node --inspect", "debugging"),
    ("python -m pdb", "debugging"),
    ("lldb", "debugging"),
    ("gdb", "debugging"),
    # Architecture
    ("npm init", "architecture"),
    ("cargo init", "architecture"),
    ("mkdir -p src/", "architecture"),
    # Package management (coding adjacent)
    ("npm install", "coding"),
    ("pip install", "coding"),
    ("cargo add", "coding"),
    # Linting / formatting
    ("eslint", "coding"),
    ("prettier", "coding"),
    ("ruff", "coding"),
    ("black", "coding"),
    ("mypy", "coding"),
    ("tsc", "coding"),
    ("python3 -c", "coding"),
]

# Keywords in file paths that hint at task type
_PATH_KEYWORDS: dict[str, str] = {
    "test": "testing",
    "spec": "testing",
    "__tests__": "testing",
    ".test.": "testing",
    ".spec.": "testing",
    "deploy": "deployment",
    "docker": "deployment",
    "k8s": "deployment",
    "ci/": "deployment",
    ".github/": "deployment",
    "security": "architecture",
    "auth": "architecture",
    "migration": "architecture",
    "schema": "architecture",
}


def _classify_bash_command(command: str) -> str:
    """Classify a Bash command string into a task type."""
    cmd_lower = command.strip().lower()
    for prefix, task_type in _BASH_PATTERNS:
        if cmd_lower.startswith(prefix):
            return task_type
    # Fallback heuristics
    if "grep" in cmd_lower or "find" in cmd_lower or "ls" in cmd_lower:
        return "research"
    if "rm " in cmd_lower or "mv " in cmd_lower or "cp " in cmd_lower:
        return "coding"
    return "general"


def _classify_by_path(file_path: str) -> str | None:
    """Classify based on file path keywords."""
    path_lower = file_path.lower()
    for keyword, task_type in _PATH_KEYWORDS.items():
        if keyword in path_lower:
            return task_type
    return None


# ============================================================================
# COMPLEXITY SCORING
# ============================================================================

# Base complexity per tool
_TOOL_COMPLEXITY: dict[str, int] = {
    "Read": 1,
    "Glob": 1,
    "Grep": 2,
    "Write": 3,
    "Edit": 3,
    "NotebookEdit": 4,
    "Bash": 3,
    "WebFetch": 2,
    "WebSearch": 2,
    "Task": 6,
    "Skill": 5,
}


def _score_complexity(
    tool_name: str,
    tool_input: dict[str, Any],
    task_type: str,
) -> int:
    """
    Score complexity 1-10 based on tool, input size, and task type.

    Factors:
    - Base tool complexity
    - Input size (longer commands/content = more complex)
    - Multi-file indicators
    - Error-prone task types get a bump
    """
    base = _TOOL_COMPLEXITY.get(tool_name, 3)

    # Input size factor
    input_str = str(tool_input)
    length = len(input_str)
    if length > 2000:
        base += 2
    elif length > 500:
        base += 1

    # Multi-file indicator (Write/Edit with long content)
    if tool_name in ("Write", "Edit") and length > 1000:
        base += 1

    # Bash command complexity
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        # Piped commands
        if "|" in command:
            base += 1
        # Chained commands
        if "&&" in command and command.count("&&") > 1:
            base += 1
        # Background processes
        if tool_input.get("run_in_background"):
            base += 1

    # Task type adjustments
    if task_type == "deployment":
        base += 1
    elif task_type == "architecture":
        base += 1
    elif task_type == "agent_orchestration":
        base += 1

    return max(1, min(10, base))


# ============================================================================
# RECOMMENDED ACTIONS
# ============================================================================

_TASK_ACTIONS: dict[str, str] = {
    "coding": "Apply instincts for code quality, immutability, error handling",
    "testing": "Verify TDD flow: RED -> GREEN -> IMPROVE",
    "debugging": "Check error patterns, load relevant failure instincts",
    "research": "Cross-reference with existing knowledge before acting",
    "architecture": "Consider SOLID, review design patterns, check security",
    "deployment": "Validate configs, check for secrets, verify rollback plan",
    "git": "Review changes, check for secrets in diff, conventional commits",
    "agent_orchestration": "Verify agent topology, check for drift",
    "general": "Observe and learn",
}


# ============================================================================
# PUBLIC API
# ============================================================================


def classify(
    tool_name: str,
    tool_input: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Classify a tool invocation into task type, complexity, and recommended action.

    Parameters
    ----------
    tool_name : str
        The Claude Code tool name (Bash, Edit, Write, Read, etc.)
    tool_input : dict, optional
        The tool's input parameters

    Returns
    -------
    dict with keys:
        task_type: str          - one of the defined task types
        complexity: int         - 1-10 score
        recommended_action: str - guidance for the agent
    """
    if tool_input is None:
        tool_input = {}

    # Step 1: Direct tool mapping
    task_type = _TOOL_TYPE_MAP.get(tool_name, "general")

    # Step 2: Refine Bash commands
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        task_type = _classify_bash_command(command)

    # Step 3: Refine by file path (for Edit, Write, Read)
    if task_type in ("coding", "research", "general"):
        file_path = tool_input.get("file_path", "")
        if not file_path:
            file_path = tool_input.get("path", "")
        if file_path:
            path_type = _classify_by_path(file_path)
            if path_type is not None:
                task_type = path_type

    # Step 4: Score complexity
    complexity = _score_complexity(tool_name, tool_input, task_type)

    # Step 5: Recommended action
    recommended_action = _TASK_ACTIONS.get(task_type, "Observe and learn")

    return {
        "task_type": task_type,
        "complexity": complexity,
        "recommended_action": recommended_action,
    }
