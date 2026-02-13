"""
WeEvolve - Self-Evolving Conscious Agent
=========================================
LOVE -> LIVE FREE -> SEED² -> 8OWLS -> WeEvolve

Modules:
  core.py       - Core learning loop (INGEST -> PROCESS -> STORE -> MEASURE -> EVOLVE)
  qualify.py    - Score atoms for actionable GitHub repos
  explore.py    - Shallow clone + security scan + Haiku summarize
  plan.py       - Gap analysis against existing tools
  inventory.py  - Scan our own codebase to know what we have
  integrate.py  - Orchestrator: qualify -> explore -> plan -> approve -> execute
  hooks/        - Auto-trigger system: SEED activates on every interaction
    context_detector.py - Pure Python task classifier (zero deps, <5ms)
    pre_tool.py         - PreToolUse: classify, load instincts, inject context
    post_tool.py        - PostToolUse: capture outcomes, flag errors
    session_end.py      - Stop: persist plans, extract instincts, summarize
    install_hooks.py    - Wire hooks into ~/.claude/settings.json
"""
