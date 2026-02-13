"""
WeEvolve Hooks Engine - Auto-Trigger System for SEED Protocol
=============================================================
Makes SEED activate on every interaction without manual commands.

Hooks:
  pre_tool.py      - PreToolUse: classify task, load instincts, inject context
  post_tool.py     - PostToolUse: capture outcome, flag errors for learning
  session_end.py   - Stop: persist plans, extract instincts, summarize
  context_detector.py - Pure classifier: task type + complexity scoring
  install_hooks.py - Installer: wire hooks into ~/.claude/settings.json

All hooks are EXTERNAL processes (subprocess, not in-agent).
All hooks run under 100ms. Zero API calls. Zero network.
"""
