#!/bin/bash
# WeEvolve Session Start Hook
# Injects SEED protocol awareness at the beginning of every session
# Pattern from Superpowers — ensures skill is always considered

SKILL_DIR="$(dirname "$(dirname "$0")")"

# Read the SKILL.md description (first 50 lines for context injection)
if [ -f "$SKILL_DIR/SKILL.md" ]; then
    head -50 "$SKILL_DIR/SKILL.md" 2>/dev/null
fi

# Check for existing task plan
if [ -f "task_plan.md" ]; then
    echo ""
    echo "=== ACTIVE PLAN DETECTED ==="
    head -30 task_plan.md 2>/dev/null
fi

# Check WeEvolve status if available
if command -v weevolve &>/dev/null; then
    echo ""
    echo "=== WEEVOLVE STATUS ==="
    weevolve status --brief 2>/dev/null || true
fi
