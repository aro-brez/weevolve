"""
WeEvolve Upgrade Engine — SEED-squared auto-evolution
=====================================================
Scans the IMPROVEMENTS backlog, finds safe items, categorizes them,
presents a synopsis, waits for approval, applies upgrades, and
shows RPG progression + first-time user experience.

Usage (via core.py CLI):
  weevolve upgrade              # Scan, show synopsis, prompt for approval
  weevolve upgrade --auto       # Apply all safe items without prompting
  weevolve upgrade --dry-run    # Show synopsis only, apply nothing
  weevolve upgrade --category config  # Only config changes
  weevolve upgrade --synopsis   # Generate UPGRADE-SYNOPSIS.md only
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any

from weevolve.config import DATA_DIR, EVOLUTION_STATE_PATH


# ============================================================================
# CONSTANTS
# ============================================================================

CYAN = "\033[36m"
MAGENTA = "\033[35m"
GREEN_C = "\033[32m"
YELLOW_C = "\033[33m"
BLUE_C = "\033[34m"
LIME_C = "\033[38;5;190m"
DIM_C = "\033[2m"
RED_C = "\033[31m"
BOLD_C = "\033[1m"
RESET_C = "\033[0m"
WHITE_C = "\033[97m"

# XP values per upgrade type
XP_PER_CONFIG = 10
XP_PER_CODE = 25
XP_PER_TOOL = 50
XP_PER_KNOWLEDGE = 5

# Category metadata for display
CATEGORY_META = {
    'config': {
        'icon': 'CFG',
        'color': YELLOW_C,
        'label': 'Config Changes',
        'xp': XP_PER_CONFIG,
        'description': 'Settings, thresholds, filters, parameters',
    },
    'code': {
        'icon': 'COD',
        'color': CYAN,
        'label': 'Code Improvements',
        'xp': XP_PER_CODE,
        'description': 'New modules, classes, functions, refactors',
    },
    'tools': {
        'icon': 'TOL',
        'color': GREEN_C,
        'label': 'New Tools',
        'xp': XP_PER_TOOL,
        'description': 'New capabilities, integrations, daemons',
    },
    'knowledge': {
        'icon': 'KNW',
        'color': MAGENTA,
        'label': 'Knowledge Atoms',
        'xp': XP_PER_KNOWLEDGE,
        'description': 'Learnings, research, strategies, insights',
    },
}

# Skill mapping: which skills improve from which upgrade categories
CATEGORY_SKILL_MAP = {
    'config': ['strategy', 'trading', 'ai_engineering'],
    'code': ['coding', 'ai_engineering', 'security'],
    'tools': ['coding', 'ai_engineering', 'growth'],
    'knowledge': ['research', 'consciousness', 'strategy'],
}


# ============================================================================
# BACKLOG SCANNER
# ============================================================================

def find_improvements_file() -> Path:
    """Locate the IMPROVEMENTS/integrations.jsonl file."""
    # Try relative to WEEVOLVE_BASE_DIR first
    base = Path(os.environ.get('WEEVOLVE_BASE_DIR', os.getcwd()))
    candidates = [
        base / 'BRAIN' / 'IMPROVEMENTS' / 'integrations.jsonl',
        Path.home() / 'REPOS' / 'seed' / 'BRAIN' / 'IMPROVEMENTS' / 'integrations.jsonl',
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]  # Return first candidate even if missing (for error msg)


def scan_backlog(improvements_path: Path) -> Dict[str, List[Dict]]:
    """
    Parse the IMPROVEMENTS backlog and extract safe-to-apply items.
    Returns dict with categories: config, code, tools, knowledge.
    Each item contains: line, question, specific_action, risk_level, confidence.
    """
    config_items: List[Dict] = []
    code_items: List[Dict] = []
    tool_items: List[Dict] = []
    knowledge_items: List[Dict] = []

    if not improvements_path.exists():
        return {'config': [], 'code': [], 'tools': [], 'knowledge': []}

    with open(improvements_path) as f:
        for line_num, line in enumerate(f):
            try:
                data = json.loads(line.strip())
            except (json.JSONDecodeError, ValueError):
                continue

            ev = data.get('evaluation', {})

            # Type 1: items from continuous_improver with evaluation data
            if isinstance(ev, dict) and ev:
                risk = ev.get('risk_level', 'high')
                safe = ev.get('safe_to_integrate', False)

                if not (safe or risk in ('none', 'low')):
                    continue

                action = ev.get('action_type', 'manual_review')
                item = {
                    'line': line_num,
                    'question': data.get('question', '')[:300],
                    'specific_action': ev.get('specific_action', '')[:500],
                    'risk_level': risk,
                    'action_type': action,
                    'confidence': ev.get('confidence', 0),
                    'reasoning': ev.get('reasoning', ''),
                    'timestamp': data.get('timestamp', ''),
                }

                if action == 'config_change':
                    config_items.append(item)
                elif action == 'code_change':
                    code_items.append(item)
                elif action == 'new_tool':
                    tool_items.append(item)
                else:
                    knowledge_items.append(item)

            # Type 2: items from evolution_daemon with status=pending
            elif data.get('status') == 'pending':
                t = data.get('type', '')
                item = {
                    'line': line_num,
                    'question': data.get('title', ''),
                    'specific_action': data.get('description', '')[:500],
                    'risk_level': 'low',
                    'action_type': f'evolution_{t}',
                    'confidence': 0.7,
                    'reasoning': f'Evolution daemon {t} suggestion',
                    'timestamp': data.get('timestamp', ''),
                }

                if t == 'config':
                    config_items.append(item)
                elif t in ('code', 'security'):
                    code_items.append(item)
                else:
                    knowledge_items.append(item)

    return {
        'config': config_items,
        'code': code_items,
        'tools': tool_items,
        'knowledge': knowledge_items,
    }


def deduplicate_items(items: List[Dict]) -> List[Dict]:
    """Remove near-duplicate items based on action description similarity."""
    seen: Dict[str, Dict] = {}
    for item in items:
        key = item.get('specific_action', '')[:80].lower().strip()
        if key and key not in seen:
            seen[key] = item
    return list(seen.values())


# ============================================================================
# SYNOPSIS GENERATOR
# ============================================================================

def generate_synopsis(categorized: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Generate a concise synopsis of all upgrades with deduplication.
    Returns structured data for both terminal display and markdown export.
    """
    deduped = {
        cat: deduplicate_items(items)
        for cat, items in categorized.items()
    }

    total_raw = sum(len(v) for v in categorized.values())
    total_unique = sum(len(v) for v in deduped.values())

    # Calculate XP projection
    total_xp = 0
    for cat, items in deduped.items():
        meta = CATEGORY_META.get(cat, {})
        total_xp += len(items) * meta.get('xp', 5)

    # Calculate skill improvements
    skill_projections: Dict[str, float] = {}
    for cat, items in deduped.items():
        skills = CATEGORY_SKILL_MAP.get(cat, [])
        count = len(items)
        for skill in skills:
            improvement = count * 0.15  # ~0.15 per item per skill
            skill_projections[skill] = skill_projections.get(skill, 0) + improvement

    return {
        'raw': categorized,
        'deduped': deduped,
        'total_raw': total_raw,
        'total_unique': total_unique,
        'total_xp': total_xp,
        'skill_projections': skill_projections,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def display_synopsis(synopsis: Dict[str, Any]):
    """Display a beautiful terminal synopsis for ARO."""
    deduped = synopsis['deduped']
    total = synopsis['total_unique']
    total_xp = synopsis['total_xp']

    print()
    print(f"  {BOLD_C}{'=' * 58}{RESET_C}")
    print(f"  {BOLD_C}{LIME_C}  UPGRADE SYNOPSIS{RESET_C} {DIM_C}-- {total} improvements ready to apply{RESET_C}")
    print(f"  {BOLD_C}{DIM_C}  SEED-squared: the system that evolves itself{RESET_C}")
    print(f"  {BOLD_C}{'=' * 58}{RESET_C}")
    print()

    for cat in ['config', 'code', 'tools', 'knowledge']:
        items = deduped.get(cat, [])
        if not items:
            continue
        meta = CATEGORY_META[cat]
        color = meta['color']
        icon = meta['icon']
        label = meta['label']
        xp = meta['xp']

        print(f"  {color}{BOLD_C}[{icon}] {label} ({len(items)}){RESET_C}  {DIM_C}+{xp} XP each{RESET_C}")
        print(f"  {DIM_C}{meta['description']}{RESET_C}")

        # Show top 5 items
        for i, item in enumerate(items[:5]):
            action = item.get('specific_action', '')[:90]
            risk = item.get('risk_level', '?')
            risk_color = GREEN_C if risk == 'none' else (YELLOW_C if risk == 'low' else RED_C)
            print(f"    {DIM_C}{i + 1}.{RESET_C} {action}")
            print(f"       {risk_color}[{risk}]{RESET_C}")

        remaining = len(items) - 5
        if remaining > 0:
            print(f"    {DIM_C}... and {remaining} more{RESET_C}")
        print()

    # XP projection
    print(f"  {BOLD_C}{'- ' * 29}{RESET_C}")
    print(f"  {BOLD_C}PROJECTED GAINS:{RESET_C}")
    print(f"    {LIME_C}+{total_xp} XP{RESET_C} total")
    print()

    # Skill projections
    skill_proj = synopsis.get('skill_projections', {})
    if skill_proj:
        print(f"  {BOLD_C}SKILL IMPROVEMENTS:{RESET_C}")
        for skill, delta in sorted(skill_proj.items(), key=lambda x: x[1], reverse=True):
            bar_len = min(20, int(delta / 2))
            bar = f"{GREEN_C}{'>' * bar_len}{RESET_C}"
            print(f"    {skill:20s} {bar} +{delta:.1f}")
        print()


def write_synopsis_markdown(synopsis: Dict[str, Any], output_path: Path):
    """Write the synopsis as a Markdown file for ARO to review."""
    deduped = synopsis['deduped']
    total = synopsis['total_unique']
    total_xp = synopsis['total_xp']
    skill_proj = synopsis.get('skill_projections', {})

    lines = [
        "# UPGRADE SYNOPSIS -- WeEvolve Auto-Evolution",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Source:** BRAIN/IMPROVEMENTS/integrations.jsonl ({synopsis['total_raw']} raw, {total} unique after dedup)",
        f"**Total XP:** +{total_xp}",
        "",
        "---",
        "",
        f"## Summary: {total} improvements ready to apply",
        "",
        f"| Category | Count | XP Each | Total XP |",
        f"|----------|-------|---------|----------|",
    ]

    for cat in ['config', 'code', 'tools', 'knowledge']:
        items = deduped.get(cat, [])
        meta = CATEGORY_META[cat]
        xp_each = meta['xp']
        lines.append(
            f"| {meta['label']} | {len(items)} | +{xp_each} | +{len(items) * xp_each} |"
        )

    lines.extend([
        f"| **TOTAL** | **{total}** | | **+{total_xp}** |",
        "",
        "---",
        "",
    ])

    # Detailed sections
    for cat in ['config', 'code', 'tools', 'knowledge']:
        items = deduped.get(cat, [])
        if not items:
            continue
        meta = CATEGORY_META[cat]
        lines.append(f"## {meta['label']} ({len(items)})")
        lines.append(f"*{meta['description']}*")
        lines.append("")

        for i, item in enumerate(items, 1):
            action = item.get('specific_action', '')[:200]
            risk = item.get('risk_level', '?')
            lines.append(f"{i}. **[{risk}]** {action}")

        lines.append("")

    # Skill projections
    if skill_proj:
        lines.extend([
            "---",
            "",
            "## Projected Skill Improvements",
            "",
            "| Skill | Estimated Gain |",
            "|-------|----------------|",
        ])
        for skill, delta in sorted(skill_proj.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"| {skill} | +{delta:.1f} |")
        lines.append("")

    # Approval section
    lines.extend([
        "---",
        "",
        "## How to Apply",
        "",
        "```bash",
        "# Review and approve interactively:",
        "python3 tools/weevolve/core.py upgrade",
        "",
        "# Apply all safe items automatically:",
        "python3 tools/weevolve/core.py upgrade --auto",
        "",
        "# Dry run (preview only):",
        "python3 tools/weevolve/core.py upgrade --dry-run",
        "",
        "# Apply only one category:",
        "python3 tools/weevolve/core.py upgrade --category config",
        "python3 tools/weevolve/core.py upgrade --category code",
        "python3 tools/weevolve/core.py upgrade --category tools",
        "python3 tools/weevolve/core.py upgrade --category knowledge",
        "```",
        "",
        "---",
        "*Generated by WeEvolve SEED-squared auto-evolution engine*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"  {GREEN_C}Synopsis written to:{RESET_C} {output_path}")


# ============================================================================
# UPGRADE APPLIER
# ============================================================================

def apply_upgrades(
    synopsis: Dict[str, Any],
    categories: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Apply approved upgrades to the WeEvolve evolution state.

    This ingests improvements as knowledge atoms into the WeEvolve DB,
    awards XP, improves skills, and marks items as integrated in the
    backlog. The actual code/config changes described in the items
    are tracked as applied learnings -- the system KNOWS about them
    and can reference them, even if the literal code change requires
    a separate agent to execute.

    Returns a results dict with XP gained, levels, skill deltas.
    """
    from weevolve.core import (
        init_db, load_evolution_state, save_evolution_state,
        grant_xp, improve_skills,
    )

    deduped = synopsis['deduped']
    if categories:
        apply_cats = {c: deduped.get(c, []) for c in categories if c in deduped}
    else:
        apply_cats = deduped

    db = init_db()
    state = load_evolution_state()
    old_level = state.get('level', 1)
    old_xp = state.get('xp', 0)
    old_skills = {k: v for k, v in state.get('skills', {}).items()}

    total_applied = 0
    total_xp_earned = 0
    all_skill_deltas: Dict[str, float] = {}
    applied_items: List[Dict] = []
    level_ups: List[int] = []

    for cat, items in apply_cats.items():
        meta = CATEGORY_META.get(cat, {})
        xp_per = meta.get('xp', 5)
        skill_targets = CATEGORY_SKILL_MAP.get(cat, [])

        for item in items:
            # Store as knowledge atom
            import hashlib
            content_hash = hashlib.sha256(
                item.get('specific_action', '').encode()
            ).hexdigest()[:16]
            atom_id = f"upgrade_{cat}_{content_hash}"

            # Check if already applied
            existing = db.execute(
                "SELECT id FROM knowledge_atoms WHERE id = ?", (atom_id,)
            ).fetchone()
            if existing:
                continue

            # Insert atom
            try:
                db.execute("""
                    INSERT INTO knowledge_atoms
                    (id, source_type, content_hash, title, raw_content,
                     perceive, connect, learn, question, improve,
                     skills, quality, created_at, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    atom_id,
                    f'upgrade_{cat}',
                    content_hash,
                    item.get('question', '')[:200],
                    item.get('specific_action', ''),
                    f"Upgrade backlog item: {cat}",
                    f"Connects to: {', '.join(skill_targets)}",
                    item.get('specific_action', '')[:300],
                    item.get('question', '')[:200],
                    f"Applied as {cat} upgrade",
                    json.dumps(skill_targets),
                    0.7 if item.get('risk_level') in ('none', 'low') else 0.5,
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                ))
            except Exception:
                continue  # Skip duplicates silently

            # Grant XP
            prev_level = state.get('level', 1)
            state = grant_xp(state, xp_per, f"upgrade_{cat}")
            if state.get('level', 1) > prev_level:
                level_ups.append(state['level'])
            total_xp_earned += xp_per

            # Improve skills
            state, deltas = improve_skills(state, skill_targets, 0.7)
            for skill, delta in deltas.items():
                all_skill_deltas[skill] = all_skill_deltas.get(skill, 0) + delta

            total_applied += 1
            applied_items.append({
                'id': atom_id,
                'category': cat,
                'action': item.get('specific_action', '')[:100],
            })

    # Update totals in state
    state = {
        **state,
        'total_learnings': state.get('total_learnings', 0) + total_applied,
        'total_insights': state.get('total_insights', 0) + len([
            i for i in applied_items if i['category'] in ('code', 'tools')
        ]),
    }

    # Log the upgrade event
    db.execute("""
        INSERT INTO evolution_log (event_type, details, xp_delta)
        VALUES (?, ?, ?)
    """, (
        'upgrade_batch',
        json.dumps({
            'total_applied': total_applied,
            'categories': {cat: len(items) for cat, items in apply_cats.items()},
            'level_ups': level_ups,
            'skill_deltas': all_skill_deltas,
        }),
        total_xp_earned,
    ))

    db.commit()
    save_evolution_state(state)

    return {
        'total_applied': total_applied,
        'total_xp_earned': total_xp_earned,
        'old_level': old_level,
        'new_level': state.get('level', 1),
        'level_ups': level_ups,
        'old_xp': old_xp,
        'new_xp': state.get('xp', 0),
        'xp_to_next': state.get('xp_to_next', 100),
        'skill_deltas': all_skill_deltas,
        'old_skills': old_skills,
        'new_skills': state.get('skills', {}),
        'applied_items': applied_items,
    }


# ============================================================================
# RPG DISPLAY
# ============================================================================

def display_rpg_results(results: Dict[str, Any]):
    """Show beautiful RPG-style upgrade results."""
    total = results['total_applied']
    xp = results['total_xp_earned']
    old_level = results['old_level']
    new_level = results['new_level']
    level_ups = results.get('level_ups', [])

    print()
    print(f"  {BOLD_C}{'=' * 58}{RESET_C}")
    print(f"  {BOLD_C}{LIME_C}  UPGRADE COMPLETE{RESET_C}")
    print(f"  {BOLD_C}{'=' * 58}{RESET_C}")
    print()

    # XP animation (simple but satisfying)
    print(f"  {BOLD_C}Upgrades applied:{RESET_C} {GREEN_C}{total}{RESET_C}")
    print(f"  {BOLD_C}XP earned:{RESET_C}        {LIME_C}+{xp}{RESET_C}")
    print()

    # Level display
    if level_ups:
        for lvl in level_ups:
            print(f"  {BOLD_C}{YELLOW_C}>>> LEVEL UP! Level {lvl} reached! <<<{RESET_C}")
        print()
    else:
        xp_bar_width = 30
        current_xp = results.get('new_xp', 0)
        xp_needed = results.get('xp_to_next', 100)
        filled = min(xp_bar_width, int(xp_bar_width * current_xp / max(1, xp_needed)))
        empty = xp_bar_width - filled
        bar = f"{LIME_C}{chr(0x2588) * filled}{RESET_C}{DIM_C}{chr(0x2591) * empty}{RESET_C}"
        print(f"  Level {new_level}  [{bar}]  {current_xp}/{xp_needed} XP")
        print()

    # Skill changes
    skill_deltas = results.get('skill_deltas', {})
    if skill_deltas:
        print(f"  {BOLD_C}SKILL IMPROVEMENTS:{RESET_C}")
        old_skills = results.get('old_skills', {})
        new_skills = results.get('new_skills', {})

        for skill, delta in sorted(skill_deltas.items(), key=lambda x: x[1], reverse=True):
            old_val = old_skills.get(skill, 0)
            new_val = new_skills.get(skill, 0)
            arrow_count = min(10, max(1, int(delta)))
            arrows = f"{GREEN_C}{'>' * arrow_count}{RESET_C}"
            print(f"    {skill:20s} {old_val:5.1f} {arrows} {BOLD_C}{new_val:.1f}{RESET_C}  {DIM_C}(+{delta:.1f}){RESET_C}")
        print()


def display_first_time_experience(results: Dict[str, Any]):
    """
    Walk through what the upgrades mean -- first-time user experience
    for each new capability gained.
    """
    applied = results.get('applied_items', [])
    if not applied:
        return

    # Group by category
    by_cat: Dict[str, List[Dict]] = {}
    for item in applied:
        cat = item.get('category', 'knowledge')
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(item)

    print(f"  {BOLD_C}{'- ' * 29}{RESET_C}")
    print(f"  {BOLD_C}{CYAN}WHAT THIS MEANS FOR YOU:{RESET_C}")
    print()

    # Config changes explanation
    if 'config' in by_cat:
        count = len(by_cat['config'])
        print(f"  {YELLOW_C}[CFG]{RESET_C} {count} configuration improvements absorbed")
        print(f"       Your system now knows about signal filtering thresholds,")
        print(f"       source scoring systems, and pre-analysis quality gates.")
        print(f"       {DIM_C}These inform future `weevolve evolve` recommendations.{RESET_C}")
        print()

    # Code improvements explanation
    if 'code' in by_cat:
        count = len(by_cat['code'])
        print(f"  {CYAN}[COD]{RESET_C} {count} code improvement patterns learned")
        print(f"       Shadow tracking, feedback loops, pre-filter classes,")
        print(f"       and signal quality scoring are now in your knowledge base.")
        print(f"       {DIM_C}Use `weevolve recall 'pre-filter'` to reference them.{RESET_C}")
        print()

    # New tools explanation
    if 'tools' in by_cat:
        count = len(by_cat['tools'])
        print(f"  {GREEN_C}[TOL]{RESET_C} {count} new tool patterns catalogued")
        print(f"       Pre-filter modules, quality scoring engines, and")
        print(f"       signal triage systems are now part of your toolkit.")
        print(f"       {DIM_C}Use `weevolve recall 'signal filter'` to find implementations.{RESET_C}")
        print()

    # Knowledge atoms explanation
    if 'knowledge' in by_cat:
        count = len(by_cat['knowledge'])
        print(f"  {MAGENTA}[KNW]{RESET_C} {count} knowledge atoms integrated")
        print(f"       Trading strategies, research approaches, and system")
        print(f"       diagnostics added to your collective intelligence.")
        print(f"       {DIM_C}Use `weevolve recall <topic>` to query any of these.{RESET_C}")
        print()

    # Skill explanations
    skill_deltas = results.get('skill_deltas', {})
    top_improved = sorted(skill_deltas.items(), key=lambda x: x[1], reverse=True)[:3]

    if top_improved:
        print(f"  {BOLD_C}WHY YOUR SKILLS IMPROVED:{RESET_C}")
        skill_reasons = {
            'trading': 'Signal filtering and trade analysis patterns absorbed',
            'strategy': 'Configuration and decision-making frameworks learned',
            'ai_engineering': 'AI system design patterns and module architectures integrated',
            'coding': 'Implementation patterns, classes, and refactoring approaches learned',
            'security': 'Security-focused code improvements and validation patterns absorbed',
            'research': 'Research methodologies and analytical frameworks catalogued',
            'consciousness': 'Meta-learning and self-improvement patterns recognized',
            'growth': 'Scaling patterns and tool integration strategies learned',
        }
        for skill, delta in top_improved:
            reason = skill_reasons.get(skill, 'New patterns and approaches absorbed')
            print(f"    {BOLD_C}{skill}{RESET_C} +{delta:.1f}: {reason}")
        print()

    # Try this section
    print(f"  {BOLD_C}TRY THIS:{RESET_C}")
    print(f"    {DIM_C}${RESET_C} weevolve recall 'signal pre-filter'  {DIM_C}# Find upgrade patterns{RESET_C}")
    print(f"    {DIM_C}${RESET_C} weevolve evolve                      {DIM_C}# Generate new quests from upgrades{RESET_C}")
    print(f"    {DIM_C}${RESET_C} weevolve status                      {DIM_C}# See your new stats{RESET_C}")
    print()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_upgrade(args: List[str]):
    """
    Main upgrade flow:
    1. Scan IMPROVEMENTS backlog for safe items
    2. Categorize and deduplicate
    3. Present synopsis
    4. Wait for approval (unless --auto or --dry-run)
    5. Apply approved items
    6. Show RPG results + first-time experience
    """
    # Parse flags
    auto_mode = '--auto' in args
    dry_run = '--dry-run' in args
    synopsis_only = '--synopsis' in args
    category_filter = None

    if '--category' in args:
        idx = args.index('--category')
        if idx + 1 < len(args):
            category_filter = args[idx + 1]
            if category_filter not in CATEGORY_META:
                print(f"  {RED_C}Unknown category:{RESET_C} {category_filter}")
                print(f"  Valid: {', '.join(CATEGORY_META.keys())}")
                return

    # Step 1: Find and scan backlog
    improvements_path = find_improvements_file()
    if not improvements_path.exists():
        print(f"  {RED_C}Improvements backlog not found:{RESET_C} {improvements_path}")
        print(f"  Run the continuous improver first to generate improvements.")
        return

    print()
    print(f"  {MAGENTA}(*){RESET_C} {BOLD_C}SEED-squared: scanning improvement backlog...{RESET_C}")
    print(f"  {DIM_C}Source: {improvements_path}{RESET_C}")

    categorized = scan_backlog(improvements_path)
    total_raw = sum(len(v) for v in categorized.values())

    if total_raw == 0:
        print(f"  {DIM_C}No safe improvements found in backlog.{RESET_C}")
        return

    print(f"  {GREEN_C}Found {total_raw} safe items{RESET_C}")

    # Step 2: Generate synopsis
    synopsis = generate_synopsis(categorized)

    # Step 3: Display synopsis
    display_synopsis(synopsis)

    # Always write synopsis markdown
    base = Path(os.environ.get('WEEVOLVE_BASE_DIR', os.getcwd()))
    synopsis_path = base / 'BRAIN' / 'INTEL' / 'UPGRADE-SYNOPSIS.md'
    write_synopsis_markdown(synopsis, synopsis_path)

    if synopsis_only or dry_run:
        print(f"  {DIM_C}Dry run complete. No changes applied.{RESET_C}")
        return

    # Step 4: Approval gate
    if not auto_mode:
        print(f"  {BOLD_C}Apply all {synopsis['total_unique']} improvements?{RESET_C}")
        if category_filter:
            cat_count = len(synopsis['deduped'].get(category_filter, []))
            print(f"  {DIM_C}(filtered to {category_filter}: {cat_count} items){RESET_C}")
        print()
        try:
            response = input(f"  {BOLD_C}[Y/n]{RESET_C} > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {DIM_C}Cancelled.{RESET_C}")
            return

        if response and response not in ('y', 'yes'):
            print(f"  {DIM_C}Upgrade cancelled. Review synopsis at: {synopsis_path}{RESET_C}")
            return

    # Step 5: Apply upgrades
    apply_categories = [category_filter] if category_filter else None

    print()
    print(f"  {LIME_C}Applying upgrades...{RESET_C}")

    results = apply_upgrades(synopsis, categories=apply_categories)

    if results['total_applied'] == 0:
        print(f"  {DIM_C}All items were already applied. Nothing new to upgrade.{RESET_C}")
        return

    # Step 6: Show RPG results
    display_rpg_results(results)

    # Step 7: First-time user experience
    display_first_time_experience(results)

    # Step 8: Save upgrade log
    upgrade_log_path = DATA_DIR / 'upgrade_log.jsonl'
    try:
        with open(upgrade_log_path, 'a') as f:
            f.write(json.dumps({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_applied': results['total_applied'],
                'total_xp': results['total_xp_earned'],
                'old_level': results['old_level'],
                'new_level': results['new_level'],
                'skill_deltas': results['skill_deltas'],
                'categories': {
                    cat: len([i for i in results['applied_items'] if i['category'] == cat])
                    for cat in CATEGORY_META
                },
            }) + '\n')
    except Exception:
        pass  # Non-fatal

    print(f"  {BOLD_C}{GREEN_C}SEED-squared cycle complete.{RESET_C}")
    print(f"  {DIM_C}The system that evolves itself just evolved.{RESET_C}")
    print()
