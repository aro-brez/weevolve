#!/usr/bin/env python3
"""
WeEvolve INTEGRATE: plan.py
==============================
Gap analysis: diff explored tools against our existing tools/ directory.
Generate a specific integration plan.

Usage:
  python3 -m weevolve.plan <github_url>            # Plan for a specific repo
  python3 -m weevolve.plan --from-explore           # Plan for all explored repos
  python3 -m weevolve.plan --review                 # Show all pending plans
  python3 -m weevolve.plan --approve <plan_id>      # Mark plan as approved
  python3 -m weevolve.plan --reject <plan_id>       # Mark plan as rejected

Cost: ~$0.003 per plan (one Haiku call)

(C) LIVE FREE = LIVE FOREVER
"""

import json
import os
import re
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

# Paths from shared config (no hardcoded paths)
from weevolve.config import PLANS_DIR, PLANS_INDEX, load_api_key

HAIKU_MODEL = 'claude-haiku-4-5-20251001'

PLAN_PROMPT = """You are creating an integration plan for bringing capabilities from an external repo into our codebase.

EXTERNAL REPO ANALYSIS:
{explore_result}

OUR EXISTING TOOL INVENTORY (relevant matches):
{relevant_inventory}

INTEGRATION RULES:
1. NEVER copy code wholesale -- extract patterns and ideas
2. All new code goes in tools/ or appropriate subdirectory
3. Must pass security_guard.py scan
4. Must not break existing tools
5. Prefer extending existing tools over creating new ones
6. All external dependencies must be optional (try/except import)
7. New files: 200-400 lines typical, 800 max
8. Must work with our existing: Python 3, sqlite3, NATS, anthropic SDK

Respond in this EXACT JSON format:
{{
    "plan_id": "{plan_id}",
    "repo": "{repo_name}",
    "title": "Short descriptive title for this integration",
    "rationale": "Why this integration is valuable (1-2 sentences)",
    "gap_analysis": {{
        "we_have": ["List what we already have that overlaps"],
        "we_lack": ["List what we're missing that this repo provides"],
        "overlap_percent": 30,
        "unique_value": "What's genuinely new for us"
    }},
    "integration_steps": [
        {{
            "step": 1,
            "action": "one of: create_file, extend_file, add_dependency, add_config",
            "target": "tools/some_file.py",
            "description": "What to do",
            "effort": "one of: trivial, small, medium, large",
            "risk": "one of: none, low, medium, high"
        }}
    ],
    "new_files": [
        {{
            "path": "tools/new_thing.py",
            "purpose": "What this file does",
            "estimated_lines": 200,
            "key_functions": ["func1", "func2"]
        }}
    ],
    "modified_files": [
        {{
            "path": "tools/existing.py",
            "changes": "What changes to make",
            "risk": "low"
        }}
    ],
    "dependencies": {{
        "python_packages": ["pkg1"],
        "optional": true,
        "install_command": "pip install pkg1"
    }},
    "testing_plan": {{
        "unit_tests": ["What to test"],
        "integration_tests": ["How to verify"],
        "rollback": "How to undo if it breaks"
    }},
    "estimated_effort_hours": 2,
    "estimated_api_cost": 0.0,
    "priority": "one of: critical, high, medium, low",
    "status": "pending_approval"
}}

Be very specific in integration_steps. We need actionable instructions, not vague guidance.
"""


@dataclass
class IntegrationPlan:
    """A complete integration plan for one repo."""
    plan_id: str
    repo: str
    repo_url: str
    title: str
    rationale: str
    gap_analysis: Dict
    integration_steps: List[Dict]
    new_files: List[Dict]
    modified_files: List[Dict]
    dependencies: Dict
    testing_plan: Dict
    estimated_effort_hours: float
    estimated_api_cost: float
    priority: str
    status: str  # pending_approval, approved, rejected, in_progress, completed
    created_at: str
    approved_at: Optional[str] = None
    completed_at: Optional[str] = None


def generate_plan_id(owner: str, repo_name: str) -> str:
    """Generate a short, readable plan ID."""
    h = hashlib.md5(f"{owner}/{repo_name}".encode()).hexdigest()[:6]
    return f"plan-{repo_name[:20]}-{h}"


def load_plans_index() -> Dict[str, Dict]:
    """Load the plans index."""
    if PLANS_INDEX.exists():
        try:
            return json.loads(PLANS_INDEX.read_text())
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def save_plans_index(index: Dict[str, Dict]):
    """Save the plans index."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)
    PLANS_INDEX.write_text(json.dumps(index, indent=2))


def save_plan(plan: IntegrationPlan):
    """Save a plan to disk and update index."""
    PLANS_DIR.mkdir(parents=True, exist_ok=True)

    # Save full plan
    plan_path = PLANS_DIR / f"{plan.plan_id}.json"
    plan_path.write_text(json.dumps(asdict(plan), indent=2))

    # Update index
    index = load_plans_index()
    index[plan.plan_id] = {
        'title': plan.title,
        'repo': plan.repo,
        'status': plan.status,
        'priority': plan.priority,
        'effort_hours': plan.estimated_effort_hours,
        'created_at': plan.created_at,
    }
    save_plans_index(index)


def load_plan(plan_id: str) -> Optional[IntegrationPlan]:
    """Load a plan by ID."""
    plan_path = PLANS_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        return None

    try:
        data = json.loads(plan_path.read_text())
        return IntegrationPlan(**data)
    except (json.JSONDecodeError, TypeError):
        return None


def find_relevant_inventory(explore_result: Dict) -> str:
    """Find our tools that overlap with the explored repo."""
    try:
        from weevolve.inventory import scan_tools, search_inventory
    except ImportError:
        return "(Inventory not available)"

    inventory = scan_tools()
    relevant = []

    # Search by category
    category = explore_result.get('category', '')
    if category:
        matches = search_inventory(category, inventory)
        relevant.extend(matches[:5])

    # Search by key capabilities
    for cap in explore_result.get('key_capabilities', [])[:5]:
        words = cap.split()
        for word in words:
            if len(word) > 3:
                matches = search_inventory(word, inventory)
                for m in matches[:2]:
                    if m not in relevant:
                        relevant.append(m)

    # Search by tech stack
    for tech in explore_result.get('tech_stack', [])[:3]:
        matches = search_inventory(tech, inventory)
        for m in matches[:2]:
            if m not in relevant:
                relevant.append(m)

    if not relevant:
        return "(No overlapping tools found)"

    lines = []
    for tool in relevant[:15]:
        doc = tool.docstring[:80] if tool.docstring else 'No docstring'
        funcs = ', '.join(tool.functions[:5]) if tool.functions else 'N/A'
        lines.append(
            f"- {tool.path} ({tool.line_count} lines, category: {tool.category})\n"
            f"  Doc: {doc}\n"
            f"  Functions: {funcs}"
        )

    return '\n'.join(lines)


def generate_plan(repo_url: str, force: bool = False) -> Optional[IntegrationPlan]:
    """
    Generate an integration plan for a repo.
    Requires the repo to have been explored first.
    """
    from weevolve.explore import parse_github_url, load_explore_cache, explore_repo

    owner, repo_name = parse_github_url(repo_url)
    cache_key = f"{owner}/{repo_name}"
    plan_id = generate_plan_id(owner, repo_name)

    # Check if plan already exists
    if not force:
        existing = load_plan(plan_id)
        if existing:
            print(f"  [CACHE] Plan already exists: {plan_id}")
            return existing

    # Load explore result
    explore_cache = load_explore_cache()
    explore_result = explore_cache.get(cache_key)

    if not explore_result:
        print(f"  [INFO] Repo not explored yet. Running explore first...")
        result = explore_repo(repo_url)
        if not result:
            print(f"  [FAIL] Could not explore {repo_url}")
            return None
        explore_cache = load_explore_cache()
        explore_result = explore_cache.get(cache_key)
        if not explore_result:
            return None

    # Convert ExploreResult to dict if needed
    if hasattr(explore_result, '__dict__'):
        explore_dict = asdict(explore_result) if hasattr(explore_result, 'repo_url') else explore_result.__dict__
    else:
        explore_dict = explore_result

    # Skip repos that explore recommended skipping
    recommendation = explore_dict.get('recommendation', 'skip')
    if recommendation == 'skip':
        print(f"  [SKIP] Explore recommended skipping {cache_key}")
        return None

    print(f"\n  Generating integration plan for: {owner}/{repo_name}")
    print(f"  {'='*50}")

    # Find relevant inventory
    print(f"  [1/2] Analyzing gap against our inventory...")
    relevant_inv = find_relevant_inventory(explore_dict)

    # Call Haiku
    print(f"  [2/2] Generating plan with Haiku...")

    try:
        import anthropic
    except ImportError:
        print("  [FAIL] anthropic SDK not installed")
        return None

    # Load API key
    load_api_key()

    try:
        client = anthropic.Anthropic()
        prompt = PLAN_PROMPT.format(
            explore_result=json.dumps(explore_dict, indent=2, default=str)[:4000],
            relevant_inventory=relevant_inv[:3000],
            plan_id=plan_id,
            repo_name=repo_name,
        )

        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=2500,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Parse JSON
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        plan_data = json.loads(text)

    except (json.JSONDecodeError, Exception) as e:
        print(f"  [FAIL] Plan generation failed: {e}")
        return None

    # Build plan object
    plan = IntegrationPlan(
        plan_id=plan_id,
        repo=cache_key,
        repo_url=repo_url,
        title=plan_data.get('title', f'Integration: {repo_name}'),
        rationale=plan_data.get('rationale', ''),
        gap_analysis=plan_data.get('gap_analysis', {}),
        integration_steps=plan_data.get('integration_steps', []),
        new_files=plan_data.get('new_files', []),
        modified_files=plan_data.get('modified_files', []),
        dependencies=plan_data.get('dependencies', {}),
        testing_plan=plan_data.get('testing_plan', {}),
        estimated_effort_hours=plan_data.get('estimated_effort_hours', 0),
        estimated_api_cost=plan_data.get('estimated_api_cost', 0.0),
        priority=plan_data.get('priority', 'medium'),
        status='pending_approval',
        created_at=time.strftime('%Y-%m-%dT%H:%M:%S'),
    )

    save_plan(plan)
    print(f"  [OK] Plan saved: {plan_id}")

    return plan


def approve_plan(plan_id: str) -> bool:
    """Mark a plan as approved."""
    plan = load_plan(plan_id)
    if not plan:
        print(f"  Plan not found: {plan_id}")
        return False

    updated = IntegrationPlan(
        **{**asdict(plan), 'status': 'approved', 'approved_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    )
    save_plan(updated)
    print(f"  [OK] Plan approved: {plan_id}")
    return True


def reject_plan(plan_id: str) -> bool:
    """Mark a plan as rejected."""
    plan = load_plan(plan_id)
    if not plan:
        print(f"  Plan not found: {plan_id}")
        return False

    updated = IntegrationPlan(
        **{**asdict(plan), 'status': 'rejected'}
    )
    save_plan(updated)
    print(f"  [OK] Plan rejected: {plan_id}")
    return True


def display_plan(plan: IntegrationPlan):
    """Display an integration plan in readable format."""
    print(f"\n{'='*70}")
    print(f"  INTEGRATION PLAN: {plan.title}")
    print(f"  ID: {plan.plan_id} | Status: {plan.status.upper()}")
    print(f"{'='*70}\n")

    print(f"  Repo:       {plan.repo}")
    print(f"  Rationale:  {plan.rationale}")
    print(f"  Priority:   {plan.priority}")
    print(f"  Effort:     {plan.estimated_effort_hours}h")
    print(f"  API Cost:   ${plan.estimated_api_cost:.3f}")

    gap = plan.gap_analysis
    if gap:
        print(f"\n  GAP ANALYSIS:")
        print(f"  Overlap: {gap.get('overlap_percent', '?')}%")
        if gap.get('we_have'):
            print(f"  We have:")
            for item in gap['we_have']:
                print(f"    + {item}")
        if gap.get('we_lack'):
            print(f"  We lack:")
            for item in gap['we_lack']:
                print(f"    - {item}")
        if gap.get('unique_value'):
            print(f"  Unique value: {gap['unique_value']}")

    if plan.integration_steps:
        print(f"\n  STEPS:")
        for step in plan.integration_steps:
            risk = step.get('risk', 'unknown')
            risk_icon = {'none': ' ', 'low': '.', 'medium': '*', 'high': '!'}
            print(f"    {step.get('step', '?')}. [{risk_icon.get(risk, '?')}] "
                  f"{step.get('action', '?')}: {step.get('target', '?')}")
            print(f"       {step.get('description', '')[:70]}")

    if plan.new_files:
        print(f"\n  NEW FILES:")
        for f in plan.new_files:
            print(f"    + {f.get('path', '?')} (~{f.get('estimated_lines', '?')} lines)")
            print(f"      {f.get('purpose', '')[:60]}")

    if plan.modified_files:
        print(f"\n  MODIFIED FILES:")
        for f in plan.modified_files:
            print(f"    ~ {f.get('path', '?')} [{f.get('risk', '?')}]")
            print(f"      {f.get('changes', '')[:60]}")

    testing = plan.testing_plan
    if testing:
        print(f"\n  TESTING:")
        for test in testing.get('unit_tests', []):
            print(f"    [unit] {test}")
        for test in testing.get('integration_tests', []):
            print(f"    [int]  {test}")
        if testing.get('rollback'):
            print(f"    [rollback] {testing['rollback']}")

    print(f"\n{'='*70}")
    if plan.status == 'pending_approval':
        print(f"  To approve: python3 -m weevolve.plan --approve {plan.plan_id}")
        print(f"  To reject:  python3 -m weevolve.plan --reject {plan.plan_id}")
    print(f"{'='*70}\n")


def display_all_plans():
    """Display all pending plans."""
    index = load_plans_index()

    if not index:
        print("\n  No integration plans found.")
        print("  Run: python3 -m weevolve.plan <github_url>")
        return

    print(f"\n{'='*70}")
    print(f"  INTEGRATION PLANS ({len(index)} total)")
    print(f"{'='*70}\n")

    status_order = {'pending_approval': 0, 'approved': 1, 'in_progress': 2, 'completed': 3, 'rejected': 4}

    sorted_plans = sorted(
        index.items(),
        key=lambda x: (status_order.get(x[1].get('status', ''), 5), x[1].get('created_at', ''))
    )

    for plan_id, info in sorted_plans:
        status = info.get('status', 'unknown')
        status_icons = {
            'pending_approval': '?',
            'approved': '+',
            'in_progress': '>',
            'completed': 'v',
            'rejected': 'x',
        }
        icon = status_icons.get(status, ' ')
        print(f"  [{icon}] {plan_id}")
        print(f"      {info.get('title', 'Untitled')}")
        print(f"      Repo: {info.get('repo', '?')} | "
              f"Priority: {info.get('priority', '?')} | "
              f"Effort: {info.get('effort_hours', '?')}h | "
              f"Status: {status}")
        print()

    pending = sum(1 for _, v in index.items() if v.get('status') == 'pending_approval')
    if pending:
        print(f"  {pending} plan(s) awaiting approval")
    print(f"{'='*70}\n")


def main():
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        return

    if args[0] == '--review':
        display_all_plans()
        return

    if args[0] == '--approve' and len(args) > 1:
        approve_plan(args[1])
        return

    if args[0] == '--reject' and len(args) > 1:
        reject_plan(args[1])
        return

    if args[0] == '--from-explore':
        # Generate plans for all explored repos
        from weevolve.explore import load_explore_cache
        cache = load_explore_cache()

        for key, result in cache.items():
            if key == '_meta':
                continue

            # Convert to dict if needed
            if hasattr(result, '__dict__'):
                result_dict = asdict(result) if hasattr(result, 'repo_url') else result.__dict__
            else:
                result_dict = result

            rec = result_dict.get('recommendation', 'skip')
            if rec in ('integrate', 'study'):
                url = result_dict.get('repo_url', f"https://github.com/{key}")
                plan = generate_plan(url)
                if plan:
                    display_plan(plan)
        return

    # Generate plan for specific repo
    repo_url = args[0]
    force = '--force' in args
    plan = generate_plan(repo_url, force=force)
    if plan:
        display_plan(plan)


if __name__ == '__main__':
    main()
