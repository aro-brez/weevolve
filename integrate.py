#!/usr/bin/env python3
"""
WeEvolve INTEGRATE: integrate.py
===================================
Orchestrator: qualify -> explore -> plan -> approve -> execute.
ALWAYS requires human approval before any code execution.

Usage:
  python3 -m weevolve.integrate qualify              # Score atoms for repos
  python3 -m weevolve.integrate explore [url]        # Explore a repo or top qualified
  python3 -m weevolve.integrate plan [url]           # Generate integration plan
  python3 -m weevolve.integrate review               # Review all pending plans
  python3 -m weevolve.integrate execute <plan_id>    # Execute approved plan
  python3 -m weevolve.integrate pipeline             # Full pipeline (qualify -> explore -> plan)
  python3 -m weevolve.integrate status               # Dashboard of all integrations
  python3 -m weevolve.integrate cost                 # Show cost tracking

(C) LIVE FREE = LIVE FOREVER
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

# Paths from shared config (no hardcoded paths)
from weevolve.config import COST_LOG, PLANS_DIR

# Cost tracking
COST_PER_EXPLORE = 0.002   # One Haiku call
COST_PER_PLAN = 0.003      # One Haiku call (longer prompt)
DAILY_BUDGET = 0.25         # $0.25/day max


def load_cost_tracker() -> Dict:
    """Load daily cost tracking."""
    if COST_LOG.exists():
        try:
            return json.loads(COST_LOG.read_text())
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        'daily_totals': {},
        'total_spent': 0.0,
        'total_explores': 0,
        'total_plans': 0,
    }


def save_cost_tracker(tracker: Dict):
    """Save cost tracking."""
    COST_LOG.parent.mkdir(parents=True, exist_ok=True)
    COST_LOG.write_text(json.dumps(tracker, indent=2))


def record_cost(amount: float, operation: str):
    """Record API cost."""
    tracker = load_cost_tracker()
    today = time.strftime('%Y-%m-%d')

    daily = tracker.get('daily_totals', {})
    daily[today] = daily.get(today, 0.0) + amount
    tracker['daily_totals'] = daily
    tracker['total_spent'] = tracker.get('total_spent', 0.0) + amount

    if operation == 'explore':
        tracker['total_explores'] = tracker.get('total_explores', 0) + 1
    elif operation == 'plan':
        tracker['total_plans'] = tracker.get('total_plans', 0) + 1

    save_cost_tracker(tracker)


def check_budget() -> bool:
    """Check if we're within daily budget."""
    tracker = load_cost_tracker()
    today = time.strftime('%Y-%m-%d')
    spent_today = tracker.get('daily_totals', {}).get(today, 0.0)

    if spent_today >= DAILY_BUDGET:
        print(f"  [BUDGET] Daily limit reached: ${spent_today:.3f} / ${DAILY_BUDGET:.3f}")
        return False

    remaining = DAILY_BUDGET - spent_today
    print(f"  [BUDGET] ${spent_today:.3f} spent today, ${remaining:.3f} remaining")
    return True


# =============================================================================
# Pipeline Commands
# =============================================================================

def cmd_qualify(args: List[str]):
    """Run qualification on knowledge atoms."""
    from weevolve.qualify import qualify_atoms, display_qualified

    min_score = 0.3
    limit = 30

    i = 0
    while i < len(args):
        if args[i] == '--min-score' and i + 1 < len(args):
            min_score = float(args[i + 1])
            i += 2
        elif args[i] == '--limit' and i + 1 < len(args):
            limit = int(args[i + 1])
            i += 2
        else:
            i += 1

    atoms = qualify_atoms(min_score=min_score, limit=limit)
    display_qualified(atoms)

    if atoms:
        print(f"  Next step: python3 -m weevolve.integrate explore")


def cmd_explore(args: List[str]):
    """Explore repos (from URL or from qualify results)."""
    if not check_budget():
        return

    from weevolve.explore import explore_repo, display_result

    if args and not args[0].startswith('--'):
        # Explore specific URL
        result = explore_repo(args[0], force='--force' in args)
        if result:
            display_result(result)
            record_cost(COST_PER_EXPLORE, 'explore')
    else:
        # Explore from qualify results
        from weevolve.qualify import qualify_atoms

        limit = 5
        if '--limit' in args:
            idx = args.index('--limit')
            if idx + 1 < len(args):
                limit = int(args[idx + 1])

        qualified = qualify_atoms(min_score=0.4, limit=limit)
        if not qualified:
            print("  No qualified atoms. Run: python3 -m weevolve.integrate qualify")
            return

        explored = 0
        for atom in qualified:
            if not check_budget():
                print(f"  Explored {explored} repos before budget limit.")
                break

            result = explore_repo(atom.primary_url)
            if result:
                display_result(result)
                record_cost(COST_PER_EXPLORE, 'explore')
                explored += 1

        print(f"\n  Explored {explored} repos.")
        print(f"  Next step: python3 -m weevolve.integrate plan")


def cmd_plan(args: List[str]):
    """Generate integration plans."""
    if not check_budget():
        return

    from weevolve.plan import generate_plan, display_plan

    if args and not args[0].startswith('--'):
        # Plan for specific URL
        plan = generate_plan(args[0], force='--force' in args)
        if plan:
            display_plan(plan)
            record_cost(COST_PER_PLAN, 'plan')
    else:
        # Plan from explore results
        from weevolve.explore import load_explore_cache

        cache = load_explore_cache()
        planned = 0

        for key, result in cache.items():
            if key == '_meta':
                continue

            if not check_budget():
                break

            result_dict = asdict(result) if hasattr(result, 'repo_url') else result
            rec = result_dict.get('recommendation', 'skip')

            if rec in ('integrate', 'study'):
                url = result_dict.get('repo_url', f"https://github.com/{key}")
                plan = generate_plan(url)
                if plan:
                    display_plan(plan)
                    record_cost(COST_PER_PLAN, 'plan')
                    planned += 1

        print(f"\n  Generated {planned} plans.")
        print(f"  Next step: python3 -m weevolve.integrate review")


def cmd_review(args: List[str]):
    """Review all pending plans."""
    from weevolve.plan import display_all_plans, load_plan, display_plan

    if args and not args[0].startswith('--'):
        # Show specific plan
        plan = load_plan(args[0])
        if plan:
            display_plan(plan)
        else:
            print(f"  Plan not found: {args[0]}")
    else:
        display_all_plans()


def cmd_execute(args: List[str]):
    """
    Execute an approved integration plan.
    CRITICAL: This only works on plans with status='approved'.
    Human must approve first via: python3 -m weevolve.plan --approve <plan_id>
    """
    if not args:
        print("  Usage: python3 -m weevolve.integrate execute <plan_id>")
        print("  First approve: python3 -m weevolve.plan --approve <plan_id>")
        return

    plan_id = args[0]

    from weevolve.plan import load_plan, save_plan, IntegrationPlan

    plan = load_plan(plan_id)
    if not plan:
        print(f"  [FAIL] Plan not found: {plan_id}")
        return

    # CRITICAL: Only execute approved plans
    if plan.status != 'approved':
        print(f"  [BLOCKED] Plan status is '{plan.status}', must be 'approved'.")
        print(f"  To approve: python3 -m weevolve.plan --approve {plan_id}")
        return

    print(f"\n{'='*70}")
    print(f"  EXECUTING INTEGRATION PLAN")
    print(f"  {plan.title}")
    print(f"  ID: {plan_id}")
    print(f"{'='*70}\n")

    # Confirm execution
    if '--yes' not in args:
        print(f"  This will:")
        for step in plan.integration_steps:
            print(f"    {step.get('step', '?')}. {step.get('action', '?')}: {step.get('target', '?')}")
            print(f"       {step.get('description', '')[:70]}")

        if plan.new_files:
            print(f"\n  New files:")
            for f in plan.new_files:
                print(f"    + {f.get('path', '?')}")

        if plan.modified_files:
            print(f"\n  Modified files:")
            for f in plan.modified_files:
                print(f"    ~ {f.get('path', '?')}")

        print(f"\n  IMPORTANT: This requires a developer (human or Claude Code)")
        print(f"  to implement each step. This orchestrator does NOT auto-generate code.")
        print()
        print(f"  The plan has been saved to:")
        print(f"    {PLANS_DIR / f'{plan_id}.json'}")
        print()
        print(f"  To mark as in-progress:")
        print(f"    python3 -m weevolve.integrate mark-progress {plan_id}")
        print()
        print(f"  To mark as complete:")
        print(f"    python3 -m weevolve.integrate mark-complete {plan_id}")
        return

    # Mark as in-progress
    updated = IntegrationPlan(**{**asdict(plan), 'status': 'in_progress'})
    save_plan(updated)
    print(f"  [OK] Plan marked as in_progress")

    # The actual code writing happens via Claude Code or a developer
    # following the steps in the plan. This is intentionally NOT automated.
    print(f"\n  Implementation steps are in the plan JSON.")
    print(f"  A developer should follow each step and verify.")
    print(f"  When done: python3 -m weevolve.integrate mark-complete {plan_id}")


def cmd_mark_progress(args: List[str]):
    """Mark a plan as in-progress."""
    if not args:
        print("  Usage: python3 -m weevolve.integrate mark-progress <plan_id>")
        return

    from weevolve.plan import load_plan, save_plan, IntegrationPlan

    plan = load_plan(args[0])
    if not plan:
        print(f"  Plan not found: {args[0]}")
        return

    updated = IntegrationPlan(**{**asdict(plan), 'status': 'in_progress'})
    save_plan(updated)
    print(f"  [OK] Plan marked as in_progress: {args[0]}")


def cmd_mark_complete(args: List[str]):
    """Mark a plan as completed."""
    if not args:
        print("  Usage: python3 -m weevolve.integrate mark-complete <plan_id>")
        return

    from weevolve.plan import load_plan, save_plan, IntegrationPlan

    plan = load_plan(args[0])
    if not plan:
        print(f"  Plan not found: {args[0]}")
        return

    updated = IntegrationPlan(
        **{**asdict(plan), 'status': 'completed', 'completed_at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    )
    save_plan(updated)
    print(f"  [OK] Plan marked as completed: {args[0]}")


def cmd_pipeline(args: List[str]):
    """
    Run the full pipeline: qualify -> explore -> plan.
    Stops before execution (requires human approval).
    """
    limit = 3  # Default: process top 3

    if '--limit' in args:
        idx = args.index('--limit')
        if idx + 1 < len(args):
            limit = int(args[idx + 1])

    print(f"\n{'='*70}")
    print(f"  WeEvolve INTEGRATE PIPELINE")
    print(f"  qualify -> explore -> plan -> [HUMAN APPROVAL] -> execute")
    print(f"{'='*70}\n")

    if not check_budget():
        return

    # Phase 1: Qualify
    print(f"\n  PHASE 1: QUALIFY")
    print(f"  {'-'*40}")
    from weevolve.qualify import qualify_atoms, display_qualified

    qualified = qualify_atoms(min_score=0.4, limit=limit * 2)
    display_qualified(qualified[:limit])

    if not qualified:
        print("  No qualified atoms found.")
        return

    # Phase 2: Explore
    print(f"\n  PHASE 2: EXPLORE")
    print(f"  {'-'*40}")
    from weevolve.explore import explore_repo, display_result

    explored = []
    for atom in qualified[:limit]:
        if not check_budget():
            break

        result = explore_repo(atom.primary_url)
        if result:
            display_result(result)
            record_cost(COST_PER_EXPLORE, 'explore')

            # Only plan for integrate/study recommendations
            result_dict = asdict(result) if hasattr(result, 'repo_url') else result
            rec = result_dict.get('recommendation', 'skip')
            if rec in ('integrate', 'study'):
                explored.append(atom)

    if not explored:
        print("  No repos recommended for integration.")
        return

    # Phase 3: Plan
    print(f"\n  PHASE 3: PLAN")
    print(f"  {'-'*40}")
    from weevolve.plan import generate_plan, display_plan

    plans_created = 0
    for atom in explored:
        if not check_budget():
            break

        plan = generate_plan(atom.primary_url)
        if plan:
            display_plan(plan)
            record_cost(COST_PER_PLAN, 'plan')
            plans_created += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Qualified:  {len(qualified)} atoms")
    print(f"  Explored:   {len(explored)} repos")
    print(f"  Plans:      {plans_created} integration plans")
    print()
    print(f"  NEXT STEPS:")
    print(f"  1. Review plans: python3 -m weevolve.integrate review")
    print(f"  2. Approve:      python3 -m weevolve.plan --approve <plan_id>")
    print(f"  3. Execute:      python3 -m weevolve.integrate execute <plan_id>")
    print(f"{'='*70}\n")


def cmd_status(args: List[str]):
    """Show integration status dashboard."""
    from weevolve.plan import load_plans_index
    from weevolve.explore import load_explore_cache

    tracker = load_cost_tracker()
    plans_index = load_plans_index()
    explore_cache = load_explore_cache()

    # Count by status
    status_counts = {}
    for _, info in plans_index.items():
        status = info.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1

    today = time.strftime('%Y-%m-%d')
    spent_today = tracker.get('daily_totals', {}).get(today, 0.0)

    explored_count = sum(1 for k in explore_cache if k != '_meta')

    print(f"\n{'='*70}")
    print(f"  WeEvolve INTEGRATE STATUS")
    print(f"{'='*70}\n")

    print(f"  PIPELINE:")
    print(f"    Explored repos:    {explored_count}")
    print(f"    Integration plans: {len(plans_index)}")
    for status, count in sorted(status_counts.items()):
        print(f"      {status:20s} {count}")

    print(f"\n  COST:")
    print(f"    Today:     ${spent_today:.3f} / ${DAILY_BUDGET:.3f}")
    print(f"    Total:     ${tracker.get('total_spent', 0):.3f}")
    print(f"    Explores:  {tracker.get('total_explores', 0)}")
    print(f"    Plans:     {tracker.get('total_plans', 0)}")

    print(f"\n  COMMANDS:")
    print(f"    qualify     Score atoms for repos")
    print(f"    explore     Clone + scan + analyze repos")
    print(f"    plan        Generate integration plans")
    print(f"    review      View all plans")
    print(f"    execute     Execute approved plans")
    print(f"    pipeline    Run full pipeline (qualify->explore->plan)")
    print(f"    cost        Detailed cost breakdown")
    print(f"\n{'='*70}\n")


def cmd_cost(args: List[str]):
    """Show detailed cost tracking."""
    tracker = load_cost_tracker()

    print(f"\n{'='*70}")
    print(f"  WeEvolve INTEGRATE COST TRACKER")
    print(f"{'='*70}\n")

    print(f"  Daily Budget:   ${DAILY_BUDGET:.3f}")
    print(f"  Total Spent:    ${tracker.get('total_spent', 0):.3f}")
    print(f"  Total Explores: {tracker.get('total_explores', 0)}")
    print(f"  Total Plans:    {tracker.get('total_plans', 0)}")

    daily = tracker.get('daily_totals', {})
    if daily:
        print(f"\n  Daily Breakdown:")
        for date in sorted(daily.keys(), reverse=True)[:10]:
            amount = daily[date]
            bar = '#' * int(amount / DAILY_BUDGET * 30)
            print(f"    {date}: ${amount:.3f} [{bar}]")

    print(f"\n  Cost per operation:")
    print(f"    Explore: ~${COST_PER_EXPLORE:.3f}")
    print(f"    Plan:    ~${COST_PER_PLAN:.3f}")
    print(f"    Qualify: $0.000 (zero cost)")
    print(f"    Review:  $0.000 (zero cost)")
    print(f"\n{'='*70}\n")


# =============================================================================
# __main__ entry point
# =============================================================================

COMMANDS = {
    'qualify': cmd_qualify,
    'explore': cmd_explore,
    'plan': cmd_plan,
    'review': cmd_review,
    'execute': cmd_execute,
    'pipeline': cmd_pipeline,
    'status': cmd_status,
    'cost': cmd_cost,
    'mark-progress': cmd_mark_progress,
    'mark-complete': cmd_mark_complete,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('--help', '-h', 'help'):
        print(__doc__)
        return

    cmd = sys.argv[1].lower()
    remaining_args = sys.argv[2:]

    if cmd in COMMANDS:
        COMMANDS[cmd](remaining_args)
    else:
        print(f"  Unknown command: {cmd}")
        print(f"  Available: {', '.join(COMMANDS.keys())}")
        print()
        print(__doc__)


if __name__ == '__main__':
    main()
