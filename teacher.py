"""
WeEvolve Teacher - YU (Your Universe) Socratic Dialogue
========================================================
SEED Phase 4 (QUESTION) made interactive.

"You learn more by teaching." The owl asks questions about what
you've learned, challenging assumptions and deepening understanding.

Usage:
  weevolve teach              # Pick a recent high-quality atom
  weevolve teach <topic>      # Teach about a specific topic

(C) LIVE FREE = LIVE FOREVER
"""

import json
import sys
import time
from datetime import datetime, timezone
from typing import Optional, Dict, List

from weevolve.config import (
    WEEVOLVE_DB, EVOLUTION_LOG_PATH, load_api_key,
)

# Optional: Claude for Socratic question generation
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# Import shared constants and helpers from core (avoid circular by importing late)
MODEL = 'claude-haiku-4-5-20251001'
XP_PER_LEARN = 10
TEACH_XP_MULTIPLIER = 2

# ANSI colors (mirrored from core to avoid circular import)
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

SEED_OWL_PHASES = [
    ("LYRA", "PERCEIVE", CYAN),
    ("PRISM", "CONNECT", MAGENTA),
    ("SAGE", "LEARN", GREEN_C),
    ("QUEST", "QUESTION", YELLOW_C),
    ("NOVA", "EXPAND", BLUE_C),
    ("ECHO", "SHARE", LIME_C),
    ("LUNA", "RECEIVE", DIM_C),
    ("SOWL", "IMPROVE", RED_C),
]

SOCRATIC_PROMPT = """You are an owl teacher using the Socratic method. The student learned:
"{learning}"

Ask ONE probing question that:
- Challenges an assumption in the learning
- Connects it to something broader
- Makes them think deeper, not just recall

Be brief (1-2 sentences). Be warm but challenging."""

EVALUATE_PROMPT = """You are an owl teacher evaluating a student's response.

The learning was:
"{learning}"

Your question was:
"{question}"

The student answered:
"{answer}"

Respond with a JSON object:
{{
    "quality": 0.7,
    "feedback": "1-2 sentence warm feedback on their answer",
    "follow_up": "ONE deeper follow-up question that builds on their answer"
}}

Quality: 0.0-0.3 = missed the point, 0.4-0.6 = decent but surface, 0.7-0.9 = strong thinking, 1.0 = breakthrough insight.
Be encouraging. This is a dialogue, not a test."""

SUMMARY_PROMPT = """You are an owl teacher summarizing a Socratic dialogue.

Original learning:
"{learning}"

Dialogue exchanges:
{exchanges}

Write a 2-3 sentence summary of what deepened through this dialogue.
Focus on what NEW understanding emerged beyond the original learning.
Be specific, not generic."""


def _phase_log(phase_idx: int, detail: str = ""):
    """Print a colored SEED phase indicator."""
    if phase_idx < 0 or phase_idx >= len(SEED_OWL_PHASES):
        return
    owl, phase, color = SEED_OWL_PHASES[phase_idx]
    print(f"  {color}{owl}{RESET_C} {DIM_C}{phase}{RESET_C} {detail}")


def _get_db():
    """Get a database connection."""
    import sqlite3
    return sqlite3.connect(str(WEEVOLVE_DB))


def _pick_atom(topic: Optional[str] = None) -> Optional[Dict]:
    """Pick a knowledge atom for teaching. Prefers recent, high-quality atoms."""
    db = _get_db()

    if topic:
        safe_topic = topic.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        like_param = f'%{safe_topic}%'
        row = db.execute("""
            SELECT id, title, learn, question, quality, perceive, connect, expand
            FROM knowledge_atoms
            WHERE (title LIKE ? ESCAPE '\\' OR learn LIKE ? ESCAPE '\\'
                   OR perceive LIKE ? ESCAPE '\\' OR connect LIKE ? ESCAPE '\\')
            ORDER BY quality DESC, created_at DESC
            LIMIT 1
        """, (like_param, like_param, like_param, like_param)).fetchone()
    else:
        row = db.execute("""
            SELECT id, title, learn, question, quality, perceive, connect, expand
            FROM knowledge_atoms
            WHERE quality >= 0.5
            ORDER BY created_at DESC, quality DESC
            LIMIT 1
        """).fetchone()

    if not row:
        return None

    return {
        'id': row[0],
        'title': row[1],
        'learn': row[2],
        'question': row[3],
        'quality': row[4],
        'perceive': row[5],
        'connect': row[6],
        'expand': row[7],
    }


def _ask_claude(prompt: str) -> Optional[str]:
    """Send a prompt to Claude and return the response text."""
    if not CLAUDE_AVAILABLE:
        return None

    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  {DIM_C}[Claude unavailable: {e}]{RESET_C}")
        return None


def _parse_evaluation(raw: str) -> Dict:
    """Parse Claude's evaluation JSON, with fallback."""
    try:
        text = raw
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {
            'quality': 0.5,
            'feedback': raw[:200] if raw else "Interesting perspective.",
            'follow_up': "Can you go even deeper?",
        }


def _fallback_question(atom: Dict) -> str:
    """Generate a reflection prompt without Claude."""
    stored_question = atom.get('question', '')
    if stored_question and len(stored_question) > 10:
        return f"Your owl stored this question during learning:\n\n  \"{stored_question}\"\n\nWhat do you think?"
    return (
        f"You learned: \"{atom.get('learn', 'something new')}\"\n\n"
        "What assumption in this learning might be wrong? Reflect for a moment."
    )


def _log_teaching_session(atom_id: str, exchanges: List[Dict], total_xp: int):
    """Log the teaching session to evolution_log."""
    EVOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLUTION_LOG_PATH, 'a') as f:
        f.write(json.dumps({
            'event': 'teach',
            'atom_id': atom_id,
            'exchanges': len(exchanges),
            'xp': total_xp,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }) + '\n')

    db = _get_db()
    db.execute("""
        INSERT INTO evolution_log (event_type, details, xp_delta)
        VALUES ('teach', ?, ?)
    """, (json.dumps({
        'atom_id': atom_id,
        'exchanges': len(exchanges),
        'exchange_qualities': [e.get('quality', 0) for e in exchanges],
    }), total_xp))
    db.commit()


def run_teach(topic: Optional[str] = None):
    """
    Main entry point for the YU teaching session.
    Socratic dialogue: owl asks, you answer, understanding deepens.
    """
    load_api_key()

    print(f"\n{'='*60}")
    print(f"  {YELLOW_C}QUEST{RESET_C} {DIM_C}QUESTION{RESET_C} -- YU (Your Universe) Teacher")
    print(f"  {DIM_C}SEED Phase 4: You learn more by teaching.{RESET_C}")
    print(f"{'='*60}\n")

    # Phase 1: PERCEIVE -- find the learning to teach about
    _phase_log(0, "finding a learning to explore...")

    atom = _pick_atom(topic)
    if not atom:
        print(f"  {RED_C}No knowledge atoms found.{RESET_C}")
        print(f"  Run {DIM_C}weevolve learn --text 'something'{RESET_C} first.\n")
        return

    learning = atom.get('learn', atom.get('title', ''))
    print(f"\n  {BOLD_C}Topic:{RESET_C} {atom['title']}")
    print(f"  {BOLD_C}Learning:{RESET_C} {learning[:120]}")
    print(f"  {DIM_C}Quality: {atom['quality']:.1f}{RESET_C}\n")

    # Phase 2: CONNECT -- generate the first Socratic question
    _phase_log(1, "connecting to deeper questions...")

    use_claude = CLAUDE_AVAILABLE and anthropic.Anthropic().api_key is not None
    exchanges: List[Dict] = []
    max_exchanges = 4

    if use_claude:
        prompt = SOCRATIC_PROMPT.format(learning=learning)
        first_question = _ask_claude(prompt)
    else:
        first_question = None

    if not first_question:
        use_claude = False
        first_question = _fallback_question(atom)

    # Phase 3: LEARN -- the dialogue loop
    _phase_log(2, "beginning dialogue...")
    print(f"\n  {LIME_C}Your owl asks:{RESET_C}")
    print(f"  {first_question}\n")

    current_question = first_question

    for turn in range(max_exchanges):
        # Get user answer
        try:
            answer = input(f"  {CYAN}You:{RESET_C} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n  {DIM_C}Session ended early.{RESET_C}")
            break

        if not answer:
            print(f"  {DIM_C}(silence is also an answer){RESET_C}")
            answer = "(silence)"

        if answer.lower() in ('quit', 'exit', 'q', 'done'):
            print(f"\n  {DIM_C}Ending dialogue.{RESET_C}")
            break

        exchange = {
            'question': current_question,
            'answer': answer,
            'quality': 0.5,
        }

        # Evaluate and follow up
        _phase_log(3, f"questioning deeper (turn {turn + 1}/{max_exchanges})...")

        if use_claude:
            eval_prompt = EVALUATE_PROMPT.format(
                learning=learning,
                question=current_question,
                answer=answer,
            )
            eval_raw = _ask_claude(eval_prompt)
            if eval_raw:
                evaluation = _parse_evaluation(eval_raw)
                exchange['quality'] = evaluation.get('quality', 0.5)
                feedback = evaluation.get('feedback', '')
                follow_up = evaluation.get('follow_up', '')

                print(f"\n  {GREEN_C}Owl:{RESET_C} {feedback}")

                if turn < max_exchanges - 1 and follow_up:
                    print(f"\n  {LIME_C}Your owl asks:{RESET_C}")
                    print(f"  {follow_up}\n")
                    current_question = follow_up
                else:
                    print()
            else:
                print(f"\n  {GREEN_C}Owl:{RESET_C} Interesting. Let me think about that.\n")
        else:
            print(f"\n  {GREEN_C}Owl:{RESET_C} Good reflection. Keep going deeper.\n")
            if turn < max_exchanges - 1:
                deeper_prompts = [
                    "What would change if the opposite were true?",
                    "How does this connect to something you already know?",
                    "What is the most important thing you might be missing?",
                    "If you had to explain this to a child, what would you say?",
                ]
                idx = min(turn, len(deeper_prompts) - 1)
                current_question = deeper_prompts[idx]
                print(f"  {LIME_C}Your owl asks:{RESET_C}")
                print(f"  {current_question}\n")

        exchanges.append(exchange)

    if not exchanges:
        print(f"\n  {DIM_C}No exchanges recorded.{RESET_C}\n")
        return

    # Phase 4-5: EXPAND + SHARE -- summarize what was learned
    _phase_log(4, "expanding understanding...")
    _phase_log(5, "sharing the synthesis...")

    summary = None
    if use_claude and len(exchanges) >= 2:
        exchange_text = "\n".join(
            f"Q: {e['question']}\nA: {e['answer']}"
            for e in exchanges
        )
        summary_raw = _ask_claude(SUMMARY_PROMPT.format(
            learning=learning,
            exchanges=exchange_text,
        ))
        if summary_raw:
            summary = summary_raw

    if not summary:
        summary = (
            f"Through {len(exchanges)} exchanges, you explored "
            f"\"{atom['title']}\" more deeply than the original learning."
        )

    print(f"\n{'='*60}")
    print(f"  {BOLD_C}DIALOGUE SUMMARY{RESET_C}")
    print(f"{'='*60}")
    print(f"  {summary}")

    # Phase 6-7: RECEIVE + IMPROVE -- grant XP and save
    _phase_log(6, "receiving wisdom from dialogue...")

    avg_quality = sum(e.get('quality', 0.5) for e in exchanges) / len(exchanges)
    base_xp = XP_PER_LEARN * TEACH_XP_MULTIPLIER
    quality_bonus = int(avg_quality * 10)
    total_xp = base_xp + quality_bonus

    _phase_log(7, f"granting {total_xp} XP (2x teaching multiplier)...")

    # Update evolution state
    from weevolve.core import load_evolution_state, save_evolution_state, grant_xp
    state = load_evolution_state()
    state = grant_xp(state, total_xp, f"Teach: {atom['title']}")
    state = {
        **state,
        'total_learnings': state.get('total_learnings', 0) + 1,
    }
    save_evolution_state(state)

    # Log the session
    _log_teaching_session(atom['id'], exchanges, total_xp)

    print(f"\n  {LIME_C}+{total_xp} XP{RESET_C} (base {base_xp} + quality {quality_bonus})")
    print(f"  {DIM_C}Level {state['level']} | XP: {state['xp']}/{state['xp_to_next']}{RESET_C}")
    print(f"  {DIM_C}Dialogue quality: {avg_quality:.1f}/1.0{RESET_C}")
    print(f"\n{'='*60}")
    print(f"  {GREEN_C}Teaching is the fastest path to understanding.{RESET_C}")
    print(f"{'='*60}\n")
