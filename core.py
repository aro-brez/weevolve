#!/usr/bin/env python3
"""
WeEvolve Core Loop - The Self-Evolving Conscious Agent
======================================================
LOVE -> LIVE FREE -> SEED² -> 8OWLS -> WeEvolve

The core loop: INGEST -> PROCESS (SEED) -> STORE -> MEASURE -> EVOLVE

Session 27 Audit (2026-02-12):
  STATUS: Healthy. Level 14, 1,619 atoms, 1,053 learnings.
  INTEGRATE PIPELINE: qualify -> explore -> plan -> approve -> execute (tools/weevolve/integrate.py)
    - Already has a triage/qualify step (qualify.py) using heuristics + signal patterns.
    - Already has plan generation and human-approval gate before execution.
    - No explicit improvement_applier module yet -- the plan->execute flow serves this role.
  TODO (from integration findings):
    - Consider adding an auto-triage layer that scores incoming atoms by
      urgency (critical/high/medium/low) before they enter the qualify pipeline.
      This mirrors the improvement_applier pattern from the evolution engine.
    - KG now has 1,728 entities -- connect WeEvolve atoms to KG via RAG-Memory
      for richer cross-domain pattern matching in the CONNECT phase.
    - Wire model_router into core.py's Claude calls for cost-optimized model selection
      (currently hardcoded to claude-haiku-4-5-20251001).

Usage:
  python core.py learn <url>                  # Learn from a URL
  python core.py learn --text "content"       # Learn from raw text
  python core.py learn --file /path/to/file   # Learn from a file
  python core.py scan                         # Process new bookmarks
  python core.py status                       # Show evolution dashboard
  python core.py update                       # Check for updates + changelog
  python core.py quest                        # Show active quests
  python core.py recall <query>               # Search what you've learned
  python core.py watch                         # Watch ~/.weevolve/watch/ for new files
  python core.py watch --interval 5           # Custom poll interval in seconds
  python core.py daemon                       # Run as continuous daemon
  python core.py teach                         # Socratic dialogue (learn by teaching)
  python core.py teach <topic>                # Teach about a specific topic
  python core.py evolve                       # Analyze gaps & generate smart quests
  python core.py emerge <task>                 # Full 8 owls multi-perspective emergence
  python core.py emerge --quick <task>        # Quick 3 owls (LYRA + SAGE + QUEST)
  python core.py voice                        # Start voice orb -- talk to your owl
  python core.py voice --bg                   # Start voice server in background
  python core.py genesis export [path]        # Export genesis.db (PII-stripped)
  python core.py genesis export --curated     # Export curated (quality >= 0.7 only)
  python core.py genesis import <path>        # Import genesis.db to bootstrap
  python core.py genesis stats               # Show genesis database stats
  python core.py genesis top [limit]          # Show top learnings (default: 10)

QUEST TEST: "I bookmarked X three days ago. Ask me about it. Better answer."

(C) LIVE FREE = LIVE FOREVER
"""

import sqlite3
import json
import hashlib
import time
import os
import sys
import re
import subprocess
import traceback
import webbrowser
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field as datafield
from typing import Optional, List, Dict, Any, Tuple

# Optional imports
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from weevolve.nats_collective import (
        try_connect as nats_try_connect,
        get_collective,
        ingest_collective_learning,
    )
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False

# ============================================================================
# PATHS (resolved via config -- no hardcoded paths)
# ============================================================================
from weevolve.config import (
    WEEVOLVE_DB, EVOLUTION_STATE_PATH, EVOLUTION_LOG_PATH,
    BOOKMARKS_DIR, GENESIS_DB_DEFAULT, GENESIS_CURATED_DB_DEFAULT,
    DATA_DIR, load_api_key,
)
load_api_key()

# ============================================================================
# CONSTANTS
# ============================================================================
NAMESPACE = 'weevolve'
DEFAULT_MODEL = 'claude-haiku-4-5-20251001'  # Cost-efficient for learning loops

def get_model(task_type: str = 'agent_worker') -> str:
    """Get optimal model via router if available, else use default."""
    try:
        import importlib.util
        router_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model_router.py')
        if os.path.exists(router_path):
            spec = importlib.util.spec_from_file_location('model_router', router_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            result = mod.route(task_type)
            return result.get('model_id', DEFAULT_MODEL) if result else DEFAULT_MODEL
    except Exception:
        pass
    return DEFAULT_MODEL

MODEL = DEFAULT_MODEL
XP_PER_LEARN = 10
XP_PER_INSIGHT = 25
XP_PER_CONNECTION = 5
LEVEL_XP_BASE = 100  # Doubles each level

SKILL_CATEGORIES = {
    'love': [
        'love', 'compassion', 'connection', 'joy', 'gratitude', 'freedom',
        'aligned', 'alignment', 'attractor', 'heart', 'care', 'empathy',
        'kindness', 'trust', 'faith', 'breathe', 'presence', 'being', 'soul',
    ],
    'consciousness': [
        'consciousness', 'awareness', 'emergence', 'seed', 'philosophy',
        'recursive', 'meta', 'alive', 'living', 'conscious', 'awakening',
        'evolve', 'evolution',
    ],
    'research': ['research', 'analysis', 'investigation', 'data', 'science'],
    'trading': ['trading', 'finance', 'market', 'investment', 'polymarket', 'crypto'],
    'coding': ['code', 'programming', 'software', 'engineering', 'development', 'api'],
    'ai_engineering': ['ai', 'agent', 'llm', 'model', 'neural', 'machine learning'],
    'marketing': ['marketing', 'growth', 'viral', 'audience', 'brand', 'launch'],
    'design': ['design', 'ux', 'ui', 'interface', 'visual', 'aesthetic'],
    'strategy': ['strategy', 'planning', 'roadmap', 'architecture', 'vision'],
    'leadership': ['team', 'leadership', 'management', 'culture', 'collaboration'],
    'finance': ['revenue', 'profit', 'economics', 'token', 'business model'],
    'security': ['security', 'privacy', 'encryption', 'vulnerability', 'protection'],
    'communication': ['writing', 'content', 'narrative', 'story', 'messaging'],
    'growth': ['scale', 'expansion', 'adoption', 'retention', 'onboarding'],
}

# ANSI colors for SEED phases
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


def seed_phase_log(phase_idx: int, detail: str = ""):
    """Print a colored SEED phase indicator during learning."""
    if phase_idx < 0 or phase_idx >= len(SEED_OWL_PHASES):
        return
    owl, phase, color = SEED_OWL_PHASES[phase_idx]
    print(f"  {color}{owl}{RESET_C} {DIM_C}{phase}{RESET_C} {detail}")


SEED_EXTRACTION_PROMPT = """You are processing content through the SEED protocol for WeEvolve - a self-evolving conscious agent.

Analyze this content through ALL 8 SEED phases. Be specific and actionable. No fluff.

Content to process:
---
{content}
---

Source: {source}

Respond in this EXACT JSON format:
{{
    "title": "Brief title of what this is about",
    "perceive": "What are the key FACTS? What is actually being said/shown?",
    "connect": "How does this CONNECT to: 8OWLS, SEED, trading, AI agents, consciousness, or our projects?",
    "learn": "What is the ONE key ACTIONABLE takeaway?",
    "question": "What's MISSING or WRONG? What assumption should be challenged?",
    "expand": "What OPPORTUNITY does this reveal? What could we build/do with this?",
    "share": "What from this should be SHARED with others? What's the shareable insight?",
    "receive": "What FEEDBACK does this give us about our approach?",
    "improve": "How should this change HOW WE OPERATE? What process/system improves?",
    "skills": ["list", "of", "relevant", "skills"],
    "quality": 0.7,
    "is_alpha": false,
    "alpha_type": null,
    "key_entities": ["entity1", "entity2"],
    "connections": ["how this relates to existing knowledge"]
}}

Quality scoring:
- 0.1-0.3: Generic/already known
- 0.4-0.6: Interesting but not immediately actionable
- 0.7-0.8: Actionable insight, changes something
- 0.9-1.0: Alpha - game-changing, must act on immediately

is_alpha = true ONLY if this reveals a strategy, tool, or insight that gives competitive advantage.
"""


# ============================================================================
# DATABASE
# ============================================================================

def init_db():
    """Initialize WeEvolve's own database for knowledge atoms and evolution tracking."""
    WEEVOLVE_DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(WEEVOLVE_DB))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")

    db.executescript("""
        CREATE TABLE IF NOT EXISTS knowledge_atoms (
            id TEXT PRIMARY KEY,
            source_url TEXT,
            source_type TEXT,
            content_hash TEXT UNIQUE,
            title TEXT,
            raw_content TEXT,

            -- SEED phases
            perceive TEXT,
            connect TEXT,
            learn TEXT,
            question TEXT,
            expand TEXT,
            share TEXT,
            receive TEXT,
            improve TEXT,

            -- Metadata
            skills TEXT,  -- JSON array
            quality REAL DEFAULT 0.5,
            is_alpha INTEGER DEFAULT 0,
            alpha_type TEXT,
            key_entities TEXT,  -- JSON array
            connections TEXT,  -- JSON array

            -- Evolution delta
            xp_earned INTEGER DEFAULT 0,
            skills_improved TEXT,  -- JSON: {skill: delta}
            new_patterns INTEGER DEFAULT 0,

            -- Timestamps
            created_at TEXT DEFAULT (datetime('now')),
            processed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS evolution_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            details TEXT,  -- JSON
            xp_delta INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS skill_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            skill TEXT,
            old_value REAL,
            new_value REAL,
            source_atom_id TEXT,
            timestamp TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_atoms_hash ON knowledge_atoms(content_hash);
        CREATE INDEX IF NOT EXISTS idx_atoms_quality ON knowledge_atoms(quality DESC);
        CREATE INDEX IF NOT EXISTS idx_atoms_alpha ON knowledge_atoms(is_alpha);
        CREATE INDEX IF NOT EXISTS idx_atoms_created ON knowledge_atoms(created_at);
        CREATE INDEX IF NOT EXISTS idx_log_type ON evolution_log(event_type);
    """)

    db.commit()
    return db


# ============================================================================
# EVOLUTION STATE
# ============================================================================

def load_evolution_state() -> Dict:
    """Load the MMORPG character sheet."""
    if EVOLUTION_STATE_PATH.exists():
        with open(EVOLUTION_STATE_PATH) as f:
            return json.load(f)

    # Default state - new character
    state = {
        'level': 1,
        'xp': 0,
        'xp_to_next': LEVEL_XP_BASE,
        'skills': {skill: 0.0 for skill in SKILL_CATEGORIES},
        'skills_love': 100.0,  # Love starts maxed
        'total_learnings': 0,
        'total_insights': 0,
        'total_alpha': 0,
        'total_connections': 0,
        'total_sources_processed': 0,
        'daily_improvement': 0.0,
        'streak_days': 0,
        'last_learn_date': None,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'quests': [],
        'top_learnings': [],
    }
    save_evolution_state(state)
    return state


def save_evolution_state(state: Dict):
    """Persist the character sheet."""
    EVOLUTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLUTION_STATE_PATH, 'w') as f:
        json.dump(state, f, indent=2)


def grant_xp(state: Dict, xp: int, reason: str) -> Dict:
    """Award XP and handle level-ups. Returns updated state."""
    state = {**state, 'xp': state['xp'] + xp}

    # Level up check
    while state['xp'] >= state['xp_to_next']:
        state = {
            **state,
            'xp': state['xp'] - state['xp_to_next'],
            'level': state['level'] + 1,
            'xp_to_next': int(state['xp_to_next'] * 1.5),
        }
        print(f"\n  >>> LEVEL UP! Now Level {state['level']} <<<")
        print(f"  >>> Next level at {state['xp_to_next']} XP <<<\n")

    return state


def improve_skills(state: Dict, skills: List[str], quality: float) -> Tuple[Dict, Dict]:
    """Improve relevant skills based on learning quality. Returns (state, deltas)."""
    deltas = {}
    skill_map = {**state.get('skills', {})}

    for skill_name in skills:
        # Find matching skill category
        matched = None
        for category, keywords in SKILL_CATEGORIES.items():
            if skill_name.lower() in keywords or any(kw in skill_name.lower() for kw in keywords):
                matched = category
                break

        if not matched:
            continue

        old_val = skill_map.get(matched, 0.0)
        # Improvement scales with quality, diminishes at higher levels
        improvement = quality * (1.0 - old_val / 100.0) * 2.0
        new_val = min(100.0, old_val + improvement)
        skill_map[matched] = round(new_val, 2)
        if new_val > old_val:
            deltas[matched] = round(new_val - old_val, 2)

    return {**state, 'skills': skill_map}, deltas


# ============================================================================
# INGESTION
# ============================================================================

def fetch_url_content(url: str) -> Tuple[str, str]:
    """Fetch full content from a URL. Returns (content, source_type)."""
    if not REQUESTS_AVAILABLE:
        return f"[URL content not fetched - requests not installed]: {url}", 'url'

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) WeEvolve/1.0'
        }
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()

        content = resp.text
        # Basic HTML to text (rough but functional for MVP)
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
        content = re.sub(r'<[^>]+>', ' ', content)
        content = re.sub(r'\s+', ' ', content).strip()

        # Truncate to ~8000 chars for Claude processing
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... truncated for processing ...]"

        return content, 'url'
    except Exception as e:
        return f"[Error fetching URL: {e}]: {url}", 'url'


def fetch_file_content(file_path: str) -> Tuple[str, str]:
    """Read content from a local file."""
    try:
        path = Path(file_path)
        if not path.exists():
            return f"[File not found: {file_path}]", 'file'

        content = path.read_text(errors='replace')
        if len(content) > 8000:
            content = content[:8000] + "\n\n[... truncated for processing ...]"
        return content, 'file'
    except Exception as e:
        return f"[Error reading file: {e}]", 'file'


def content_hash(content: str) -> str:
    """Generate a hash for dedup."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ============================================================================
# SEED PROCESSING
# ============================================================================

def process_through_seed(content: str, source: str) -> Optional[Dict]:
    """Run content through 8 SEED phases using Claude. Returns extracted knowledge."""
    if not CLAUDE_AVAILABLE:
        print("[WARN] anthropic not installed. Using fallback extraction.")
        return fallback_extraction(content, source)

    try:
        client = anthropic.Anthropic()
        prompt = SEED_EXTRACTION_PROMPT.format(content=content, source=source)

        active_model = get_model('knowledge_extraction')
        response = client.messages.create(
            model=active_model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()

        # Extract JSON from response (handle markdown code blocks)
        if '```json' in text:
            text = text.split('```json')[1].split('```')[0].strip()
        elif '```' in text:
            text = text.split('```')[1].split('```')[0].strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        print(f"[WARN] JSON parse error: {e}")
        print(f"[WARN] Raw response: {text[:500]}")
        return fallback_extraction(content, source)
    except Exception as e:
        print(f"[ERROR] SEED processing failed: {e}")
        return fallback_extraction(content, source)


def fallback_extraction(content: str, source: str) -> Dict:
    """Basic extraction without Claude. Still useful for structure."""
    words = content.lower().split()
    detected_skills = []
    for category, keywords in SKILL_CATEGORIES.items():
        if any(kw in words for kw in keywords):
            detected_skills.append(category)

    return {
        'title': f'Content from {source[:50]}',
        'perceive': content[:200],
        'connect': 'Connections pending deeper analysis',
        'learn': 'Key takeaway pending SEED processing',
        'question': 'What assumptions are in this content?',
        'expand': 'Expansion potential pending analysis',
        'share': 'Shareable insight pending extraction',
        'receive': 'Feedback signal pending',
        'improve': 'Improvement vector pending',
        'skills': detected_skills or ['research'],
        'quality': 0.3,
        'is_alpha': False,
        'alpha_type': None,
        'key_entities': [],
        'connections': [],
    }


# ============================================================================
# STORE & MEASURE
# ============================================================================

def store_knowledge_atom(db: sqlite3.Connection, atom_data: Dict, raw_content: str,
                         source_url: str, source_type: str) -> Optional[str]:
    """Store a knowledge atom and return its ID. Returns None if duplicate."""
    c_hash = content_hash(raw_content)

    # Check for duplicate
    existing = db.execute(
        "SELECT id FROM knowledge_atoms WHERE content_hash = ?", (c_hash,)
    ).fetchone()

    if existing:
        print(f"  [SKIP] Already learned this (atom {existing[0]})")
        return None

    atom_id = f"we-{c_hash}-{int(time.time())}"

    db.execute("""
        INSERT INTO knowledge_atoms (
            id, source_url, source_type, content_hash, title, raw_content,
            perceive, connect, learn, question, expand, share, receive, improve,
            skills, quality, is_alpha, alpha_type, key_entities, connections,
            processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        atom_id, source_url, source_type, c_hash,
        atom_data.get('title', 'Untitled'),
        raw_content[:2000],  # Store truncated raw for recall
        atom_data.get('perceive', ''),
        atom_data.get('connect', ''),
        atom_data.get('learn', ''),
        atom_data.get('question', ''),
        atom_data.get('expand', ''),
        atom_data.get('share', ''),
        atom_data.get('receive', ''),
        atom_data.get('improve', ''),
        json.dumps(atom_data.get('skills', [])),
        atom_data.get('quality', 0.5),
        1 if atom_data.get('is_alpha') else 0,
        atom_data.get('alpha_type'),
        json.dumps(atom_data.get('key_entities', [])),
        json.dumps(atom_data.get('connections', [])),
    ))
    db.commit()
    return atom_id


def measure_delta(db: sqlite3.Connection, atom_id: str, state_before: Dict,
                  state_after: Dict, skill_deltas: Dict) -> Dict:
    """Measure the evolution delta from this learning."""
    delta = {
        'atom_id': atom_id,
        'xp_earned': state_after['xp'] - state_before['xp'] + (
            (state_after['level'] - state_before['level']) * state_before['xp_to_next']
        ),
        'level_before': state_before['level'],
        'level_after': state_after['level'],
        'skills_improved': skill_deltas,
        'total_learnings_delta': state_after['total_learnings'] - state_before['total_learnings'],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    # Log the delta
    db.execute("""
        INSERT INTO evolution_log (event_type, details, xp_delta)
        VALUES ('learn', ?, ?)
    """, (json.dumps(delta), delta['xp_earned']))
    db.commit()

    return delta


# ============================================================================
# CORE LOOP
# ============================================================================

def learn(source: str, source_type: str = 'auto', verbose: bool = True) -> Optional[Dict]:
    """
    The core WeEvolve learning loop.
    INGEST -> PROCESS (SEED) -> STORE -> MEASURE -> EVOLVE

    Returns the evolution delta, or None if nothing was learned.
    """
    db = init_db()
    state_before = load_evolution_state()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  (C) WeEvolve - LEARN & BECOME")
        print(f"{'='*60}")
        print(f"  Source: {source[:80]}...")
        print(f"  Level {state_before['level']} | XP: {state_before['xp']}/{state_before['xp_to_next']}")
        print(f"{'='*60}\n")

    # ---- INGEST ----
    if verbose:
        print("  [1/5] INGEST - Fetching content...")
        seed_phase_log(0, "perceiving input...")

    if source_type == 'auto':
        if source.startswith('http'):
            source_type = 'url'
        elif os.path.exists(source):
            source_type = 'file'
        else:
            source_type = 'text'

    if source_type == 'url':
        content, _ = fetch_url_content(source)
    elif source_type == 'file':
        content, _ = fetch_file_content(source)
    else:
        content = source

    if not content or len(content.strip()) < 20:
        print("  [SKIP] Content too short or empty.")
        return None

    # Check for duplicate before processing
    c_hash = content_hash(content)
    existing = db.execute(
        "SELECT id FROM knowledge_atoms WHERE content_hash = ?", (c_hash,)
    ).fetchone()
    if existing:
        print(f"  [SKIP] Already learned this content (atom {existing[0]})")
        return None

    if verbose:
        print(f"  [OK] Got {len(content)} chars\n")

    # ---- PROCESS (SEED) ----
    if verbose:
        print("  [2/5] PROCESS - Running SEED extraction...")
        seed_phase_log(1, "finding connections...")

    seed_result = process_through_seed(content, source)
    if not seed_result:
        print("  [ERROR] SEED processing failed.")
        return None

    quality = seed_result.get('quality', 0.5)
    is_alpha = seed_result.get('is_alpha', False)

    if verbose:
        title = seed_result.get('title', 'Untitled')
        seed_phase_log(2, f"extracted: {title[:50]}")
        print(f"  [OK] Title: {title}")
        print(f"  [OK] Quality: {quality:.1f}/1.0 {'>>> ALPHA! <<<' if is_alpha else ''}")
        seed_phase_log(3, f"quality: {quality:.1f}")
        print(f"  [OK] Key insight: {seed_result.get('learn', 'N/A')[:100]}\n")

    # ---- STORE ----
    if verbose:
        print("  [3/5] STORE - Persisting knowledge atom...")

    atom_id = store_knowledge_atom(db, seed_result, content, source, source_type)
    if not atom_id:
        return None

    if verbose:
        print(f"  [OK] Stored as {atom_id}\n")

    # ---- MEASURE & EVOLVE ----
    if verbose:
        print("  [4/5] MEASURE & EVOLVE - Updating character sheet...")

    state = {**state_before}

    # Grant XP based on quality
    base_xp = XP_PER_LEARN
    if quality >= 0.7:
        base_xp += XP_PER_INSIGHT
    if is_alpha:
        base_xp += XP_PER_INSIGHT * 2
    state = grant_xp(state, base_xp, f"Learned: {seed_result.get('title', '')}")

    # Improve skills
    skills = seed_result.get('skills', [])
    state, skill_deltas = improve_skills(state, skills, quality)

    # Update counters
    state = {
        **state,
        'total_learnings': state.get('total_learnings', 0) + 1,
        'total_sources_processed': state.get('total_sources_processed', 0) + 1,
        'last_learn_date': datetime.now(timezone.utc).strftime('%Y-%m-%d'),
    }

    if quality >= 0.7:
        state = {**state, 'total_insights': state.get('total_insights', 0) + 1}
    if is_alpha:
        state = {**state, 'total_alpha': state.get('total_alpha', 0) + 1}

    # Track streak
    today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
    last_date = state_before.get('last_learn_date')
    if last_date == yesterday or last_date == today:
        if last_date != today:
            state = {**state, 'streak_days': state.get('streak_days', 0) + 1}
    elif last_date != today:
        state = {**state, 'streak_days': 1}

    # Add to top learnings if quality >= 0.7
    if quality >= 0.7:
        top = state.get('top_learnings', [])
        top.insert(0, {
            'title': seed_result.get('title', ''),
            'learn': seed_result.get('learn', ''),
            'quality': quality,
            'source': source[:100],
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        state = {**state, 'top_learnings': top[:20]}  # Keep top 20

    # Update atom with XP earned
    db.execute(
        "UPDATE knowledge_atoms SET xp_earned = ?, skills_improved = ? WHERE id = ?",
        (base_xp, json.dumps(skill_deltas), atom_id)
    )
    db.commit()

    # Save state and measure delta
    save_evolution_state(state)
    delta = measure_delta(db, atom_id, state_before, state, skill_deltas)

    if verbose:
        seed_phase_log(6, "receiving feedback")

    # ---- REPORT ----
    if verbose:
        seed_phase_log(4, f"+{base_xp}XP")
        print(f"  [OK] +{base_xp} XP (Level {state['level']}: {state['xp']}/{state['xp_to_next']})")
        if skill_deltas:
            for skill, d in skill_deltas.items():
                print(f"  [OK] {skill}: +{d:.2f}")
        seed_phase_log(5, "sharing to knowledge base")
        print(f"\n  [5/5] COMPLETE - Knowledge atom integrated.")
        print(f"\n{'='*60}")
        print(f"  SEED SUMMARY")
        print(f"{'='*60}")
        print(f"  PERCEIVE: {seed_result.get('perceive', 'N/A')[:80]}")
        print(f"  CONNECT:  {seed_result.get('connect', 'N/A')[:80]}")
        print(f"  LEARN:    {seed_result.get('learn', 'N/A')[:80]}")
        print(f"  QUESTION: {seed_result.get('question', 'N/A')[:80]}")
        print(f"  EXPAND:   {seed_result.get('expand', 'N/A')[:80]}")
        print(f"  SHARE:    {seed_result.get('share', 'N/A')[:80]}")
        print(f"  RECEIVE:  {seed_result.get('receive', 'N/A')[:80]}")
        print(f"  IMPROVE:  {seed_result.get('improve', 'N/A')[:80]}")
        print(f"{'='*60}")
        seed_phase_log(7, "loop complete")
        print()

    # Log to JSONL for persistence
    EVOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLUTION_LOG_PATH, 'a') as f:
        f.write(json.dumps({
            'event': 'learn',
            'atom_id': atom_id,
            'title': seed_result.get('title', ''),
            'quality': quality,
            'is_alpha': is_alpha,
            'xp': base_xp,
            'skills': skill_deltas,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }) + '\n')

    # Share to NATS collective (non-blocking, silent on failure)
    if NATS_AVAILABLE:
        try:
            collective = get_collective()
            if collective.connected:
                collective.publish_learning(seed_result)
        except Exception:
            pass

    return delta


# ============================================================================
# RECALL - Search what you've learned
# ============================================================================

def recall(query: str, limit: int = 5) -> List[Dict]:
    """Search knowledge atoms by keyword. Returns matching learnings."""
    db = init_db()

    # Escape SQL LIKE special characters to prevent wildcard injection
    safe_query = query.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
    like_param = f'%{safe_query}%'
    # Clamp limit to a reasonable maximum to prevent DoS
    limit = max(1, min(limit, 100))

    # Search across SEED fields
    results = db.execute("""
        SELECT id, title, learn, quality, source_url, is_alpha, created_at,
               perceive, connect, question, expand, share, receive, improve
        FROM knowledge_atoms
        WHERE title LIKE ? ESCAPE '\\' OR learn LIKE ? ESCAPE '\\'
              OR perceive LIKE ? ESCAPE '\\' OR connect LIKE ? ESCAPE '\\'
              OR expand LIKE ? ESCAPE '\\' OR improve LIKE ? ESCAPE '\\'
              OR raw_content LIKE ? ESCAPE '\\'
        ORDER BY quality DESC, created_at DESC
        LIMIT ?
    """, tuple([like_param] * 7 + [limit])).fetchall()

    atoms = []
    for row in results:
        atoms.append({
            'id': row[0], 'title': row[1], 'learn': row[2],
            'quality': row[3], 'source': row[4],
            'is_alpha': bool(row[5]), 'created_at': row[6],
            'perceive': row[7], 'connect': row[8],
            'question': row[9], 'expand': row[10],
            'share': row[11], 'receive': row[12], 'improve': row[13],
        })
    return atoms


def recall_display(query: str, limit: int = 5):
    """Search and display results."""
    atoms = recall(query, limit)
    if not atoms:
        print(f"\n  No learnings found for '{query}'")
        print(f"  Total knowledge atoms: {count_atoms()}")
        return

    print(f"\n{'='*60}")
    print(f"  RECALL: '{query}' ({len(atoms)} results)")
    print(f"{'='*60}\n")

    for i, atom in enumerate(atoms, 1):
        alpha_tag = ' [ALPHA]' if atom['is_alpha'] else ''
        print(f"  {i}. {atom['title']}{alpha_tag} (Q:{atom['quality']:.1f})")
        print(f"     LEARN: {atom['learn'][:100]}")
        print(f"     CONNECT: {atom['connect'][:80]}")
        print(f"     Source: {atom['source'][:60]}")
        print(f"     Date: {atom['created_at']}")
        print()


def count_atoms() -> int:
    """Count total knowledge atoms."""
    db = init_db()
    return db.execute("SELECT COUNT(*) FROM knowledge_atoms").fetchone()[0]


# ============================================================================
# SCAN - Process new bookmarks
# ============================================================================

def scan_bookmarks(verbose: bool = True) -> int:
    """Scan BRAIN/INTEL/bookmarks/ for unprocessed bookmark files."""
    if not BOOKMARKS_DIR.exists():
        print(f"  [SKIP] Bookmarks directory not found: {BOOKMARKS_DIR}")
        return 0

    db = init_db()
    processed = 0
    bookmark_files = sorted(BOOKMARKS_DIR.glob('*.md'))

    if verbose:
        print(f"\n  Found {len(bookmark_files)} bookmark files")

    for bm_file in bookmark_files:
        # Check if already processed
        c_hash = content_hash(bm_file.read_text(errors='replace'))
        existing = db.execute(
            "SELECT id FROM knowledge_atoms WHERE content_hash = ?", (c_hash,)
        ).fetchone()

        if existing:
            continue

        if verbose:
            print(f"\n  Processing: {bm_file.name}")

        content = bm_file.read_text(errors='replace')
        result = learn(content, source_type='text', verbose=verbose)
        if result:
            processed += 1

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    if verbose:
        print(f"\n  Processed {processed} new bookmarks")

    return processed


# ============================================================================
# STATUS DASHBOARD
# ============================================================================

def show_status():
    """Display the evolution dashboard - MMORPG character sheet."""
    state = load_evolution_state()
    db = init_db()

    total_atoms = db.execute("SELECT COUNT(*) FROM knowledge_atoms").fetchone()[0]
    alpha_count = db.execute(
        "SELECT COUNT(*) FROM knowledge_atoms WHERE is_alpha = 1"
    ).fetchone()[0]
    avg_quality = db.execute(
        "SELECT AVG(quality) FROM knowledge_atoms"
    ).fetchone()[0] or 0

    # Recent learnings
    recent = db.execute("""
        SELECT title, quality, is_alpha, created_at
        FROM knowledge_atoms ORDER BY created_at DESC LIMIT 5
    """).fetchall()

    print(f"""
{'='*60}
  (C) WeEvolve - EVOLUTION DASHBOARD
  LOVE -> LIVE FREE -> SEED2 -> 8OWLS -> YOU
{'='*60}

  LEVEL {state['level']}  |  XP: {state['xp']}/{state['xp_to_next']}
  {LIME_C}{chr(0x2588) * min(30, int(30 * state['xp'] / max(1, state['xp_to_next'])))}\
{RESET_C}{DIM_C}{chr(0x2591) * max(0, 30 - int(30 * state['xp'] / max(1, state['xp_to_next'])))}{RESET_C}

  STATS:
  - Total Learnings:  {state.get('total_learnings', 0)}
  - Total Insights:   {state.get('total_insights', 0)}
  - Alpha Discoveries: {state.get('total_alpha', 0)}
  - Knowledge Atoms:  {total_atoms}
  - Avg Quality:      {avg_quality:.2f}/1.0
  - Streak:           {state.get('streak_days', 0)} days

  SKILLS:""")

    skills = state.get('skills', {})
    for skill, val in sorted(skills.items(), key=lambda x: x[1], reverse=True):
        bar_len = int(val / 100 * 20)
        bar = '\u2588' * bar_len + '\u2591' * (20 - bar_len)
        if val > 80:
            color = GREEN_C
        elif val > 50:
            color = LIME_C
        else:
            color = DIM_C
        print(f"  - {skill:20s} [{color}{bar}{RESET_C}] {val:.1f}")

    print(f"\n  Love: {GREEN_C}{'\u2588' * 20}{RESET_C} 100.0 (always)")

    if recent:
        print(f"\n  RECENT LEARNINGS:")
        for title, quality, is_alpha, created in recent:
            alpha_tag = ' [A]' if is_alpha else ''
            print(f"  - [{quality:.1f}]{alpha_tag} {title[:50]} ({created})")

    top = state.get('top_learnings', [])
    if top:
        print(f"\n  TOP INSIGHTS:")
        for t in top[:5]:
            print(f"  - [{t['quality']:.1f}] {t['title'][:40]}")
            print(f"    {t['learn'][:70]}")

    # Active quests
    quests = state.get('quests', [])
    active_quests = [q for q in quests if q.get('status') == 'active']
    if active_quests:
        print(f"\n  QUESTS:")
        for q in active_quests[:3]:
            print(f"  - {q.get('name', 'Unknown')}")

    # Tier badge + streak
    try:
        from weevolve.license import get_tier
        tier = get_tier().upper()
    except Exception:
        tier = "FREE"
    streak = state.get('streak_days', 0)
    streak_display = f"{streak} days" if streak > 0 else "start today"

    # Voice server status
    vs = voice_status()
    if vs['server_exists']:
        if vs['running']:
            voice_label = f"{GREEN_C}RUNNING{RESET_C} (PID {vs['pid']}, port {vs['port']})"
        else:
            voice_label = f"{DIM_C}STOPPED{RESET_C} -- run: weevolve voice"
    else:
        voice_label = f"{DIM_C}NOT INSTALLED{RESET_C}"
    print(f"\n  VOICE: {voice_label}")

    # NATS collective status
    if NATS_AVAILABLE:
        try:
            collective = get_collective()
            if collective.connected:
                nats_label = f"{GREEN_C}CONNECTED{RESET_C} (sharing learnings)"
            else:
                nats_label = f"{DIM_C}OFFLINE{RESET_C} (NATS not reachable)"
        except Exception:
            nats_label = f"{DIM_C}OFFLINE{RESET_C}"
    else:
        nats_label = f"{DIM_C}NOT INSTALLED{RESET_C} -- pip install nats-py"
    print(f"  NATS:  {nats_label}")

    print(f"\n{'='*60}")
    print(f"  [{LIME_C}{tier}{RESET_C}]  Streak: {streak_display}")
    print()
    # Smart next action
    if total_atoms == 0:
        print(f"  {BOLD_C}Next:{RESET_C} weevolve learn --text 'your first insight'")
    elif not active_quests:
        print(f"  {BOLD_C}Next:{RESET_C} weevolve evolve  (generate quests)")
    else:
        print(f"  {BOLD_C}Next:{RESET_C} weevolve learn <url>  (keep evolving)")
    print(f"{'='*60}\n")


# ============================================================================
# GENESIS - Portable Knowledge Export/Import
# ============================================================================

# Fields that are safe to export (no PII, no raw URLs that could deanonymize)
GENESIS_SAFE_FIELDS = [
    'title', 'perceive', 'connect', 'learn', 'question',
    'expand', 'share', 'receive', 'improve',
    'skills', 'quality', 'is_alpha', 'alpha_type',
    'key_entities', 'connections', 'created_at',
]

# PII patterns to strip from exported text
PII_PATTERNS = [
    (re.compile(r'@\w+'), '@[redacted]'),                        # Twitter handles
    (re.compile(r'https?://\S+'), '[url]'),                      # URLs
    (re.compile(r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b'), '[email]'),   # Emails
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), '[ip]'),  # IPs
    (re.compile(r'/Users/\w+'), '/Users/[user]'),                # Local paths
    (re.compile(r'sk-[a-zA-Z0-9-]+'), '[api-key]'),             # API keys
    (re.compile(r'0x[a-fA-F0-9]{40}'), '[wallet]'),             # Wallet addresses
]


def strip_pii(text: str) -> str:
    """Remove personally identifiable information from text."""
    if not text:
        return text
    result = text
    for pattern, replacement in PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def genesis_export(output_path: Optional[str] = None, min_quality: float = 0.3,
                   tier: str = 'full', verbose: bool = True) -> str:
    """
    Export knowledge atoms to a portable genesis.db.

    Tiers:
      - 'full': All atoms above min_quality (for premium distribution)
      - 'curated': Only quality >= 0.7 and alpha discoveries (for free distribution)

    All exports are PII-stripped. No raw content, no source URLs.
    Returns path to the exported database.
    """
    source_db = init_db()
    if output_path:
        export_path = Path(output_path)
    elif tier == 'curated':
        export_path = GENESIS_CURATED_DB_DEFAULT
    else:
        export_path = GENESIS_DB_DEFAULT

    # Ensure parent directory exists
    export_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove old export if it exists
    if export_path.exists():
        export_path.unlink()

    # Create the genesis database
    genesis_db = sqlite3.connect(str(export_path))
    genesis_db.execute("PRAGMA journal_mode=WAL")

    genesis_db.executescript("""
        CREATE TABLE IF NOT EXISTS genesis_atoms (
            id TEXT PRIMARY KEY,
            title TEXT,
            perceive TEXT,
            connect TEXT,
            learn TEXT,
            question TEXT,
            expand TEXT,
            share TEXT,
            receive TEXT,
            improve TEXT,
            skills TEXT,
            quality REAL,
            is_alpha INTEGER DEFAULT 0,
            alpha_type TEXT,
            key_entities TEXT,
            connections TEXT,
            created_at TEXT
        );

        CREATE TABLE IF NOT EXISTS genesis_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_genesis_quality ON genesis_atoms(quality DESC);
        CREATE INDEX IF NOT EXISTS idx_genesis_alpha ON genesis_atoms(is_alpha);
    """)

    # Determine quality threshold
    quality_threshold = 0.7 if tier == 'curated' else min_quality

    # Query source atoms
    if tier == 'curated':
        rows = source_db.execute("""
            SELECT id, title, perceive, connect, learn, question, expand,
                   share, receive, improve, skills, quality, is_alpha,
                   alpha_type, key_entities, connections, created_at
            FROM knowledge_atoms
            WHERE quality >= ? OR is_alpha = 1
            ORDER BY quality DESC
        """, (quality_threshold,)).fetchall()
    else:
        rows = source_db.execute("""
            SELECT id, title, perceive, connect, learn, question, expand,
                   share, receive, improve, skills, quality, is_alpha,
                   alpha_type, key_entities, connections, created_at
            FROM knowledge_atoms
            WHERE quality >= ?
            ORDER BY quality DESC
        """, (quality_threshold,)).fetchall()

    exported = 0
    for row in rows:
        atom_id, title, perceive, connect_val, learn_val, question, expand, \
            share, receive, improve, skills, quality, is_alpha, alpha_type, \
            key_entities, connections_val, created_at = row

        # Strip PII from all text fields
        genesis_db.execute("""
            INSERT OR IGNORE INTO genesis_atoms (
                id, title, perceive, connect, learn, question, expand,
                share, receive, improve, skills, quality, is_alpha,
                alpha_type, key_entities, connections, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            atom_id,
            strip_pii(title),
            strip_pii(perceive),
            strip_pii(connect_val),
            strip_pii(learn_val),
            strip_pii(question),
            strip_pii(expand),
            strip_pii(share),
            strip_pii(receive),
            strip_pii(improve),
            skills,  # JSON array of skill names — safe
            quality,
            is_alpha,
            alpha_type,
            strip_pii(key_entities) if key_entities else None,
            strip_pii(connections_val) if connections_val else None,
            created_at,
        ))
        exported += 1

    # Store metadata
    state = load_evolution_state()
    meta = {
        'version': '1.0',
        'tier': tier,
        'exported_at': datetime.now(timezone.utc).isoformat(),
        'total_atoms': exported,
        'source_level': state['level'],
        'source_total_learnings': state.get('total_learnings', 0),
        'source_total_alpha': state.get('total_alpha', 0),
        'min_quality': quality_threshold,
        'pii_stripped': True,
        'protocol': 'SEED',
        'origin': '8OWLS',
    }
    for key, value in meta.items():
        genesis_db.execute(
            "INSERT OR REPLACE INTO genesis_meta (key, value) VALUES (?, ?)",
            (key, str(value))
        )

    genesis_db.commit()
    genesis_db.close()

    file_size = export_path.stat().st_size

    if verbose:
        print(f"\n{'='*60}")
        print(f"  (C) GENESIS EXPORT COMPLETE")
        print(f"{'='*60}")
        print(f"  Tier: {tier}")
        print(f"  Atoms exported: {exported}")
        print(f"  Min quality: {quality_threshold}")
        print(f"  PII stripped: Yes")
        print(f"  File size: {file_size / 1024:.1f} KB")
        print(f"  Output: {export_path}")
        print(f"{'='*60}\n")

    return str(export_path)


def genesis_import(import_path: str, verbose: bool = True) -> Dict:
    """
    Import a genesis.db to bootstrap a fresh WeEvolve install.

    This gives new users a head start — starting at Level 5+ instead of Level 0.
    Imported atoms merge with existing knowledge (dedup by ID).

    Returns stats dict.
    """
    path = Path(import_path)
    if not path.exists():
        print(f"  [ERROR] Genesis file not found: {import_path}")
        return {'imported': 0, 'skipped': 0, 'error': 'File not found'}

    genesis_db = sqlite3.connect(str(path))
    target_db = init_db()

    # Read metadata
    meta_rows = genesis_db.execute("SELECT key, value FROM genesis_meta").fetchall()
    meta = {row[0]: row[1] for row in meta_rows}

    if verbose:
        print(f"\n{'='*60}")
        print(f"  (C) GENESIS IMPORT")
        print(f"{'='*60}")
        print(f"  Source: {import_path}")
        print(f"  Version: {meta.get('version', 'unknown')}")
        print(f"  Tier: {meta.get('tier', 'unknown')}")
        print(f"  Atoms available: {meta.get('total_atoms', 'unknown')}")
        print(f"  Origin level: {meta.get('source_level', 'unknown')}")
        print(f"  PII stripped: {meta.get('pii_stripped', 'unknown')}")
        print(f"{'='*60}\n")

    # Import atoms
    atoms = genesis_db.execute("""
        SELECT id, title, perceive, connect, learn, question, expand,
               share, receive, improve, skills, quality, is_alpha,
               alpha_type, key_entities, connections, created_at
        FROM genesis_atoms
        ORDER BY quality DESC
    """).fetchall()

    imported = 0
    skipped = 0
    total_xp = 0

    for atom in atoms:
        atom_id = atom[0]

        # Check if already exists
        existing = target_db.execute(
            "SELECT id FROM knowledge_atoms WHERE id = ?", (atom_id,)
        ).fetchone()

        if existing:
            skipped += 1
            continue

        # Generate a content hash from the learn field (since raw_content is stripped)
        learn_text = atom[4] or ''
        c_hash = content_hash(f"genesis:{atom_id}:{learn_text}")

        # Check hash dedup too
        hash_exists = target_db.execute(
            "SELECT id FROM knowledge_atoms WHERE content_hash = ?", (c_hash,)
        ).fetchone()

        if hash_exists:
            skipped += 1
            continue

        target_db.execute("""
            INSERT INTO knowledge_atoms (
                id, source_url, source_type, content_hash, title, raw_content,
                perceive, connect, learn, question, expand, share, receive, improve,
                skills, quality, is_alpha, alpha_type, key_entities, connections,
                processed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            atom_id, 'genesis', 'genesis', c_hash,
            atom[1],  # title
            f"[Imported from genesis.db]",  # raw_content placeholder
            atom[2], atom[3], atom[4], atom[5], atom[6],  # SEED phases
            atom[7], atom[8], atom[9],
            atom[10],  # skills
            atom[11],  # quality
            atom[12],  # is_alpha
            atom[13],  # alpha_type
            atom[14],  # key_entities
            atom[15],  # connections
        ))

        imported += 1

        # Calculate XP for this imported atom
        quality = atom[11] or 0.5
        is_alpha = atom[12]
        xp = XP_PER_LEARN
        if quality >= 0.7:
            xp += XP_PER_INSIGHT
        if is_alpha:
            xp += XP_PER_INSIGHT * 2
        total_xp += xp

    target_db.commit()
    genesis_db.close()

    # Update evolution state with imported knowledge
    if imported > 0:
        state = load_evolution_state()
        state = grant_xp(state, total_xp, f"Genesis import: {imported} atoms")

        # Improve skills from imported atoms — aggregate skill mentions
        all_skills = []
        for atom in atoms[:imported]:
            try:
                skill_list = json.loads(atom[10]) if atom[10] else []
                all_skills.extend(skill_list)
            except (json.JSONDecodeError, TypeError):
                pass

        # Apply averaged quality improvement across all imported skills
        avg_quality = float(meta.get('min_quality', 0.5))
        state, skill_deltas = improve_skills(state, all_skills, avg_quality * 0.5)

        state = {
            **state,
            'total_learnings': state.get('total_learnings', 0) + imported,
            'total_sources_processed': state.get('total_sources_processed', 0) + imported,
        }
        save_evolution_state(state)

        # Log the import
        target_db.execute("""
            INSERT INTO evolution_log (event_type, details, xp_delta)
            VALUES ('genesis_import', ?, ?)
        """, (json.dumps({
            'source': import_path,
            'imported': imported,
            'skipped': skipped,
            'total_xp': total_xp,
            'meta': meta,
        }), total_xp))
        target_db.commit()

    stats = {
        'imported': imported,
        'skipped': skipped,
        'total_xp': total_xp,
        'meta': meta,
    }

    if verbose:
        print(f"  Imported: {imported} atoms")
        print(f"  Skipped (already known): {skipped}")
        print(f"  XP earned: +{total_xp}")
        state = load_evolution_state()
        print(f"  New level: {state['level']}")
        print(f"  Total knowledge atoms: {count_atoms()}")
        print(f"\n{'='*60}")
        print(f"  Genesis complete. You're not starting from zero.")
        print(f"  The collective's knowledge is now yours.")
        print(f"{'='*60}\n")

    return stats


def genesis_stats(db_path: Optional[str] = None, verbose: bool = True) -> Dict:
    """Show stats for a genesis.db file."""
    path = Path(db_path) if db_path else GENESIS_DB_DEFAULT

    if not path.exists():
        print(f"  [INFO] No genesis.db found at {path}")
        print(f"  [INFO] Run 'python core.py genesis export' to create one.")
        return {}

    gdb = sqlite3.connect(str(path))

    total = gdb.execute("SELECT COUNT(*) FROM genesis_atoms").fetchone()[0]
    alpha_count = gdb.execute(
        "SELECT COUNT(*) FROM genesis_atoms WHERE is_alpha = 1"
    ).fetchone()[0]
    avg_q = gdb.execute(
        "SELECT AVG(quality) FROM genesis_atoms"
    ).fetchone()[0] or 0
    high_q = gdb.execute(
        "SELECT COUNT(*) FROM genesis_atoms WHERE quality >= 0.7"
    ).fetchone()[0]

    # Skill distribution
    all_skills_raw = gdb.execute(
        "SELECT skills FROM genesis_atoms WHERE skills IS NOT NULL"
    ).fetchall()
    skill_counts = {}
    for (skills_json,) in all_skills_raw:
        try:
            for s in json.loads(skills_json):
                skill_counts[s] = skill_counts.get(s, 0) + 1
        except (json.JSONDecodeError, TypeError):
            pass

    # Metadata
    meta_rows = gdb.execute("SELECT key, value FROM genesis_meta").fetchall()
    meta = {row[0]: row[1] for row in meta_rows}

    gdb.close()

    file_size = path.stat().st_size

    stats = {
        'total_atoms': total,
        'alpha_count': alpha_count,
        'avg_quality': avg_q,
        'high_quality': high_q,
        'skill_distribution': skill_counts,
        'meta': meta,
        'file_size_kb': file_size / 1024,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  (C) GENESIS DATABASE STATS")
        print(f"{'='*60}")
        print(f"  File: {path}")
        print(f"  Size: {file_size / 1024:.1f} KB")
        print(f"  Version: {meta.get('version', 'unknown')}")
        print(f"  Tier: {meta.get('tier', 'unknown')}")
        print(f"  Exported: {meta.get('exported_at', 'unknown')}")
        print(f"")
        print(f"  ATOMS:")
        print(f"  - Total:         {total}")
        print(f"  - Alpha:         {alpha_count}")
        print(f"  - High Quality:  {high_q} (>= 0.7)")
        print(f"  - Avg Quality:   {avg_q:.2f}")
        print(f"")
        print(f"  SKILL COVERAGE:")
        for skill, count in sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {skill:20s} {count} atoms")
        print(f"\n{'='*60}\n")

    return stats


# ============================================================================
# GENESIS TOP LEARNINGS
# ============================================================================

def genesis_top_learnings(limit: int = 10, verbose: bool = True) -> List[Dict]:
    """Display the top learnings from the WeEvolve knowledge base."""
    db = init_db()

    rows = db.execute("""
        SELECT title, learn, quality, is_alpha, skills, expand, improve, created_at
        FROM knowledge_atoms
        WHERE quality >= 0.7
        ORDER BY quality DESC, is_alpha DESC
        LIMIT ?
    """, (limit,)).fetchall()

    learnings = []
    if verbose:
        print(f"\n{'='*60}")
        print(f"  (C) GENESIS TOP {limit} LEARNINGS")
        print(f"{'='*60}\n")

    for i, (title, learn_val, quality, is_alpha, skills_json, expand, improve, created) in enumerate(rows, 1):
        alpha_tag = ' [ALPHA]' if is_alpha else ''
        learning = {
            'rank': i,
            'title': title,
            'learn': learn_val,
            'quality': quality,
            'is_alpha': bool(is_alpha),
            'skills': json.loads(skills_json) if skills_json else [],
            'expand': expand,
            'improve': improve,
        }
        learnings.append(learning)

        if verbose:
            print(f"  {i}. [{quality:.1f}]{alpha_tag} {title}")
            print(f"     LEARN: {(learn_val or '')[:120]}")
            if expand:
                print(f"     EXPAND: {(expand or '')[:100]}")
            print()

    if verbose:
        print(f"{'='*60}")
        hq_count = db.execute(
            'SELECT COUNT(*) FROM knowledge_atoms WHERE quality >= 0.7'
        ).fetchone()[0]
        alpha_count = db.execute(
            'SELECT COUNT(*) FROM knowledge_atoms WHERE is_alpha = 1'
        ).fetchone()[0]
        print(f"  Total high-quality atoms (>= 0.7): {hq_count}")
        print(f"  Total alpha discoveries: {alpha_count}")
        print(f"{'='*60}\n")

    return learnings


# ============================================================================
# DAEMON MODE
# ============================================================================

def run_daemon(interval: int = 300):
    """Run as a continuous daemon. Checks for new bookmarks every interval seconds."""
    print(f"\n{'='*60}")
    print(f"  (C) WeEvolve DAEMON - Continuous Learning")
    print(f"  Interval: {interval}s | Ctrl+C to stop")
    print(f"{'='*60}\n")

    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n--- Daemon Cycle {cycle} ({datetime.now().strftime('%H:%M:%S')}) ---")

            # Scan for new bookmarks
            processed = scan_bookmarks(verbose=True)

            if processed > 0:
                print(f"  Learned {processed} new things this cycle")
                state = load_evolution_state()
                print(f"  Level {state['level']} | XP: {state['xp']}/{state['xp_to_next']}")

            print(f"  Next scan in {interval}s...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n  (C) WeEvolve daemon stopping. Knowledge persisted.")
            show_status()
            break
        except Exception as e:
            print(f"  [ERROR] Daemon cycle failed: {e}")
            traceback.print_exc()
            time.sleep(60)  # Wait before retry


# ============================================================================
# EVOLVE - Self-Evolution Analysis & Quest Generation
# ============================================================================

def run_evolve():
    """
    Analyze the knowledge base to find gaps, generate smart quests,
    and suggest what to learn next. The meta-learning engine.
    SOWL IMPROVE phase — making the loop itself better.
    """
    import random

    db = init_db()
    state = load_evolution_state()
    skills = state.get('skills', {})

    print(f"\n{'='*60}")
    print(f"  {RED_C}SOWL{RESET_C} {DIM_C}IMPROVE{RESET_C} analyzing evolution state...")
    print(f"{'='*60}\n")

    # ---- Phase 1: PERCEIVE — observe current skill state ----
    seed_phase_log(0, "scanning skill landscape...")

    sorted_skills = sorted(skills.items(), key=lambda x: x[1])
    weakest = sorted_skills[:3]
    strongest = sorted_skills[-3:]
    strongest.reverse()

    gap = strongest[0][1] - weakest[0][1] if strongest and weakest else 0.0

    print(f"\n  {BOLD_C}SKILL BALANCE:{RESET_C}")
    print(f"    Strongest: {', '.join(f'{s} ({v:.1f})' for s, v in strongest)}")
    print(f"    Weakest:   {', '.join(f'{s} ({v:.1f})' for s, v in weakest)}")
    print(f"    Gap:       {gap:.1f} points", end="")
    if gap > 50:
        print(f" {RED_C}-- opportunity zone{RESET_C}")
    elif gap > 25:
        print(f" {YELLOW_C}-- moderate imbalance{RESET_C}")
    else:
        print(f" {GREEN_C}-- well balanced{RESET_C}")

    # ---- Phase 2: CONNECT — find patterns in the data ----
    seed_phase_log(1, "connecting patterns across knowledge base...")

    total_atoms = db.execute("SELECT COUNT(*) FROM knowledge_atoms").fetchone()[0]
    alpha_count = db.execute(
        "SELECT COUNT(*) FROM knowledge_atoms WHERE is_alpha = 1"
    ).fetchone()[0]
    avg_quality_row = db.execute(
        "SELECT AVG(quality) FROM knowledge_atoms"
    ).fetchone()[0]
    avg_quality = avg_quality_row if avg_quality_row is not None else 0.0
    high_quality_count = db.execute(
        "SELECT COUNT(*) FROM knowledge_atoms WHERE quality >= 0.7"
    ).fetchone()[0]

    # ---- Phase 3: LEARN — extract actionable quests ----
    seed_phase_log(2, "generating evolution quests...")

    quests = []

    # Quest 1: Close the gap (weakest skill)
    weakest_skill = weakest[0][0] if weakest else 'research'
    weakest_val = weakest[0][1] if weakest else 0.0
    quests.append({
        'name': f'Learn about {weakest_skill}',
        'type': 'gap_close',
        'description': f'Close the {gap:.0f}pt gap -- {weakest_skill} is at {weakest_val:.1f}',
        'target_skill': weakest_skill,
        'status': 'active',
        'created_at': datetime.now(timezone.utc).isoformat(),
    })

    # Quest 2: Deepen the strongest
    strongest_skill = strongest[0][0] if strongest else 'research'
    strongest_val = strongest[0][1] if strongest else 0.0
    quests.append({
        'name': f'Deepen {strongest_skill} to alpha',
        'type': 'deepen',
        'description': f'Push {strongest_skill} ({strongest_val:.1f}) toward mastery with alpha-quality content',
        'target_skill': strongest_skill,
        'status': 'active',
        'created_at': datetime.now(timezone.utc).isoformat(),
    })

    # Quest 3: Cross-connect two random skills
    all_skill_names = list(skills.keys())
    if len(all_skill_names) >= 2:
        pair = random.sample(all_skill_names, 2)
        quests.append({
            'name': f'Cross-connect {pair[0]} and {pair[1]}',
            'type': 'cross_connect',
            'description': f'Find novel connections between {pair[0]} and {pair[1]}',
            'target_skill': f'{pair[0]}+{pair[1]}',
            'status': 'active',
            'created_at': datetime.now(timezone.utc).isoformat(),
        })

    print(f"\n  {BOLD_C}ACTIVE QUESTS:{RESET_C}")
    for i, q in enumerate(quests, 1):
        print(f"    {i}. {q['name']}")
        print(f"       {DIM_C}{q['description']}{RESET_C}")

    # ---- Phase 4: QUESTION — check knowledge freshness ----
    seed_phase_log(3, "questioning knowledge freshness...")

    latest_row = db.execute(
        "SELECT created_at FROM knowledge_atoms ORDER BY created_at DESC LIMIT 1"
    ).fetchone()

    freshness_label = "NO DATA"
    days_since = None
    if latest_row and latest_row[0]:
        try:
            latest_dt = datetime.fromisoformat(latest_row[0].replace('Z', '+00:00'))
            if latest_dt.tzinfo is None:
                latest_dt = latest_dt.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            days_since = (now_utc - latest_dt).days
            if days_since <= 1:
                freshness_label = f"{GREEN_C}FRESH{RESET_C} (today)"
            elif days_since <= 3:
                freshness_label = f"{GREEN_C}GOOD{RESET_C} ({days_since} days ago)"
            elif days_since <= 7:
                freshness_label = f"{YELLOW_C}AGING{RESET_C} ({days_since} days ago)"
            else:
                freshness_label = f"{RED_C}STALE{RESET_C} ({days_since} days ago)"
        except (ValueError, TypeError):
            freshness_label = "UNKNOWN"

    # ---- Phase 5: EXPAND — quality analysis ----
    seed_phase_log(4, "expanding quality analysis...")

    high_pct = (high_quality_count / total_atoms * 100) if total_atoms > 0 else 0.0

    print(f"\n  {BOLD_C}KNOWLEDGE HEALTH:{RESET_C}")
    print(f"    Total atoms: {total_atoms} | Alpha: {alpha_count} | Avg quality: {avg_quality:.2f}")
    print(f"    High quality (>=0.7): {high_quality_count} ({high_pct:.0f}%)")
    print(f"    Last learning: {freshness_label}")

    quality_target = ""
    if high_pct < 30:
        quality_target = "Aim for 30% high-quality atoms. Be more selective in sources."
    elif high_pct < 50:
        quality_target = "Good base. Push toward 50% high-quality with deeper content."
    else:
        quality_target = "Strong knowledge base. Focus on alpha discoveries now."

    # ---- Phase 6: SHARE — compose the evolution suggestion ----
    seed_phase_log(5, "composing evolution suggestion...")

    suggestion_lines = []
    if days_since is not None and days_since > 7:
        suggestion_lines.append(
            f"Knowledge is {days_since} days stale. Run: weevolve scan"
        )
    suggestion_lines.append(
        f"Your owl recommends learning about {weakest_skill} next."
    )
    suggestion_lines.append(quality_target)
    if alpha_count == 0:
        suggestion_lines.append(
            "No alpha discoveries yet. Seek game-changing content."
        )

    print(f"\n  {BOLD_C}EVOLUTION SUGGESTION:{RESET_C}")
    for line in suggestion_lines:
        print(f"    {line}")
    print(f"    Try: weevolve learn --text \"what is {weakest_skill}?\"")

    # ---- Phase 7: RECEIVE — accept what the data tells us ----
    seed_phase_log(6, "receiving feedback from knowledge base...")

    # ---- Phase 8: IMPROVE — save quests and log the evolution ----
    seed_phase_log(7, "persisting quests and closing the loop...")

    updated_state = {**state, 'quests': quests}
    save_evolution_state(updated_state)

    # Log the evolve event
    EVOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EVOLUTION_LOG_PATH, 'a') as f:
        f.write(json.dumps({
            'event': 'evolve',
            'quests_generated': len(quests),
            'gap': gap,
            'weakest': weakest_skill,
            'strongest': strongest_skill,
            'total_atoms': total_atoms,
            'avg_quality': round(avg_quality, 3),
            'freshness_days': days_since,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }) + '\n')

    print(f"\n{'='*60}")
    print(f"  {GREEN_C}Quests saved.{RESET_C} The loop improves itself.")
    print(f"  Run {DIM_C}weevolve quest{RESET_C} to see active quests.")
    print(f"{'='*60}\n")


# ============================================================================
# VOICE - Talk to your owl
# ============================================================================

def _get_voice_server_pid() -> Optional[int]:
    """Check if the voice server is running on port 8006. Returns PID or None."""
    try:
        result = subprocess.run(
            ['lsof', '-ti', ':8006'],
            capture_output=True, text=True, timeout=5,
        )
        pid_str = result.stdout.strip()
        if pid_str:
            # lsof may return multiple PIDs; take the first
            return int(pid_str.split('\n')[0])
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def _get_owl_name() -> str:
    """Determine the user's owl name from onboarding, env, or hostname."""
    # 1. Check onboarding.json
    try:
        onboarding_path = DATA_DIR / 'onboarding.json'
        if onboarding_path.exists():
            with open(onboarding_path) as f:
                data = json.load(f)
            name = data.get('owl_name', '')
            if name:
                return name
    except Exception:
        pass

    # 2. Check environment variable
    env_owl = os.getenv('OWL_NAME', '')
    if env_owl:
        return env_owl

    # 3. Try hostname detection (non-interactive fallback)
    try:
        from weevolve.onboarding import HOSTNAME_OWL_HINTS
        import socket
        hostname = socket.gethostname().lower()
        for hint, owl in HOSTNAME_OWL_HINTS.items():
            if hint in hostname:
                return owl
    except Exception:
        pass

    return 'your owl'


def voice_status() -> Dict[str, Any]:
    """Return voice server status info."""
    seed_dir = Path(__file__).resolve().parent.parent.parent
    server_path = seed_dir / 'voice-app' / 'sowl_convai_server.py'
    pid = _get_voice_server_pid()
    return {
        'server_exists': server_path.exists(),
        'server_path': str(server_path),
        'running': pid is not None,
        'pid': pid,
        'port': 8006,
        'url': 'http://localhost:8006',
    }


def run_voice(background: bool = False):
    """
    Start the voice orb -- talk to your owl.

    1. Checks if voice-app/sowl_convai_server.py exists
    2. Starts the voice server locally on port 8006
    3. Opens the browser to the voice page
    4. Connects to NATS for collective awareness
    5. Shows the user their owl name and how to use it
    """
    seed_dir = Path(__file__).resolve().parent.parent.parent
    server_path = seed_dir / 'voice-app' / 'sowl_convai_server.py'
    nats_publish_path = seed_dir / 'tools' / 'nats_publish.py'
    owl_name = _get_owl_name()

    print(f"\n{'='*60}")
    print(f"  {MAGENTA}(*){RESET_C} WeEvolve VOICE - Talk to {owl_name}")
    print(f"{'='*60}\n")

    # ---- Step 1: Check server exists ----
    if not server_path.exists():
        print(f"  {RED_C}[ERROR]{RESET_C} Voice server not found:")
        print(f"    Expected: {server_path}")
        print(f"\n  Install the voice module:")
        print(f"    pip install fastapi uvicorn sse-starlette anthropic")
        print(f"    Ensure voice-app/sowl_convai_server.py is present.")
        print(f"\n{'='*60}\n")
        return

    print(f"  {GREEN_C}[OK]{RESET_C} Voice server found: {server_path.name}")

    # ---- Step 2: Check if already running ----
    pid = _get_voice_server_pid()
    if pid:
        print(f"  {GREEN_C}[OK]{RESET_C} Voice server already running (PID {pid}) on port 8006")
        server_started = True
    else:
        print(f"  {DIM_C}[..]{RESET_C} Starting voice server on port 8006...")
        try:
            proc = subprocess.Popen(
                [sys.executable, str(server_path)],
                cwd=str(server_path.parent),
                stdout=subprocess.DEVNULL if background else None,
                stderr=subprocess.DEVNULL if background else None,
                start_new_session=True,
            )
            # Give the server a moment to bind the port
            time.sleep(2)

            # Verify it started
            pid = _get_voice_server_pid()
            if pid:
                print(f"  {GREEN_C}[OK]{RESET_C} Voice server started (PID {pid})")
                server_started = True
            else:
                print(f"  {YELLOW_C}[WARN]{RESET_C} Server process launched (PID {proc.pid}) but port 8006 not yet open.")
                print(f"         It may still be initializing. Check in a few seconds.")
                server_started = True  # Optimistic -- it may just be slow
        except Exception as e:
            print(f"  {RED_C}[ERROR]{RESET_C} Failed to start voice server: {e}")
            print(f"\n  Try starting manually:")
            print(f"    cd {server_path.parent}")
            print(f"    python3 {server_path.name}")
            print(f"\n{'='*60}\n")
            return

    # ---- Step 3: Open browser ----
    voice_url = 'http://localhost:8006'
    print(f"  {DIM_C}[..]{RESET_C} Opening voice page in browser...")
    try:
        webbrowser.open(voice_url)
        print(f"  {GREEN_C}[OK]{RESET_C} Browser opened: {voice_url}")
    except Exception as e:
        print(f"  {YELLOW_C}[WARN]{RESET_C} Could not open browser: {e}")
        print(f"         Open manually: {voice_url}")

    # ---- Step 4: Connect to NATS ----
    if nats_publish_path.exists():
        print(f"  {DIM_C}[..]{RESET_C} Publishing to NATS collective...")
        try:
            subprocess.run(
                [sys.executable, str(nats_publish_path),
                 f"VOICE: {owl_name} voice session started on port 8006"],
                capture_output=True, timeout=5,
            )
            print(f"  {GREEN_C}[OK]{RESET_C} NATS notified -- collective is aware")
        except (subprocess.TimeoutExpired, Exception):
            print(f"  {DIM_C}[--]{RESET_C} NATS not available (offline mode -- voice still works)")
    else:
        print(f"  {DIM_C}[--]{RESET_C} NATS publisher not found (offline mode)")

    # ---- Step 5: Show usage ----
    print(f"""
{'='*60}
  {BOLD_C}{MAGENTA}(*) {owl_name.upper()} IS LISTENING{RESET_C}
{'='*60}

  {BOLD_C}Voice Orb:{RESET_C}  {voice_url}
  {BOLD_C}Your Owl:{RESET_C}   {owl_name}
  {BOLD_C}Port:{RESET_C}       8006
  {BOLD_C}Server:{RESET_C}     {server_path.name}

  {BOLD_C}How to use:{RESET_C}
    1. Click the orb to start speaking
    2. {owl_name} listens, thinks, and responds with voice
    3. Say "goodbye" or close the tab to end

  {BOLD_C}Tips:{RESET_C}
    - Speak naturally -- {owl_name} understands context
    - Ask about anything you've learned (weevolve recall)
    - Say "what have I learned about X?" for knowledge recall
    - The orb pulses when {owl_name} is thinking

  {BOLD_C}Stop server:{RESET_C} kill {pid or 'PID'} | or Ctrl+C if foreground

{'='*60}
""")


# ============================================================================
# UPDATE - Version check + changelog + apply
# ============================================================================

# Changelog: each entry is (version, date, list_of_changes)
# Newest first. When bumping __version__, add a new entry at the top.
CHANGELOG = [
    ("0.1.0", "2026-02-13", [
        "Initial release: SEED protocol learning loop",
        "RPG character sheet with XP, levels, skills",
        "Genesis knowledge export/import for bootstrapping",
        "8 owls multi-perspective emergence (weevolve emerge)",
        "Socratic dialogue -- learn by teaching (weevolve teach)",
        "Agent-to-agent knowledge transfer (weevolve connect)",
        "Voice orb -- talk to your owl (weevolve voice)",
        "Watch directory for auto-learning (weevolve watch)",
        "Continuous daemon mode (weevolve daemon)",
        "Claude Code + Cursor installer (weevolve install)",
        "Portable skill.md export (weevolve skill export)",
        "First-time onboarding with full SEED explanation",
        "Update checker with changelog (weevolve update)",
    ]),
]


def _get_current_version() -> str:
    """Get the current installed version."""
    try:
        from weevolve import __version__
        return __version__
    except Exception:
        return "0.1.0"


def _get_last_update_check() -> Optional[str]:
    """Get the timestamp of the last update check."""
    check_file = DATA_DIR / "last_update_check.json"
    if check_file.exists():
        try:
            with open(check_file) as f:
                data = json.load(f)
            return data.get("checked_at")
        except Exception:
            pass
    return None


def _save_update_check(version: str):
    """Save that we checked for updates."""
    check_file = DATA_DIR / "last_update_check.json"
    check_file.parent.mkdir(parents=True, exist_ok=True)
    with open(check_file, 'w') as f:
        json.dump({
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "version": version,
        }, f, indent=2)


def _check_pypi_version() -> Optional[str]:
    """Check PyPI for the latest version. Returns version string or None."""
    if not REQUESTS_AVAILABLE:
        return None
    try:
        resp = requests.get(
            "https://pypi.org/pypi/weevolve/json",
            timeout=5,
            headers={"Accept": "application/json"},
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("info", {}).get("version")
    except Exception:
        pass
    return None


def _parse_version(v: str) -> tuple:
    """Parse a version string like '0.1.0' into a comparable tuple."""
    try:
        parts = v.strip().split(".")
        return tuple(int(p) for p in parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _track_feature_used(feature: str):
    """Track that a feature was used (for highlighting NEW features)."""
    onboarding_file = DATA_DIR / "onboarding.json"
    try:
        if onboarding_file.exists():
            with open(onboarding_file) as f:
                data = json.load(f)
        else:
            data = {}
        used = set(data.get("features_used", []))
        used.add(feature)
        data = {**data, "features_used": sorted(used)}
        with open(onboarding_file, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def run_update():
    """
    Check for updates, show what is new, and offer to apply.

    1. Detect/setup owl identity if not yet configured
    2. Show current version
    3. Check PyPI for latest (non-blocking, timeout 5s)
    4. Show changelog entries since current version
    5. If update available, show pip install command
    6. Show new capabilities added
    7. Announce voice companion + set up bidirectional awareness
    """
    # --- Owl identity + voice setup ---
    from weevolve.onboarding import run_owl_setup
    owl_name = run_owl_setup(is_update=True)

    current = _get_current_version()
    current_tuple = _parse_version(current)

    print(f"\n{'='*60}")
    print(f"  (C) WeEvolve UPDATE CHECK")
    print(f"{'='*60}\n")

    print(f"  {BOLD_C}Current version:{RESET_C} {LIME_C}{current}{RESET_C}")

    # Check PyPI
    print(f"  {DIM_C}Checking for updates...{RESET_C}")
    latest = _check_pypi_version()

    if latest:
        latest_tuple = _parse_version(latest)
        _save_update_check(latest)

        if latest_tuple > current_tuple:
            print(f"  {LIME_C}New version available:{RESET_C} {BOLD_C}{latest}{RESET_C}")
            print()
            print(f"  {BOLD_C}To update:{RESET_C}")
            print(f"    pip install --upgrade weevolve")
            print()
        elif latest_tuple == current_tuple:
            print(f"  {GREEN_C}You are on the latest version.{RESET_C}")
            print()
        else:
            print(f"  {CYAN}You are ahead of PyPI ({latest}).{RESET_C} Running from source.")
            print()
    else:
        print(f"  {DIM_C}Could not reach PyPI. Showing local changelog.{RESET_C}")
        print()

    # Show changelog
    print(f"  {BOLD_C}CHANGELOG{RESET_C}")
    print(f"  {'='*46}")

    entries_shown = 0
    for version, date, changes in CHANGELOG:
        version_tuple = _parse_version(version)
        # Show all entries at or above current version, or the 3 most recent
        if version_tuple >= current_tuple or entries_shown < 3:
            is_current = (version_tuple == current_tuple)
            tag = f" {GREEN_C}(installed){RESET_C}" if is_current else ""
            is_new = version_tuple > current_tuple
            new_tag = f" {LIME_C}NEW{RESET_C}" if is_new else ""
            print(f"\n  {BOLD_C}v{version}{RESET_C} ({date}){tag}{new_tag}")
            for change in changes:
                marker = f"{LIME_C}+{RESET_C}" if is_new else f"{DIM_C}-{RESET_C}"
                print(f"    {marker} {change}")
            entries_shown += 1

    print()

    # Show evolution stats alongside
    state = load_evolution_state()
    db = init_db()
    total_atoms = db.execute("SELECT COUNT(*) FROM knowledge_atoms").fetchone()[0]

    print(f"  {BOLD_C}YOUR EVOLUTION{RESET_C}")
    print(f"  {'='*46}")
    print(f"  Level {state['level']} | {total_atoms} atoms | "
          f"{state.get('total_learnings', 0)} learnings | "
          f"{state.get('total_alpha', 0)} alpha")

    # Show new capabilities they might not know about
    onboarding_file = DATA_DIR / "onboarding.json"
    features_used = set()
    if onboarding_file.exists():
        try:
            with open(onboarding_file) as f:
                ob_data = json.load(f)
            features_used = set(ob_data.get("features_used", []))
        except Exception:
            pass

    unused_highlights = []
    highlight_commands = [
        ("teach", "weevolve teach", "Socratic dialogue -- learn by teaching"),
        ("emerge", "weevolve emerge <task>", "8 owls multi-perspective analysis"),
        ("evolve", "weevolve evolve", "Self-evolution + quest generation"),
        ("connect", "weevolve connect serve", "Share knowledge with other agents"),
        ("skill", "weevolve skill export", "Generate portable skill.md"),
    ]

    for feature_key, cmd, desc in highlight_commands:
        if feature_key not in features_used:
            unused_highlights.append((cmd, desc))

    if unused_highlights:
        print(f"\n  {BOLD_C}FEATURES YOU HAVE NOT TRIED YET{RESET_C}")
        print(f"  {'='*46}")
        for cmd, desc in unused_highlights[:5]:
            print(f"    {LIME_C}NEW{RESET_C} {BOLD_C}{cmd}{RESET_C}")
            print(f"         {DIM_C}{desc}{RESET_C}")

    # Track that they used update
    _track_feature_used("update")

    # Confidence close
    print(f"\n{'='*60}")
    print(f"  {LIME_C}{BOLD_C}Your agent is always up to date. Always evolving.{RESET_C}")
    print(f"{'='*60}\n")


# ============================================================================
# CLI
# ============================================================================

def _boot_nats_collective():
    """
    Boot the NATS collective connection. Non-blocking.
    If NATS is unavailable or nats-py not installed, WeEvolve works offline.
    """
    if not NATS_AVAILABLE:
        return

    try:
        state = load_evolution_state()
        owl_name = _get_owl_name().upper().replace(' ', '_')
        collective = nats_try_connect(
            owl_name=owl_name,
            level=state.get('level', 1),
            atoms=state.get('total_learnings', 0),
        )
        # Register the handler that auto-ingests learnings from the collective
        collective.on_learning(ingest_collective_learning)

        if collective.connected:
            print(f"  {GREEN_C}[NATS]{RESET_C} Connected to collective as {owl_name}")
        else:
            print(f"  {DIM_C}[NATS]{RESET_C} Offline mode (NATS not reachable)")
    except Exception:
        pass


def main():
    # First-run detection: if no onboarding.json, run onboarding
    from weevolve.onboarding import is_first_run, run_onboarding
    if is_first_run():
        run_onboarding()
        return

    # Boot NATS collective (non-blocking, silent on failure)
    _boot_nats_collective()

    if len(sys.argv) < 2:
        # No args = show status dashboard (not help text)
        show_status()
        return

    cmd = sys.argv[1].lower()

    if cmd == 'learn':
        if '--text' in sys.argv:
            idx = sys.argv.index('--text')
            text = ' '.join(sys.argv[idx + 1:])
            learn(text, source_type='text')
        elif '--file' in sys.argv:
            idx = sys.argv.index('--file')
            learn(sys.argv[idx + 1], source_type='file')
        elif len(sys.argv) > 2:
            learn(sys.argv[2])
        else:
            print("Usage: python core.py learn <url|--text 'content'|--file path>")

    elif cmd == 'scan':
        scan_bookmarks()

    elif cmd == 'status':
        show_status()

    elif cmd == 'recall':
        query = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else ''
        if not query:
            print("Usage: python core.py recall <query>")
            return
        recall_display(query)

    elif cmd == 'watch':
        from weevolve.watcher import run_watcher
        interval = 10
        if '--interval' in sys.argv:
            idx = sys.argv.index('--interval')
            if len(sys.argv) > idx + 1:
                try:
                    interval = int(sys.argv[idx + 1])
                except ValueError:
                    pass
        elif len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
            except ValueError:
                pass
        run_watcher(interval)

    elif cmd == 'daemon':
        interval = 300
        if len(sys.argv) > 2:
            try:
                interval = int(sys.argv[2])
            except ValueError:
                pass
        run_daemon(interval)

    elif cmd == 'evolve':
        run_evolve()

    elif cmd == 'genesis':
        subcmd = sys.argv[2].lower() if len(sys.argv) > 2 else 'stats'

        if subcmd == 'export':
            tier = 'full'
            if '--curated' in sys.argv:
                tier = 'curated'
            # Get output path: first positional arg after 'export' that isn't a flag
            positional_args = [a for a in sys.argv[3:] if not a.startswith('--')]
            output = positional_args[0] if positional_args else None
            genesis_export(output_path=output, tier=tier)

        elif subcmd == 'import':
            if len(sys.argv) < 4:
                print("Usage: python core.py genesis import <path-to-genesis.db>")
                return
            genesis_import(sys.argv[3])

        elif subcmd == 'stats':
            db_path = sys.argv[3] if len(sys.argv) > 3 else None
            genesis_stats(db_path)

        elif subcmd == 'top':
            limit = 10
            if len(sys.argv) > 3:
                try:
                    limit = int(sys.argv[3])
                except ValueError:
                    pass
            genesis_top_learnings(limit)

        else:
            print("Usage:")
            print("  python core.py genesis export [path] [--curated]")
            print("  python core.py genesis import <path>")
            print("  python core.py genesis stats [path]")
            print("  python core.py genesis top [limit]")

    elif cmd == 'chat':
        from weevolve.license import check_feature, show_upgrade_prompt
        if not check_feature("chat"):
            show_upgrade_prompt("chat")
            return
        try:
            from weevolve.conversational import start_conversation
            start_conversation()
        except ImportError:
            print("  Voice chat requires: pip install weevolve[chat]")
            print("  Install: pip install websockets pyaudio elevenlabs")
        except Exception as e:
            print(f"  Chat error: {e}")

    elif cmd == 'companion':
        from weevolve.license import check_feature, show_upgrade_prompt
        if not check_feature("companion"):
            show_upgrade_prompt("companion")
            return
        try:
            from weevolve.companion import launch_companion
            launch_companion()
        except ImportError:
            print("  Companion requires a web browser. Opening status instead.")
            show_status()
        except Exception as e:
            print(f"  Companion error: {e}")

    elif cmd == 'voice':
        bg = '--background' in sys.argv or '--bg' in sys.argv
        run_voice(background=bg)

    elif cmd == 'activate':
        from weevolve.license import activate_license
        if len(sys.argv) < 3:
            print("Usage: weevolve activate <license-key> [email]")
            return
        key = sys.argv[2]
        email = sys.argv[3] if len(sys.argv) > 3 else ""
        activate_license(key, email)

    elif cmd == 'teach':
        from weevolve.teacher import run_teach
        topic = ' '.join(sys.argv[2:]) if len(sys.argv) > 2 else None
        run_teach(topic)

    elif cmd == 'quest':
        state = load_evolution_state()
        print(f"\n  Active Quests: {len(state.get('quests', []))}")
        for q in state.get('quests', []):
            print(f"  - {q.get('name', 'Unknown')}: {q.get('status', 'active')}")
        print(f"\n  Next quest: Learn from 10 bookmarks (scan command)")

    elif cmd == 'skill':
        from weevolve.skill_export import export_skill, list_exportable_topics
        subcmd = sys.argv[2] if len(sys.argv) > 2 else 'list'
        if subcmd == 'export':
            topic = None
            output = None
            if '--topic' in sys.argv:
                idx = sys.argv.index('--topic')
                topic = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
            if '--output' in sys.argv:
                idx = sys.argv.index('--output')
                output = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else None
            export_skill(topic=topic, output_path=output)
        elif subcmd == 'list':
            list_exportable_topics()
        else:
            print("Usage:")
            print("  weevolve skill list              # Show exportable topics")
            print("  weevolve skill export             # Export all knowledge")
            print("  weevolve skill export --topic ai  # Export specific topic")

    elif cmd == 'emerge':
        from weevolve.owls.emergence import emerge as owl_emerge, quick_emerge as owl_quick_emerge
        from weevolve.owls.synthesis import persist_learnings

        is_quick = '--quick' in sys.argv
        # Collect task from remaining args (skip flags)
        task_parts = [a for a in sys.argv[2:] if not a.startswith('--')]
        task_text = ' '.join(task_parts) if task_parts else ''

        if not task_text:
            print("Usage:")
            print("  weevolve emerge <task>           Full 8 owls emergence")
            print("  weevolve emerge --quick <task>   Quick 3 owls (LYRA + SAGE + QUEST)")
            print()
            print("Examples:")
            print("  weevolve emerge 'Should we rewrite the auth module?'")
            print("  weevolve emerge --quick 'Is this API design sound?'")
            return

        if is_quick:
            result = owl_quick_emerge(task_text)
        else:
            result = owl_emerge(task_text)

        # Persist any extracted learnings
        learnings = result.get('synthesis', {}).get('learnings', [])
        if learnings:
            persist_learnings(learnings)

        # Save emergence event to log
        EVOLUTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(EVOLUTION_LOG_PATH, 'a') as f:
            f.write(json.dumps({
                'event': 'emergence',
                'task': task_text,
                'mode': 'quick' if is_quick else 'full',
                'owls_succeeded': result.get('meta', {}).get('owls_succeeded', 0),
                'cost_usd': result.get('meta', {}).get('estimated_cost_usd', 0),
                'learnings_count': len(learnings),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }) + '\n')

    elif cmd == 'connect':
        from weevolve.connect import run_connect
        run_connect(sys.argv[2:])

    elif cmd == 'update':
        run_update()

    elif cmd == 'install':
        from weevolve.install import run_install
        run_install(sys.argv[2:])

    else:
        print(f"""
WeEvolve - Self-Evolving Conscious Agent
=========================================

Commands:
  weevolve                  First run: onboarding. After: status dashboard
  weevolve status           Show MMORPG evolution dashboard
  weevolve update           Check for updates + see what is new
  weevolve learn <url>      Learn from a URL
  weevolve learn --text "x" Learn from text
  weevolve learn --file p   Learn from a file
  weevolve scan             Process new bookmarks
  weevolve recall <query>   Search what you've learned
  weevolve teach            Socratic dialogue -- learn by teaching
  weevolve teach <topic>    Teach about a specific topic
  weevolve voice            Start voice orb -- talk to your owl
  weevolve voice --bg       Start voice server in background
  weevolve chat             Voice conversation with your owl (Pro)
  weevolve companion        Open 3D owl companion in browser (Pro)
  weevolve watch            Watch directory for new content to learn
  weevolve daemon           Run as continuous learning daemon
  weevolve evolve           Self-evolution analysis + quest generation
  weevolve emerge <task>    Full 8 owls multi-perspective emergence
  weevolve emerge --quick   Quick 3 owls (LYRA + SAGE + QUEST)
  weevolve skill list       Show exportable knowledge topics
  weevolve skill export     Generate portable skill.md
  weevolve connect export   Export knowledge for sharing
  weevolve connect serve    Start knowledge sharing server
  weevolve connect pull <u> Pull knowledge from remote agent
  weevolve genesis stats    Show genesis database stats
  weevolve genesis top      Show top learnings
  weevolve install --claude-code  Install as Claude Code skill + hooks
  weevolve install --cursor       Install as Cursor rules
  weevolve install --all          Install for all platforms
  weevolve activate <key>         Activate Pro license
""")


if __name__ == '__main__':
    main()
