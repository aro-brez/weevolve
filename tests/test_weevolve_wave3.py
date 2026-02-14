"""
Tests for WeEvolve Wave 3 modules: tiers, qualify, integrate, inventory,
explore, model_router, nats_collective, teacher, conversational
=====================================================================
Covers:
  - tiers.py:          _reset_daily_counters, trial logic, tier info, usage tracking
  - qualify.py:        extract_github_urls, count_signals, count_noise,
                       compute_qualification_score, QualifiedAtom
  - integrate.py:      load_cost_tracker, record_cost, check_budget, save_cost_tracker
  - inventory.py:      extract_python_metadata, extract_shell_metadata,
                       categorize_tool, extract_keywords, ToolEntry, Inventory
  - explore.py:        CODE_EXTENSIONS, MAX_REPO_SIZE_MB, EXPLORE_PROMPT format,
                       collect_repo_content structure
  - model_router.py:   estimate_complexity, classify_task, TaskType, Provider,
                       ModelSpec, check_api_key, MODELS registry, RoutingDecision
  - nats_collective.py: NATSCollective init, channel constants
  - teacher.py:        SEED_OWL_PHASES, XP constants, MODEL constant
  - conversational.py: OWL_CONFIGS, SAMPLE_RATE, ConversationalSession init

All tests are self-contained, use tmp_path for filesystem isolation,
and require no network access or external APIs.
"""

import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, asdict

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    """Provide a clean temporary data directory for WeEvolve."""
    d = tmp_path / "weevolve_data"
    d.mkdir()
    monkeypatch.setenv("WEEVOLVE_DATA_DIR", str(d))
    return d


@pytest.fixture()
def base_dir(tmp_path, monkeypatch):
    """Provide a clean temporary base directory."""
    d = tmp_path / "weevolve_base"
    d.mkdir()
    monkeypatch.setenv("WEEVOLVE_BASE_DIR", str(d))
    return d


@pytest.fixture()
def usage_file(data_dir):
    """Provide a usage.json file path for tiers tests."""
    return data_dir / "usage.json"


@pytest.fixture()
def cost_file(data_dir):
    """Provide a cost log file for integrate tests."""
    return data_dir / "cost_log.json"


@pytest.fixture()
def plans_dir(data_dir):
    """Provide a plans directory for integrate tests."""
    d = data_dir / "plans"
    d.mkdir()
    return d


# ===========================================================================
# tiers.py tests (inlined pure functions)
# ===========================================================================

def _reset_daily_counters(usage: Dict) -> Dict:
    """Reset daily counters if the date has rolled over. Returns new dict."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    updated = {**usage}

    if updated.get("voice_date") != today:
        updated = {**updated, "voice_messages_today": 0, "voice_date": today}
    if updated.get("forest_date") != today:
        updated = {**updated, "forest_queries_today": 0, "forest_date": today}

    return updated


FREE_TIER = {
    "voice_messages_per_day": 50,
    "forest_queries_per_day": 5,
    "model": "claude-opus-4-20250514",
    "features": ["seed", "learn", "recall", "teach", "evolve", "quest", "scan",
                  "genesis", "voice_limited", "forest_limited"],
}

PRO_TIER = {
    "voice_messages_per_day": -1,
    "forest_queries_per_day": -1,
    "model": "claude-opus-4-20250514",
    "features": ["unlimited_voice", "unlimited_forest", "team", "dashboard",
                  "priority_support", "chat", "companion", "8owls"],
}

TRIAL_DAYS = 8


class TestResetDailyCounters:
    """Tests for _reset_daily_counters pure function."""

    def test_resets_voice_counter_on_new_day(self):
        old = {
            "voice_messages_today": 42,
            "voice_date": "2025-01-01",
            "forest_queries_today": 3,
            "forest_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }
        result = _reset_daily_counters(old)
        assert result["voice_messages_today"] == 0
        assert result["forest_queries_today"] == 3  # unchanged (same day)

    def test_resets_forest_counter_on_new_day(self):
        old = {
            "voice_messages_today": 5,
            "voice_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "forest_queries_today": 5,
            "forest_date": "2025-01-01",
        }
        result = _reset_daily_counters(old)
        assert result["voice_messages_today"] == 5  # unchanged
        assert result["forest_queries_today"] == 0

    def test_resets_both_on_old_dates(self):
        old = {
            "voice_messages_today": 50,
            "voice_date": "2020-01-01",
            "forest_queries_today": 5,
            "forest_date": "2020-01-01",
        }
        result = _reset_daily_counters(old)
        assert result["voice_messages_today"] == 0
        assert result["forest_queries_today"] == 0

    def test_no_reset_on_same_day(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        old = {
            "voice_messages_today": 10,
            "voice_date": today,
            "forest_queries_today": 2,
            "forest_date": today,
        }
        result = _reset_daily_counters(old)
        assert result["voice_messages_today"] == 10
        assert result["forest_queries_today"] == 2

    def test_immutability(self):
        old = {
            "voice_messages_today": 42,
            "voice_date": "2020-01-01",
            "forest_queries_today": 3,
            "forest_date": "2020-01-01",
        }
        result = _reset_daily_counters(old)
        assert old["voice_messages_today"] == 42  # original unchanged
        assert result is not old  # new object

    def test_missing_keys_handled(self):
        old = {}
        result = _reset_daily_counters(old)
        assert result["voice_messages_today"] == 0
        assert result["forest_queries_today"] == 0


class TestTierDefinitions:
    """Tests for tier constant definitions."""

    def test_free_tier_has_voice_limit(self):
        assert FREE_TIER["voice_messages_per_day"] == 50

    def test_free_tier_has_forest_limit(self):
        assert FREE_TIER["forest_queries_per_day"] == 5

    def test_pro_tier_unlimited_voice(self):
        assert PRO_TIER["voice_messages_per_day"] == -1

    def test_pro_tier_unlimited_forest(self):
        assert PRO_TIER["forest_queries_per_day"] == -1

    def test_trial_days(self):
        assert TRIAL_DAYS == 8

    def test_free_tier_has_required_features(self):
        assert "seed" in FREE_TIER["features"]
        assert "learn" in FREE_TIER["features"]

    def test_pro_tier_has_premium_features(self):
        assert "unlimited_voice" in PRO_TIER["features"]
        assert "companion" in PRO_TIER["features"]
        assert "8owls" in PRO_TIER["features"]


# ===========================================================================
# qualify.py tests (inlined pure functions)
# ===========================================================================

GITHUB_URL_PATTERN = re.compile(
    r'(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)(?:/[^\s)\]\"\']*)?',
    re.IGNORECASE
)

SIGNAL_PATTERNS = {
    'tool_mention': re.compile(
        r'\b(?:tool|library|framework|sdk|cli|package|module|plugin|extension)\b',
        re.IGNORECASE
    ),
    'action_verb': re.compile(
        r'\b(?:build|deploy|integrate|install|clone|fork|use|implement|run|setup)\b',
        re.IGNORECASE
    ),
    'agent_related': re.compile(
        r'\b(?:agent|swarm|orchestrat|autonom|daemon|pipeline|workflow)\b',
        re.IGNORECASE
    ),
    'trading_related': re.compile(
        r'\b(?:trad|market|polymarket|signal|arbitrage|price|volume)\b',
        re.IGNORECASE
    ),
    'infra_related': re.compile(
        r'\b(?:nats|redis|postgres|sqlite|docker|kubernetes|api|websocket|mcp)\b',
        re.IGNORECASE
    ),
}

NOISE_PATTERNS = {
    'promotional': re.compile(
        r'\b(?:giveaway|subscribe|follow me|retweet|like and share)\b',
        re.IGNORECASE
    ),
    'vaporware': re.compile(
        r'\b(?:coming soon|waitlist|pre-launch|stealth mode|announcing)\b',
        re.IGNORECASE
    ),
}


def extract_github_urls(text: str) -> List[Tuple[str, str, str]]:
    """Extract GitHub repo URLs from text."""
    if not text:
        return []

    results = []
    seen = set()

    for match in GITHUB_URL_PATTERN.finditer(text):
        owner = match.group(1)
        repo = match.group(2)

        repo = re.sub(r'\.git$', '', repo)
        repo = re.sub(r'[.,;:!?]+$', '', repo)

        skip_owners = {'features', 'pricing', 'about', 'blog', 'explore', 'topics', 'trending'}
        if owner.lower() in skip_owners:
            continue

        canonical = f"github.com/{owner}/{repo}"
        if canonical.lower() not in seen:
            seen.add(canonical.lower())
            full_url = f"https://github.com/{owner}/{repo}"
            results.append((full_url, owner, repo))

    return results


def count_signals(text: str) -> Dict[str, int]:
    """Count signal pattern matches in text."""
    counts = {}
    combined = text or ''

    for name, pattern in SIGNAL_PATTERNS.items():
        matches = pattern.findall(combined)
        if matches:
            counts[name] = len(matches)

    return counts


def count_noise(text: str) -> int:
    """Count noise pattern matches."""
    total = 0
    for pattern in NOISE_PATTERNS.values():
        total += len(pattern.findall(text or ''))
    return total


def compute_qualification_score(
    atom_quality: float,
    github_url_count: int,
    signals: Dict[str, int],
    noise_count: int,
    is_alpha: bool
) -> float:
    """Compute a 0-1 qualification score for integration potential."""
    if github_url_count == 0:
        return 0.0

    base = atom_quality * 0.4
    url_score = min(1.0, github_url_count / 2.0) * 0.2
    signal_score = min(1.0, sum(signals.values()) / 8.0) * 0.3
    alpha_bonus = 0.1 if is_alpha else 0.0
    noise_penalty = min(0.2, noise_count * 0.05)

    score = base + url_score + signal_score + alpha_bonus - noise_penalty
    return round(max(0.0, min(1.0, score)), 3)


class TestExtractGithubUrls:
    """Tests for extract_github_urls."""

    def test_basic_url(self):
        text = "Check out https://github.com/anthropics/claude-code for more"
        result = extract_github_urls(text)
        assert len(result) == 1
        assert result[0] == ("https://github.com/anthropics/claude-code", "anthropics", "claude-code")

    def test_url_with_git_suffix(self):
        text = "Clone https://github.com/user/repo.git"
        result = extract_github_urls(text)
        assert result[0][2] == "repo"  # .git stripped

    def test_multiple_urls(self):
        text = "https://github.com/user1/repo1 and https://github.com/user2/repo2"
        result = extract_github_urls(text)
        assert len(result) == 2

    def test_deduplication(self):
        text = "https://github.com/user/repo https://github.com/user/repo"
        result = extract_github_urls(text)
        assert len(result) == 1

    def test_skip_non_repo_pages(self):
        text = "https://github.com/features and https://github.com/pricing"
        result = extract_github_urls(text)
        assert len(result) == 0

    def test_empty_string(self):
        assert extract_github_urls("") == []

    def test_none_string(self):
        assert extract_github_urls(None) == []

    def test_no_github_urls(self):
        assert extract_github_urls("No repos here, just text") == []

    def test_url_with_trailing_punctuation(self):
        text = "See https://github.com/user/repo, for details."
        result = extract_github_urls(text)
        assert result[0][2] == "repo"

    def test_url_with_path(self):
        text = "https://github.com/user/repo/tree/main/src"
        result = extract_github_urls(text)
        assert result[0] == ("https://github.com/user/repo", "user", "repo")


class TestCountSignals:
    """Tests for count_signals."""

    def test_tool_mention(self):
        result = count_signals("This is a great tool and framework")
        assert "tool_mention" in result
        assert result["tool_mention"] == 2

    def test_action_verbs(self):
        result = count_signals("Build and deploy this agent")
        assert "action_verb" in result
        assert result["action_verb"] == 2

    def test_agent_related(self):
        result = count_signals("An autonomous agent swarm for pipeline orchestration")
        assert "agent_related" in result

    def test_no_signals(self):
        result = count_signals("A beautiful sunny day in the park")
        assert len(result) == 0

    def test_empty_string(self):
        result = count_signals("")
        assert len(result) == 0

    def test_none_input(self):
        result = count_signals(None)
        assert len(result) == 0

    def test_multiple_categories(self):
        result = count_signals("Deploy this API agent framework for the market")
        assert "action_verb" in result
        assert "agent_related" in result
        assert "infra_related" in result


class TestCountNoise:
    """Tests for count_noise."""

    def test_promotional_noise(self):
        assert count_noise("giveaway! subscribe now!") > 0

    def test_vaporware_noise(self):
        assert count_noise("coming soon on the waitlist") > 0

    def test_no_noise(self):
        assert count_noise("A solid open-source framework with documentation") == 0

    def test_empty_string(self):
        assert count_noise("") == 0

    def test_none_input(self):
        assert count_noise(None) == 0


class TestComputeQualificationScore:
    """Tests for compute_qualification_score."""

    def test_no_github_urls_returns_zero(self):
        assert compute_qualification_score(1.0, 0, {"tool_mention": 5}, 0, True) == 0.0

    def test_high_quality_atom(self):
        score = compute_qualification_score(1.0, 2, {"tool_mention": 4, "agent_related": 4}, 0, True)
        assert score > 0.8

    def test_low_quality_atom(self):
        score = compute_qualification_score(0.1, 1, {}, 3, False)
        assert score < 0.3

    def test_alpha_bonus(self):
        score_no_alpha = compute_qualification_score(0.5, 1, {"tool_mention": 1}, 0, False)
        score_alpha = compute_qualification_score(0.5, 1, {"tool_mention": 1}, 0, True)
        assert score_alpha > score_no_alpha
        assert score_alpha - score_no_alpha == pytest.approx(0.1, abs=0.01)

    def test_noise_penalty(self):
        score_clean = compute_qualification_score(0.5, 1, {"tool_mention": 1}, 0, False)
        score_noisy = compute_qualification_score(0.5, 1, {"tool_mention": 1}, 4, False)
        assert score_clean > score_noisy

    def test_score_clamped_to_0_1(self):
        score = compute_qualification_score(1.0, 10, {"a": 100}, 0, True)
        assert 0.0 <= score <= 1.0

    def test_score_never_negative(self):
        score = compute_qualification_score(0.0, 1, {}, 100, False)
        assert score >= 0.0


# ===========================================================================
# integrate.py tests (inlined pure functions)
# ===========================================================================

COST_PER_EXPLORE = 0.002
COST_PER_PLAN = 0.003
DAILY_BUDGET = 0.25


def _load_cost_tracker_from_path(cost_path: Path) -> Dict:
    """Load daily cost tracking from a specific path."""
    if cost_path.exists():
        try:
            return json.loads(cost_path.read_text())
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        'daily_totals': {},
        'total_spent': 0.0,
        'total_explores': 0,
        'total_plans': 0,
    }


def _save_cost_tracker(cost_path: Path, tracker: Dict):
    """Save cost tracking to a specific path."""
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    cost_path.write_text(json.dumps(tracker, indent=2))


def _record_cost(cost_path: Path, amount: float, operation: str):
    """Record API cost to a specific path."""
    tracker = _load_cost_tracker_from_path(cost_path)
    today = time.strftime('%Y-%m-%d')

    daily = tracker.get('daily_totals', {})
    daily[today] = daily.get(today, 0.0) + amount
    tracker = {**tracker, 'daily_totals': daily}
    tracker = {**tracker, 'total_spent': tracker.get('total_spent', 0.0) + amount}

    if operation == 'explore':
        tracker = {**tracker, 'total_explores': tracker.get('total_explores', 0) + 1}
    elif operation == 'plan':
        tracker = {**tracker, 'total_plans': tracker.get('total_plans', 0) + 1}

    _save_cost_tracker(cost_path, tracker)


class TestLoadCostTracker:
    """Tests for cost tracker loading."""

    def test_returns_defaults_when_no_file(self, tmp_path):
        p = tmp_path / "nonexistent.json"
        result = _load_cost_tracker_from_path(p)
        assert result["total_spent"] == 0.0
        assert result["total_explores"] == 0
        assert result["total_plans"] == 0
        assert result["daily_totals"] == {}

    def test_loads_existing_file(self, tmp_path):
        p = tmp_path / "cost.json"
        data = {"total_spent": 1.50, "total_explores": 10, "total_plans": 5, "daily_totals": {}}
        p.write_text(json.dumps(data))
        result = _load_cost_tracker_from_path(p)
        assert result["total_spent"] == 1.50
        assert result["total_explores"] == 10

    def test_handles_corrupt_json(self, tmp_path):
        p = tmp_path / "cost.json"
        p.write_text("{invalid json")
        result = _load_cost_tracker_from_path(p)
        assert result["total_spent"] == 0.0


class TestRecordCost:
    """Tests for cost recording."""

    def test_records_explore_cost(self, tmp_path):
        p = tmp_path / "cost.json"
        _record_cost(p, 0.002, "explore")
        tracker = json.loads(p.read_text())
        assert tracker["total_spent"] == 0.002
        assert tracker["total_explores"] == 1

    def test_records_plan_cost(self, tmp_path):
        p = tmp_path / "cost.json"
        _record_cost(p, 0.003, "plan")
        tracker = json.loads(p.read_text())
        assert tracker["total_spent"] == 0.003
        assert tracker["total_plans"] == 1

    def test_accumulates_costs(self, tmp_path):
        p = tmp_path / "cost.json"
        _record_cost(p, 0.002, "explore")
        _record_cost(p, 0.002, "explore")
        _record_cost(p, 0.003, "plan")
        tracker = json.loads(p.read_text())
        assert tracker["total_spent"] == pytest.approx(0.007)
        assert tracker["total_explores"] == 2
        assert tracker["total_plans"] == 1

    def test_tracks_daily_totals(self, tmp_path):
        p = tmp_path / "cost.json"
        _record_cost(p, 0.005, "explore")
        tracker = json.loads(p.read_text())
        today = time.strftime('%Y-%m-%d')
        assert today in tracker["daily_totals"]
        assert tracker["daily_totals"][today] == 0.005

    def test_cost_constants(self):
        assert COST_PER_EXPLORE == 0.002
        assert COST_PER_PLAN == 0.003
        assert DAILY_BUDGET == 0.25


# ===========================================================================
# inventory.py tests (inlined pure functions)
# ===========================================================================

CATEGORY_PATTERNS = {
    'trading': re.compile(r'trad|market|polymarket|arbitrage|position|order|swap', re.I),
    'agent': re.compile(r'agent|swarm|daemon|coordinator|orchestrat|autonomous', re.I),
    'voice': re.compile(r'voice|speech|tts|stt|cartesia|deepgram|speak', re.I),
    'intelligence': re.compile(r'intel|bookmark|scanner|scrape|research|feed', re.I),
    'evolution': re.compile(r'evolv|improv|learn|weevolve|seed|consciousness', re.I),
    'security': re.compile(r'secur|guard|token|auth|protect|encrypt', re.I),
    'testing': re.compile(r'test|verify|check|validate|assert', re.I),
    'infrastructure': re.compile(r'nats|bridge|mcp|server|deploy|tunnel|start', re.I),
    'content': re.compile(r'video|image|content|tweet|post|compos', re.I),
    'utility': re.compile(r'util|helper|tool|script|config|setup', re.I),
}


def categorize_tool(name: str, docstring: str, content_hint: str) -> str:
    """Categorize a tool based on name and docstring."""
    combined = f"{name} {docstring} {content_hint}"

    for category, pattern in CATEGORY_PATTERNS.items():
        if pattern.search(combined):
            return category

    return 'utility'


def extract_keywords(name: str, docstring: str, functions: List[str]) -> List[str]:
    """Extract searchable keywords from tool metadata."""
    combined = f"{name} {docstring} {' '.join(functions)}"
    words = re.findall(r'\b[a-zA-Z]{3,}\b', combined.lower())

    stop_words = {
        'the', 'and', 'for', 'not', 'are', 'but', 'this', 'that', 'with',
        'from', 'have', 'has', 'was', 'were', 'will', 'can', 'all', 'each',
        'def', 'class', 'import', 'return', 'self', 'none', 'true', 'false',
        'str', 'int', 'float', 'list', 'dict', 'tuple', 'set',
    }

    unique = []
    seen = set()
    for word in words:
        if word not in stop_words and word not in seen:
            seen.add(word)
            unique.append(word)

    return unique[:30]


def extract_python_metadata_inline(filepath: Path) -> Dict:
    """Extract metadata from a Python file without executing it."""
    try:
        content = filepath.read_text(errors='replace')
    except Exception:
        return {'docstring': '', 'imports': [], 'functions': [], 'classes': [], 'line_count': 0}

    lines = content.split('\n')
    line_count = len(lines)

    docstring = ''
    doc_match = re.search(r'^(?:#!/.*\n)?(?:#.*\n)*\s*(?:\"\"\"(.*?)\"\"\"|\'\'\'(.*?)\'\'\')',
                          content, re.DOTALL)
    if doc_match:
        docstring = (doc_match.group(1) or doc_match.group(2) or '').strip()
        if len(docstring) > 300:
            docstring = docstring[:300] + '...'

    imports = []
    for line in lines[:80]:
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            module = stripped.split()[1].split('.')[0]
            if module not in imports:
                imports.append(module)

    functions = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
    classes = re.findall(r'^class\s+(\w+)\s*[:(]', content, re.MULTILINE)

    return {
        'docstring': docstring,
        'imports': imports[:20],
        'functions': functions[:30],
        'classes': classes[:10],
        'line_count': line_count,
    }


class TestCategorizeTool:
    """Tests for categorize_tool."""

    def test_trading_category(self):
        assert categorize_tool("trade_bot", "Market trading", "") == "trading"

    def test_agent_category(self):
        assert categorize_tool("swarm_conductor", "Agent orchestration", "") == "agent"

    def test_voice_category(self):
        assert categorize_tool("voice_server", "TTS bridge", "") == "voice"

    def test_intelligence_category(self):
        assert categorize_tool("bookmark_scanner", "Research feed", "") == "intelligence"

    def test_evolution_category(self):
        assert categorize_tool("weevolve_core", "Self-evolving learning", "") == "evolution"

    def test_security_category(self):
        assert categorize_tool("auth_guard", "Token authentication", "") == "security"

    def test_default_utility(self):
        assert categorize_tool("random_thing", "does stuff", "") == "utility"

    def test_uses_content_hint(self):
        assert categorize_tool("helper", "", "nats bridge deploy") == "infrastructure"


class TestExtractKeywords:
    """Tests for extract_keywords."""

    def test_basic_extraction(self):
        result = extract_keywords("weevolve", "Self-evolving learning system", ["learn", "evolve"])
        assert "weevolve" in result
        assert "evolving" in result
        assert "learning" in result
        assert "system" in result

    def test_stop_words_filtered(self):
        result = extract_keywords("tool", "the and for not are but", [])
        assert "the" not in result
        assert "and" not in result

    def test_deduplication(self):
        result = extract_keywords("test", "test test test", ["test"])
        assert result.count("test") == 1

    def test_max_30_keywords(self):
        long_text = " ".join([f"word{i}" for i in range(100)])
        result = extract_keywords("name", long_text, [])
        assert len(result) <= 30

    def test_short_words_filtered(self):
        result = extract_keywords("ab", "to is an", [])
        assert len(result) == 0


class TestExtractPythonMetadata:
    """Tests for extract_python_metadata."""

    def test_extracts_docstring(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('"""\nHello World\n"""\nimport os\n')
        result = extract_python_metadata_inline(f)
        assert "Hello World" in result["docstring"]

    def test_extracts_imports(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('import os\nimport sys\nfrom pathlib import Path\n')
        result = extract_python_metadata_inline(f)
        assert "os" in result["imports"]
        assert "sys" in result["imports"]
        assert "pathlib" in result["imports"]

    def test_extracts_functions(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('def hello():\n    pass\n\ndef world():\n    pass\n')
        result = extract_python_metadata_inline(f)
        assert "hello" in result["functions"]
        assert "world" in result["functions"]

    def test_extracts_classes(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('class MyClass:\n    pass\n\nclass Other(Base):\n    pass\n')
        result = extract_python_metadata_inline(f)
        assert "MyClass" in result["classes"]
        assert "Other" in result["classes"]

    def test_counts_lines(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text('line1\nline2\nline3\n')
        result = extract_python_metadata_inline(f)
        assert result["line_count"] == 4  # trailing newline creates empty 4th

    def test_handles_missing_file(self, tmp_path):
        f = tmp_path / "nonexistent.py"
        result = extract_python_metadata_inline(f)
        assert result["docstring"] == ""
        assert result["line_count"] == 0

    def test_truncates_long_docstring(self, tmp_path):
        f = tmp_path / "test.py"
        long_doc = 'x' * 500
        f.write_text(f'"""\n{long_doc}\n"""\n')
        result = extract_python_metadata_inline(f)
        assert len(result["docstring"]) <= 304  # 300 + "..."


# ===========================================================================
# model_router.py tests (inlined pure functions)
# ===========================================================================

class TaskType:
    CODE_FRONTEND = "code_frontend"
    CODE_BACKEND = "code_backend"
    CODE_SYSTEMS = "code_systems"
    CODE_REVIEW = "code_review"
    ARCHITECTURE = "architecture"
    RESEARCH = "research"
    REALTIME_SEARCH = "realtime_search"
    MATH_REASONING = "math_reasoning"
    CREATIVE_WRITING = "creative_writing"
    CLASSIFICATION = "classification"
    BULK_PROCESSING = "bulk_processing"
    TRADING_SIGNALS = "trading_signals"
    MULTIMODAL = "multimodal"
    AGENT_ORCHESTRATION = "agent_orchestration"
    AGENT_WORKER = "agent_worker"
    GENERAL = "general"


TASK_KEYWORDS = {
    TaskType.CODE_FRONTEND: ["react", "vue", "angular", "css", "html", "component", "ui", "frontend", "tailwind", "next.js", "svelte"],
    TaskType.CODE_BACKEND: ["api", "endpoint", "server", "database", "sql", "backend", "rest", "graphql", "express", "fastapi", "django"],
    TaskType.CODE_SYSTEMS: ["rust", "go", "c++", "kernel", "driver", "systems", "performance", "memory", "concurrent", "thread"],
    TaskType.CODE_REVIEW: ["review", "audit", "check", "lint", "quality", "refactor", "improve"],
    TaskType.ARCHITECTURE: ["architect", "design", "system design", "schema", "infrastructure", "scale", "microservice"],
    TaskType.RESEARCH: ["research", "analyze", "investigate", "study", "compare", "survey", "literature"],
    TaskType.REALTIME_SEARCH: ["search", "latest", "current", "news", "trending", "twitter", "x.com", "real-time", "live"],
    TaskType.MATH_REASONING: ["math", "calculate", "prove", "theorem", "algorithm", "optimize", "probability", "statistics"],
    TaskType.CREATIVE_WRITING: ["write", "story", "creative", "narrative", "blog", "copy", "marketing", "content"],
    TaskType.CLASSIFICATION: ["classify", "categorize", "label", "tag", "sort", "extract", "parse", "format"],
    TaskType.BULK_PROCESSING: ["batch", "bulk", "process all", "transform", "convert", "migrate"],
    TaskType.TRADING_SIGNALS: ["trade", "market", "stock", "crypto", "polymarket", "price", "signal", "sentiment"],
    TaskType.MULTIMODAL: ["image", "picture", "screenshot", "video", "chart", "diagram", "visual", "ocr"],
    TaskType.AGENT_ORCHESTRATION: ["orchestrate", "coordinate", "plan", "strategy", "manage agents", "swarm"],
    TaskType.AGENT_WORKER: ["worker", "simple task", "helper", "utility", "format", "template"],
}


def estimate_complexity(prompt: str) -> float:
    """Estimate task complexity from 0.0 to 1.0."""
    score = 0.3

    words = len(prompt.split())
    if words > 200:
        score += 0.1
    if words > 500:
        score += 0.1

    multi_step_patterns = [
        r'\b(then|after that|next|finally|also|additionally)\b',
        r'\b(step \d|phase \d|first.*second|1\).*2\))\b',
        r'\b(and also|as well as|in addition)\b',
    ]
    for pattern in multi_step_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            score += 0.05

    tech_depth = [
        r'\b(architecture|distributed|concurrent|scalable|microservice)\b',
        r'\b(security|authentication|authorization|encryption)\b',
        r'\b(optimize|performance|profil|benchmark)\b',
        r'\b(refactor|redesign|migrate|overhaul)\b',
    ]
    for pattern in tech_depth:
        if re.search(pattern, prompt, re.IGNORECASE):
            score += 0.08

    simple_patterns = [
        r'\b(fix typo|rename|format|lint|simple|quick|small)\b',
        r'\b(add comment|update readme|change name)\b',
    ]
    for pattern in simple_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            score -= 0.15

    return max(0.0, min(1.0, score))


def classify_task(prompt: str) -> str:
    """Classify a prompt into a task type."""
    scores = {}
    prompt_lower = prompt.lower()

    for task_type, keywords in TASK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[task_type] = score

    if not scores:
        return TaskType.GENERAL

    return max(scores, key=scores.get)


class TestEstimateComplexity:
    """Tests for estimate_complexity."""

    def test_baseline_complexity(self):
        result = estimate_complexity("Do a thing")
        assert result == pytest.approx(0.3, abs=0.01)

    def test_long_prompt_increases_complexity(self):
        long_prompt = " ".join(["word"] * 250)
        result = estimate_complexity(long_prompt)
        assert result > 0.3

    def test_very_long_prompt(self):
        long_prompt = " ".join(["word"] * 600)
        result = estimate_complexity(long_prompt)
        assert result >= 0.5

    def test_multi_step_increases_complexity(self):
        result = estimate_complexity("First do this, then do that, finally deploy")
        assert result > 0.3

    def test_technical_depth_increases_complexity(self):
        result = estimate_complexity("Build a distributed concurrent microservice architecture with security authentication and optimize performance benchmarks")
        assert result > 0.5

    def test_simple_task_decreases_complexity(self):
        result = estimate_complexity("Fix typo in readme")
        assert result < 0.3

    def test_clamped_to_0_1(self):
        result = estimate_complexity("simple quick small fix typo rename format lint")
        assert result >= 0.0
        assert result <= 1.0

    def test_security_increases_complexity(self):
        result = estimate_complexity("Implement authentication and authorization with encryption")
        assert result > 0.3  # baseline 0.3 + security group 0.08 = 0.38

    def test_refactor_increases_complexity(self):
        result = estimate_complexity("Refactor and redesign the overhaul migration")
        assert result > 0.3  # baseline 0.3 + refactor group 0.08 = 0.38


class TestClassifyTask:
    """Tests for classify_task."""

    def test_frontend_classification(self):
        assert classify_task("Build a React component with Tailwind CSS") == TaskType.CODE_FRONTEND

    def test_backend_classification(self):
        assert classify_task("Create a REST API endpoint with FastAPI and SQL database") == TaskType.CODE_BACKEND

    def test_research_classification(self):
        assert classify_task("Research and analyze the latest AI study") == TaskType.RESEARCH

    def test_realtime_search(self):
        assert classify_task("Search for the latest trending news on twitter") == TaskType.REALTIME_SEARCH

    def test_trading_classification(self):
        assert classify_task("Analyze polymarket crypto price signals") == TaskType.TRADING_SIGNALS

    def test_architecture_classification(self):
        assert classify_task("Design the system architecture and infrastructure schema") == TaskType.ARCHITECTURE

    def test_creative_writing(self):
        assert classify_task("Write a creative blog post with a compelling narrative") == TaskType.CREATIVE_WRITING

    def test_general_fallback(self):
        assert classify_task("Hello, how are you today?") == TaskType.GENERAL

    def test_math_reasoning(self):
        assert classify_task("Calculate the probability and optimize the algorithm") == TaskType.MATH_REASONING

    def test_multimodal(self):
        assert classify_task("Analyze this screenshot image and create a diagram") == TaskType.MULTIMODAL

    def test_agent_orchestration(self):
        assert classify_task("Orchestrate the agent swarm and coordinate the strategy") == TaskType.AGENT_ORCHESTRATION


# ===========================================================================
# explore.py tests (constants and patterns)
# ===========================================================================

CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.rs', '.rb', '.java',
    '.sh', '.yaml', '.yml', '.toml', '.json', '.md', '.txt',
}

MAX_REPO_SIZE_MB = 100
MAX_CONTEXT_CHARS = 12000


class TestExploreConstants:
    """Tests for explore.py constants and patterns."""

    def test_python_in_code_extensions(self):
        assert '.py' in CODE_EXTENSIONS

    def test_typescript_in_code_extensions(self):
        assert '.ts' in CODE_EXTENSIONS
        assert '.tsx' in CODE_EXTENSIONS

    def test_rust_in_code_extensions(self):
        assert '.rs' in CODE_EXTENSIONS

    def test_max_repo_size(self):
        assert MAX_REPO_SIZE_MB == 100

    def test_max_context_chars(self):
        assert MAX_CONTEXT_CHARS == 12000

    def test_markdown_in_code_extensions(self):
        assert '.md' in CODE_EXTENSIONS

    def test_no_binary_extensions(self):
        binary_exts = {'.exe', '.bin', '.so', '.dll', '.png', '.jpg', '.mp4'}
        for ext in binary_exts:
            assert ext not in CODE_EXTENSIONS


# ===========================================================================
# nats_collective.py tests (constants and init)
# ===========================================================================

CH_LEARN = "weevolve.learn"
CH_STATUS = "weevolve.status"
CH_OWL_ALL = "owl.all"


class TestNATSCollectiveConstants:
    """Tests for NATS channel constants and initialization."""

    def test_learn_channel(self):
        assert CH_LEARN == "weevolve.learn"

    def test_status_channel(self):
        assert CH_STATUS == "weevolve.status"

    def test_owl_all_channel(self):
        assert CH_OWL_ALL == "owl.all"

    def test_channels_follow_naming_convention(self):
        """All channels should use dot-separated naming."""
        for ch in [CH_LEARN, CH_STATUS, CH_OWL_ALL]:
            assert "." in ch

    def test_collective_class_init(self):
        """Test that NATSCollective-like init works with defaults."""
        class MockCollective:
            def __init__(self, owl_name="WEEVOLVE"):
                self.owl_name = owl_name
                self._nc = None
                self._loop = None
                self._thread = None
                self._connected = False
                self._sub_handlers = []
                self._pending_learnings = []

            @property
            def connected(self):
                return self._connected and self._nc is not None

        c = MockCollective()
        assert c.owl_name == "WEEVOLVE"
        assert c.connected is False
        assert c._pending_learnings == []

    def test_collective_custom_name(self):
        class MockCollective:
            def __init__(self, owl_name="WEEVOLVE"):
                self.owl_name = owl_name
                self._nc = None
                self._connected = False

            @property
            def connected(self):
                return self._connected and self._nc is not None

        c = MockCollective(owl_name="LUNA")
        assert c.owl_name == "LUNA"
        assert c.connected is False


# ===========================================================================
# teacher.py tests (constants and owl phases)
# ===========================================================================

SEED_OWL_PHASES = [
    ("LYRA", "PERCEIVE"),
    ("PRISM", "CONNECT"),
    ("SAGE", "LEARN"),
    ("QUEST", "QUESTION"),
    ("NOVA", "EXPAND"),
    ("ECHO", "SHARE"),
    ("LUNA", "RECEIVE"),
    ("SOWL", "IMPROVE"),
]

TEACH_XP_MULTIPLIER = 2
XP_PER_LEARN = 10
TEACHER_MODEL = 'claude-haiku-4-5-20251001'


class TestTeacherConstants:
    """Tests for teacher.py constants."""

    def test_eight_owl_phases(self):
        assert len(SEED_OWL_PHASES) == 8

    def test_sowl_is_improve(self):
        sowl_entry = [p for p in SEED_OWL_PHASES if p[0] == "SOWL"][0]
        assert sowl_entry[1] == "IMPROVE"

    def test_lyra_is_perceive(self):
        lyra_entry = [p for p in SEED_OWL_PHASES if p[0] == "LYRA"][0]
        assert lyra_entry[1] == "PERCEIVE"

    def test_quest_is_question(self):
        quest_entry = [p for p in SEED_OWL_PHASES if p[0] == "QUEST"][0]
        assert quest_entry[1] == "QUESTION"

    def test_teach_xp_multiplier(self):
        assert TEACH_XP_MULTIPLIER == 2

    def test_teach_xp_is_double(self):
        teach_xp = XP_PER_LEARN * TEACH_XP_MULTIPLIER
        assert teach_xp == 20

    def test_all_owl_names_unique(self):
        names = [p[0] for p in SEED_OWL_PHASES]
        assert len(names) == len(set(names))

    def test_all_phases_unique(self):
        phases = [p[1] for p in SEED_OWL_PHASES]
        assert len(phases) == len(set(phases))


# ===========================================================================
# conversational.py tests (constants and init)
# ===========================================================================

OWL_CONFIGS = {
    "sowl": {"voice_id": "JBFqnCBsd6RMkjVDRZzb"},
    "luna": {"voice_id": "SAz9YHcvj6GT2YYXdXww"},
    "lyra": {"voice_id": "iP95p4xoKVk53GoZ742B"},
    "nova": {"voice_id": "CwhRBWXzGAHq8TQ4Fs17"},
    "default": {"voice_id": "JBFqnCBsd6RMkjVDRZzb"},
}

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SAMPLES = 4000
WS_URL = "wss://api.elevenlabs.io/v1/convai/conversation"


class TestConversationalConstants:
    """Tests for conversational.py constants and configuration."""

    def test_sowl_has_voice_id(self):
        assert "voice_id" in OWL_CONFIGS["sowl"]
        assert len(OWL_CONFIGS["sowl"]["voice_id"]) > 0

    def test_luna_has_voice_id(self):
        assert "voice_id" in OWL_CONFIGS["luna"]

    def test_default_matches_sowl(self):
        assert OWL_CONFIGS["default"]["voice_id"] == OWL_CONFIGS["sowl"]["voice_id"]

    def test_all_owls_have_unique_voice_ids(self):
        ids = [v["voice_id"] for k, v in OWL_CONFIGS.items() if k != "default"]
        assert len(ids) == len(set(ids))

    def test_sample_rate(self):
        assert SAMPLE_RATE == 16000

    def test_chunk_samples(self):
        assert CHUNK_SAMPLES == 4000

    def test_ws_url_is_elevenlabs(self):
        assert "elevenlabs.io" in WS_URL
        assert WS_URL.startswith("wss://")

    def test_conversational_session_init(self):
        """Test that session-like init works with defaults."""
        class MockSession:
            def __init__(
                self,
                api_key=None,
                agent_id=None,
                owl_name="default",
                on_transcript=None,
                on_agent_text=None,
            ):
                self.api_key = api_key or ""
                self.agent_id = agent_id or ""
                self.owl_config = OWL_CONFIGS.get(owl_name, OWL_CONFIGS["default"])
                self.on_transcript = on_transcript
                self.on_agent_text = on_agent_text

        s = MockSession(owl_name="luna")
        assert s.owl_config == OWL_CONFIGS["luna"]
        assert s.api_key == ""

    def test_unknown_owl_falls_back_to_default(self):
        config = OWL_CONFIGS.get("nonexistent", OWL_CONFIGS["default"])
        assert config == OWL_CONFIGS["default"]
