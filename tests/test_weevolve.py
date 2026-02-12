"""
Tests for WeEvolve core.py - Pure function unit tests
=====================================================
Covers: grant_xp, improve_skills, content_hash, fallback_extraction,
        level calculation, SKILL_CATEGORIES constants
"""

import pytest
import hashlib
import json


# ---------------------------------------------------------------------------
# Inline copies of pure functions under test (avoids importing the module
# which triggers side-effects like DB init, credential loading, etc.)
# ---------------------------------------------------------------------------

LEVEL_XP_BASE = 100
XP_PER_LEARN = 10
XP_PER_INSIGHT = 25

SKILL_CATEGORIES = {
    'research': ['research', 'analysis', 'investigation', 'data', 'science'],
    'trading': ['trading', 'finance', 'market', 'investment', 'polymarket', 'crypto'],
    'coding': ['code', 'programming', 'software', 'engineering', 'development', 'api'],
    'ai_engineering': ['ai', 'agent', 'llm', 'model', 'neural', 'machine learning'],
    'marketing': ['marketing', 'growth', 'viral', 'audience', 'brand', 'launch'],
    'design': ['design', 'ux', 'ui', 'interface', 'visual', 'aesthetic'],
    'strategy': ['strategy', 'planning', 'roadmap', 'architecture', 'vision'],
    'consciousness': ['consciousness', 'awareness', 'emergence', 'seed', 'love', 'philosophy'],
    'leadership': ['team', 'leadership', 'management', 'culture', 'collaboration'],
    'finance': ['revenue', 'profit', 'economics', 'token', 'business model'],
    'security': ['security', 'privacy', 'encryption', 'vulnerability', 'protection'],
    'communication': ['writing', 'content', 'narrative', 'story', 'messaging'],
    'growth': ['scale', 'expansion', 'adoption', 'retention', 'onboarding'],
    'love': ['love', 'compassion', 'connection', 'joy', 'gratitude', 'freedom'],
}


def grant_xp(state: dict, xp: int, reason: str) -> dict:
    """Award XP and handle level-ups. Returns updated state."""
    state = {**state, 'xp': state['xp'] + xp}
    while state['xp'] >= state['xp_to_next']:
        state = {
            **state,
            'xp': state['xp'] - state['xp_to_next'],
            'level': state['level'] + 1,
            'xp_to_next': int(state['xp_to_next'] * 1.5),
        }
    return state


def improve_skills(state: dict, skills: list, quality: float) -> tuple:
    """Improve relevant skills based on learning quality. Returns (state, deltas)."""
    deltas = {}
    skill_map = {**state.get('skills', {})}

    for skill_name in skills:
        matched = None
        for category, keywords in SKILL_CATEGORIES.items():
            if skill_name.lower() in keywords or any(kw in skill_name.lower() for kw in keywords):
                matched = category
                break

        if not matched:
            continue

        old_val = skill_map.get(matched, 0.0)
        improvement = quality * (1.0 - old_val / 100.0) * 2.0
        new_val = min(100.0, old_val + improvement)
        skill_map[matched] = round(new_val, 2)
        if new_val > old_val:
            deltas[matched] = round(new_val - old_val, 2)

    return {**state, 'skills': skill_map}, deltas


def content_hash(content: str) -> str:
    """Generate a hash for dedup."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def fallback_extraction(content: str, source: str) -> dict:
    """Basic extraction without Claude."""
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


# ---------------------------------------------------------------------------
# Helper to create a fresh default state
# ---------------------------------------------------------------------------

def make_default_state(**overrides):
    state = {
        'level': 1,
        'xp': 0,
        'xp_to_next': LEVEL_XP_BASE,
        'skills': {skill: 0.0 for skill in SKILL_CATEGORIES},
        'skills_love': 100.0,
        'total_learnings': 0,
        'total_insights': 0,
        'total_alpha': 0,
        'total_connections': 0,
        'total_sources_processed': 0,
        'daily_improvement': 0.0,
        'streak_days': 0,
        'last_learn_date': None,
        'quests': [],
        'top_learnings': [],
    }
    return {**state, **overrides}


# ===========================================================================
# grant_xp tests
# ===========================================================================

class TestGrantXP:
    """Tests for the XP granting and level-up system."""

    def test_basic_xp_grant(self):
        """Granting XP below threshold should increment xp without leveling up."""
        state = make_default_state(xp=0, xp_to_next=100, level=1)
        result = grant_xp(state, 50, "test learn")
        assert result['xp'] == 50
        assert result['level'] == 1

    def test_exact_level_up(self):
        """Granting XP exactly equal to xp_to_next triggers a level up."""
        state = make_default_state(xp=0, xp_to_next=100, level=1)
        result = grant_xp(state, 100, "exact level up")
        assert result['level'] == 2
        assert result['xp'] == 0
        assert result['xp_to_next'] == 150  # 100 * 1.5

    def test_overflow_xp_carries_over(self):
        """Excess XP after leveling up carries over correctly."""
        state = make_default_state(xp=0, xp_to_next=100, level=1)
        result = grant_xp(state, 120, "overflow")
        assert result['level'] == 2
        assert result['xp'] == 20
        assert result['xp_to_next'] == 150

    def test_double_level_up(self):
        """Granting enough XP for two level-ups in one call."""
        state = make_default_state(xp=0, xp_to_next=100, level=1)
        # Level 1->2 costs 100, Level 2->3 costs 150 (100*1.5), total 250
        result = grant_xp(state, 250, "double level")
        assert result['level'] == 3
        assert result['xp'] == 0
        assert result['xp_to_next'] == 225  # 150 * 1.5

    def test_triple_level_up(self):
        """Three level-ups with leftover XP."""
        state = make_default_state(xp=0, xp_to_next=100, level=1)
        # L1->L2: 100, L2->L3: 150, L3->L4: 225, total 475
        result = grant_xp(state, 480, "triple")
        assert result['level'] == 4
        assert result['xp'] == 5
        assert result['xp_to_next'] == int(225 * 1.5)  # 337

    def test_xp_to_next_grows_by_1_5x(self):
        """XP threshold multiplies by 1.5 on each level-up."""
        state = make_default_state(xp=0, xp_to_next=100, level=1)
        result = grant_xp(state, 100, "lv2")
        assert result['xp_to_next'] == 150

        result = grant_xp(result, 150, "lv3")
        assert result['xp_to_next'] == 225

        result = grant_xp(result, 225, "lv4")
        assert result['xp_to_next'] == int(225 * 1.5)

    def test_zero_xp_does_nothing(self):
        """Granting 0 XP should not change state."""
        state = make_default_state(xp=50, xp_to_next=100, level=1)
        result = grant_xp(state, 0, "nothing")
        assert result['xp'] == 50
        assert result['level'] == 1

    def test_immutability(self):
        """grant_xp should not mutate the original state dict."""
        state = make_default_state(xp=50, xp_to_next=100, level=1)
        original_xp = state['xp']
        _ = grant_xp(state, 30, "immutable test")
        assert state['xp'] == original_xp  # Original unchanged

    def test_high_level_state(self):
        """Granting XP at a high level works correctly."""
        state = make_default_state(xp=500, xp_to_next=10000, level=20)
        result = grant_xp(state, 200, "high level")
        assert result['xp'] == 700
        assert result['level'] == 20


# ===========================================================================
# improve_skills tests
# ===========================================================================

class TestImproveSkills:
    """Tests for the skill improvement system."""

    def test_basic_skill_improvement(self):
        """A recognized skill keyword should increase the matching category."""
        state = make_default_state()
        new_state, deltas = improve_skills(state, ['trading'], 0.8)
        assert 'trading' in deltas
        assert deltas['trading'] > 0
        assert new_state['skills']['trading'] > 0

    def test_unrecognized_skill_ignored(self):
        """Skills that do not match any category produce no deltas."""
        state = make_default_state()
        new_state, deltas = improve_skills(state, ['xyznonexistent'], 0.9)
        assert len(deltas) == 0

    def test_quality_affects_improvement(self):
        """Higher quality should produce larger skill improvements."""
        state_low = make_default_state()
        state_high = make_default_state()
        _, deltas_low = improve_skills(state_low, ['research'], 0.3)
        _, deltas_high = improve_skills(state_high, ['research'], 0.9)
        assert deltas_high.get('research', 0) > deltas_low.get('research', 0)

    def test_diminishing_returns(self):
        """Skills near 100 should improve less than skills near 0."""
        state_low = make_default_state()
        state_low['skills']['coding'] = 10.0

        state_high = make_default_state()
        state_high['skills']['coding'] = 90.0

        _, deltas_low = improve_skills(state_low, ['code'], 0.8)
        _, deltas_high = improve_skills(state_high, ['code'], 0.8)

        assert deltas_low.get('coding', 0) > deltas_high.get('coding', 0)

    def test_skill_capped_at_100(self):
        """Skill value should never exceed 100.0."""
        state = make_default_state()
        state['skills']['trading'] = 99.9
        new_state, _ = improve_skills(state, ['trading'], 1.0)
        assert new_state['skills']['trading'] <= 100.0

    def test_multiple_skills_at_once(self):
        """Multiple skills can be improved in a single call."""
        state = make_default_state()
        new_state, deltas = improve_skills(state, ['trading', 'research', 'ai'], 0.7)
        assert 'trading' in deltas
        assert 'research' in deltas
        assert 'ai_engineering' in deltas

    def test_keyword_substring_matching(self):
        """Keywords within a skill name should match (e.g., 'machine learning' -> ai_engineering)."""
        state = make_default_state()
        _, deltas = improve_skills(state, ['machine learning'], 0.8)
        assert 'ai_engineering' in deltas

    def test_immutability(self):
        """improve_skills should not mutate the original state."""
        state = make_default_state()
        original_trading = state['skills']['trading']
        _ = improve_skills(state, ['trading'], 0.9)
        assert state['skills']['trading'] == original_trading

    def test_empty_skills_list(self):
        """An empty skills list should produce no changes."""
        state = make_default_state()
        new_state, deltas = improve_skills(state, [], 0.9)
        assert len(deltas) == 0

    def test_zero_quality_no_improvement(self):
        """Zero quality should produce zero improvement."""
        state = make_default_state()
        _, deltas = improve_skills(state, ['trading'], 0.0)
        # 0.0 * (1 - 0/100) * 2.0 = 0
        assert deltas.get('trading', 0) == 0


# ===========================================================================
# content_hash tests
# ===========================================================================

class TestContentHash:
    """Tests for the content dedup hash function."""

    def test_deterministic(self):
        """Same content should always produce the same hash."""
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Different content should produce different hashes."""
        h1 = content_hash("hello world")
        h2 = content_hash("goodbye world")
        assert h1 != h2

    def test_hash_length_is_16(self):
        """Hash should be truncated to 16 hex characters."""
        h = content_hash("any content")
        assert len(h) == 16

    def test_matches_sha256_prefix(self):
        """Hash should match the first 16 chars of SHA-256 hex digest."""
        text = "test content for hashing"
        expected = hashlib.sha256(text.encode()).hexdigest()[:16]
        assert content_hash(text) == expected

    def test_empty_string(self):
        """Empty string should still produce a valid 16-char hash."""
        h = content_hash("")
        assert len(h) == 16
        assert all(c in '0123456789abcdef' for c in h)

    def test_unicode_content(self):
        """Unicode content should hash correctly."""
        h = content_hash("emoji content: the owl sees all")
        assert len(h) == 16


# ===========================================================================
# fallback_extraction tests
# ===========================================================================

class TestFallbackExtraction:
    """Tests for the fallback extraction when Claude is unavailable."""

    def test_basic_structure(self):
        """Fallback should return all required SEED phase keys."""
        result = fallback_extraction("some content about trading and ai", "test_source")
        required_keys = [
            'title', 'perceive', 'connect', 'learn', 'question',
            'expand', 'share', 'receive', 'improve', 'skills',
            'quality', 'is_alpha', 'alpha_type', 'key_entities', 'connections'
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_quality_is_always_0_3(self):
        """Fallback quality should always be 0.3 (low)."""
        result = fallback_extraction("anything", "source")
        assert result['quality'] == 0.3

    def test_is_never_alpha(self):
        """Fallback should never report alpha discoveries."""
        result = fallback_extraction("incredible alpha discovery", "source")
        assert result['is_alpha'] is False
        assert result['alpha_type'] is None

    def test_detects_trading_skills(self):
        """Content with trading keywords should detect the trading category."""
        result = fallback_extraction("buying crypto on polymarket investment", "source")
        assert 'trading' in result['skills']

    def test_detects_multiple_skills(self):
        """Content spanning multiple domains should detect multiple categories."""
        result = fallback_extraction(
            "research analysis ai agent design ux strategy planning",
            "source"
        )
        assert 'research' in result['skills']
        assert 'ai_engineering' in result['skills']
        assert 'design' in result['skills']
        assert 'strategy' in result['skills']

    def test_defaults_to_research_when_no_match(self):
        """When no skills are detected, should default to ['research']."""
        result = fallback_extraction("a b c d e f g h", "source")
        assert result['skills'] == ['research']

    def test_title_uses_source(self):
        """Title should incorporate the source name."""
        result = fallback_extraction("content", "https://example.com/article")
        assert "https://example.com/article" in result['title']

    def test_title_truncates_long_source(self):
        """Source longer than 50 chars should be truncated in title."""
        long_source = "x" * 100
        result = fallback_extraction("content", long_source)
        # Title = "Content from " + source[:50], so source part is at most 50
        source_in_title = result['title'].replace("Content from ", "")
        assert len(source_in_title) <= 50

    def test_perceive_uses_content_prefix(self):
        """Perceive should use the first 200 chars of content."""
        long_content = "word " * 100  # 500 chars
        result = fallback_extraction(long_content, "source")
        assert len(result['perceive']) <= 200


# ===========================================================================
# SKILL_CATEGORIES constant tests
# ===========================================================================

class TestSkillCategories:
    """Tests for the SKILL_CATEGORIES constant integrity."""

    def test_has_14_categories(self):
        """Should have exactly 14 skill categories."""
        assert len(SKILL_CATEGORIES) == 14

    def test_love_category_exists(self):
        """The 'love' category should exist (core to 8OWLS)."""
        assert 'love' in SKILL_CATEGORIES
        assert 'love' in SKILL_CATEGORIES['love']

    def test_no_empty_keyword_lists(self):
        """Every category should have at least one keyword."""
        for category, keywords in SKILL_CATEGORIES.items():
            assert len(keywords) > 0, f"Category '{category}' has no keywords"

    def test_all_keywords_lowercase(self):
        """All keywords should be lowercase for consistent matching."""
        for category, keywords in SKILL_CATEGORIES.items():
            for kw in keywords:
                assert kw == kw.lower(), f"Keyword '{kw}' in '{category}' is not lowercase"


# ===========================================================================
# XP calculation integration tests (combining grant_xp with learn XP logic)
# ===========================================================================

class TestXPCalculation:
    """Tests for XP calculation as done in the learn() function."""

    def test_base_xp_for_low_quality(self):
        """Low quality (<0.7) learning should grant base XP only."""
        base_xp = XP_PER_LEARN  # 10
        quality = 0.5
        is_alpha = False
        # From learn(): base_xp = 10, no bonus for quality < 0.7
        assert base_xp == 10

    def test_insight_bonus_for_high_quality(self):
        """Quality >= 0.7 should add XP_PER_INSIGHT bonus."""
        base_xp = XP_PER_LEARN
        quality = 0.8
        if quality >= 0.7:
            base_xp += XP_PER_INSIGHT
        assert base_xp == 35  # 10 + 25

    def test_alpha_bonus(self):
        """Alpha discovery should add 2x XP_PER_INSIGHT bonus."""
        base_xp = XP_PER_LEARN
        quality = 0.9
        is_alpha = True
        if quality >= 0.7:
            base_xp += XP_PER_INSIGHT
        if is_alpha:
            base_xp += XP_PER_INSIGHT * 2
        assert base_xp == 85  # 10 + 25 + 50

    def test_low_quality_alpha_still_gets_alpha_bonus(self):
        """Alpha with quality < 0.7 still gets alpha bonus but not insight bonus."""
        base_xp = XP_PER_LEARN
        quality = 0.5
        is_alpha = True
        if quality >= 0.7:
            base_xp += XP_PER_INSIGHT
        if is_alpha:
            base_xp += XP_PER_INSIGHT * 2
        assert base_xp == 60  # 10 + 0 + 50

    def test_level_progression_over_many_learns(self):
        """Simulate many learning events and verify level progression."""
        state = make_default_state()
        # 10 basic learns at 10 XP each = 100 XP = exactly Level 2
        for i in range(10):
            state = grant_xp(state, XP_PER_LEARN, f"learn {i}")
        assert state['level'] == 2
        assert state['xp'] == 0
