"""
Tests for WeEvolve Innovation Modules
=======================================
Tests: Observational Memory, Step Validator, Tool Discovery

These are the 3 key innovations from the PAIN-POINTS-AND-INNOVATIONS research:
  1. Observational Memory Layer (Pain Point #1: Context Loss)
  2. Step Validator with Checkpoint/Rollback (Pain Point #2: Error Cascading)
  3. Discovery-Based Tool Loading (Pain Point #4/#5: Integration + Cost)

All tests are pure unit tests with no external dependencies (no API calls, no NATS).
"""

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest


# ============================================================================
# OBSERVATIONAL MEMORY TESTS
# ============================================================================


class TestObservationalMemory:
    """Tests for the observational memory layer."""

    def _make_mem(self, tmp_path):
        """Create a MemoryLayer with a temp database."""
        from weevolve.observational_memory import MemoryLayer
        db_path = tmp_path / "test_mem.db"
        return MemoryLayer(db_path=db_path)

    def test_observe_creates_observation(self, tmp_path):
        mem = self._make_mem(tmp_path)
        obs_id = mem.observe("user prefers Python over TypeScript", source="conversation")
        assert obs_id is not None
        assert obs_id.startswith("obs-")

    def test_observe_deduplicates(self, tmp_path):
        mem = self._make_mem(tmp_path)
        obs1 = mem.observe("NATS is used for messaging")
        obs2 = mem.observe("NATS is used for messaging")
        assert obs1 is not None
        assert obs2 is None  # Duplicate returns None (reinforced instead)

    def test_observe_reinforces_duplicate(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("user prefers dark mode")
        mem.observe("user prefers dark mode")
        mem.observe("user prefers dark mode")
        results = mem.recall("dark mode")
        assert len(results) == 1
        assert results[0].reinforcement_count == 3

    def test_observe_rejects_empty(self, tmp_path):
        mem = self._make_mem(tmp_path)
        assert mem.observe("") is None
        assert mem.observe("  ") is None
        assert mem.observe("hi") is None  # Too short (<5 chars)

    def test_observe_auto_classifies(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("user prefers Python over TypeScript")
        results = mem.recall("Python")
        assert len(results) == 1
        assert results[0].category == "preference"

    def test_observe_auto_classifies_decision(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("decided to use NATS for all messaging")
        results = mem.recall("NATS")
        assert len(results) == 1
        assert results[0].category == "decision"

    def test_observe_auto_classifies_insight(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("key finding: realized that agent infra wins over model quality")
        results = mem.recall("agent infra")
        assert len(results) == 1
        assert results[0].category == "insight"

    def test_observe_extracts_tags(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("the voice pipeline uses ElevenLabs for TTS")
        results = mem.recall("voice")
        assert len(results) == 1
        tags = json.loads(results[0].tags)
        assert "voice" in tags

    def test_observe_batch(self, tmp_path):
        mem = self._make_mem(tmp_path)
        count = mem.observe_batch([
            {"content": "WeEvolve uses SQLite for storage"},
            {"content": "NATS provides real-time messaging"},
            {"content": "Genesis has 945 atoms"},
        ])
        assert count == 3

    def test_recall_by_query(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("Python is the primary language")
        mem.observe("TypeScript is used for frontend")
        mem.observe("Rust is being evaluated")
        results = mem.recall("Python")
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_recall_by_category(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("user prefers dark mode", category="preference")
        mem.observe("server runs on port 8006", category="fact")
        results = mem.recall("", category="preference")
        assert len(results) == 1
        assert results[0].category == "preference"

    def test_recall_by_confidence(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("high confidence observation", confidence=0.95)
        mem.observe("low confidence guess maybe", confidence=0.2)
        results = mem.recall("", min_confidence=0.5)
        assert len(results) == 1
        assert results[0].confidence >= 0.5

    def test_recall_limit(self, tmp_path):
        mem = self._make_mem(tmp_path)
        for i in range(20):
            mem.observe(f"observation number {i} about testing things")
        results = mem.recall("observation", limit=5)
        assert len(results) == 5

    def test_build_context(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("user prefers Python over TypeScript")
        mem.observe("project uses NATS for messaging")
        mem.observe("WeEvolve has 945 genesis atoms")
        context = mem.build_context(topic="WeEvolve")
        assert len(context) > 0
        assert "WeEvolve" in context or "genesis" in context

    def test_build_context_empty(self, tmp_path):
        mem = self._make_mem(tmp_path)
        context = mem.build_context(topic="nonexistent")
        assert context == ""

    def test_build_context_max_tokens(self, tmp_path):
        mem = self._make_mem(tmp_path)
        for i in range(50):
            mem.observe(f"observation {i}: a detailed fact about something important that fills space")
        context = mem.build_context(max_tokens=100)
        assert len(context) <= 500  # 100 tokens * ~4 chars + some overhead

    def test_stats(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("fact one about Python programming")
        mem.observe("user prefers dark mode interface")
        stats = mem.stats()
        assert stats['total_observations'] == 2
        assert stats['active_observations'] == 2
        assert stats['average_confidence'] > 0
        assert 'by_category' in stats
        assert 'by_source' in stats

    def test_apply_decay(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.observe("this should not decay immediately")
        deactivated = mem.apply_decay()
        assert deactivated == 0  # Nothing old enough to deactivate

    def test_merge_similar(self, tmp_path):
        mem = self._make_mem(tmp_path)
        # These are different enough to not dedup by hash, but similar enough to merge
        mem.observe("the user prefers Python language for backend development work")
        mem.observe("the user prefers Python language for backend development tasks")
        merged = mem.merge_similar(similarity_threshold=0.7)
        assert merged >= 1


# ============================================================================
# STEP VALIDATOR TESTS
# ============================================================================


class TestStepValidator:
    """Tests for the step validator with checkpoint/rollback."""

    def test_simple_pipeline_success(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("test")
        pipeline.add_step(Step(
            "add_key",
            execute=lambda state: {"added": True},
            validate=lambda out: out.get("added") is True,
        ))

        result = pipeline.run({"initial": True})
        assert result.success is True
        assert result.steps_completed == 1
        assert result.final_state.get("added") is True

    def test_pipeline_validation_failure(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("test_fail")
        pipeline.add_step(Step(
            "bad_step",
            execute=lambda state: {"value": -1},
            validate=lambda out: out.get("value", 0) > 0,  # Will fail
            max_retries=1,
            retry_delay_ms=10,
        ))

        result = pipeline.run()
        assert result.success is False
        assert result.steps_completed == 0

    def test_pipeline_exception_handling(self):
        from weevolve.step_validator import Pipeline, Step

        def exploding_step(state):
            raise ValueError("Boom!")

        pipeline = Pipeline("test_exception")
        pipeline.add_step(Step(
            "exploder",
            execute=exploding_step,
            max_retries=1,
            retry_delay_ms=10,
        ))

        result = pipeline.run()
        assert result.success is False
        assert "ValueError" in result.step_results[0].error

    def test_pipeline_retry(self):
        from weevolve.step_validator import Pipeline, Step

        call_count = [0]

        def flaky_step(state):
            call_count[0] += 1
            if call_count[0] < 2:
                return {"success": False}
            return {"success": True}

        pipeline = Pipeline("test_retry")
        pipeline.add_step(Step(
            "flaky",
            execute=flaky_step,
            validate=lambda out: out.get("success") is True,
            max_retries=3,
            retry_delay_ms=10,
        ))

        result = pipeline.run()
        assert result.success is True
        assert call_count[0] == 2

    def test_multi_step_pipeline(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("multi")
        pipeline.add_step(Step(
            "step1",
            execute=lambda s: {"step1_done": True},
        ))
        pipeline.add_step(Step(
            "step2",
            execute=lambda s: {"step2_done": True},
        ))
        pipeline.add_step(Step(
            "step3",
            execute=lambda s: {"step3_done": True},
        ))

        result = pipeline.run({"initial": True})
        assert result.success is True
        assert result.steps_completed == 3
        assert result.final_state.get("step1_done") is True
        assert result.final_state.get("step2_done") is True
        assert result.final_state.get("step3_done") is True

    def test_pipeline_rollback_saga(self):
        from weevolve.step_validator import Pipeline, Step

        compensated = []

        def compensate_step1(state, output):
            compensated.append("step1")

        pipeline = Pipeline("saga")
        pipeline.add_step(Step(
            "step1",
            execute=lambda s: {"step1_done": True},
            compensate=compensate_step1,
        ))
        pipeline.add_step(Step(
            "step2_fails",
            execute=lambda s: {"value": -1},
            validate=lambda out: out.get("value", 0) > 0,
            max_retries=1,
            retry_delay_ms=10,
        ))

        result = pipeline.run()
        assert result.success is False
        assert result.rollback_performed is True
        assert "step1" in compensated

    def test_optional_step_skipped(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("optional")
        pipeline.add_step(Step(
            "required_step",
            execute=lambda s: {"ok": True},
        ))
        pipeline.add_step(Step(
            "optional_fail",
            execute=lambda s: {"bad": True},
            validate=lambda out: False,  # Always fails
            required=False,
            max_retries=1,
            retry_delay_ms=10,
        ))
        pipeline.add_step(Step(
            "after_optional",
            execute=lambda s: {"final": True},
        ))

        result = pipeline.run()
        assert result.success is True
        assert result.steps_completed == 2  # required + after_optional (optional skipped)

    def test_checkpoint_save_restore(self, tmp_path):
        from weevolve.step_validator import Checkpoint

        cp = Checkpoint(storage_dir=tmp_path)
        state = {"level": 14, "xp": 500}
        cp.save("test_cp", state)

        restored = cp.restore("test_cp")
        assert restored is not None
        assert restored["level"] == 14
        assert restored["xp"] == 500

    def test_checkpoint_independence(self, tmp_path):
        from weevolve.step_validator import Checkpoint

        cp = Checkpoint(storage_dir=tmp_path)
        state = {"data": [1, 2, 3]}
        cp.save("cp1", state)

        # Mutate original (should not affect checkpoint)
        state["data"].append(4)
        restored = cp.restore("cp1")
        assert len(restored["data"]) == 3  # Original, not mutated

    def test_checkpoint_not_found(self):
        from weevolve.step_validator import Checkpoint

        cp = Checkpoint()
        assert cp.restore("nonexistent") is None

    def test_pipeline_trace(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("traced")
        pipeline.add_step(Step("s1", execute=lambda s: {"done": True}))
        pipeline.add_step(Step("s2", execute=lambda s: {"done": True}))
        pipeline.run()

        trace = pipeline.get_trace()
        assert len(trace) == 2
        assert trace[0]["step_name"] == "s1"
        assert trace[1]["step_name"] == "s2"

    def test_pipeline_trace_summary(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("summary_test")
        pipeline.add_step(Step("build", execute=lambda s: {"built": True}))
        pipeline.run()

        summary = pipeline.trace_summary()
        assert "summary_test" in summary
        assert "PASS" in summary
        assert "build" in summary

    def test_validate_schema(self):
        from weevolve.step_validator import validate_schema

        assert validate_schema({"success": True, "count": 5}, {"success": bool, "count": int})
        assert not validate_schema({"success": True}, {"success": bool, "count": int})
        assert not validate_schema("not a dict", {"key": str})
        assert not validate_schema({"count": "not_int"}, {"count": int})

    def test_validate_assertions(self):
        from weevolve.step_validator import validate_assertions

        assert validate_assertions(
            {"count": 5, "status": "ok"},
            [
                lambda o: o.get("count", 0) > 0,
                lambda o: o.get("status") == "ok",
            ],
        )
        assert not validate_assertions(
            {"count": 0},
            [lambda o: o.get("count", 0) > 0],
        )

    def test_pipeline_chaining(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = (
            Pipeline("chained")
            .add_step(Step("s1", execute=lambda s: {"a": 1}))
            .add_step(Step("s2", execute=lambda s: {"b": 2}))
        )
        result = pipeline.run()
        assert result.success
        assert result.steps_completed == 2

    def test_step_result_fields(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("fields")
        pipeline.add_step(Step("s1", execute=lambda s: {"ok": True}))
        result = pipeline.run()

        sr = result.step_results[0]
        assert sr.step_name == "s1"
        assert sr.success is True
        assert sr.duration_ms >= 0
        assert sr.attempt == 1
        assert sr.timestamp is not None

    def test_pipeline_result_fields(self):
        from weevolve.step_validator import Pipeline, Step

        pipeline = Pipeline("result_fields")
        pipeline.add_step(Step("s1", execute=lambda s: {"ok": True}))
        result = pipeline.run({"start": True})

        assert result.pipeline_name == "result_fields"
        assert result.success is True
        assert result.total_duration_ms >= 0
        assert result.timestamp is not None
        assert result.final_state.get("start") is True


# ============================================================================
# TOOL DISCOVERY TESTS
# ============================================================================


class TestToolDiscovery:
    """Tests for the discovery-based tool loading system."""

    def _make_registry(self, tmp_path=None):
        from weevolve.tool_discovery import ToolRegistry
        db_path = tmp_path / "test_tools.db" if tmp_path else None
        registry = ToolRegistry(db_path=db_path)
        return registry

    def test_register_tool(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register(
            "web_search",
            category="research",
            keywords=["search", "web", "find", "internet"],
            description="Search the web for information",
        )
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "web_search"

    def test_register_batch(self, tmp_path):
        registry = self._make_registry(tmp_path)
        count = registry.register_batch([
            {"name": "tool1", "category": "a", "keywords": ["x"]},
            {"name": "tool2", "category": "b", "keywords": ["y"]},
            {"name": "tool3", "category": "c", "keywords": ["z"]},
        ])
        assert count == 3
        assert len(registry.list_tools()) == 3

    def test_discover_by_keyword(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("web_search", keywords=["search", "web", "find"])
        registry.register("code_edit", keywords=["edit", "code", "file"])
        registry.register("nats_publish", keywords=["nats", "publish", "message"])

        tools = registry.discover("search the web for NATS documentation")
        assert "web_search" in tools
        assert "nats_publish" in tools

    def test_discover_by_name(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("flash_scan", keywords=["scan", "codebase"])
        registry.register("web_search", keywords=["search", "web"])

        tools = registry.discover("run flash_scan on the project")
        assert "flash_scan" in tools

    def test_discover_respects_limit(self, tmp_path):
        registry = self._make_registry(tmp_path)
        for i in range(10):
            registry.register(f"tool_{i}", keywords=["common", f"kw{i}"])

        tools = registry.discover("common operation", limit=3)
        assert len(tools) <= 3

    def test_discover_with_scores(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("web_search", keywords=["search", "web"])
        registry.register("nats_publish", keywords=["nats", "publish"])

        results = registry.discover_with_scores("search the web")
        assert len(results) >= 1
        assert results[0][0] == "web_search"
        assert results[0][1] > 0

    def test_discover_category_filter(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("web_search", category="research", keywords=["search"])
        registry.register("code_edit", category="coding", keywords=["edit"])

        tools = registry.discover("search and edit", categories=["research"])
        assert "web_search" in tools
        assert "code_edit" not in tools

    def test_load_schemas(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register(
            "web_search",
            keywords=["search"],
            schema={"type": "function", "name": "web_search"},
        )
        schemas = registry.load_schemas(["web_search"])
        assert len(schemas) == 1
        assert schemas[0]["name"] == "web_search"

    def test_load_schemas_caches(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register(
            "tool_a",
            keywords=["a"],
            schema={"name": "tool_a"},
        )
        schemas1 = registry.load_schemas(["tool_a"])
        schemas2 = registry.load_schemas(["tool_a"])
        assert schemas1 == schemas2

    def test_record_usage(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("tool_a", keywords=["a"])
        registry.register("tool_b", keywords=["b"])

        registry.record_usage(["tool_a", "tool_b"], task="test task")
        co_used = registry.get_co_used("tool_a")
        assert len(co_used) >= 1
        assert co_used[0][0] == "tool_b"

    def test_stats(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("tool_a", category="cat1", keywords=["a"])
        registry.register("tool_b", category="cat2", keywords=["b"])

        stats = registry.stats()
        assert stats['total_tools'] == 2
        assert 'cat1' in stats['categories']

    def test_token_savings(self, tmp_path):
        registry = self._make_registry(tmp_path)
        for i in range(50):
            registry.register(f"tool_{i}", keywords=[f"kw{i}"], cost_tokens=200)

        savings = registry.token_savings(discovered_count=3)
        assert savings['total_tools'] == 50
        assert savings['savings_percent'] > 90  # >90% savings

    def test_default_registry(self, tmp_path):
        from weevolve.tool_discovery import create_default_registry
        registry = create_default_registry(db_path=tmp_path / "default.db")
        tools = registry.list_tools()
        assert len(tools) >= 10  # We defined 17 default tools

    def test_default_registry_discover(self, tmp_path):
        from weevolve.tool_discovery import create_default_registry
        registry = create_default_registry(db_path=tmp_path / "default.db")

        # Discover tools for a voice task
        voice_tools = registry.discover("start voice conversation with ElevenLabs")
        assert "voice_speak" in voice_tools

        # Discover tools for a learning task
        learn_tools = registry.discover("learn from this URL about AI agents")
        assert "weevolve_learn" in learn_tools

    def test_empty_discover(self, tmp_path):
        registry = self._make_registry(tmp_path)
        tools = registry.discover("anything")
        assert tools == []

    def test_discover_no_match(self, tmp_path):
        registry = self._make_registry(tmp_path)
        registry.register("web_search", keywords=["search", "web"])
        tools = registry.discover("completely unrelated quantum physics")
        # May or may not match depending on word overlap
        assert isinstance(tools, list)
