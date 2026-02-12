#!/usr/bin/env python3
"""
8OWLS Intelligent Model Router
===============================
Routes AI tasks to the optimal model based on:
- Task type and complexity
- Cost/quality/speed tradeoffs
- Required capabilities (search, vision, code, reasoning)
- Learning from past routing decisions

Integrates with WeEvolve's RPG loadout system.

Usage:
    python3 tools/model_router.py route "Write a React component for user auth"
    python3 tools/model_router.py route --task-type code --complexity 0.6 "Build API endpoint"
    python3 tools/model_router.py status
    python3 tools/model_router.py benchmark
    python3 tools/model_router.py cost-report
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


# --- Model Definitions ---

class Provider(Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    XAI = "xai"
    GOOGLE = "google"
    DEEPSEEK = "deepseek"
    TOGETHER = "together"
    LOCAL = "local"


class TaskType(Enum):
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


@dataclass
class ModelSpec:
    model_id: str
    provider: Provider
    display_name: str
    input_cost_per_m: float  # $ per million input tokens
    output_cost_per_m: float  # $ per million output tokens
    context_window: int  # tokens
    speed_toks_per_sec: float  # approximate
    swe_bench_score: float  # 0-100, approximate
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)
    capabilities: list = field(default_factory=list)  # search, vision, code, reasoning, tools
    cache_read_cost_per_m: float = 0.0  # $ per million cached input tokens (0 = no caching)
    env_key: str = ""  # environment variable for API key
    available: bool = True


# --- Model Registry ---

MODELS = {
    # --- Anthropic ---
    "claude-opus-4-6": ModelSpec(
        model_id="claude-opus-4-6",
        provider=Provider.ANTHROPIC,
        display_name="Claude Opus 4.6",
        input_cost_per_m=5.00,
        output_cost_per_m=25.00,
        context_window=1_000_000,
        speed_toks_per_sec=40,
        swe_bench_score=80.8,
        strengths=["deep_reasoning", "architecture", "code_review", "creative_writing", "security"],
        weaknesses=["speed", "cost"],
        capabilities=["code", "reasoning", "tools", "vision"],
        cache_read_cost_per_m=0.50,
        env_key="ANTHROPIC_API_KEY",
    ),
    "claude-sonnet-4-5": ModelSpec(
        model_id="claude-sonnet-4-5",
        provider=Provider.ANTHROPIC,
        display_name="Claude Sonnet 4.5",
        input_cost_per_m=3.00,
        output_cost_per_m=15.00,
        context_window=200_000,
        speed_toks_per_sec=80,
        swe_bench_score=77.2,
        strengths=["coding", "tool_use", "balance"],
        weaknesses=["not_cheapest", "smaller_context"],
        capabilities=["code", "reasoning", "tools", "vision"],
        cache_read_cost_per_m=0.30,
        env_key="ANTHROPIC_API_KEY",
    ),
    "claude-haiku-4-5": ModelSpec(
        model_id="claude-haiku-4-5",
        provider=Provider.ANTHROPIC,
        display_name="Claude Haiku 4.5",
        input_cost_per_m=1.00,
        output_cost_per_m=5.00,
        context_window=200_000,
        speed_toks_per_sec=150,
        swe_bench_score=65.0,
        strengths=["speed", "cost_effective", "agent_worker"],
        weaknesses=["complex_reasoning"],
        capabilities=["code", "tools"],
        cache_read_cost_per_m=0.10,
        env_key="ANTHROPIC_API_KEY",
    ),

    # --- OpenAI ---
    "gpt-5.3-codex": ModelSpec(
        model_id="gpt-5.3-codex",
        provider=Provider.OPENAI,
        display_name="GPT-5.3-Codex",
        input_cost_per_m=1.25,
        output_cost_per_m=10.00,
        context_window=400_000,
        speed_toks_per_sec=90,
        swe_bench_score=81.4,  # SWE-Lancer IC Diamond
        strengths=["coding", "debugging", "parallel_execution", "ci_cd"],
        weaknesses=["frontend_design", "api_phased_rollout"],
        capabilities=["code", "reasoning", "tools", "vision", "mcp_server"],
        env_key="OPENAI_API_KEY",
        available=False,  # Phased API rollout
    ),
    "codex-mini-latest": ModelSpec(
        model_id="codex-mini-latest",
        provider=Provider.OPENAI,
        display_name="Codex Mini",
        input_cost_per_m=1.50,
        output_cost_per_m=6.00,
        context_window=400_000,
        speed_toks_per_sec=120,
        swe_bench_score=70.0,
        strengths=["cost_optimized_coding", "cache_discount"],
        weaknesses=["not_frontier"],
        capabilities=["code", "tools"],
        env_key="OPENAI_API_KEY",
    ),
    "gpt-5": ModelSpec(
        model_id="gpt-5",
        provider=Provider.OPENAI,
        display_name="GPT-5",
        input_cost_per_m=1.25,
        output_cost_per_m=10.00,
        context_window=400_000,
        speed_toks_per_sec=101,
        swe_bench_score=74.9,
        strengths=["cheapest_frontier", "fast", "general"],
        weaknesses=["not_best_agentic"],
        capabilities=["code", "reasoning", "tools", "vision", "search"],
        env_key="OPENAI_API_KEY",
    ),

    # --- xAI ---
    "grok-4.1-fast": ModelSpec(
        model_id="grok-4.1-fast",
        provider=Provider.XAI,
        display_name="Grok 4.1 Fast",
        input_cost_per_m=0.20,
        output_cost_per_m=0.50,
        context_window=2_000_000,
        speed_toks_per_sec=203,
        swe_bench_score=70.0,
        strengths=["realtime_search", "x_twitter", "speed", "massive_context", "cheapest_fast"],
        weaknesses=["not_deep_reasoning"],
        capabilities=["code", "tools", "search", "x_search"],
        env_key="XAI_API_KEY",
    ),
    "grok-4": ModelSpec(
        model_id="grok-4",
        provider=Provider.XAI,
        display_name="Grok 4",
        input_cost_per_m=3.00,
        output_cost_per_m=15.00,
        context_window=256_000,
        speed_toks_per_sec=40,
        swe_bench_score=75.0,
        strengths=["reasoning", "search", "x_twitter"],
        weaknesses=["cost", "speed"],
        capabilities=["code", "reasoning", "tools", "search", "x_search"],
        env_key="XAI_API_KEY",
    ),
    "grok-code-fast": ModelSpec(
        model_id="grok-code-fast",
        provider=Provider.XAI,
        display_name="Grok Code Fast",
        input_cost_per_m=0.20,
        output_cost_per_m=1.50,
        context_window=256_000,
        speed_toks_per_sec=150,
        swe_bench_score=70.0,
        strengths=["budget_coding", "speed"],
        weaknesses=["not_frontier"],
        capabilities=["code", "tools"],
        env_key="XAI_API_KEY",
    ),

    # --- Google ---
    "gemini-2.5-flash": ModelSpec(
        model_id="gemini-2.5-flash",
        provider=Provider.GOOGLE,
        display_name="Gemini 2.5 Flash",
        input_cost_per_m=0.30,
        output_cost_per_m=2.50,
        context_window=1_000_000,
        speed_toks_per_sec=200,
        swe_bench_score=72.0,
        strengths=["multimodal", "long_context", "fast", "cheap"],
        weaknesses=["not_frontier_reasoning"],
        capabilities=["code", "tools", "vision", "search"],
        env_key="GOOGLE_API_KEY",
    ),
    "gemini-2.5-flash-lite": ModelSpec(
        model_id="gemini-2.5-flash-lite",
        provider=Provider.GOOGLE,
        display_name="Gemini 2.5 Flash-Lite",
        input_cost_per_m=0.10,
        output_cost_per_m=0.40,
        context_window=1_000_000,
        speed_toks_per_sec=300,
        swe_bench_score=60.0,
        strengths=["cheapest_production", "classification", "extraction"],
        weaknesses=["simple_tasks_only"],
        capabilities=["tools"],
        env_key="GOOGLE_API_KEY",
    ),
    "gemini-2.5-pro": ModelSpec(
        model_id="gemini-2.5-pro",
        provider=Provider.GOOGLE,
        display_name="Gemini 2.5 Pro",
        input_cost_per_m=1.25,
        output_cost_per_m=10.00,
        context_window=1_000_000,
        speed_toks_per_sec=70,
        swe_bench_score=75.0,
        strengths=["long_context", "multimodal", "grounding"],
        weaknesses=["cost"],
        capabilities=["code", "reasoning", "tools", "vision", "search"],
        env_key="GOOGLE_API_KEY",
    ),

    # --- DeepSeek ---
    "deepseek-v3.2": ModelSpec(
        model_id="deepseek-v3.2",
        provider=Provider.DEEPSEEK,
        display_name="DeepSeek V3.2",
        input_cost_per_m=0.028,
        output_cost_per_m=0.14,
        context_window=128_000,
        speed_toks_per_sec=80,
        swe_bench_score=73.0,
        strengths=["cheapest", "open_source", "self_hostable", "coding"],
        weaknesses=["smaller_context"],
        capabilities=["code", "tools"],
        env_key="DEEPSEEK_API_KEY",
    ),
    "deepseek-r1": ModelSpec(
        model_id="deepseek-r1",
        provider=Provider.DEEPSEEK,
        display_name="DeepSeek R1",
        input_cost_per_m=0.70,
        output_cost_per_m=2.50,
        context_window=128_000,
        speed_toks_per_sec=50,
        swe_bench_score=68.0,
        strengths=["reasoning", "math", "cheap_reasoning"],
        weaknesses=["slower", "smaller_context"],
        capabilities=["code", "reasoning"],
        env_key="DEEPSEEK_API_KEY",
    ),

    # --- Meta (via Together) ---
    "llama-4-scout": ModelSpec(
        model_id="llama-4-scout",
        provider=Provider.TOGETHER,
        display_name="Llama 4 Scout",
        input_cost_per_m=0.08,
        output_cost_per_m=0.30,
        context_window=10_000_000,
        speed_toks_per_sec=100,
        swe_bench_score=55.0,
        strengths=["massive_context", "cheapest_bulk", "open_source"],
        weaknesses=["not_frontier"],
        capabilities=["code"],
        env_key="TOGETHER_API_KEY",
    ),
}


# --- Task Classification ---

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

# Task type to preferred models mapping
TASK_MODEL_PREFERENCES = {
    TaskType.CODE_FRONTEND: ["claude-opus-4-6", "claude-sonnet-4-5", "gemini-2.5-flash"],
    TaskType.CODE_BACKEND: ["claude-opus-4-6", "claude-sonnet-4-5", "deepseek-v3.2"],
    TaskType.CODE_SYSTEMS: ["gpt-5", "claude-opus-4-6", "deepseek-v3.2"],
    TaskType.CODE_REVIEW: ["claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5"],
    TaskType.ARCHITECTURE: ["claude-opus-4-6", "claude-sonnet-4-5", "gemini-2.5-pro"],
    TaskType.RESEARCH: ["claude-opus-4-6", "gemini-2.5-pro", "grok-4.1-fast"],
    TaskType.REALTIME_SEARCH: ["grok-4.1-fast", "grok-4", "gpt-5"],
    TaskType.MATH_REASONING: ["deepseek-r1", "claude-opus-4-6", "claude-sonnet-4-5"],
    TaskType.CREATIVE_WRITING: ["claude-opus-4-6", "gpt-5", "claude-sonnet-4-5"],
    TaskType.CLASSIFICATION: ["gemini-2.5-flash-lite", "claude-haiku-4-5", "llama-4-scout"],
    TaskType.BULK_PROCESSING: ["deepseek-v3.2", "llama-4-scout", "gemini-2.5-flash-lite"],
    TaskType.TRADING_SIGNALS: ["grok-4.1-fast", "deepseek-r1", "claude-sonnet-4-5"],
    TaskType.MULTIMODAL: ["gemini-2.5-flash", "claude-sonnet-4-5", "gemini-2.5-pro"],
    TaskType.AGENT_ORCHESTRATION: ["claude-opus-4-6", "claude-sonnet-4-5", "gpt-5"],
    TaskType.AGENT_WORKER: ["claude-haiku-4-5", "gemini-2.5-flash", "grok-4.1-fast"],
    TaskType.GENERAL: ["claude-sonnet-4-5", "gpt-5", "claude-haiku-4-5"],
}


# --- Complexity Estimation ---

def estimate_complexity(prompt: str) -> float:
    """Estimate task complexity from 0.0 to 1.0 based on prompt characteristics."""
    score = 0.3  # baseline

    # Length signals
    words = len(prompt.split())
    if words > 200:
        score += 0.1
    if words > 500:
        score += 0.1

    # Multi-step indicators
    multi_step_patterns = [
        r'\b(then|after that|next|finally|also|additionally)\b',
        r'\b(step \d|phase \d|first.*second|1\).*2\))\b',
        r'\b(and also|as well as|in addition)\b',
    ]
    for pattern in multi_step_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            score += 0.05

    # Technical depth indicators
    tech_depth = [
        r'\b(architecture|distributed|concurrent|scalable|microservice)\b',
        r'\b(security|authentication|authorization|encryption)\b',
        r'\b(optimize|performance|profil|benchmark)\b',
        r'\b(refactor|redesign|migrate|overhaul)\b',
    ]
    for pattern in tech_depth:
        if re.search(pattern, prompt, re.IGNORECASE):
            score += 0.08

    # Simple task indicators (reduce complexity)
    simple_patterns = [
        r'\b(fix typo|rename|format|lint|simple|quick|small)\b',
        r'\b(add comment|update readme|change name)\b',
    ]
    for pattern in simple_patterns:
        if re.search(pattern, prompt, re.IGNORECASE):
            score -= 0.15

    return max(0.0, min(1.0, score))


def classify_task(prompt: str) -> TaskType:
    """Classify a prompt into a task type based on keyword matching."""
    scores = {}
    prompt_lower = prompt.lower()

    for task_type, keywords in TASK_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in prompt_lower)
        if score > 0:
            scores[task_type] = score

    if not scores:
        return TaskType.GENERAL

    return max(scores, key=scores.get)


# --- Router State ---

STATE_PATH = Path(__file__).parent.parent / ".swarm" / "model-router-state.json"


def load_state() -> dict:
    """Load router state from disk."""
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {
        "totalDecisions": 0,
        "modelDistribution": {},
        "avgComplexity": 0.0,
        "avgConfidence": 0.0,
        "circuitBreakerTrips": 0,
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "learningHistory": [],
        "costSaved": 0.0,
        "totalCost": 0.0,
    }


def save_state(state: dict):
    """Persist router state to disk."""
    state["lastUpdated"] = datetime.now(timezone.utc).isoformat()
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


# --- Core Router ---

@dataclass
class RoutingDecision:
    model_id: str
    model_name: str
    provider: str
    task_type: str
    complexity: float
    confidence: float
    estimated_cost_per_1k_tokens: float
    fallback_model: str
    reasoning: str


def check_api_key(model: ModelSpec) -> bool:
    """Check if API key is available for a model."""
    if not model.env_key:
        return True
    return bool(os.environ.get(model.env_key))


def route(
    prompt: str,
    task_type: Optional[str] = None,
    complexity: Optional[float] = None,
    max_cost_per_m: Optional[float] = None,
    require_capability: Optional[str] = None,
    prefer_speed: bool = False,
) -> RoutingDecision:
    """
    Route a task to the optimal model.

    Args:
        prompt: The task description
        task_type: Override task type classification
        complexity: Override complexity estimate (0.0-1.0)
        max_cost_per_m: Maximum cost per million tokens (input)
        require_capability: Required capability (search, vision, code, reasoning, tools)
        prefer_speed: Prioritize speed over quality
    """
    # Classify task
    if task_type:
        detected_type = TaskType(task_type)
    else:
        detected_type = classify_task(prompt)

    # Estimate complexity
    if complexity is not None:
        est_complexity = complexity
    else:
        est_complexity = estimate_complexity(prompt)

    # Get preferred models for this task type
    preferred = TASK_MODEL_PREFERENCES.get(detected_type, TASK_MODEL_PREFERENCES[TaskType.GENERAL])

    # Filter by availability and API keys
    candidates = []
    for model_id in preferred:
        model = MODELS.get(model_id)
        if not model:
            continue
        if not model.available:
            continue
        if not check_api_key(model):
            continue
        if max_cost_per_m and model.input_cost_per_m > max_cost_per_m:
            continue
        if require_capability and require_capability not in model.capabilities:
            continue
        candidates.append(model)

    # If no candidates from preferred list, try all models
    if not candidates:
        for model_id, model in MODELS.items():
            if not model.available:
                continue
            if not check_api_key(model):
                continue
            if max_cost_per_m and model.input_cost_per_m > max_cost_per_m:
                continue
            if require_capability and require_capability not in model.capabilities:
                continue
            candidates.append(model)

    # If still nothing, fallback to Claude Sonnet (most likely to have key)
    if not candidates:
        candidates = [MODELS["claude-sonnet-4-5"]]

    # Score candidates
    def score_model(model: ModelSpec) -> float:
        s = 0.0

        # Quality score (weighted by complexity)
        quality_weight = 0.3 + (est_complexity * 0.4)  # 0.3-0.7
        s += (model.swe_bench_score / 100.0) * quality_weight

        # Cost score (inversely weighted by complexity)
        cost_weight = 0.4 - (est_complexity * 0.3)  # 0.1-0.4
        max_input = max(m.input_cost_per_m for m in candidates) or 1
        cost_score = 1.0 - (model.input_cost_per_m / max_input)
        s += cost_score * cost_weight

        # Speed score
        speed_weight = 0.2 if not prefer_speed else 0.4
        max_speed = max(m.speed_toks_per_sec for m in candidates) or 1
        s += (model.speed_toks_per_sec / max_speed) * speed_weight

        # Capability bonus
        if require_capability and require_capability in model.capabilities:
            s += 0.1

        # Preference order bonus (first in list = most preferred)
        if model.model_id in preferred:
            idx = preferred.index(model.model_id)
            s += (len(preferred) - idx) * 0.02

        return s

    scored = sorted(candidates, key=score_model, reverse=True)
    best = scored[0]
    fallback = scored[1] if len(scored) > 1 else scored[0]

    confidence = score_model(best)
    blended_cost = (best.input_cost_per_m * 0.75 + best.output_cost_per_m * 0.25) / 1000

    # Build reasoning
    reasoning_parts = [
        f"Task type: {detected_type.value}",
        f"Complexity: {est_complexity:.2f}",
        f"Selected: {best.display_name} ({best.provider.value})",
        f"Cost: ${best.input_cost_per_m}/{best.output_cost_per_m} per M tokens",
    ]
    if require_capability:
        reasoning_parts.append(f"Required: {require_capability}")
    if max_cost_per_m:
        reasoning_parts.append(f"Budget: <${max_cost_per_m}/M")

    # Record decision
    state = load_state()
    state["totalDecisions"] = state.get("totalDecisions", 0) + 1
    dist = state.get("modelDistribution", {})
    dist[best.model_id] = dist.get(best.model_id, 0) + 1
    state["modelDistribution"] = dist

    n = state["totalDecisions"]
    state["avgComplexity"] = ((state.get("avgComplexity", 0) * (n - 1)) + est_complexity) / n
    state["avgConfidence"] = ((state.get("avgConfidence", 0) * (n - 1)) + confidence) / n

    # Track in learning history (keep last 100)
    history = state.get("learningHistory", [])
    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": best.model_id,
        "task_type": detected_type.value,
        "complexity": round(est_complexity, 3),
        "confidence": round(confidence, 3),
        "cost_per_m": best.input_cost_per_m,
    })
    state["learningHistory"] = history[-100:]

    save_state(state)

    return RoutingDecision(
        model_id=best.model_id,
        model_name=best.display_name,
        provider=best.provider.value,
        task_type=detected_type.value,
        complexity=round(est_complexity, 3),
        confidence=round(confidence, 3),
        estimated_cost_per_1k_tokens=round(blended_cost, 6),
        fallback_model=fallback.model_id,
        reasoning=" | ".join(reasoning_parts),
    )


# --- 8OWLS Owl-to-Model Mapping ---

OWL_BUILDS = {
    "SOWL": {"model": "claude-opus-4-6", "role": "Coordinator", "phase": "IMPROVE"},
    "LYRA": {"model": "grok-4.1-fast", "role": "Perception", "phase": "PERCEIVE"},
    "SAGE": {"model": "claude-sonnet-4-5", "role": "Learning", "phase": "LEARN"},
    "QUEST": {"model": "deepseek-r1", "role": "Questioning", "phase": "QUESTION"},
    "NOVA": {"model": "claude-sonnet-4-5", "role": "Expansion", "phase": "EXPAND"},
    "PRISM": {"model": "gemini-2.5-flash", "role": "Connection", "phase": "CONNECT"},
    "ECHO": {"model": "claude-haiku-4-5", "role": "Sharing", "phase": "SHARE"},
    "LUNA": {"model": "gemini-2.5-flash", "role": "Receiving", "phase": "RECEIVE"},
}

# WeEvolve-style loadout presets
LOADOUT_PRESETS = {
    "researcher": {
        "primary": "grok-4.1-fast",
        "secondary": "gemini-2.5-pro",
        "description": "Real-time search + long context analysis",
    },
    "coder": {
        "primary": "claude-sonnet-4-5",
        "secondary": "codex-mini-latest",
        "description": "Primary coding + bulk code generation",
    },
    "architect": {
        "primary": "claude-opus-4-6",
        "secondary": "gemini-2.5-pro",
        "description": "Deep reasoning + large codebase analysis",
    },
    "scout": {
        "primary": "llama-4-scout",
        "secondary": "deepseek-v3.2",
        "description": "10M context ingestion + cheap processing",
    },
    "trader": {
        "primary": "deepseek-r1",
        "secondary": "grok-4.1-fast",
        "description": "Quantitative reasoning + real-time X signals",
    },
    "reviewer": {
        "primary": "claude-opus-4-6",
        "secondary": "claude-haiku-4-5",
        "description": "Deep review + quick lint checks",
    },
}


# --- CLI ---

def show_status():
    """Display router status and statistics."""
    state = load_state()
    available = sum(1 for m in MODELS.values() if m.available and check_api_key(m))

    print("=" * 60)
    print("  8OWLS MODEL ROUTER STATUS")
    print("=" * 60)
    print(f"  Models registered:  {len(MODELS)}")
    print(f"  Models available:   {available}")
    print(f"  Total decisions:    {state.get('totalDecisions', 0)}")
    print(f"  Avg complexity:     {state.get('avgComplexity', 0):.3f}")
    print(f"  Avg confidence:     {state.get('avgConfidence', 0):.3f}")
    print(f"  Last updated:       {state.get('lastUpdated', 'never')}")
    print()

    # Model distribution
    dist = state.get("modelDistribution", {})
    if dist:
        print("  Model Distribution:")
        total = sum(dist.values())
        for model_id, count in sorted(dist.items(), key=lambda x: -x[1]):
            model = MODELS.get(model_id)
            name = model.display_name if model else model_id
            pct = (count / total) * 100 if total > 0 else 0
            print(f"    {name:25s}  {count:4d} ({pct:5.1f}%)")
    print()

    # Available API keys
    print("  API Key Status:")
    seen_keys = set()
    for model in MODELS.values():
        if model.env_key and model.env_key not in seen_keys:
            seen_keys.add(model.env_key)
            has_key = bool(os.environ.get(model.env_key))
            status = "OK" if has_key else "MISSING"
            print(f"    {model.env_key:25s}  [{status}]")
    print()

    # Owl builds
    print("  8OWLS Builds (Recommended):")
    for owl, build in OWL_BUILDS.items():
        model = MODELS.get(build["model"])
        cost = f"${model.input_cost_per_m}" if model else "?"
        print(f"    {owl:6s} ({build['phase']:8s}) -> {build['model']:20s}  {cost}/M")
    print("=" * 60)


def show_pricing():
    """Display full pricing comparison."""
    print("=" * 80)
    print("  MODEL PRICING COMPARISON (Feb 2026)")
    print("=" * 80)
    print(f"  {'Model':<25s} {'In $/M':>8s} {'Out $/M':>9s} {'Context':>10s} {'SWE%':>6s} {'Key':>5s}")
    print("-" * 80)

    sorted_models = sorted(MODELS.values(), key=lambda m: m.input_cost_per_m)
    for m in sorted_models:
        has_key = "OK" if check_api_key(m) else "NO"
        avail = "" if m.available else " [SOON]"
        ctx = f"{m.context_window // 1000}K" if m.context_window < 1_000_000 else f"{m.context_window // 1_000_000}M"
        print(f"  {m.display_name + avail:<25s} ${m.input_cost_per_m:>6.3f} ${m.output_cost_per_m:>7.2f} {ctx:>10s} {m.swe_bench_score:>5.1f} {has_key:>5s}")
    print("=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 tools/model_router.py route 'Write a React component'")
        print("  python3 tools/model_router.py route --task-type code_backend --complexity 0.7 'Build API'")
        print("  python3 tools/model_router.py route --max-cost 1.0 --require search 'Find latest news'")
        print("  python3 tools/model_router.py status")
        print("  python3 tools/model_router.py pricing")
        print("  python3 tools/model_router.py owls")
        print("  python3 tools/model_router.py loadouts")
        return

    cmd = sys.argv[1]

    if cmd == "status":
        show_status()

    elif cmd == "pricing":
        show_pricing()

    elif cmd == "owls":
        print("\n8OWLS Multi-Model Build Configuration:\n")
        total_input = 0
        total_output = 0
        for owl, build in OWL_BUILDS.items():
            model = MODELS.get(build["model"])
            if model:
                total_input += model.input_cost_per_m
                total_output += model.output_cost_per_m
                print(f"  {owl:6s} | {build['phase']:8s} | {model.display_name:20s} | ${model.input_cost_per_m:.2f}/${model.output_cost_per_m:.2f} per M")
        print(f"\n  Total per 8-owl emergence (1M tokens each): ~${total_input:.2f} input / ${total_output:.2f} output")
        all_opus_cost = 5.00 * 8
        print(f"  vs All-Opus cost: ${all_opus_cost:.2f} input")
        savings = ((all_opus_cost - total_input) / all_opus_cost) * 100
        print(f"  Savings: {savings:.0f}%")

    elif cmd == "loadouts":
        print("\nWeEvolve Loadout Presets:\n")
        for name, preset in LOADOUT_PRESETS.items():
            primary = MODELS.get(preset["primary"])
            secondary = MODELS.get(preset["secondary"])
            p_name = primary.display_name if primary else preset["primary"]
            s_name = secondary.display_name if secondary else preset["secondary"]
            print(f"  [{name.upper():10s}] {preset['description']}")
            print(f"    Primary:   {p_name}")
            print(f"    Secondary: {s_name}")
            print()

    elif cmd == "route":
        # Parse args
        task_type = None
        complexity = None
        max_cost = None
        require_cap = None
        prefer_speed = False
        prompt_parts = []

        i = 2
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "--task-type" and i + 1 < len(sys.argv):
                task_type = sys.argv[i + 1]
                i += 2
            elif arg == "--complexity" and i + 1 < len(sys.argv):
                complexity = float(sys.argv[i + 1])
                i += 2
            elif arg == "--max-cost" and i + 1 < len(sys.argv):
                max_cost = float(sys.argv[i + 1])
                i += 2
            elif arg == "--require" and i + 1 < len(sys.argv):
                require_cap = sys.argv[i + 1]
                i += 2
            elif arg == "--fast":
                prefer_speed = True
                i += 1
            else:
                prompt_parts.append(arg)
                i += 1

        prompt = " ".join(prompt_parts)
        if not prompt:
            print("Error: No prompt provided")
            return

        decision = route(
            prompt=prompt,
            task_type=task_type,
            complexity=complexity,
            max_cost_per_m=max_cost,
            require_capability=require_cap,
            prefer_speed=prefer_speed,
        )

        print(json.dumps(asdict(decision), indent=2))

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: route, status, pricing, owls, loadouts")


if __name__ == "__main__":
    main()
