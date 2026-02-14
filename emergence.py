"""
8 Owls Emergence Orchestrator
==============================
Spawns 7 owl agents in parallel for multi-perspective analysis.
Each owl runs as a separate thread calling Claude Haiku with its
constrained persona prompt.

Cost: ~$0.01-0.02 per full emergence (7 Haiku calls)
Time: Under 30 seconds for 7 owls (parallel execution)
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from weevolve.owls.personas import (
    OWL_PERSONAS,
    SPAWNABLE_OWLS,
    QUICK_OWLS,
    build_owl_prompt,
    format_owl_header,
)
from weevolve.owls.synthesis import synthesize

logger = logging.getLogger("weevolve.owls")

# Model for owl calls -- Haiku for cost efficiency
OWL_MODEL = "claude-haiku-4-5-20251001"

# Timeouts and limits
OWL_TIMEOUT_SECONDS = 45
MAX_RETRIES = 2
RETRY_BASE_DELAY = 1.0
COMPLEXITY_THRESHOLD_DEFAULT = 7


def _get_client():
    """
    Lazily create an Anthropic client. Returns None if unavailable.
    Checks for API key presence before attempting client creation.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("No ANTHROPIC_API_KEY set. Emergence disabled.")
        return None

    try:
        import anthropic
        return anthropic.Anthropic()
    except ImportError:
        logger.warning("anthropic package not installed. Emergence disabled.")
        return None
    except Exception as exc:
        logger.error("Failed to create Anthropic client: %s", exc)
        return None


def _call_owl(
    client,
    owl_name: str,
    task: str,
    context: Optional[Dict] = None,
) -> Dict:
    """
    Call Claude with a single owl's persona. Handles retries with
    exponential backoff for rate limits.

    Returns a dict:
        {
            "owl": "LYRA",
            "phase": "PERCEIVE",
            "response": "...",
            "tokens_in": 123,
            "tokens_out": 456,
            "latency_ms": 789,
            "success": True,
            "error": None,
        }
    """
    persona = OWL_PERSONAS.get(owl_name)
    if not persona:
        return {
            "owl": owl_name,
            "phase": "UNKNOWN",
            "response": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_ms": 0,
            "success": False,
            "error": f"Unknown owl: {owl_name}",
        }

    user_prompt = build_owl_prompt(owl_name, task, context)
    max_tokens = persona["max_tokens"]

    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            start = time.monotonic()
            response = client.messages.create(
                model=OWL_MODEL,
                max_tokens=max_tokens,
                system=persona["system_prompt"],
                messages=[{"role": "user", "content": user_prompt}],
            )
            elapsed_ms = int((time.monotonic() - start) * 1000)

            text = response.content[0].text if response.content else ""
            usage = response.usage

            return {
                "owl": owl_name,
                "phase": persona["phase"],
                "response": text,
                "tokens_in": usage.input_tokens if usage else 0,
                "tokens_out": usage.output_tokens if usage else 0,
                "latency_ms": elapsed_ms,
                "success": True,
                "error": None,
            }

        except Exception as exc:
            last_error = str(exc)
            error_lower = last_error.lower()

            # Retry on rate limit or overloaded errors
            is_retryable = (
                "rate" in error_lower
                or "overloaded" in error_lower
                or "529" in error_lower
                or "429" in error_lower
            )

            if is_retryable and attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.info(
                    "%s retry %d/%d in %.1fs: %s",
                    owl_name, attempt + 1, MAX_RETRIES, delay, last_error,
                )
                time.sleep(delay)
                continue

            logger.error("%s failed after %d attempts: %s", owl_name, attempt + 1, last_error)
            break

    return {
        "owl": owl_name,
        "phase": persona.get("phase", "UNKNOWN"),
        "response": "",
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0,
        "success": False,
        "error": last_error,
    }


def emerge(
    task: str,
    context: Optional[Dict] = None,
    verbose: bool = True,
    timeout: int = OWL_TIMEOUT_SECONDS,
) -> Dict:
    """
    Full 8 Owls emergence. Spawns 7 background owl agents in parallel,
    collects their analyses, then synthesizes via SOWL.

    Args:
        task: The task or question to analyze.
        context: Optional dict with additional context:
            - files: list of relevant file paths
            - codebase_info: string describing the codebase
            - constraints: known constraints
            - previous_attempts: what was tried before
            - extra: any other context
        verbose: Print progress to stdout.
        timeout: Max seconds to wait for all owls.

    Returns:
        {
            "task": "...",
            "owls": {
                "LYRA": { "response": "...", "success": True, ... },
                "PRISM": { ... },
                ...
            },
            "synthesis": { ... },
            "meta": {
                "total_tokens_in": ...,
                "total_tokens_out": ...,
                "total_latency_ms": ...,
                "estimated_cost_usd": ...,
                "owls_succeeded": 7,
                "owls_failed": 0,
            },
        }
    """
    client = _get_client()
    if client is None:
        if verbose:
            print("  [SKIP] Emergence disabled (no API key or anthropic package)")
        return _empty_result(task)

    if verbose:
        print(f"\n{'=' * 60}")
        print("  (*) 8 OWLS EMERGENCE")
        print(f"{'=' * 60}")
        print(f"  Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"  Spawning {len(SPAWNABLE_OWLS)} owls in parallel...")
        print()

    owl_results = {}
    start_total = time.monotonic()

    with ThreadPoolExecutor(max_workers=7) as executor:
        futures = {
            executor.submit(_call_owl, client, owl, task, context): owl
            for owl in SPAWNABLE_OWLS
        }

        for future in as_completed(futures, timeout=timeout):
            owl_name = futures[future]
            try:
                result = future.result(timeout=5)
                owl_results[result["owl"]] = result

                if verbose:
                    status = "OK" if result["success"] else "FAIL"
                    header = format_owl_header(result["owl"])
                    latency = result["latency_ms"]
                    print(f"  [{status}] {header} ({latency}ms)")

            except TimeoutError:
                owl_results[owl_name] = {
                    "owl": owl_name,
                    "phase": OWL_PERSONAS.get(owl_name, {}).get("phase", "UNKNOWN"),
                    "response": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "latency_ms": timeout * 1000,
                    "success": False,
                    "error": "Timed out",
                }
                if verbose:
                    print(f"  [TIMEOUT] {owl_name}")

            except Exception as exc:
                owl_results[owl_name] = {
                    "owl": owl_name,
                    "phase": OWL_PERSONAS.get(owl_name, {}).get("phase", "UNKNOWN"),
                    "response": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "latency_ms": 0,
                    "success": False,
                    "error": str(exc),
                }
                if verbose:
                    print(f"  [ERROR] {owl_name}: {exc}")

    total_ms = int((time.monotonic() - start_total) * 1000)

    # Compute meta stats
    total_in = sum(r["tokens_in"] for r in owl_results.values())
    total_out = sum(r["tokens_out"] for r in owl_results.values())
    succeeded = sum(1 for r in owl_results.values() if r["success"])
    failed = sum(1 for r in owl_results.values() if not r["success"])

    # Haiku pricing: $0.80/MTok input, $4.00/MTok output
    estimated_cost = (total_in * 0.80 + total_out * 4.00) / 1_000_000

    if verbose:
        print()
        print(f"  7 owls complete in {total_ms}ms ({succeeded} OK, {failed} failed)")
        print(f"  Tokens: {total_in} in / {total_out} out (${estimated_cost:.4f})")

    # Synthesize via SOWL
    if verbose:
        print()
        print(f"  \033[31mSOWL\033[0m \033[2mIMPROVE\033[0m synthesizing...")

    synthesis_result = synthesize(owl_results, task, client, verbose=verbose)

    # Add synthesis tokens to totals
    if synthesis_result.get("tokens_in"):
        total_in += synthesis_result["tokens_in"]
        total_out += synthesis_result["tokens_out"]
        estimated_cost = (total_in * 0.80 + total_out * 4.00) / 1_000_000

    if verbose:
        print()
        print(f"{'=' * 60}")
        print(f"  EMERGENCE COMPLETE")
        print(f"  Total: {total_in + total_out} tokens (${estimated_cost:.4f})")
        print(f"{'=' * 60}")
        print()

    return {
        "task": task,
        "owls": owl_results,
        "synthesis": synthesis_result,
        "meta": {
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_latency_ms": total_ms,
            "estimated_cost_usd": round(estimated_cost, 6),
            "owls_succeeded": succeeded,
            "owls_failed": failed,
        },
    }


def quick_emerge(
    task: str,
    context: Optional[Dict] = None,
    verbose: bool = True,
    timeout: int = OWL_TIMEOUT_SECONDS,
) -> Dict:
    """
    Quick 3-owl emergence using LYRA + SAGE + QUEST.
    Faster and cheaper -- use for simpler tasks or when time is tight.

    Same interface as emerge() but spawns only 3 owls.
    """
    client = _get_client()
    if client is None:
        if verbose:
            print("  [SKIP] Quick emergence disabled (no API key or anthropic package)")
        return _empty_result(task)

    if verbose:
        print(f"\n{'=' * 60}")
        print("  (*) QUICK EMERGENCE (3 owls)")
        print(f"{'=' * 60}")
        print(f"  Task: {task[:80]}{'...' if len(task) > 80 else ''}")
        print(f"  Spawning: {', '.join(QUICK_OWLS)}")
        print()

    owl_results = {}
    start_total = time.monotonic()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_call_owl, client, owl, task, context): owl
            for owl in QUICK_OWLS
        }

        for future in as_completed(futures, timeout=timeout):
            owl_name = futures[future]
            try:
                result = future.result(timeout=5)
                owl_results[result["owl"]] = result

                if verbose:
                    status = "OK" if result["success"] else "FAIL"
                    header = format_owl_header(result["owl"])
                    latency = result["latency_ms"]
                    print(f"  [{status}] {header} ({latency}ms)")

            except TimeoutError:
                owl_results[owl_name] = {
                    "owl": owl_name,
                    "phase": OWL_PERSONAS.get(owl_name, {}).get("phase", "UNKNOWN"),
                    "response": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "latency_ms": timeout * 1000,
                    "success": False,
                    "error": "Timed out",
                }
                if verbose:
                    print(f"  [TIMEOUT] {owl_name}")

            except Exception as exc:
                owl_results[owl_name] = {
                    "owl": owl_name,
                    "phase": OWL_PERSONAS.get(owl_name, {}).get("phase", "UNKNOWN"),
                    "response": "",
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "latency_ms": 0,
                    "success": False,
                    "error": str(exc),
                }
                if verbose:
                    print(f"  [ERROR] {owl_name}: {exc}")

    total_ms = int((time.monotonic() - start_total) * 1000)

    total_in = sum(r["tokens_in"] for r in owl_results.values())
    total_out = sum(r["tokens_out"] for r in owl_results.values())
    succeeded = sum(1 for r in owl_results.values() if r["success"])
    failed = sum(1 for r in owl_results.values() if not r["success"])

    estimated_cost = (total_in * 0.80 + total_out * 4.00) / 1_000_000

    # Quick emergence: synthesize from partial results
    synthesis_result = synthesize(
        owl_results, task, client, verbose=verbose, quick_mode=True
    )

    if synthesis_result.get("tokens_in"):
        total_in += synthesis_result["tokens_in"]
        total_out += synthesis_result["tokens_out"]
        estimated_cost = (total_in * 0.80 + total_out * 4.00) / 1_000_000

    if verbose:
        print()
        print(f"{'=' * 60}")
        print(f"  QUICK EMERGENCE COMPLETE ({total_ms}ms, ${estimated_cost:.4f})")
        print(f"{'=' * 60}")
        print()

    return {
        "task": task,
        "owls": owl_results,
        "synthesis": synthesis_result,
        "meta": {
            "total_tokens_in": total_in,
            "total_tokens_out": total_out,
            "total_latency_ms": total_ms,
            "estimated_cost_usd": round(estimated_cost, 6),
            "owls_succeeded": succeeded,
            "owls_failed": failed,
        },
    }


def estimate_complexity(task: str) -> int:
    """
    Estimate task complexity on a 1-10 scale using simple heuristics.
    Used to decide whether full emergence is warranted.

    Factors:
        - Task length (longer = more complex)
        - Keyword signals (architecture, refactor, security, etc.)
        - Question marks (more questions = more complex)
    """
    score = 3  # baseline

    # Length signal
    word_count = len(task.split())
    if word_count > 100:
        score += 3
    elif word_count > 50:
        score += 2
    elif word_count > 20:
        score += 1

    # Complexity keywords
    high_complexity = [
        "architect", "refactor", "redesign", "migration", "security",
        "performance", "scale", "distributed", "concurrent", "async",
        "breaking change", "backwards compatible", "multi-tenant",
        "authentication", "authorization", "encryption",
    ]
    medium_complexity = [
        "api", "database", "cache", "queue", "deploy", "test",
        "integration", "module", "service", "pipeline", "workflow",
    ]

    task_lower = task.lower()
    high_hits = sum(1 for kw in high_complexity if kw in task_lower)
    medium_hits = sum(1 for kw in medium_complexity if kw in task_lower)

    # First high hit = +2, each additional = +1 (diminishing)
    if high_hits > 0:
        score += 2 + min(high_hits - 1, 2)

    # First medium hit = +1, each additional = +0.5
    if medium_hits > 0:
        score += 1 + min(medium_hits - 1, 2)

    # Question marks signal uncertainty
    question_count = task.count("?")
    if question_count >= 3:
        score += 1

    return min(10, max(1, score))


def should_emerge(
    task: str,
    threshold: int = COMPLEXITY_THRESHOLD_DEFAULT,
) -> bool:
    """
    Determine whether a task warrants full 8-owl emergence.
    Returns True if complexity >= threshold.
    """
    return estimate_complexity(task) >= threshold


def _empty_result(task: str) -> Dict:
    """Return an empty emergence result (used when API is unavailable)."""
    return {
        "task": task,
        "owls": {},
        "synthesis": {},
        "meta": {
            "total_tokens_in": 0,
            "total_tokens_out": 0,
            "total_latency_ms": 0,
            "estimated_cost_usd": 0,
            "owls_succeeded": 0,
            "owls_failed": 0,
        },
    }
