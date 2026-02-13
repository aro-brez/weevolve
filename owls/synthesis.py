"""
Owl Synthesis - Combines 7 owl reports into a final recommendation.
====================================================================
Takes the raw owl outputs, feeds them to SOWL's persona for synthesis,
and extracts learnings worth persisting to the knowledge base.
"""

import time
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timezone

from weevolve.owls.personas import OWL_PERSONAS, format_owl_header

logger = logging.getLogger("weevolve.owls")

OWL_MODEL = "claude-haiku-4-5-20251001"

# Reset code for terminal colors
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def synthesize(
    owl_results: Dict[str, Dict],
    task: str,
    client,
    verbose: bool = True,
    quick_mode: bool = False,
) -> Dict:
    """
    Synthesize owl reports into a final recommendation using SOWL's persona.

    Args:
        owl_results: Dict mapping owl names to their result dicts.
        task: The original task string.
        client: An initialized Anthropic client.
        verbose: Print synthesis output.
        quick_mode: If True, uses a lighter synthesis prompt for 3-owl results.

    Returns:
        {
            "recommendation": "...",
            "synthesis_text": "...",
            "learnings": [ ... ],
            "tokens_in": ...,
            "tokens_out": ...,
            "latency_ms": ...,
            "success": True,
        }
    """
    # Build the combined report from all successful owls
    report_sections = []
    successful_owls = []

    for owl_name, result in sorted(owl_results.items()):
        if not result.get("success") or not result.get("response"):
            continue
        successful_owls.append(owl_name)
        phase = result.get("phase", "UNKNOWN")
        report_sections.append(
            f"=== {owl_name} ({phase}) ===\n{result['response']}"
        )

    if not report_sections:
        logger.warning("No successful owl reports to synthesize")
        return _empty_synthesis()

    combined_report = "\n\n".join(report_sections)

    # Build synthesis prompt
    if quick_mode:
        synthesis_prompt = _build_quick_synthesis_prompt(task, combined_report, successful_owls)
        max_tokens = 500
    else:
        synthesis_prompt = _build_full_synthesis_prompt(task, combined_report)
        max_tokens = OWL_PERSONAS["SOWL"]["max_tokens"]

    system_prompt = OWL_PERSONAS["SOWL"]["system_prompt"]

    try:
        start = time.monotonic()
        response = client.messages.create(
            model=OWL_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": synthesis_prompt}],
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        text = response.content[0].text if response.content else ""
        usage = response.usage

        # Extract learnings from the synthesis
        learnings = extract_learnings(text, task)

        # Extract the recommendation line
        recommendation = _extract_section(text, "RECOMMENDATION")

        if verbose:
            _print_synthesis(text, learnings, elapsed_ms)

        return {
            "recommendation": recommendation,
            "synthesis_text": text,
            "learnings": learnings,
            "tokens_in": usage.input_tokens if usage else 0,
            "tokens_out": usage.output_tokens if usage else 0,
            "latency_ms": elapsed_ms,
            "success": True,
        }

    except Exception as exc:
        logger.error("Synthesis failed: %s", exc)
        if verbose:
            print(f"  [ERROR] Synthesis failed: {exc}")
        return _empty_synthesis(error=str(exc))


def extract_learnings(synthesis_text: str, task: str) -> List[Dict]:
    """
    Extract persistable learnings from the synthesis text.
    Looks for the LEARNINGS TO PERSIST section and parses each item.
    """
    learnings = []

    section = _extract_section(synthesis_text, "LEARNINGS TO PERSIST")
    if not section:
        section = _extract_section(synthesis_text, "LEARNINGS")

    if not section:
        return learnings

    # Parse bullet points or numbered items
    lines = section.strip().split("\n")
    for line in lines:
        cleaned = line.strip().lstrip("-").lstrip("0123456789.").strip()
        if cleaned and len(cleaned) > 10:
            learnings.append({
                "insight": cleaned,
                "source": "emergence",
                "task": task[:200],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    return learnings


def persist_learnings(learnings: List[Dict], verbose: bool = True) -> int:
    """
    Persist extracted learnings to the WeEvolve knowledge base.
    Uses the core learn() function with source_type='text'.

    Returns the number of learnings successfully persisted.
    """
    if not learnings:
        return 0

    persisted = 0
    try:
        from weevolve.core import learn as core_learn
    except ImportError:
        logger.warning("Cannot import weevolve.core.learn -- skipping persistence")
        return 0

    for learning in learnings:
        content = (
            f"Emergence Learning: {learning['insight']}\n"
            f"Task: {learning['task']}\n"
            f"Source: 8 Owls Emergence"
        )
        try:
            result = core_learn(content, source_type="text", verbose=False)
            if result is not None:
                persisted += 1
        except Exception as exc:
            logger.error("Failed to persist learning: %s", exc)

    if verbose and persisted > 0:
        print(f"  {GREEN}Persisted {persisted} learning(s) to knowledge base{RESET}")

    return persisted


def format_emergence_report(emergence_result: Dict) -> str:
    """
    Format a complete emergence result into a readable text report.
    Useful for logging, sharing, or piping to other tools.
    """
    lines = []
    task = emergence_result.get("task", "Unknown task")
    meta = emergence_result.get("meta", {})

    lines.append("=" * 60)
    lines.append("8 OWLS EMERGENCE REPORT")
    lines.append("=" * 60)
    lines.append(f"Task: {task}")
    lines.append(
        f"Owls: {meta.get('owls_succeeded', 0)} succeeded, "
        f"{meta.get('owls_failed', 0)} failed"
    )
    lines.append(
        f"Cost: ${meta.get('estimated_cost_usd', 0):.4f} | "
        f"Time: {meta.get('total_latency_ms', 0)}ms"
    )
    lines.append("")

    # Individual owl reports
    owls = emergence_result.get("owls", {})
    for owl_name in ["LYRA", "PRISM", "SAGE", "QUEST", "NOVA", "ECHO", "LUNA"]:
        result = owls.get(owl_name)
        if not result:
            continue

        phase = result.get("phase", "UNKNOWN")
        status = "OK" if result.get("success") else "FAILED"
        lines.append(f"--- {owl_name} ({phase}) [{status}] ---")
        if result.get("response"):
            lines.append(result["response"])
        elif result.get("error"):
            lines.append(f"Error: {result['error']}")
        lines.append("")

    # Synthesis
    synthesis = emergence_result.get("synthesis", {})
    if synthesis.get("synthesis_text"):
        lines.append("--- SOWL (IMPROVE) [SYNTHESIS] ---")
        lines.append(synthesis["synthesis_text"])
        lines.append("")

    # Learnings
    learnings = synthesis.get("learnings", [])
    if learnings:
        lines.append("EXTRACTED LEARNINGS:")
        for i, learning in enumerate(learnings, 1):
            lines.append(f"  {i}. {learning['insight']}")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def _build_full_synthesis_prompt(task: str, combined_report: str) -> str:
    """Build the user prompt for full 7-owl synthesis."""
    return (
        f"ORIGINAL TASK: {task}\n\n"
        f"Below are reports from 7 owl analysts. "
        f"Synthesize them into a final recommendation.\n\n"
        f"{combined_report}"
    )


def _build_quick_synthesis_prompt(
    task: str, combined_report: str, owls_used: List[str]
) -> str:
    """Build the user prompt for quick 3-owl synthesis."""
    owl_list = ", ".join(owls_used)
    return (
        f"ORIGINAL TASK: {task}\n\n"
        f"This is a QUICK emergence with only {owl_list}. "
        f"Provide a focused synthesis -- shorter than a full emergence.\n\n"
        f"Focus on:\n"
        f"- SYNTHESIS: 1-2 sentences combining the signals.\n"
        f"- RECOMMENDATION: One action.\n"
        f"- CRITICAL RISKS: Top 1 risk from QUEST.\n\n"
        f"{combined_report}"
    )


def _extract_section(text: str, section_name: str) -> str:
    """Extract content after a section header like 'RECOMMENDATION:' from text."""
    upper_text = text.upper()
    header_variants = [
        f"{section_name.upper()}:",
        f"**{section_name.upper()}**:",
        f"**{section_name.upper()}:**",
        f"{section_name.upper()} -",
    ]

    start_idx = -1
    for variant in header_variants:
        idx = upper_text.find(variant)
        if idx >= 0:
            start_idx = idx + len(variant)
            break

    if start_idx < 0:
        return ""

    # Find the next section header (or end of text)
    remaining = text[start_idx:]
    end_markers = [
        "\nSYNTHESIS:", "\nRECOMMENDATION:", "\nCRITICAL RISKS:",
        "\nQUICK WINS:", "\nLEARNINGS", "\nMETA:",
        "\n**SYNTHESIS", "\n**RECOMMENDATION", "\n**CRITICAL",
        "\n**QUICK", "\n**LEARNINGS", "\n**META",
    ]

    end_idx = len(remaining)
    for marker in end_markers:
        pos = remaining.upper().find(marker.upper())
        if 0 < pos < end_idx:
            end_idx = pos

    return remaining[:end_idx].strip()


def _print_synthesis(text: str, learnings: List[Dict], latency_ms: int):
    """Pretty-print the synthesis to stdout."""
    print(f"\n  {RED}SOWL{RESET} {DIM}IMPROVE{RESET} ({latency_ms}ms)")
    print(f"  {'~' * 50}")

    # Print each section with light formatting
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            print()
            continue

        # Highlight section headers
        upper = stripped.upper()
        is_header = any(
            upper.startswith(h) for h in [
                "SYNTHESIS", "RECOMMENDATION", "CRITICAL",
                "QUICK WIN", "LEARNING", "META",
                "**SYNTHESIS", "**RECOMMENDATION", "**CRITICAL",
                "**QUICK", "**LEARNING", "**META",
            ]
        )

        if is_header:
            print(f"  {BOLD}{stripped}{RESET}")
        else:
            print(f"  {stripped}")

    if learnings:
        print(f"\n  {GREEN}Extracted {len(learnings)} learning(s) to persist{RESET}")


def _empty_synthesis(error: Optional[str] = None) -> Dict:
    """Return an empty synthesis result."""
    return {
        "recommendation": "",
        "synthesis_text": "",
        "learnings": [],
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": 0,
        "success": False,
        "error": error,
    }
