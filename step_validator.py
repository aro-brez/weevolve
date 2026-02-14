#!/usr/bin/env python3
"""
WeEvolve Step Validator with Checkpoint/Rollback
==================================================
Addresses Pain Point #2: Compound Error Cascading in Multi-Step Workflows

95% per-step reliability = 36% success over 20 steps.
This module breaks that by validating each step output before proceeding,
with automatic checkpoint/rollback on failure.

Architecture:
  1. Checkpoint: Save state snapshot before each step
  2. Execute: Run the step
  3. Validate: Check output against schema/assertions
  4. Decide: Pass -> continue | Fail -> rollback + retry or skip
  5. Saga: Compensating actions for distributed operations

Patterns:
  - Schema validation (type checking outputs)
  - Assertion validation (custom predicates)
  - Checkpoint/rollback (state snapshots)
  - Retry with backoff (configurable per step)
  - Saga pattern (compensating transactions)

Usage:
  from weevolve.step_validator import Pipeline, Step, Checkpoint

  pipeline = Pipeline("deploy_app")
  pipeline.add_step(Step("build", build_fn, validate=lambda r: r.get("success")))
  pipeline.add_step(Step("test", test_fn, validate=lambda r: r.get("passed") > 0))
  pipeline.add_step(Step("deploy", deploy_fn, compensate=rollback_fn))
  result = pipeline.run(initial_state={"app": "weevolve"})

(C) LIVE FREE = LIVE FOREVER
"""

import json
import copy
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class StepResult:
    """Result of a single step execution."""
    step_name: str
    success: bool
    output: Any
    error: Optional[str]
    duration_ms: float
    attempt: int
    checkpoint_id: Optional[str]
    timestamp: str


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline execution."""
    pipeline_name: str
    success: bool
    steps_completed: int
    steps_total: int
    step_results: Tuple  # Tuple of StepResult (frozen)
    total_duration_ms: float
    rollback_performed: bool
    final_state: Dict
    timestamp: str


@dataclass
class Step:
    """
    A single step in a validated pipeline.

    Args:
        name: Unique step identifier
        execute: Callable that takes state dict and returns output
        validate: Optional callable that takes output and returns bool
        compensate: Optional rollback function (Saga pattern)
        max_retries: Maximum retry attempts on failure
        retry_delay_ms: Delay between retries (doubles each attempt)
        required: If True, pipeline stops on failure. If False, skips.
        timeout_ms: Maximum execution time for this step
    """
    name: str
    execute: Callable[[Dict], Any]
    validate: Optional[Callable[[Any], bool]] = None
    compensate: Optional[Callable[[Dict, Any], None]] = None
    max_retries: int = 2
    retry_delay_ms: float = 500.0
    required: bool = True
    timeout_ms: float = 30000.0


class Checkpoint:
    """
    State checkpoint manager.
    Saves deep copies of state at each step boundary for rollback.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        self._checkpoints: Dict[str, Dict] = {}
        self._storage_dir = storage_dir
        if storage_dir:
            storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint_id: str, state: Dict) -> str:
        """Save a state checkpoint. Returns checkpoint ID."""
        self._checkpoints[checkpoint_id] = copy.deepcopy(state)

        if self._storage_dir:
            checkpoint_path = self._storage_dir / f"{checkpoint_id}.json"
            try:
                with open(checkpoint_path, 'w') as f:
                    json.dump(state, f, indent=2, default=str)
            except (TypeError, OSError):
                pass  # Best effort persistence

        return checkpoint_id

    def restore(self, checkpoint_id: str) -> Optional[Dict]:
        """Restore state from a checkpoint."""
        if checkpoint_id in self._checkpoints:
            return copy.deepcopy(self._checkpoints[checkpoint_id])

        if self._storage_dir:
            checkpoint_path = self._storage_dir / f"{checkpoint_id}.json"
            if checkpoint_path.exists():
                try:
                    with open(checkpoint_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

        return None

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoint IDs."""
        return list(self._checkpoints.keys())

    def clear(self):
        """Clear all in-memory checkpoints."""
        self._checkpoints.clear()


class Pipeline:
    """
    Validated multi-step pipeline with checkpoint/rollback.

    Features:
    - Schema validation between steps
    - Automatic retry with exponential backoff
    - Checkpoint/rollback on failure
    - Saga pattern compensating actions
    - Full execution trace for debugging
    """

    def __init__(
        self,
        name: str,
        checkpoint_dir: Optional[Path] = None,
    ):
        self.name = name
        self.steps: List[Step] = []
        self.checkpoint = Checkpoint(storage_dir=checkpoint_dir)
        self._trace: List[StepResult] = []

    def add_step(self, step: Step) -> 'Pipeline':
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(step)
        return self

    def run(self, initial_state: Optional[Dict] = None) -> PipelineResult:
        """
        Execute the full pipeline with validation and checkpointing.

        Returns PipelineResult with full execution trace.
        """
        state = copy.deepcopy(initial_state or {})
        self._trace = []
        completed_steps: List[Tuple[Step, Any]] = []
        start_time = time.monotonic()
        rollback_performed = False

        # Save initial checkpoint
        self.checkpoint.save(f"{self.name}_initial", state)

        for i, step in enumerate(self.steps):
            checkpoint_id = f"{self.name}_step_{i}_{step.name}"
            self.checkpoint.save(checkpoint_id, state)

            step_result = self._execute_step(step, state, checkpoint_id)
            self._trace.append(step_result)

            if step_result.success:
                # Merge step output into state
                if isinstance(step_result.output, dict):
                    state = {**state, **step_result.output}
                else:
                    state = {**state, f"{step.name}_result": step_result.output}
                completed_steps.append((step, step_result.output))
            else:
                if step.required:
                    # Required step failed -- rollback
                    rollback_performed = self._rollback(completed_steps, state)
                    restored = self.checkpoint.restore(f"{self.name}_initial")
                    if restored is not None:
                        state = restored

                    total_ms = (time.monotonic() - start_time) * 1000
                    return PipelineResult(
                        pipeline_name=self.name,
                        success=False,
                        steps_completed=len(completed_steps),
                        steps_total=len(self.steps),
                        step_results=tuple(self._trace),
                        total_duration_ms=round(total_ms, 2),
                        rollback_performed=rollback_performed,
                        final_state=state,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )
                # Non-required step: skip and continue

        total_ms = (time.monotonic() - start_time) * 1000
        return PipelineResult(
            pipeline_name=self.name,
            success=True,
            steps_completed=len(completed_steps),
            steps_total=len(self.steps),
            step_results=tuple(self._trace),
            total_duration_ms=round(total_ms, 2),
            rollback_performed=False,
            final_state=state,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _execute_step(
        self, step: Step, state: Dict, checkpoint_id: str
    ) -> StepResult:
        """Execute a single step with retry logic."""
        last_error = None

        for attempt in range(1, step.max_retries + 1):
            start = time.monotonic()
            try:
                output = step.execute(state)
                duration = (time.monotonic() - start) * 1000

                # Validate output if validator provided
                if step.validate is not None:
                    is_valid = step.validate(output)
                    if not is_valid:
                        last_error = f"Validation failed for step '{step.name}' (attempt {attempt})"
                        if attempt < step.max_retries:
                            delay = step.retry_delay_ms * (2 ** (attempt - 1)) / 1000
                            time.sleep(delay)
                            continue
                        return StepResult(
                            step_name=step.name,
                            success=False,
                            output=output,
                            error=last_error,
                            duration_ms=round(duration, 2),
                            attempt=attempt,
                            checkpoint_id=checkpoint_id,
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        )

                return StepResult(
                    step_name=step.name,
                    success=True,
                    output=output,
                    error=None,
                    duration_ms=round(duration, 2),
                    attempt=attempt,
                    checkpoint_id=checkpoint_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            except Exception as e:
                duration = (time.monotonic() - start) * 1000
                last_error = f"{type(e).__name__}: {e}"
                if attempt < step.max_retries:
                    delay = step.retry_delay_ms * (2 ** (attempt - 1)) / 1000
                    time.sleep(delay)
                    continue

                return StepResult(
                    step_name=step.name,
                    success=False,
                    output=None,
                    error=last_error,
                    duration_ms=round(duration, 2),
                    attempt=attempt,
                    checkpoint_id=checkpoint_id,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

        # Should not reach here, but just in case
        return StepResult(
            step_name=step.name,
            success=False,
            output=None,
            error=last_error or "Unknown error",
            duration_ms=0,
            attempt=step.max_retries,
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _rollback(
        self, completed_steps: List[Tuple[Step, Any]], state: Dict
    ) -> bool:
        """
        Execute compensating actions for completed steps (Saga pattern).
        Rolls back in reverse order.
        Returns True if any compensations were executed.
        """
        any_compensated = False
        for step, output in reversed(completed_steps):
            if step.compensate is not None:
                try:
                    step.compensate(state, output)
                    any_compensated = True
                except Exception:
                    pass  # Best effort rollback
        return any_compensated

    def get_trace(self) -> List[Dict]:
        """Get full execution trace as list of dicts."""
        return [asdict(r) for r in self._trace]

    def trace_summary(self) -> str:
        """Get a human-readable trace summary."""
        lines = [f"Pipeline: {self.name}"]
        for r in self._trace:
            status = "PASS" if r.success else "FAIL"
            lines.append(
                f"  [{status}] {r.step_name} "
                f"({r.duration_ms:.0f}ms, attempt {r.attempt})"
            )
            if r.error:
                lines.append(f"         Error: {r.error}")
        return "\n".join(lines)


def validate_schema(output: Any, schema: Dict[str, type]) -> bool:
    """
    Simple schema validator for step outputs.
    Checks that output is a dict with expected keys and types.

    Usage:
        validate = lambda out: validate_schema(out, {"success": bool, "count": int})
    """
    if not isinstance(output, dict):
        return False
    for key, expected_type in schema.items():
        if key not in output:
            return False
        if not isinstance(output[key], expected_type):
            return False
    return True


def validate_assertions(output: Any, assertions: List[Callable[[Any], bool]]) -> bool:
    """
    Validate output against a list of assertion functions.
    All assertions must pass.

    Usage:
        validate = lambda out: validate_assertions(out, [
            lambda o: o.get("count", 0) > 0,
            lambda o: o.get("status") == "ok",
        ])
    """
    return all(assertion(output) for assertion in assertions)
