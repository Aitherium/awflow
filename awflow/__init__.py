"""awflow: deterministic workflow runtime with journaled replay.

A lightweight engine for multi-agent orchestration on the AitherOS fleet.
Supports journaling, deterministic replay, and resume from interruption.

Usage:
    async def my_workflow():
        result = await agent("prompt")
        items = await parallel([coro1(), coro2()])
        summary = await pipeline(items, stage1, stage2)
        return summary

    result = await run_workflow(my_workflow, budget_tokens=100000)

Public API:
    run_workflow(script, *, journal_path=None, resume_from=None, budget_tokens=1000000)
    agent, parallel, pipeline, phase, log
    get_budget(), set_budget(total), set_concurrency_cap(n)
    Budget, BudgetExhausted, JournalError
"""

import logging
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

# Import sibling modules with error handling
try:
    from . import dispatch, journal, runtime  # noqa: F401 - re-exported; see __all__
except ImportError as e:
    raise ImportError(
        f"Failed to import awflow submodules. "
        f"This usually means the package is partially built. Error: {e}"
    ) from e

# Context variable to track the current runtime (async-safe)
_current_runtime: ContextVar[Optional[runtime.WorkflowRuntime]] = ContextVar(
    "_current_runtime", default=None
)


# ===== Public Exceptions =====


class BudgetExhausted(Exception):  # noqa: N818 - a state, not an error class
    """Raised when the token budget is exhausted before an agent call."""

    pass


class JournalError(Exception):
    """Raised when the journal is corrupt or unrecoverable."""

    pass


# ===== Public Budget Class =====


class Budget:
    """Tracks token budget: total, spent(), remaining().

    This is a thin wrapper exposing the runtime's Budget object.
    """

    def __init__(self, budget_obj: runtime.Budget):
        self._budget = budget_obj

    @property
    def total(self) -> int:
        """Total token budget."""
        return self._budget.total

    def spent(self) -> int:
        """Tokens spent so far."""
        return self._budget.spent()

    def remaining(self) -> int:
        """Tokens remaining."""
        return self._budget.remaining()


# ===== Public Primitives (ContextVar-based dispatch) =====


async def agent(
    prompt: str,
    *,
    schema: Optional[Any] = None,
    label: Optional[str] = None,
    phase: Optional[str] = None,
    model: Optional[str] = None,
    effort: Optional[int] = None,
) -> Any:
    """Call an LLM agent.

    Args:
        prompt: the request text
        schema: optional JSON Schema (dict or pydantic class); response is validated
        label: narrative label for this call (included in determinism hash)
        phase: phase name (included in determinism hash)
        model: backend model name (included in determinism hash)
        effort: reasoning effort tier (included in determinism hash)

    Returns:
        str when no schema; validated dict when schema is given. Returns None after retries.

    Raises:
        RuntimeError: if called outside run_workflow()
        BudgetExhausted: if token budget is exhausted
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("agent() called outside of run_workflow()")
    return await rt.agent(
        prompt, schema=schema, label=label, phase=phase, model=model, effort=effort
    )


async def parallel(thunks: list[Awaitable[Any]]) -> list[Any]:
    """Execute multiple coroutines concurrently with a barrier.

    All thunks start before any completes. A failing thunk becomes None in the result.
    parallel() itself never raises; exceptions become None.

    Args:
        thunks: list of awaitable objects (max 4096)

    Returns:
        list[T | None]: result from each thunk, or None if that thunk raised

    Raises:
        RuntimeError: if called outside run_workflow()
        ValueError: if len(thunks) > 4096
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("parallel() called outside of run_workflow()")
    return await rt.parallel(thunks)


async def pipeline(
    items: list[Any], *stages: Callable[..., Awaitable[Any]]
) -> list[Any]:
    """Flow each item through multiple stages independently (no barrier between stages).

    Items may be at different stages simultaneously. A stage that raises drops that
    item to None and skips remaining stages for it. Pipeline itself never raises.

    Each stage receives (prevResult, originalItem, index).

    Args:
        items: list of items to process (max 4096)
        *stages: async callables (prevResult, item, index) -> nextResult

    Returns:
        list[T | None]: final result for each item, or None if dropped

    Raises:
        RuntimeError: if called outside run_workflow()
        ValueError: if len(items) > 4096 or len(stages) == 0
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("pipeline() called outside of run_workflow()")
    return await rt.pipeline(items, *stages)


async def phase(title: str) -> None:
    """Log a phase marker. Journaled but no operational effect.

    Args:
        title: phase name (must not be empty)

    Raises:
        RuntimeError: if called outside run_workflow()
        ValueError: if title is empty
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("phase() called outside of run_workflow()")
    return await rt.phase(title)


async def log(msg: str) -> None:
    """Log a narrative message. Journaled but no operational effect.

    Args:
        msg: message text (must not be empty)

    Raises:
        RuntimeError: if called outside run_workflow()
        ValueError: if msg is empty
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("log() called outside of run_workflow()")
    return await rt.log(msg)


def get_budget() -> Budget:
    """Get the current budget (only valid inside run_workflow).

    Returns:
        Budget object with total, spent(), remaining()

    Raises:
        RuntimeError: if called outside run_workflow()
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("get_budget() called outside of run_workflow()")
    return Budget(rt.budget)


def set_budget(total: int) -> None:
    """Set the current budget (only valid inside run_workflow).

    Args:
        total: new budget in tokens

    Raises:
        RuntimeError: if called outside run_workflow()
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("set_budget() called outside of run_workflow()")
    rt.set_budget_total(total)


def set_concurrency_cap(n: int) -> None:
    """Set the concurrency cap (only valid inside run_workflow).

    Default: min(16, cpu_count - 2). Excess work queues rather than erroring.

    Args:
        n: new concurrency cap

    Raises:
        RuntimeError: if called outside run_workflow()
    """
    rt = _current_runtime.get()
    if rt is None:
        raise RuntimeError("set_concurrency_cap() called outside of run_workflow()")
    rt.set_concurrency_cap(n)


# ===== Main Entry Point =====


async def run_workflow(
    script: Callable[..., Awaitable[Any]],
    *,
    journal_path: Optional[Path | str] = None,
    resume_from: Optional[str] = None,
    budget_tokens: int = 1000000,
    mirror: bool = True,
) -> Any:
    """Execute a workflow script with journaling and optional resume.

    The script is an async callable that receives injected context via ContextVar.
    All awflow primitives (agent, parallel, pipeline, etc.) are available inside.

    Args:
        script: async callable with no parameters
        journal_path: directory where to write the journal
                     (default: /data/workflows if on fleet, else ~/.awflow)
        resume_from: run_id to resume from (default: None, start fresh)
        budget_tokens: total token budget (default: 1000000)
        mirror: whether to mirror this run as an expedition (default: True);
               can be disabled via env AITHER_AWFLOW_MIRROR=0

    Returns:
        The return value of the script

    Raises:
        JournalError: if the journal is corrupt
        ValueError: if journal_path is not writable or script is not callable
        BudgetExhausted: if budget is exceeded during execution
    """
    if not callable(script):
        raise ValueError("script must be callable")

    # Check if mirroring is disabled via env var
    import os
    if os.environ.get("AITHER_AWFLOW_MIRROR", "1").lower() in ("0", "false"):
        mirror = False

    # Resolve journal root directory
    if journal_path is None:
        # Use /data/workflows if it exists (fleet containers), else ~/.awflow
        if Path("/data/workflows").exists():
            journal_root = Path("/data/workflows")
        else:
            journal_root = Path.home() / "awflow"
            journal_root.mkdir(parents=True, exist_ok=True)
    else:
        journal_root = Path(journal_path)
        journal_root.mkdir(parents=True, exist_ok=True)

    # Use resume_from as run_id, or generate a new one
    run_id = resume_from or str(uuid.uuid4())

    # Create journal
    try:
        jnl = journal.Journal(run_id=run_id, root=journal_root)
    except Exception as e:
        if isinstance(e, journal.JournalError):
            raise JournalError(f"failed to initialize journal: {e}") from e
        raise ValueError(f"failed to initialize journal: {e}") from e

    # Create runtime
    try:
        rt = runtime.WorkflowRuntime(
            journal=jnl,
            budget_tokens=budget_tokens,
            run_id=run_id,
            mirror_enabled=mirror,
        )
    except Exception as e:
        if isinstance(e, runtime.BudgetExhausted):
            raise BudgetExhausted(str(e)) from e
        raise ValueError(f"failed to initialize runtime: {e}") from e

    # Run script with runtime in context
    token = _current_runtime.set(rt)
    try:
        # Write WORKFLOW_START record
        from datetime import datetime, timezone
        await jnl.write_record({
            "type": "WORKFLOW_START",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": 0,
            "version": "1",
            "script_name": script.__name__ if hasattr(script, "__name__") else "unknown",
            "budget_tokens": budget_tokens,
            "journal_path": str(journal_root),
        })
        rt.sequence += 1

        # Create mirrored expedition if enabled
        if mirror:
            rt.mirror_expedition_id = await rt._create_mirrored_expedition(script)

        # Execute the workflow script
        result = await script()

        # Write WORKFLOW_END record
        await jnl.write_record({
            "type": "WORKFLOW_END",
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": rt.sequence,
            "version": "1",
            "status": "completed",
            "error": None,
            "total_calls": rt.agent_call_count,
            "total_tokens": rt.budget.spent(),
            "total_duration_ms": 0,  # Would need to track actual duration
        })

        # Mirror completion
        if mirror and rt.mirror_expedition_id:
            await rt._post_mirror_event("completed", {
                "ts": datetime.now(timezone.utc).isoformat(),
                "result": str(result)[:2000] if result else "",
                "status": "completed",
            })

        return result
    except (BudgetExhausted, JournalError) as e:
        # Write ERROR record for expected exceptions
        error_msg = str(e) if e else "BudgetExhausted or JournalError"
        try:
            from datetime import datetime, timezone
            await jnl.write_record({
                "type": "ERROR",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": rt.sequence,
                "version": "1",
                "message": error_msg,
            })
        except Exception as exc:
            # NEVER silent. A dropped journal record breaks RESUME -- the one
            # thing this package exists to provide -- and it breaks it
            # invisibly: the run completes, the result is right, and the replay
            # is short by one call with nothing to say so. Log loudly and count
            # it; the caller can still finish the run.
            logger.error(
                "[awflow] JOURNAL WRITE FAILED (resume will be incomplete): %s",
                exc,
            )

        # Mirror failure
        if mirror and rt.mirror_expedition_id:
            try:
                await rt._post_mirror_event("failed", {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "message": error_msg,
                    "status": "failed",
                })
            except Exception as exc:
                logger.error("[awflow] MIRROR EVENT FAILED: %s", exc)

        raise
    except Exception as e:
        logger.exception("Workflow failed")
        # Write ERROR record
        try:
            from datetime import datetime, timezone
            await jnl.write_record({
                "type": "ERROR",
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sequence": rt.sequence,
                "version": "1",
                "message": str(e),
            })
        except Exception as exc:
            # NEVER silent. A dropped journal record breaks RESUME -- the one
            # thing this package exists to provide -- and it breaks it
            # invisibly: the run completes, the result is right, and the replay
            # is short by one call with nothing to say so. Log loudly and count
            # it; the caller can still finish the run.
            logger.error(
                "[awflow] JOURNAL WRITE FAILED (resume will be incomplete): %s",
                exc,
            )

        # Mirror failure
        if mirror and rt.mirror_expedition_id:
            try:
                await rt._post_mirror_event("failed", {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "message": str(e),
                    "status": "failed",
                })
            except Exception as exc:
                logger.error("[awflow] MIRROR EVENT FAILED: %s", exc)

        raise
    finally:
        _current_runtime.reset(token)


__all__ = [
    "run_workflow",
    "agent",
    "parallel",
    "pipeline",
    "phase",
    "log",
    "Budget",
    "get_budget",
    "set_budget",
    "set_concurrency_cap",
    "BudgetExhausted",
    "JournalError",
]
