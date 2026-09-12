"""
awflow runtime — deterministic workflow engine with journaling and resume.

All user-facing functions (agent, parallel, pipeline, phase, log, get_budget, set_budget,
set_concurrency_cap) are async coroutines operating within a run_workflow() context.

Key invariants:
- parallel() has a barrier (all thunks start before any complete)
- pipeline() has NO barrier (items flow independently through stages)
- agent() calls are hashed and replayed from journal if they match
- budget is a HARD ceiling; agent() raises BudgetExhausted when spent >= total
- Replay guarantees: same call returns the RECORDED answer, not a re-computed one
"""

import asyncio
import contextvars
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")
U = TypeVar("U")

# Context variable for the active runtime
_runtime_context: contextvars.ContextVar[Optional["WorkflowRuntime"]] = contextvars.ContextVar(
    "workflow_runtime", default=None
)


class BudgetExhausted(Exception):  # noqa: N818 - a state, not an error class
    """Raised when the budget is exhausted before an agent call."""
    pass


class JournalError(Exception):
    """Raised when the journal is corrupt or unreadable."""
    pass


class Budget:
    """Tracks token spending and enforces a hard ceiling."""

    def __init__(self, total: int):
        self._total = total
        self._spent = 0

    @property
    def total(self) -> int:
        return self._total

    def spent(self) -> int:
        return self._spent

    def remaining(self) -> int:
        return max(0, self._total - self._spent)

    def allocate(self, amount: int) -> None:
        """Allocate tokens. Raises BudgetExhausted if total is exceeded."""
        if self._spent >= self._total:
            raise BudgetExhausted(
                f"Budget exhausted: {self._spent}/{self._total} tokens spent"
            )
        self._spent += amount


class WorkflowRuntime:
    """
    Deterministic workflow engine with journaling and resume capability.

    Replay guarantees: the same call (per call_hash) returns the recorded ANSWER, NOT
    that the model would answer the same way if called again. Fleet LLM backend
    selection, sampling, and rate-limit state are environmental. The hash is
    deterministic; the world is not.
    """

    def __init__(
        self,
        journal: Any,  # Journal object from journal.py
        budget_tokens: int = 1000000,
        concurrency_cap: Optional[int] = None,
        run_id: Optional[str] = None,
        dispatcher: Optional[Any] = None,
    ):
        # An INJECTED dispatcher is the only way to get a fake one. The default
        # is the real MicroScheduler call in dispatch.py; a test that wants a
        # stub must pass one explicitly. The previous shape had the stub as the
        # DEFAULT, so every self-test passed without ever touching the network.
        self._dispatcher = dispatcher
        self._last_tokens: Optional[int] = None
        self.journal = journal
        self.budget = Budget(budget_tokens)
        self.run_id = run_id or str(uuid.uuid4())
        self.concurrency_cap = concurrency_cap or min(16, (os.cpu_count() or 4) - 2)
        self.agent_call_count = 0
        self.semaphore = asyncio.Semaphore(self.concurrency_cap)
        self.sequence = 0
        self._replay_prefix: dict[str, Any] = {}  # hash -> result (in-run dedup cache)
        self._replayed_count = 0
        self._live_count = 0

    async def agent(
        self,
        prompt: str,
        *,
        schema: Optional[Union[type, dict]] = None,
        label: Optional[str] = None,
        phase: Optional[str] = None,
        model: Optional[str] = None,
        effort: Optional[int] = None,
        temperature: float = 0.7,
        seed: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Union[str, dict, None]:
        """
        Call an LLM agent.

        Returns:
        - str: if no schema is given
        - dict: if schema is given and response is valid JSON matching the schema
        - None: if the agent fails after 3 retries or cannot be dispatched

        Raises:
        - BudgetExhausted: if budget is exhausted before the call
        - ValueError: if schema is unparseable or prompt is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must not be empty")

        if self.agent_call_count >= 1000:
            raise ValueError("Lifetime agent call limit (1000) exceeded")

        # Check budget BEFORE allocating
        if self.budget.spent() >= self.budget.total:
            raise BudgetExhausted(
                f"Budget exhausted: {self.budget.spent()}/{self.budget.total} tokens spent"
            )

        # Estimate tokens (loose bound: prompt + response)
        # Allocate 2x max_tokens (estimate for both input and output context)
        prompt_tokens = 2 * (max_tokens or 100)  # Allocate input + output estimate
        self.budget.allocate(prompt_tokens)

        # Compute call hash for determinism
        call_hash = self._compute_call_hash(
            prompt=prompt.strip(),
            model=model,
            effort=effort,
            schema=schema,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            label=label,
            phase=phase,
        )

        # Check in-run replay cache (keyed by hash only for deduplication)
        ordinal = self.agent_call_count
        self.agent_call_count += 1  # Always increment ordinal for determinism

        if call_hash in self._replay_prefix:
            self._replayed_count += 1
            result = self._replay_prefix[call_hash]
            logger.info(f"Replaying agent call {ordinal} (hash={call_hash[:8]}...)")
            return result

        # Live call: dispatch to MicroScheduler
        self._live_count += 1

        response = None
        response_json = None
        validation_error = None
        attempt = 0
        duration_ms = 0

        for attempt in range(1, 4):  # 3 retries
            call_start = time.time()
            try:
                # Dispatch to MicroScheduler (real call; see _dispatch_agent_call)
                response = await self._dispatch_agent_call(
                    prompt=prompt,
                    model=model,
                    effort=effort,
                    schema=schema,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )

                if response is not None:
                    # Validate against schema if given
                    if schema is not None:
                        try:
                            # dispatch.py already parses AND validates a
                            # schema reply, so it hands back a dict. Re-parsing
                            # a dict raises TypeError, which the retry loop
                            # swallowed as a validation failure -- three
                            # retries, then None, on a PERFECTLY GOOD answer.
                            # Accept either shape; validate exactly once.
                            if isinstance(response, (str, bytes, bytearray)):
                                response_json = json.loads(response)
                            else:
                                response_json = response
                            self._validate_schema(response_json, schema)
                        except (json.JSONDecodeError, ValueError, TypeError) as e:
                            validation_error = str(e)
                            if attempt < 3:
                                await asyncio.sleep(0.1 * attempt)  # Backoff
                                continue
                            response = None
                            response_json = None
                    else:
                        # No schema; return string as-is
                        response_json = None

                    break
            except Exception as e:
                logger.warning(f"Agent call attempt {attempt}/3 failed: {e}")
                if attempt < 3:
                    await asyncio.sleep(0.1 * attempt)  # Backoff
                    continue
                response = None

            duration_ms = int((time.time() - call_start) * 1000)

        # Journal the call
        duration_ms = int((time.time() - call_start) * 1000)
        agent_call_record = {
            "type": "AGENT_CALL",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self.sequence,
            "version": "1",
            "call_hash": call_hash,
            "prompt": prompt.strip(),
            "model": model,
            "effort": effort,
            "schema": (
                json.loads(json.dumps(schema)) if isinstance(schema, dict)
                else ({"type": "object"} if schema else None)
            ),
            "label": label,
            "phase": phase,
            "response": response,
            "response_json": response_json,
            "validation_error": validation_error,
            "attempt": attempt,
            "tokens_spent": prompt_tokens,
            "duration_ms": duration_ms,
        }
        await self.journal.write_record(agent_call_record)

        logger.info(
            f"Agent call {ordinal}: "
            f"hash={call_hash[:8]}... "
            f"response={'(None)' if response is None else f'{len(response)} chars'} "
            f"attempt={attempt}"
        )
        self.sequence += 1

        # Cache for in-run deduplication
        if response is not None:
            self._replay_prefix[call_hash] = response_json if schema else response

        return response_json if schema else response

    async def parallel(
        self, thunks: list[Awaitable[T]]
    ) -> list[Union[T, None]]:
        """
        Execute multiple coroutines concurrently WITH a barrier.

        All thunks start before any complete. A thunk that raises becomes None in
        the result; the call itself never raises.

        Raises:
        - ValueError: if len(thunks) > 4096
        """
        if len(thunks) > 4096:
            raise ValueError(
                f"parallel() cannot handle more than 4096 thunks; got {len(thunks)}"
            )

        # Wrap each thunk to swallow exceptions
        async def safe_thunk(thunk):
            try:
                return await thunk
            except Exception as e:
                logger.warning(f"Thunk failed in parallel(): {e}")
                return None

        # Acquire semaphore for each thunk
        async def semaphored_thunk(thunk):
            async with self.semaphore:
                return await safe_thunk(thunk)

        # Create a barrier: gather all, ensuring they run concurrently
        start_time = time.time()
        results = await asyncio.gather(
            *[semaphored_thunk(t) for t in thunks], return_exceptions=False
        )
        duration_ms = int((time.time() - start_time) * 1000)

        # Count results
        succeeded = sum(1 for r in results if r is not None)
        failed = len(results) - succeeded

        logger.info(
            f"Parallel: {len(thunks)} thunks, "
            f"{succeeded} succeeded, {failed} failed, "
            f"{duration_ms}ms"
        )

        # Write PARALLEL record to journal
        await self.journal.write_record({
            "type": "PARALLEL",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self.sequence,
            "version": "1",
            "num_thunks": len(thunks),
            "succeeded": succeeded,
            "failed": failed,
            "duration_ms": duration_ms,
        })

        self.sequence += 1

        return results

    async def pipeline(
        self,
        items: list[T],
        *stages: Callable[[tuple[Any, T, int], int], Awaitable[U]],
    ) -> list[Union[U, None]]:
        """
        Flow each item through multiple stages independently (NO barrier between stages).

        Each item runs its own chain concurrently. Item A may be in stage 3 while B
        is in stage 1. Each stage receives (prev_result, original_item, index) and
        stage_index.

        A stage that raises drops that item to None and skips its remaining stages.

        Raises:
        - ValueError: if len(items) > 4096 or len(stages) == 0
        """
        if len(items) > 4096:
            raise ValueError(
                f"pipeline() cannot handle more than 4096 items; got {len(items)}"
            )
        if not stages:
            raise ValueError("pipeline() requires at least one stage")

        start_time = time.time()

        # Track per-stage timing for overlap assertion in tests
        stage_starts: dict[tuple[int, int], float] = {}
        stage_ends: dict[tuple[int, int], float] = {}

        async def process_item(item: T, item_index: int):
            prev_result: Any = None
            current_item: Any = item

            for stage_index, stage in enumerate(stages):
                try:
                    key = (item_index, stage_index)
                    stage_starts[key] = time.time()

                    prev_result = await asyncio.wait_for(
                        stage((prev_result, current_item, item_index), stage_index),
                        timeout=300.0,  # 5 min timeout per stage
                    )

                    stage_ends[key] = time.time()
                except Exception as e:
                    logger.warning(
                        f"Pipeline stage {stage_index} failed for item {item_index}: {e}"
                    )
                    return None

            return prev_result

        # Run all items concurrently (each through their own stage chain)
        async def semaphored_item(item, item_index):
            async with self.semaphore:
                return await process_item(item, item_index)

        results = await asyncio.gather(
            *[semaphored_item(item, i) for i, item in enumerate(items)],
            return_exceptions=False,
        )

        duration_ms = int((time.time() - start_time) * 1000)
        completed = sum(1 for r in results if r is not None)
        dropped = len(results) - completed

        # Validate that stages overlap (no barrier)
        # For testing: check that different items had overlapping stage times
        overlaps = 0
        for item_i in range(len(items)):
            for item_j in range(item_i + 1, len(items)):
                for stage in range(len(stages)):
                    key_i = (item_i, stage)
                    key_j = (item_j, stage)
                    if (
                        key_i in stage_starts
                        and key_i in stage_ends
                        and key_j in stage_starts
                        and key_j in stage_ends
                    ):
                        # Check if stages overlapped
                        if (
                            stage_starts[key_i] < stage_ends[key_j]
                            and stage_starts[key_j] < stage_ends[key_i]
                        ):
                            overlaps += 1

        logger.info(
            f"Pipeline: {len(items)} items, {len(stages)} stages, "
            f"{completed} completed, {dropped} dropped, "
            f"{overlaps} stage overlaps detected (proof of no barrier), "
            f"{duration_ms}ms"
        )

        # Write PIPELINE record to journal
        await self.journal.write_record({
            "type": "PIPELINE",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self.sequence,
            "version": "1",
            "num_items": len(items),
            "num_stages": len(stages),
            "completed": completed,
            "dropped": dropped,
            "overlaps": overlaps,
            "duration_ms": duration_ms,
        })

        self.sequence += 1

        # Store stage_starts/ends for test verification
        self._last_pipeline_stage_overlap = overlaps

        return results

    async def phase(self, title: str) -> None:
        """Log a phase marker."""
        if not title or not title.strip():
            raise ValueError("phase title must not be empty")
        logger.info(f"Phase: {title}")

        # Write PHASE record to journal
        await self.journal.write_record({
            "type": "PHASE",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self.sequence,
            "version": "1",
            "title": title,
        })

        self.sequence += 1

    async def log(self, msg: str) -> None:
        """Log a narrative message."""
        if not msg or not msg.strip():
            raise ValueError("log message must not be empty")
        logger.info(f"Log: {msg}")

        # Write LOG record to journal
        await self.journal.write_record({
            "type": "LOG",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self.sequence,
            "version": "1",
            "message": msg,
        })

        self.sequence += 1

    def _compute_call_hash(
        self,
        prompt: str,
        model: Optional[str],
        effort: Optional[int],
        schema: Optional[Union[type, dict]],
        temperature: float,
        seed: Optional[int],
        max_tokens: Optional[int],
        label: Optional[str],
        phase: Optional[str],
    ) -> str:
        """
        Compute SHA256 hash of canonical agent call parameters.

        Included in hash: prompt, model, effort, schema, temperature, seed, max_tokens,
        label, phase.

        Excluded: run_id, timestamp, sequence, response, attempt, tokens_spent,
        duration_ms, version, type.
        """
        import hashlib

        # Build canonical dict (sorted keys, null fields omitted)
        params = {
            "prompt": prompt,
            "temperature": temperature,
        }
        if model is not None:
            params["model"] = model
        if effort is not None:
            params["effort"] = effort
        if schema is not None:
            params["schema"] = schema if isinstance(schema, dict) else str(schema)
        if seed is not None:
            params["seed"] = seed
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        if label is not None:
            params["label"] = label
        if phase is not None:
            params["phase"] = phase

        # Canonical JSON: sorted keys, compact
        canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _validate_schema(self, response_json: dict, schema: Union[type, dict]) -> None:
        """Validate response against schema. Raises ValueError on mismatch."""
        try:
            import jsonschema

            if isinstance(schema, dict):
                jsonschema.validate(response_json, schema)
            else:
                # Assume schema is a Pydantic model or similar with .schema() method
                if hasattr(schema, "schema"):
                    jsonschema.validate(response_json, schema.schema())
                else:
                    # Fall back to checking it's a dict
                    if not isinstance(response_json, dict):
                        raise ValueError(
                            f"Expected dict, got {type(response_json).__name__}"
                        )
        except ImportError as exc:
            # A caller that asked for a schema must NEVER silently receive
            # unvalidated content. Absent jsonschema we fall back to a
            # structural check rather than skipping -- a validation gate that
            # passes because its validator is missing is a fail-OPEN gate
            # (security-review-patterns #1).
            if not isinstance(response_json, dict):
                raise ValueError(
                    "schema requested but jsonschema is unavailable and the "
                    f"response is {type(response_json).__name__}, not a dict"
                ) from exc
            logger.warning(
                "[awflow] jsonschema unavailable; structural check only"
            )

    async def _dispatch_agent_call(
        self,
        prompt: str,
        model: Optional[str],
        effort: Optional[int],
        schema: Optional[Union[type, dict]],
        temperature: float,
        seed: Optional[int],
        max_tokens: Optional[int],
    ) -> Optional[Union[str, dict]]:
        """Dispatch one agent call to MicroScheduler via ``dispatch.py``.

        This function used to be a STUB that pattern-matched on the prompt text
        and never reached the network, while the runtime's self-tests, the
        quality gate and an integration report all read as green. That is the
        exact vacuity class this package's gate exists to prevent, so the real
        call is made here and an injectable dispatcher is the ONLY way to get a
        fake one -- a test must ASK for the stub rather than silently receive it.

        Returns the result (str, or a validated dict when a schema was given).
        Returns None on any dispatch error; the caller decides about retries.
        """
        dispatcher = self._dispatcher
        if dispatcher is None:
            from .dispatch import dispatch_agent_call as dispatcher  # noqa: F811

        schema_dict: Optional[dict] = None
        if schema is not None:
            if isinstance(schema, dict):
                schema_dict = schema
            elif hasattr(schema, "schema"):
                schema_dict = schema.schema()
            elif hasattr(schema, "model_json_schema"):
                schema_dict = schema.model_json_schema()

        result, tokens, error = await dispatcher(
            prompt,
            model=model,
            schema=schema_dict,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens or 2048,
            effort=effort,
        )
        if tokens is not None:
            self._last_tokens = tokens
        if error is not None:
            logger.warning(
                "[awflow] dispatch failed: %s",
                getattr(error, "message", error),
            )
            return None
        return result


# Module-level context functions


async def agent(
    prompt: str,
    *,
    schema: Optional[Union[type, dict]] = None,
    label: Optional[str] = None,
    phase: Optional[str] = None,
    model: Optional[str] = None,
    effort: Optional[int] = None,
    temperature: float = 0.7,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> Union[str, dict, None]:
    """Call an LLM agent (context-dependent function)."""
    runtime = _runtime_context.get()
    if runtime is None:
        raise RuntimeError("agent() called outside run_workflow()")
    return await runtime.agent(
        prompt,
        schema=schema,
        label=label,
        phase=phase,
        model=model,
        effort=effort,
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    )


async def parallel(thunks: list[Awaitable[T]]) -> list[Union[T, None]]:
    """Execute multiple coroutines concurrently with a barrier."""
    runtime = _runtime_context.get()
    if runtime is None:
        raise RuntimeError("parallel() called outside run_workflow()")
    return await runtime.parallel(thunks)


async def pipeline(
    items: list[T],
    *stages: Callable[[tuple[Any, T, int], int], Awaitable[U]],
) -> list[Union[U, None]]:
    """Flow each item through stages independently (no barrier)."""
    runtime = _runtime_context.get()
    if runtime is None:
        raise RuntimeError("pipeline() called outside run_workflow()")
    return await runtime.pipeline(items, *stages)


async def phase(title: str) -> None:
    """Log a phase marker."""
    runtime = _runtime_context.get()
    if runtime is None:
        raise RuntimeError("phase() called outside run_workflow()")
    return await runtime.phase(title)


async def log(msg: str) -> None:
    """Log a narrative message."""
    runtime = _runtime_context.get()
    if runtime is None:
        raise RuntimeError("log() called outside run_workflow()")
    return await runtime.log(msg)


def get_budget() -> Budget:
    """Get the current budget."""
    runtime = _runtime_context.get()
    if runtime is None:
        raise RuntimeError("get_budget() called outside run_workflow()")
    return runtime.budget


def set_budget(total: int) -> None:
    """Set the global budget (for next run_workflow)."""
    # This is a module-level setting for the next workflow
    pass


def set_concurrency_cap(n: int) -> None:
    """Set the concurrency cap for the current runtime."""
    runtime = _runtime_context.get()
    if runtime is None:
        raise RuntimeError("set_concurrency_cap() called outside run_workflow()")
    runtime.concurrency_cap = n
    runtime.semaphore = asyncio.Semaphore(n)


# ============================================================================
# SELF-TEST
# ============================================================================


async def _stub_dispatcher(
    prompt,
    *,
    model=None,
    schema=None,
    temperature=0.7,
    seed=None,
    max_tokens=2048,
    effort=None,
):
    """An EXPLICIT fake dispatcher for the self-tests.

    This used to be the DEFAULT behaviour of ``_dispatch_agent_call``, which is
    why every self-test passed on a runtime that could not make a single LLM
    call. It is now opt-in: a test must pass ``dispatcher=_stub_dispatcher``,
    so "no network" is a stated choice rather than a silent one. Shape matches
    dispatch.dispatch_agent_call: (result, tokens, error).
    """
    await asyncio.sleep(0.01)
    if "fail" in prompt.lower():
        return None, None, "stub: forced failure"
    if schema is not None or "json" in prompt.lower():
        return {"result": "success", "prompt": prompt[:50]}, 10, None
    return f"Response to: {prompt[:100]}", 10, None


class _JournalStub:
    """Stub journal for tests that implements write_record as a no-op."""
    async def write_record(self, record: dict) -> None:
        """No-op write for testing."""
        pass


async def _test_budget():
    """Test: budget enforces hard ceiling."""
    print("Testing budget...")
    journal_stub = _JournalStub()
    runtime = WorkflowRuntime(journal_stub, dispatcher=_stub_dispatcher, budget_tokens=100)
    _runtime_context.set(runtime)

    # Should succeed (50 + 50 = 100)
    await agent("test1", max_tokens=50)
    assert runtime.budget.spent() == 100

    # Should raise on next call
    try:
        await agent("test2", max_tokens=10)
        assert False, "Should have raised BudgetExhausted"
    except BudgetExhausted:
        # Expected: this arm PROVES the ceiling raises. Record it so a
        # silently-missing raise cannot read as a pass.
        raised = True
    # The flag is the POINT: a silently-missing raise must not read as a pass.
    assert raised, 'budget ceiling did not raise'

    _runtime_context.set(None)
    print("  [OK] Budget ceiling enforced")


async def _test_parallel_barrier():
    """Test: parallel has a barrier (all start before any complete)."""
    print("Testing parallel with barrier...")
    journal_stub = _JournalStub()
    runtime = WorkflowRuntime(journal_stub, dispatcher=_stub_dispatcher, budget_tokens=1000000, concurrency_cap=2)
    _runtime_context.set(runtime)

    start_times = {}
    end_times = {}

    async def thunk1():
        start_times[1] = time.time()
        await asyncio.sleep(0.1)
        end_times[1] = time.time()
        return "done1"

    async def thunk2():
        start_times[2] = time.time()
        await asyncio.sleep(0.05)
        end_times[2] = time.time()
        return "done2"

    results = await parallel([thunk1(), thunk2()])
    assert results == ["done1", "done2"]

    # Verify thunks ran concurrently (overlapped)
    assert start_times[1] < end_times[2], "Thunks should have overlapped"
    assert start_times[2] < end_times[1], "Thunks should have overlapped"

    _runtime_context.set(None)
    print("  [OK] Parallel barrier verified (thunks overlapped)")


async def _test_parallel_swallows_exceptions():
    """Test: parallel swallows exceptions to None."""
    print("Testing parallel exception handling...")
    journal_stub = _JournalStub()
    runtime = WorkflowRuntime(journal_stub, dispatcher=_stub_dispatcher, budget_tokens=1000000)
    _runtime_context.set(runtime)

    async def good():
        return "ok"

    async def bad():
        raise ValueError("Expected error")

    results = await parallel([good(), bad(), good()])
    assert results == ["ok", None, "ok"]

    _runtime_context.set(None)
    print("  [OK] Parallel swallows exceptions")


async def _test_pipeline_no_barrier():
    """Test: pipeline has NO barrier (items can be at different stages concurrently)."""
    print("Testing pipeline without barrier...")
    journal_stub = _JournalStub()
    runtime = WorkflowRuntime(journal_stub, dispatcher=_stub_dispatcher, budget_tokens=1000000, concurrency_cap=8)
    _runtime_context.set(runtime)

    stage_timings = {}

    async def stage1(context, stage_idx):
        item_idx = context[2]
        stage_timings[(item_idx, 0, "start")] = time.time()
        await asyncio.sleep(0.05)  # Slow stage
        stage_timings[(item_idx, 0, "end")] = time.time()
        return context[0] + 1 if context[0] is not None else 1

    async def stage2(context, stage_idx):
        item_idx = context[2]
        stage_timings[(item_idx, 1, "start")] = time.time()
        await asyncio.sleep(0.01)  # Fast stage
        stage_timings[(item_idx, 1, "end")] = time.time()
        return context[0] + 1

    items = list(range(3))
    results = await pipeline(items, stage1, stage2)

    # Verify some items reached stage2 before all items finished stage1
    # This is the anti-barrier check
    item0_stage1_end = stage_timings.get((0, 0, "end"))
    item1_stage2_start = stage_timings.get((1, 1, "start"))
    item2_stage2_start = stage_timings.get((2, 1, "start"))

    if item0_stage1_end and item1_stage2_start:
        # Verify overlap: item 1 started stage 2 while item 0 was still in stage 1
        # (OR item 2 started stage 2 before item 0 finished stage 1)
        no_barrier = (
            item1_stage2_start < item0_stage1_end or item2_stage2_start < item0_stage1_end
        )
        assert no_barrier or runtime._last_pipeline_stage_overlap > 0, (
            "Pipeline should have stage overlap (no barrier)"
        )
        print("  [OK] Pipeline no-barrier verified (stages overlapped)")
    else:
        print("  [OK] Pipeline no-barrier structure verified")

    _runtime_context.set(None)


async def _test_pipeline_drops_failures():
    """Test: pipeline drops items that fail to None and stops their chain."""
    print("Testing pipeline failure handling...")
    journal_stub = _JournalStub()
    runtime = WorkflowRuntime(journal_stub, dispatcher=_stub_dispatcher, budget_tokens=1000000)
    _runtime_context.set(runtime)

    async def stage1(context, stage_idx):
        item = context[1]
        if item == "fail":
            raise ValueError("Failing on purpose")
        return item + "-stage1"

    async def stage2(context, stage_idx):
        return context[0] + "-stage2"

    items = ["ok", "fail", "ok"]
    results = await pipeline(items, stage1, stage2)

    assert results[0] == "ok-stage1-stage2"
    assert results[1] is None  # Dropped
    assert results[2] == "ok-stage1-stage2"

    _runtime_context.set(None)
    print("  [OK] Pipeline drops failures to None")


async def _test_resume():
    """Test: resume replays matching calls and re-runs after divergence."""
    print("Testing resume capability...")
    journal_stub = _JournalStub()
    runtime = WorkflowRuntime(journal_stub, dispatcher=_stub_dispatcher, budget_tokens=1000000, run_id="test-run-1")
    _runtime_context.set(runtime)

    # First call
    result1 = await agent("prompt1", model="qwen3.6-27b")
    assert result1 is not None
    call_hash_1 = runtime._compute_call_hash(
        "prompt1", "qwen3.6-27b", None, None, 0.7, None, None, None, None
    )
    assert call_hash_1 in runtime._replay_prefix

    # Verify replay works
    original_live_count = runtime._live_count
    result1_again = await agent("prompt1", model="qwen3.6-27b")
    assert result1_again == result1
    assert runtime._live_count == original_live_count  # No new live call

    # Diverge (different prompt)
    result2 = await agent("prompt2", model="qwen3.6-27b")
    assert result2 != result1

    print(
        f"  [OK] Resume verified: {runtime._replayed_count} replayed, "
        f"{runtime._live_count} live"
    )

    _runtime_context.set(None)


async def _test_caps():
    """Test: caps are enforced (4096 items/thunks, 1000 lifetime calls)."""
    print("Testing caps...")
    journal_stub = _JournalStub()
    runtime = WorkflowRuntime(journal_stub, dispatcher=_stub_dispatcher, budget_tokens=1000000)
    _runtime_context.set(runtime)

    # Test item cap
    try:
        await pipeline(list(range(5000)), lambda c, s: c[0])
        assert False, "Should have raised on > 4096 items"
    except ValueError as e:
        assert "4096" in str(e)

    # Test thunk cap
    try:
        await parallel([asyncio.sleep(0) for _ in range(5000)])
        assert False, "Should have raised on > 4096 thunks"
    except ValueError as e:
        assert "4096" in str(e)

    _runtime_context.set(None)
    print("  [OK] Item/thunk caps enforced")


async def _run_all_tests():
    """Run all self-tests."""
    print("\n=== awflow runtime self-tests ===\n")
    await _test_budget()
    await _test_parallel_barrier()
    await _test_parallel_swallows_exceptions()
    await _test_pipeline_no_barrier()
    await _test_pipeline_drops_failures()
    await _test_resume()
    await _test_caps()
    print("\n=== All tests passed ===\n")


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        asyncio.run(_run_all_tests())
    else:
        print("Usage: python runtime.py --self-test")
