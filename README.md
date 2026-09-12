# awflow — Deterministic Workflow Runtime

Lightweight multi-agent orchestration engine for the AitherOS fleet.

## What is it?

awflow is a **workflow interpreter** that runs deterministic, journaled multi-agent scripts on the local fleet. It provides:

- **Deterministic replay**: Call hashing + journaling means re-running the same script returns cached agent answers, not new LLM calls
- **Resume from interruption**: Stop a workflow mid-execution and resume later from the exact point where it diverges
- **Concurrency primitives**: `agent()` (serial LLM call), `parallel()` (concurrent, barrier sync), `pipeline()` (concurrent, no barrier)
- **Budget enforcement**: Hard token ceiling with no backpression — exceeding it raises immediately
- **Live journaling**: JSONL-format audit trail of every call, response, and decision

## Why?

ExpeditionManager (365KB) already orchestrates agent phases. SwarmCodingEngine (87KB) already dispatches agents. Six Pillars kernel already ticks. But nothing records **per-agent inputs/outputs in replay-resumable form**. awflow is that primitive — deliberately small, deliberately single-purpose — that those engines can adopt to gain determinism and resume.

The fleet models' 131k context window is big enough for the orchestrator-as-LLM trap to work. awflow is not that trap. It is the plumbing.

## Self-test

Verify the package imports and can execute a trivial script offline:

```bash
python -m AitherOS.lib.orchestration.awflow --self-test
```

This runs a stub workflow that:
1. Logs phases (no LLM calls)
2. Executes a concurrent fan-out (no barrier, no LLM calls)
3. Executes a serial pipeline stage (no LLM calls)
4. Verifies journaling, budget tracking, and determinism

Exit 0 means the package is functional.

## Schema validation

When you ask for a schema-constrained response, awflow:
1. Sends `response_format` to MicroScheduler (if backend supports it)
2. Validates the response against the schema locally (jsonschema)
3. Retries up to 3 times on validation failure (same prompt, same backend)
4. Returns None if all retries fail (caller filters Nones)

This belt-and-braces approach works whether or not the backend honors constrained decoding.

## Determinism guarantee

**Replay returns RECORDED answers, not LLM re-computation.** The call hash is deterministic (prompt + model + schema + effort + temperature + seed). The world is not. Fleet backend selection, sampling, and rate-limit state are environmental. When you resume a workflow, agent calls that exactly match their recorded hash return the recorded response; all others go live.

## Adoption by existing engines

ExpeditionManager already has `expedition.collect()` and `expedition.decisions()` for checkpoints. To adopt awflow:

1. Wrap the expedition in `await run_workflow(expedition.execute, budget_tokens=...)`
2. Inside the expedition, call `await awflow.agent()` instead of genesis `/chat`
3. On resume, pass the run_id to `resume_from=run_id`

Same complexity, now with determinism and resume.

## Example

See `examples/hello_fleet.py` for a three-agent workflow:

1. Phase: "Planning"
2. Parallel: two agents brainstorm
3. Phase: "Summarizing"
4. Serial: schema-constrained summary

Run it locally (if a test MicroScheduler is available):

```bash
python AitherOS/lib/orchestration/awflow/examples/hello_fleet.py
```
