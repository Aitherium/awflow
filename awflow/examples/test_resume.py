"""Test resume functionality: run example 3 times to demonstrate replay."""

import asyncio
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from lib.orchestration import awflow


async def hello_fleet_run1():
    """First run - all live calls."""
    await awflow.phase("Planning")

    async def brainstorm_1():
        result = await awflow.agent(
            "List 3 innovative features for a personal AI assistant. Be creative.",
            label="brainstorm_1",
            effort=3,
        )
        return result

    async def brainstorm_2():
        result = await awflow.agent(
            "List 3 ways to make AI more accessible to non-technical users.",
            label="brainstorm_2",
            effort=3,
        )
        return result

    ideas = await awflow.parallel([brainstorm_1(), brainstorm_2()])
    print(f"[Run 1] Brainstorm results: {ideas}\n")

    await awflow.phase("Summarizing")

    async def summarize_stage(context, stage_idx):
        prev_result, item, index = context
        prompt = f"Summarize this idea in one sentence: {item}"
        result = await awflow.agent(
            prompt,
            schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["summary", "confidence"],
            },
            label=f"summarize_{index}",
            effort=2,
        )
        return result

    summaries = await awflow.pipeline(ideas, summarize_stage)
    print(f"[Run 1] Summaries: {summaries}\n")

    await awflow.log("Run 1 completed successfully")

    budget = awflow.get_budget()
    return {
        "ideas": ideas,
        "summaries": summaries,
        "budget_used": budget.spent(),
        "budget_remaining": budget.remaining(),
    }


async def hello_fleet_run2():
    """Second run - same as run 1, should replay all calls."""
    await awflow.phase("Planning")

    async def brainstorm_1():
        result = await awflow.agent(
            "List 3 innovative features for a personal AI assistant. Be creative.",
            label="brainstorm_1",
            effort=3,
        )
        return result

    async def brainstorm_2():
        result = await awflow.agent(
            "List 3 ways to make AI more accessible to non-technical users.",
            label="brainstorm_2",
            effort=3,
        )
        return result

    ideas = await awflow.parallel([brainstorm_1(), brainstorm_2()])
    print(f"[Run 2] Brainstorm results (replayed): {ideas}\n")

    await awflow.phase("Summarizing")

    async def summarize_stage(context, stage_idx):
        prev_result, item, index = context
        prompt = f"Summarize this idea in one sentence: {item}"
        result = await awflow.agent(
            prompt,
            schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["summary", "confidence"],
            },
            label=f"summarize_{index}",
            effort=2,
        )
        return result

    summaries = await awflow.pipeline(ideas, summarize_stage)
    print(f"[Run 2] Summaries (replayed): {summaries}\n")

    await awflow.log("Run 2 completed successfully")

    budget = awflow.get_budget()
    return {
        "ideas": ideas,
        "summaries": summaries,
        "budget_used": budget.spent(),
        "budget_remaining": budget.remaining(),
    }


async def hello_fleet_run3():
    """Third run - MODIFIED prompt to show divergence detection."""
    await awflow.phase("Planning")

    async def brainstorm_1():
        # CHANGED PROMPT - should show divergence
        result = await awflow.agent(
            "List 5 innovative features for a personal AI assistant. Be creative.",
            label="brainstorm_1",
            effort=3,
        )
        return result

    async def brainstorm_2():
        # Same as before - should replay
        result = await awflow.agent(
            "List 3 ways to make AI more accessible to non-technical users.",
            label="brainstorm_2",
            effort=3,
        )
        return result

    ideas = await awflow.parallel([brainstorm_1(), brainstorm_2()])
    print(f"[Run 3] Brainstorm results (first diverged, second replayed): {ideas}\n")

    await awflow.phase("Summarizing")

    async def summarize_stage(context, stage_idx):
        prev_result, item, index = context
        prompt = f"Summarize this idea in one sentence: {item}"
        result = await awflow.agent(
            prompt,
            schema={
                "type": "object",
                "properties": {
                    "summary": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["summary", "confidence"],
            },
            label=f"summarize_{index}",
            effort=2,
        )
        return result

    summaries = await awflow.pipeline(ideas, summarize_stage)
    print(f"[Run 3] Summaries: {summaries}\n")

    await awflow.log("Run 3 completed successfully")

    budget = awflow.get_budget()
    return {
        "ideas": ideas,
        "summaries": summaries,
        "budget_used": budget.spent(),
        "budget_remaining": budget.remaining(),
    }


if __name__ == "__main__":
    journal_path = Path.home() / "awflow" / "test_resume.jsonl"

    print("=" * 70)
    print("RUN 1: Initial run - all calls should be LIVE")
    print("=" * 70)
    result1 = asyncio.run(
        awflow.run_workflow(hello_fleet_run1, journal_path=journal_path, budget_tokens=100000)
    )

    # Extract run_id from the journal directory
    run_dirs = list(journal_path.glob("*/"))
    if run_dirs:
        run_id = run_dirs[-1].name
        print(f"RUN_ID: {run_id}\n")

        # Count journal lines for run 1
        journal_file = journal_path / run_id / "journal.jsonl"
        if journal_file.exists():
            with open(journal_file) as f:
                run1_lines = len(f.readlines())
            print(f"Journal lines after run 1: {run1_lines}\n")

            print("=" * 70)
            print("RUN 2: Resume run - all calls should be REPLAYED")
            print("=" * 70)
            result2 = asyncio.run(
                awflow.run_workflow(
                    hello_fleet_run2,
                    journal_path=journal_path,
                    resume_from=run_id,
                    budget_tokens=100000,
                )
            )

            # Count journal lines for run 2 (same run_id, more lines from replay verification)
            if journal_file.exists():
                with open(journal_file) as f:
                    run2_lines = len(f.readlines())
                print(f"Journal lines after run 2: {run2_lines}\n")

            print("=" * 70)
            print("RUN 3: Resume with divergence - prefix replayed, then diverged")
            print("=" * 70)
            result3 = asyncio.run(
                awflow.run_workflow(
                    hello_fleet_run3,
                    journal_path=journal_path,
                    resume_from=run_id,
                    budget_tokens=100000,
                )
            )

            # Count journal lines for run 3
            if journal_file.exists():
                with open(journal_file) as f:
                    run3_lines = len(f.readlines())
                print(f"Journal lines after run 3: {run3_lines}\n")

            print("=" * 70)
            print("RESUME PROOF:")
            print("=" * 70)
            print(f"Run 1 journal lines: {run1_lines}")
            print(f"Run 2 journal lines: {run2_lines} (same run_id, should be similar)")
            print(f"Run 3 journal lines: {run3_lines} (same run_id + new diverged calls)")
            print("\nExpected: Run 2 should show no new agent calls (all replayed),")
            print("          Run 3 should show 1 new agent call at the divergence point.")
