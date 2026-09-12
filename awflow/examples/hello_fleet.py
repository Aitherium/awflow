"""Example: hello_fleet.py

A three-agent workflow demonstrating awflow primitives:
1. Phase marker
2. Parallel fan-out (two agents brainstorm)
3. Serial pipeline (schema-constrained summary)
4. Print run_id and replay counts

This script can run offline (with stub dispatch) or live against the fleet.

Usage:
    python AitherOS/lib/orchestration/awflow/examples/hello_fleet.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path so we can import awflow
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from lib.orchestration import awflow


async def hello_fleet():
    """Example workflow: brainstorm ideas, then summarize."""

    await awflow.phase("Planning")

    # Parallel: two agents brainstorm ideas
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
    print(f"\n[Brainstorm results]\n{ideas}\n")

    await awflow.phase("Summarizing")

    # Serial: schema-constrained summary
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
    print(f"\n[Summaries]\n{summaries}\n")

    # Log completion
    await awflow.log("Workflow completed successfully")

    budget = awflow.get_budget()
    return {
        "ideas": ideas,
        "summaries": summaries,
        "budget_used": budget.spent(),
        "budget_remaining": budget.remaining(),
    }


if __name__ == "__main__":
    # Run the workflow
    result = asyncio.run(
        awflow.run_workflow(
            hello_fleet,
            journal_path=Path.home() / "awflow" / "hello_fleet_demo.jsonl",
            budget_tokens=100000,
        )
    )

    print(f"\n[Workflow Result]\n{result}\n")
    print("Run completed. Check the journal for replay details.")
