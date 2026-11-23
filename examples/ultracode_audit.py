#!/usr/bin/env python3
"""Worked example: audit the newest merged PRs for missing tests.

Read-only end to end: it asks `gh` for merged PR metadata, builds one work item
per PR (the changed-file list is the payload), then runs an ultracode workflow:
scout -> work -> verify (2 skeptics, refute by default) -> completeness critic
-> synthesize. Every step routes itself through `awflow.lanes` (local pool and
Bonsai first, the awrouter's cloud lane as overflow), the run is journaled and
mirrored as an expedition when the host wired the hooks, and the result is
written as an Agent Notebook when `lib.orchestration.NotebookStore` is present.

Usage (from the repo root, AitherOS on PYTHONPATH):

    PYTHONPATH=AitherOS/packages/awflow:AitherOS \\
        python AitherOS/packages/awflow/examples/ultracode_audit.py \\
        --repo Aitherium/AitherOS --limit 10 \\
        --journal-root ~/.awflow --out audit.json

Nothing here writes to a repository. `--limit 10` means 10 PRs, and the tool
refuses a list over its cap rather than silently auditing a subset.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from awflow import ultracode

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ultracode_audit")


def _wire_platform() -> None:
    """Give awflow the fleet's service resolver and internal-CA HTTP client.

    Standalone awflow refuses to guess a MicroScheduler endpoint (it raises and
    names AITHER_MICROSCHEDULER_URL), so a host script wires the same two
    things lib/orchestration wires. Best effort: on a laptop without the
    platform modules, AITHER_MICROSCHEDULER_URL is the escape hatch.
    """
    # NOTE: lib/orchestration/_wire_awflow() configures the COPY under
    # lib/orchestration/awflow, not the package this example imports. Both
    # copies need the same two things, so wire the package explicitly.
    try:
        from awflow import dispatch
        from lib.core.AitherHttp import AsyncClient
        from lib.core.AitherPorts import get_service_url
        dispatch.configure(get_service_url=get_service_url,
                           async_client=AsyncClient)
        logger.info("awflow wired to the platform resolver (%s)",
                    get_service_url("MicroScheduler"))
    except Exception as exc:  # noqa: BLE001 - the standalone lane is still usable
        logger.warning("platform wiring unavailable (%s); local lane needs "
                       "AITHER_MICROSCHEDULER_URL", exc)
    try:
        from awflow import runtime as awflow_runtime
        from lib.orchestration.expedition_intake import (
            create_mirror,
            post_mirror_event,
        )
        awflow_runtime.set_mirror_hooks(create=create_mirror,
                                        post=post_mirror_event)
        logger.info("expedition mirror hooks wired")
    except Exception as exc:  # noqa: BLE001 - journal survives without a mirror
        logger.warning("expedition mirror unavailable (%s); journal only", exc)


async def fetch_merged_prs(repo: str, limit: int) -> list:
    """Read-only `gh pr list` for merged PRs with their changed files.

    Async subprocess on purpose: this example runs inside the same loop as the
    workflow, and a blocking `subprocess.run` would stall every concurrent step.
    """
    argv = ["gh", "pr", "list", "--repo", repo, "--state", "merged",
            "--limit", str(limit), "--json",
            "number,title,mergedAt,files"]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(proc.communicate(), timeout=120)
    if proc.returncode != 0:
        raise RuntimeError((err or out).decode("utf-8", "replace").strip()
                           or "gh failed")
    return json.loads(out.decode("utf-8", "replace") or "[]")


def pr_items(prs: list) -> list:
    """One WorkItem per PR. Payload is a bounded changed-file list, never a diff:
    the work list is what the audit judges, and a full diff would neither fit a
    local lane nor be needed to answer 'was a test touched?'."""
    items = []
    for pr in prs:
        paths = [str((f or {}).get("path") or "") for f in pr.get("files") or []]
        paths = [p for p in paths if p]
        payload = (f"PR #{pr.get('number')} merged {pr.get('mergedAt')}\n"
                   f"TITLE: {pr.get('title')}\n"
                   f"CHANGED FILES ({len(paths)}):\n"
                   + "\n".join(f"- {p}" for p in paths[:120]))
        items.append({
            "id": f"pr-{pr.get('number')}",
            "title": f"PR #{pr.get('number')}: {pr.get('title')}",
            "payload": payload[:12_000],
        })
    return items


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Aitherium/AitherOS")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--budget", type=int, default=180_000)
    parser.add_argument("--skeptics", type=int, default=2)
    parser.add_argument("--deadline", type=float, default=45.0,
                        help="seconds a single lane attempt may take before "
                             "the step fails over (default 45)")
    parser.add_argument("--journal-root", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--no-notebook", action="store_true")
    args = parser.parse_args()

    _wire_platform()
    prs = await fetch_merged_prs(args.repo, args.limit)
    if not prs:
        logger.error("no merged PRs returned for %s", args.repo)
        return 1
    items = pr_items(prs)
    goal = (f"Audit the {len(items)} newest merged PRs in {args.repo} for missing "
            "test coverage: for each PR say whether the change should have "
            "touched a test and did not, citing the changed paths.")
    plan = ultracode.plan_workflow(
        goal, items, budget_tokens=args.budget,
        verify_skeptics=args.skeptics, scout=True, synthesize=True,
        deadline_s=args.deadline)
    logger.info("plan: %d items, %d tokens estimated, lanes %s",
                len(plan.items), plan.estimated_tokens,
                [phase.lane_order for phase in plan.phases])

    result = await ultracode.run_plan(
        plan,
        journal_root=Path(args.journal_root).expanduser()
        if args.journal_root else None,
        notebook=not args.no_notebook)
    payload = {
        "run_id": result.run_id,
        "notebook_id": result.notebook_id,
        "journal_path": result.journal_path,
        "complete": result.complete,
        "local_share": result.local_share,
        "lanes_used": result.lanes_used,
        "tokens": result.tokens,
        "cost_usd": result.cost_usd,
        "synthesis": result.synthesis,
        "notes": result.notes,
        "outcomes": result.outcomes,
    }
    text = json.dumps(payload, indent=2, default=str)
    if args.out:
        Path(args.out).expanduser().write_text(text, encoding="utf-8")
    print(text)
    logger.info("run %s notebook %s local_share %.2f",
                result.run_id, result.notebook_id, result.local_share)
    return 0 if result.outcomes else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
