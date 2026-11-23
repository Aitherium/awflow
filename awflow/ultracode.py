"""awflow.ultracode — plan and run a fan-out ("ultracode") workflow natively.

WHAT THIS IS
------------
The pattern Claude Code uses for a hard question is: scout once inline, fan out
N agents over a work list, adversarially VERIFY every finding, then synthesize.
Today that pattern lives in a harness, not in the platform: an Aither agent can
call one model at a time, but it cannot say "plan this, fan it out, prove each
claim, and give me the cited result" and have the platform do it.

This module is that capability, built on the pieces that already exist:

* :mod:`awflow.runtime` — the journaled, resumable, budgeted engine. A run is a
  journal; a resume replays the answers it already has.
* :mod:`awflow.lanes` — per-step lane selection (local pool / Bonsai first, the
  awrouter's DeepSeek cloud lane as overflow, free public GitHub runners for
  non-LLM compute). Every step routes itself; nothing here hard-codes a lane.
* expedition mirroring — ``run_workflow(mirror=True)`` already mirrors a run as
  an expedition when the host wired the hooks (``lib/orchestration`` does).

The primitives, all on top of ``step()``:

    agent_step(payload, needs)          one agent step, cheapest lane that fits
    pipeline(items, *stages)            items flow through stages, NO barrier
    parallel(thunks)                    barrier fan-out (skeptics, gather)
    verify(claim, n, ...)               n skeptics, REFUTE BY DEFAULT
    synthesize(goal, findings)          one cited answer over confirmed findings
    plan_workflow(goal, items, ...)     deterministic plan: phases, lanes, budget
    run_plan(plan, ...)                 execute it, journal it, return the result

DESIGN RULES THAT ARE NOT NEGOTIABLE
------------------------------------
* **Pipeline is the default.** Work items flow independently; a verify of item 3
  never waits for item 7's work step. The only barrier is the final synthesize,
  which by definition needs every finding.
* **Refute by default.** A claim is CONFIRMED only when every skeptic ran and
  every skeptic failed to refute it. A failed skeptic (lane down, empty answer,
  ambiguous text) counts as a refutation: an unproven claim is not confirmed.
  A work step that failed on every lane produces no claim at all.
* **No silent caps.** An oversized work list, a plan that cannot fit its budget,
  or a critic that keeps listing missing items is REFUSED or reported
  (``complete: False`` + a note), never quietly truncated.
* **Budget is a hard ceiling.** Tokens are reserved before a step and settled
  after; a reservation that would cross the ceiling raises BudgetExhausted.
  ``pipeline()`` swallows stage exceptions, so the stage re-raises the budget
  error AFTER the pipeline returns rather than losing it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import journal as _journal
from . import lanes as _lanes

logger = logging.getLogger(__name__)

#: Skeptics per claim. One skeptic is a re-read; two is an adversary plus a
#: second opinion. More than five is spend without a new question.
DEFAULT_SKEPTICS = 2
MAX_SKEPTICS = 5
#: Planning refuses a work list bigger than this instead of silently taking the
#: first N -- the cap exists to make the refusal explicit.
MAX_WORK_ITEMS = 200
#: How many times the completeness critic may add work before the run reports
#: itself incomplete. A loop, not a truncation: the cap is announced.
MAX_CRITIC_ROUNDS = 3
DEFAULT_BUDGET_TOKENS = 200_000
#: Per-step output bound. Small on purpose: a fan-out of terse findings is what
#: the local lanes can actually finish, and the synth reads the findings, not
#: transcripts.
DEFAULT_MAX_TOKENS = 320
SYNTH_MAX_TOKENS = 700
#: A finding longer than this is truncated IN THE VERIFY PROMPT only; the full
#: work answer stays in the journal and in the result.
CLAIM_PREVIEW_CHARS = 1_800

VERDICT_CONFIRMED = "confirmed"
VERDICT_REFUTED = "refuted"


class PlanError(ValueError):
    """The plan cannot be run as asked (empty goal, oversized list, no budget)."""


@dataclass(frozen=True)
class WorkItem:
    """One unit of fan-out work. ``payload`` is data the worker may read; it is
    never executed. ``needs`` overlays :class:`awflow.lanes.StepNeeds` fields."""

    id: str
    title: str
    payload: str = ""
    needs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.id or not str(self.id).strip():
            raise PlanError("work item id must not be empty")
        if not self.title or not str(self.title).strip():
            raise PlanError(f"work item {self.id!r} has no title")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "payload_chars": len(self.payload or ""),
            "needs": dict(self.needs or {}),
        }


@dataclass(frozen=True)
class PhasePlan:
    """One phase of a plan, with the ordered lanes its steps would use."""

    name: str
    kind: str  # scout | work | verify | critic | synthesize
    pattern: str  # single | pipeline | parallel
    steps: int
    lane_order: List[str]
    max_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowPlan:
    """A deterministic, inspectable plan. ``to_dict()`` is what a caller shows."""

    goal: str
    items: List[WorkItem]
    phases: List[PhasePlan]
    budget_tokens: int
    estimated_tokens: int
    verify_skeptics: int
    refute_default: bool
    public_safe: bool
    scout: bool
    synthesize: bool
    completeness_critic: bool
    critic_rounds: int
    max_items: int = MAX_WORK_ITEMS
    deadline_s: float = 90.0
    stage_timeout_s: float = 300.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "items": [item.to_dict() for item in self.items],
            "phases": [phase.to_dict() for phase in self.phases],
            "budget_tokens": self.budget_tokens,
            "estimated_tokens": self.estimated_tokens,
            "verify_skeptics": self.verify_skeptics,
            "refute_default": self.refute_default,
            "public_safe": self.public_safe,
            "scout": self.scout,
            "synthesize": self.synthesize,
            "completeness_critic": self.completeness_critic,
            "critic_rounds": self.critic_rounds,
            "max_items": self.max_items,
            "deadline_s": self.deadline_s,
            "stage_timeout_s": self.stage_timeout_s,
            "notes": list(self.notes),
        }


@dataclass
class UltraResult:
    """Everything a caller needs to act on and to audit the run."""

    run_id: str
    goal: str
    outcomes: List[Dict[str, Any]]
    synthesis: Optional[str]
    #: Every work item got a verdict (confirmed or refuted), none dropped, the
    #: completeness critic went dry. NOT "everything was confirmed".
    complete: bool
    notes: List[str]
    notebook_id: Optional[str]
    journal_path: str
    lanes_used: Dict[str, Dict[str, Any]]
    local_share: float
    tokens: int
    cost_usd: float
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt builders (PURE: deterministic input -> deterministic prompt, so a
# resumed run replays rather than re-asks)
# ---------------------------------------------------------------------------


def _needs(prompt: str, max_tokens: int, *, public_safe: bool = False,
           deadline_s: float = 90.0,
           overrides: Optional[Dict[str, Any]] = None) -> _lanes.StepNeeds:
    fields: Dict[str, Any] = {
        "kind": _lanes.KIND_LLM,
        "public_safe": bool(public_safe),
        "max_tokens": int(max_tokens),
        "deadline_s": float(deadline_s),
    }
    fields.update(overrides or {})
    # A caller may narrow `kind`, `max_tokens`, `deadline_s` or `model`; it may
    # not under-report its own size -- the lane selector needs the REAL length
    # or a huge prompt would be sent to a local lane that cannot prefill it.
    fields["input_chars"] = max(len(prompt),
                                int(fields.get("input_chars") or 0))
    return _lanes.StepNeeds(**fields)


def scout_prompt(goal: str, items: Sequence[WorkItem]) -> str:
    listing = "\n".join(f"- {item.id}: {item.title}" for item in items)
    return (
        "You are scouting a fan-out audit before it starts.\n"
        f"GOAL: {goal}\n"
        f"WORK LIST:\n{listing}\n\n"
        "Answer in at most 6 short lines: what 'done' means, what evidence a "
        "finding must cite, and the main way this audit could be fooled. No "
        "preamble, no restating the goal."
    )


def work_prompt(goal: str, item: WorkItem) -> str:
    payload = (item.payload or "").strip() or "(no extra input)"
    return (
        "You are ONE worker in a fan-out audit. Work only on your own item.\n"
        f"GOAL: {goal}\n"
        f"ITEM {item.id}: {item.title}\n"
        f"INPUT:\n{payload}\n\n"
        "Return the finding for THIS item only. Cite concrete evidence (ids, "
        "paths, numbers, votes) that is present in the input. If the item has "
        "nothing to report, answer exactly NONE and give the reason. Be terse."
    )


def skeptic_prompt(goal: str, item: WorkItem, claim: str, index: int) -> str:
    preview = (claim or "").strip()[:CLAIM_PREVIEW_CHARS] or "(empty claim)"
    return (
        f"You are skeptic {index + 1}. Your job is to REFUTE the claim below.\n"
        "A claim survives only if the evidence given does NOT contradict it and "
        "the claim does not overreach that evidence.\n"
        f"GOAL: {goal}\nITEM {item.id}: {item.title}\n"
        f"CLAIM:\n{preview}\n\n"
        "Answer with exactly one line, either\n"
        "REFUTED: <the concrete contradiction>\n"
        "or\n"
        "NOT-REFUTED"
    )


def critic_prompt(goal: str, items: Sequence[WorkItem]) -> str:
    listing = "\n".join(f"- {item.id}: {item.title}" for item in items)
    return (
        "You are the completeness critic for this plan. The work list so far:\n"
        f"{listing}\n"
        f"GOAL: {goal}\n\n"
        "Name only work items that are genuinely MISSING for the goal. Reply "
        "with one line per missing item:\n"
        "MISSING: <one sentence title>\n"
        "If nothing is missing reply exactly DONE. Do not repeat listed items."
    )


def synthesis_prompt(goal: str, outcomes: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for outcome in outcomes:
        verdict = outcome.get("verify") or {}
        if not verdict.get("confirmed"):
            continue
        claim = (outcome.get("work") or {}).get("result") or ""
        lines.append(
            f"[{outcome.get('id')}] {outcome.get('title')}\n"
            f"  verified: {verdict.get('skeptic_votes', 0)} skeptic(s), "
            f"{verdict.get('skeptics_ok', 0)} answered\n"
            f"  finding: {str(claim).strip()[:CLAIM_FULL_CHARS]}"
        )
    confirmed = "\n".join(lines) or "(no finding survived verification)"
    return (
        "Write the final answer for this goal using ONLY the confirmed findings "
        "below. Cite each finding by its item id. State plainly what did NOT "
        "survive verification. Do not add facts that are not in the findings.\n"
        f"GOAL: {goal}\n\nCONFIRMED FINDINGS:\n{confirmed}"
    )


#: Full finding text allowed into the synthesis prompt per item.
CLAIM_FULL_CHARS = 1_200


# ---------------------------------------------------------------------------
# Planning (PURE: no network, no clock, no model)
# ---------------------------------------------------------------------------


def items_from_dicts(raw: Sequence[Dict[str, Any]]) -> List[WorkItem]:
    """Validate and normalise a caller's work list. Raises PlanError, never
    silently drops an entry."""
    items: List[WorkItem] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise PlanError(f"work item {index} must be an object, got "
                            f"{type(entry).__name__}")
        items.append(WorkItem(
            id=str(entry.get("id") or f"item-{index + 1}"),
            title=str(entry.get("title") or entry.get("name") or ""),
            payload=str(entry.get("payload") or entry.get("input") or ""),
            needs=dict(entry["needs"]) if entry.get("needs") else None,
        ))
    return items


def plan_workflow(
    goal: str,
    items: Sequence[WorkItem | Dict[str, Any]] = (),
    *,
    budget_tokens: Optional[int] = None,
    verify_skeptics: int = DEFAULT_SKEPTICS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    public_safe: bool = False,
    scout: bool = True,
    synthesize: bool = True,
    completeness_critic: bool = True,
    critic_rounds: int = 1,
    max_items: int = MAX_WORK_ITEMS,
    deadline_s: float = 90.0,
    stage_timeout_s: Optional[float] = None,
    lanes: Optional[Sequence[_lanes.Lane]] = None,
) -> WorkflowPlan:
    """Build a deterministic plan: phases, per-phase lane order, token estimate.

    The planner never calls a model. It is what ``workflow_plan`` shows a caller
    before anything is spent, and what ``run_plan`` executes. A budget smaller
    than the estimate is refused here rather than stopping mid-fan-out.
    """
    goal_text = (goal or "").strip()
    if not goal_text:
        raise PlanError("goal must not be empty")

    work_items: List[WorkItem] = []
    for entry in items or ():
        work_items.append(entry if isinstance(entry, WorkItem) else WorkItem(
            id=str(entry.get("id") or f"item-{len(work_items) + 1}"),
            title=str(entry.get("title") or entry.get("name") or ""),
            payload=str(entry.get("payload") or entry.get("input") or ""),
            needs=dict(entry["needs"]) if entry.get("needs") else None,
        ))
    if not work_items:
        # A goal with no explicit break-down is still a workflow: one item.
        work_items = [WorkItem(id="goal", title=goal_text[:160], payload=goal_text)]

    if len(work_items) > max_items:
        raise PlanError(
            f"{len(work_items)} work items requested, cap is {max_items}; "
            "split the goal into batches in the caller -- the plan refuses "
            "rather than silently running the first "
            f"{max_items}")
    if not 0 <= verify_skeptics <= MAX_SKEPTICS:
        raise PlanError(f"verify_skeptics must be 0..{MAX_SKEPTICS}, got "
                        f"{verify_skeptics}")
    if critic_rounds < 0 or critic_rounds > MAX_CRITIC_ROUNDS:
        raise PlanError(f"critic_rounds must be 0..{MAX_CRITIC_ROUNDS}, got "
                        f"{critic_rounds}")

    ladder = list(lanes) if lanes is not None else _lanes.default_lanes()

    def lane_order(needs: _lanes.StepNeeds) -> List[str]:
        return [name for name, _reason in _lanes.plan(needs, ladder)]

    phases: List[PhasePlan] = []
    estimate = 0

    if scout:
        prompt = scout_prompt(goal_text, work_items)
        needs = _needs(prompt, max_tokens, deadline_s=deadline_s)
        phases.append(PhasePlan("scout", "scout", "single", 1,
                                lane_order(needs), max_tokens))
        estimate += 2 * max_tokens

    item_steps = 0
    for item in work_items:
        needs = _needs(work_prompt(goal_text, item), max_tokens,
                       public_safe=public_safe, deadline_s=deadline_s,
                       overrides=item.needs)
        if not phases or phases[-1].name != "work":
            phases.append(PhasePlan("work", "work", "pipeline", 0,
                                    lane_order(needs), max_tokens))
        phases[-1] = _with_steps(phases[-1], phases[-1].steps + 1)
        item_steps += 1
        estimate += 2 * max_tokens

    if verify_skeptics:
        sample_claim = "x" * CLAIM_PREVIEW_CHARS
        needs = _needs(skeptic_prompt(goal_text, work_items[0], sample_claim, 0),
                       max_tokens, deadline_s=deadline_s)
        phases.append(PhasePlan(
            "verify", "verify", "parallel", len(work_items) * verify_skeptics,
            lane_order(needs), max_tokens))
        estimate += len(work_items) * verify_skeptics * 2 * max_tokens

    if completeness_critic and critic_rounds:
        needs = _needs(critic_prompt(goal_text, work_items), max_tokens,
                       deadline_s=deadline_s)
        phases.append(PhasePlan(
            "critic", "critic", "single", critic_rounds, lane_order(needs),
            max_tokens))
        estimate += critic_rounds * 2 * max_tokens

    if synthesize:
        needs = _needs(synthesis_prompt(goal_text, []), SYNTH_MAX_TOKENS,
                       deadline_s=deadline_s)
        phases.append(PhasePlan("synthesize", "synthesize", "single", 1,
                                lane_order(needs), SYNTH_MAX_TOKENS))
        estimate += 2 * SYNTH_MAX_TOKENS

    budget = int(budget_tokens) if budget_tokens else max(
        DEFAULT_BUDGET_TOKENS, estimate)
    if budget < estimate:
        raise PlanError(
            f"budget {budget} tokens is below the plan's estimate {estimate}; "
            "raise budget_tokens or reduce items/skeptics -- a run that dies "
            "mid-fan-out proves nothing")

    notes = [
        "work items flow through work->verify in a pipeline (no barrier)",
        "verify is refute-by-default: any failed or ambiguous skeptic refutes",
        "every step routes itself: local lanes first, cloud as overflow",
    ]
    if any(len(item.payload) > _lanes.DEFAULT_LOCAL_MAX_INPUT_CHARS
           for item in work_items):
        notes.append("some item payloads exceed the local lane limit; those "
                     "steps will land on cloud by selection, not by override")
    if public_safe:
        notes.append("public_safe=True: only steps the caller vouched for may "
                     "use the free public runner lane")
    if verify_skeptics == 0:
        notes.append("verify_skeptics=0: nothing can be CONFIRMED; findings "
                     "are reported unverified rather than trusted")

    return WorkflowPlan(
        goal=goal_text,
        items=work_items,
        phases=phases,
        budget_tokens=budget,
        estimated_tokens=estimate,
        verify_skeptics=verify_skeptics,
        refute_default=True,
        public_safe=public_safe,
        scout=scout,
        synthesize=synthesize,
        completeness_critic=completeness_critic,
        critic_rounds=critic_rounds,
        max_items=max_items,
        deadline_s=deadline_s,
        stage_timeout_s=(float(stage_timeout_s) if stage_timeout_s else
                         max(300.0, len(work_items) * (verify_skeptics + 1)
                                     * (2 * deadline_s + 30))),
        notes=notes,
    )


def _with_steps(phase: PhasePlan, steps: int) -> PhasePlan:
    return PhasePlan(phase.name, phase.kind, phase.pattern, steps,
                     phase.lane_order, phase.max_tokens)


# ---------------------------------------------------------------------------
# Verification (PURE parsing; refute by default)
# ---------------------------------------------------------------------------


def interpret_skeptic(answer: Any, *, ok: bool) -> Tuple[bool, str]:
    """``(refuted, reason)`` for one skeptic reply.

    NOT-REFUTED is the ONLY way a skeptic abstains. A failed step, an empty
    reply, or any text that is not recognisably an abstention is a REFUTATION:
    an unproven claim is not confirmed.
    """
    if not ok:
        return True, "skeptic step failed on every lane"
    text = str(answer or "").strip()
    if not text:
        return True, "skeptic returned an empty answer"
    upper = text.upper()
    if "NOT-REFUTED" in upper or "NOT REFUTED" in upper:
        # A reply that says both (e.g. quotes the instruction) still refutes.
        if "REFUTED:" in upper or upper.startswith("REFUTED"):
            return True, "skeptic replied REFUTED"
        return False, "not refuted"
    if "REFUTED" in upper:
        return True, "skeptic replied REFUTED"
    return True, "skeptic reply was not a recognisable verdict"


def _verdict(claim: str, skeptic_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    votes = [interpret_skeptic(r.get("answer"), ok=bool(r.get("ok")))
             for r in skeptic_results]
    refuted = [reason for is_refuted, reason in votes if is_refuted]
    confirmed = bool(votes) and not refuted
    return {
        "claim": claim,
        "confirmed": confirmed,
        "refute_default": True,
        "skeptic_votes": len(votes),
        "skeptics_ok": sum(1 for r in skeptic_results if r.get("ok")),
        "refutations": refuted,
        "skeptics": list(skeptic_results),
        "reason": ("every skeptic failed to refute" if confirmed
                   else "; ".join(refuted) or "no skeptic ran"),
    }


# ---------------------------------------------------------------------------
# Budget ledger (hard ceiling around step(), which does not meter itself)
# ---------------------------------------------------------------------------


class _Ledger:
    def __init__(self, budget_tokens: int):
        self.budget = int(budget_tokens)
        self.spent = 0
        self.reserved = 0

    def charge(self, bound: int) -> None:
        from .runtime import BudgetExhausted

        if self.spent + self.reserved + bound > self.budget:
            raise BudgetExhausted(
                f"budget exhausted before step: spent={self.spent} "
                f"reserved={self.reserved} next={bound} total={self.budget}")
        self.reserved += bound

    def settle(self, bound: int, tokens: Optional[int]) -> None:
        self.reserved = max(0, self.reserved - bound)
        self.spent += int(tokens) if tokens else bound


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _result_fields(res: Any) -> Dict[str, Any]:
    return {
        "ok": bool(getattr(res, "ok", False)),
        "lane": getattr(res, "lane", "none"),
        "served_by": getattr(res, "served_by", ""),
        "result": getattr(res, "result", None),
        "error": getattr(res, "error", "") or "",
        "tokens": getattr(res, "tokens", None),
        "latency_ms": getattr(res, "latency_ms", 0),
        "cost_usd": round(float(getattr(res, "cost_usd", 0.0) or 0.0), 6),
    }


def ultracode_journal_root() -> Path:
    """Where an ultracode run's journal lives. Env override first, absolute
    default second -- never a cwd-relative path (AWF004's lesson)."""
    env = os.environ.get("AWFLOW_JOURNAL_ROOT", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".awflow"


def ultra_status_path(run_id: str) -> Path:
    root = Path(os.environ.get("ULTRACODE_STATE_DIR", "") or
                (Path.home() / ".aither" / "ultracode"))
    return root / f"{run_id}.json"


def write_ultra_status(run_id: str, payload: Dict[str, Any]) -> str:
    """Atomically write a run's status JSON. Returns its path."""
    path = ultra_status_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)
    return str(path)


def read_ultra_status(run_id: str) -> Optional[Dict[str, Any]]:
    path = ultra_status_path(run_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("[awflow.ultracode] unreadable status %s: %s", path, exc)
        return None


def list_ultra_status(limit: int = 20) -> List[Dict[str, Any]]:
    root = ultra_status_path("_").parent
    if not root.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True)[:max(1, limit)]:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return rows


async def run_plan(
    plan: WorkflowPlan,
    *,
    run_id: Optional[str] = None,
    lanes: Optional[Sequence[_lanes.Lane]] = None,
    journal_root: Optional[Path | str] = None,
    dispatcher: Optional[Any] = None,
    progress: Optional[Callable[[Dict[str, Any]], Any]] = None,
    notebook: bool = True,
    mirror: bool = True,
) -> UltraResult:
    """Execute a plan on the journaled engine; return the cited result.

    ``progress`` is called with JSON-safe event dicts (a coroutine function is
    awaited). The run is mirrored as an expedition when the host wired the
    mirror hooks (``run_workflow(mirror=True)`` does that already).
    """
    from . import pipeline as _pipeline
    from . import run_workflow as _run_workflow
    from . import step as _step
    from .lanes import NoLaneError

    if not isinstance(plan, WorkflowPlan):
        raise PlanError("run_plan expects a WorkflowPlan (see plan_workflow)")

    run_id = run_id or str(uuid.uuid4())
    ledger = _Ledger(plan.budget_tokens)
    started = datetime.now(timezone.utc).isoformat()
    all_steps: List[Dict[str, Any]] = []
    abort: Dict[str, BaseException] = {}
    events: List[Dict[str, Any]] = []

    async def emit(event: Dict[str, Any]) -> None:
        event = {"ts": datetime.now(timezone.utc).isoformat(), **event}
        events.append(event)
        if progress is None:
            return
        try:
            result = progress(event)
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:  # noqa: BLE001 - progress must never sink a run
            logger.warning("[awflow.ultracode] progress callback failed: %s", exc)

    async def call(payload: str, needs: _lanes.StepNeeds, label: str):
        bound = 2 * needs.max_tokens
        ledger.charge(bound)
        try:
            res = await _step(payload, needs, label=label)
        except NoLaneError as exc:
            res = _lanes.LaneResult(ok=False, lane="none", served_by="none",
                                    error=str(exc))
        ledger.settle(bound, getattr(res, "tokens", None))
        record = {"label": label, **_result_fields(res)}
        all_steps.append(record)
        await emit({"type": "step", "label": label, "ok": record["ok"],
                    "lane": record["lane"], "served_by": record["served_by"],
                    "error": record["error"]})
        return res

    async def work_stage(context: Tuple[Any, Any, int], _stage: int) -> Dict[str, Any]:
        item: WorkItem = context[1]
        prompt = work_prompt(plan.goal, item)
        needs = _needs(prompt, DEFAULT_MAX_TOKENS, public_safe=plan.public_safe,
                       deadline_s=plan.deadline_s, overrides=item.needs)
        try:
            res = await call(prompt, needs, f"work:{item.id}")
        except Exception as exc:  # noqa: BLE001 - pipeline would swallow BaseException-free
            abort.setdefault("exc", exc)
            return {"id": item.id, "title": item.title,
                    "work": {"ok": False, "lane": "none", "served_by": "",
                             "result": None, "error": str(exc), "tokens": None,
                             "latency_ms": 0, "cost_usd": 0.0},
                    "verify": {"confirmed": False, "refutations": [str(exc)]}}
        await emit({"type": "item_work", "id": item.id, "title": item.title,
                    "ok": bool(res.ok), "lane": res.lane,
                    "served_by": res.served_by})
        return {"id": item.id, "title": item.title, "work": _result_fields(res)}

    async def verify_stage(context: Tuple[Any, Any, int], _stage: int) -> Dict[str, Any]:
        outcome = context[0]
        item: WorkItem = context[1]
        if not isinstance(outcome, dict) or "work" not in outcome:
            return {"id": item.id, "title": item.title,
                    "work": {"ok": False, "lane": "none",
                             "error": "work stage produced no result"},
                    "verify": {"confirmed": False, "refutations": ["no work"]}}
        work = outcome["work"]
        if not work.get("ok") or not str(work.get("result") or "").strip():
            verdict = _verdict("", [])
            verdict["refutations"] = ["work step produced no claim"]
            return {**outcome, "verify": verdict}
        claim = str(work.get("result"))
        skeptic_results: List[Dict[str, Any]] = []
        if plan.verify_skeptics:
            async def one(index: int) -> Dict[str, Any]:
                prompt = skeptic_prompt(plan.goal, item, claim, index)
                needs = _needs(prompt, DEFAULT_MAX_TOKENS,
                               public_safe=plan.public_safe,
                               deadline_s=plan.deadline_s)
                res = await call(prompt, needs, f"verify:{item.id}:{index + 1}")
                return {"index": index + 1, **_result_fields(res),
                        "answer": getattr(res, "result", None)}
            try:
                skeptic_results = list(await asyncio.gather(
                    *(one(i) for i in range(plan.verify_skeptics))))
            except Exception as exc:  # noqa: BLE001 - keep the hard errors
                abort.setdefault("exc", exc)
        verdict = _verdict(claim, skeptic_results)
        await emit({"type": "item_verified", "id": item.id,
                    "confirmed": verdict["confirmed"],
                    "refutations": verdict["refutations"]})
        return {**outcome, "verify": verdict}

    async def critic_round(round_index: int, current: List[WorkItem]) -> List[WorkItem]:
        prompt = critic_prompt(plan.goal, current)
        needs = _needs(prompt, DEFAULT_MAX_TOKENS, public_safe=plan.public_safe,
                       deadline_s=plan.deadline_s)
        res = await call(prompt, needs, f"critic:{round_index + 1}")
        answer = str(getattr(res, "result", "") or "")
        upper = answer.strip().upper()
        if not res.ok or upper.startswith("DONE"):
            return []
        missing = [
            line.split(":", 1)[1].strip()
            for line in answer.splitlines()
            if line.strip().upper().startswith("MISSING:")
            and line.split(":", 1)[1].strip()
        ]
        return missing

    async def script() -> Dict[str, Any]:
        if plan.scout:
            prompt = scout_prompt(plan.goal, plan.items)
            res = await call(prompt, _needs(prompt, DEFAULT_MAX_TOKENS,
                                            public_safe=plan.public_safe,
                                            deadline_s=plan.deadline_s),
                             "scout")
            await emit({"type": "scout", "ok": bool(res.ok),
                        "lane": res.lane, "served_by": res.served_by})

        outcomes = await _pipeline(list(plan.items), work_stage, verify_stage)
        if abort.get("exc"):
            raise abort["exc"]
        results = [o for o in outcomes if isinstance(o, dict)]
        # pipeline() turns a dropped item into None; account for it, never hide it.
        dropped = len(outcomes) - len(results)
        notes: List[str] = []
        if dropped:
            notes.append(f"{dropped} item(s) were dropped by the pipeline "
                         "after a stage error")

        if plan.completeness_critic:
            for round_index in range(plan.critic_rounds):
                missing = await critic_round(round_index, plan.items)
                if abort.get("exc"):
                    raise abort["exc"]
                if not missing:
                    await emit({"type": "critic", "round": round_index + 1,
                                "dry": True})
                    break
                if len(plan.items) + len(missing) > plan.max_items:
                    raise PlanError(
                        "critic listed more missing items than the work-list "
                        f"cap ({plan.max_items}); refusing to truncate")
                new_items = [WorkItem(id=f"extra-{round_index + 1}-{i + 1}",
                                      title=title)
                             for i, title in enumerate(missing)]
                plan.items.extend(new_items)
                await emit({"type": "critic", "round": round_index + 1,
                            "dry": False, "added": [i.id for i in new_items]})
                extra = await _pipeline(new_items, work_stage, verify_stage)
                if abort.get("exc"):
                    raise abort["exc"]
                results.extend(o for o in extra if isinstance(o, dict))
            else:
                notes.append("completeness critic was still listing missing "
                             "items when its rounds ran out; the run is "
                             "reported incomplete, not silently closed")
        # `complete` means the WORKFLOW finished: every item got a verdict
        # (confirmed OR refuted), no item was dropped, and the critic went dry.
        # It is NOT "every finding was confirmed" -- a refuted claim is a
        # successful audit, not a lost run. `verify.confirmed` carries that.
        judged = all(
            (r.get("verify") or {}).get("skeptic_votes", 0) > 0
            or plan.verify_skeptics == 0
            for r in results)
        complete = bool(results) and not dropped and judged
        if any("still listing missing" in note for note in notes):
            complete = False

        synthesis: Optional[str] = None
        if plan.synthesize:
            prompt = synthesis_prompt(plan.goal, results)
            res = await call(prompt, _needs(prompt, SYNTH_MAX_TOKENS,
                                            public_safe=plan.public_safe,
                                            deadline_s=plan.deadline_s),
                             "synthesize")
            synthesis = str(res.result) if res.ok and res.result else None
            if synthesis is None:
                notes.append("synthesis step failed on every lane; confirmed "
                             "findings are still returned")
        return {"results": results, "synthesis": synthesis, "complete": complete,
                "notes": notes}

    await emit({"type": "workflow_started", "run_id": run_id,
                "goal": plan.goal, "items": len(plan.items),
                "budget_tokens": plan.budget_tokens})
    root = journal_root if journal_root is not None else ultracode_journal_root()
    Path(root).mkdir(parents=True, exist_ok=True)
    script.__name__ = "awflow_ultracode"
    try:
        payload = await _run_workflow(
            script, journal_path=root, budget_tokens=plan.budget_tokens,
            mirror=mirror, dispatcher=dispatcher,
            lanes=list(lanes) if lanes is not None else None, run_id=run_id,
            stage_timeout_s=plan.stage_timeout_s)
    except BaseException as exc:
        await emit({"type": "workflow_failed", "run_id": run_id,
                    "error": f"{type(exc).__name__}: {exc}"})
        raise

    stats = await _lane_stats(Path(root), run_id, all_steps)
    finished = datetime.now(timezone.utc).isoformat()
    result = UltraResult(
        run_id=run_id,
        goal=plan.goal,
        outcomes=payload["results"],
        synthesis=payload["synthesis"],
        complete=bool(payload["complete"]),
        notes=list(payload["notes"]),
        notebook_id=None,
        journal_path=str(Path(root) / run_id / "journal.jsonl"),
        lanes_used=stats["lanes_used"],
        local_share=stats["local_share"],
        tokens=stats["tokens"] or sum(int(s.get("tokens") or 0)
                                      for s in all_steps),
        cost_usd=round(stats["cost_usd"] or sum(
            float(s.get("cost_usd") or 0) for s in all_steps), 6),
        started_at=started,
        finished_at=finished,
    )
    if notebook:
        result.notebook_id, note = _write_notebook(plan, result, events)
        if note:
            result.notes.append(note)
    await emit({"type": "workflow_finished", "run_id": run_id,
                "complete": result.complete, "local_share": result.local_share,
                "tokens": result.tokens, "cost_usd": result.cost_usd,
                "notebook_id": result.notebook_id})
    return result


async def _lane_stats(root: Path, run_id: str,
                      fallback: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Read the run's journal and count where every attempt actually ran.

    Tokens and cost are summed over ATTEMPTS, not winning lanes, because a
    failed cloud attempt is still spend. ``local_share`` is winners only: the
    share of completed steps whose attempt ran on the local lane.
    """
    lanes_used: Dict[str, Dict[str, Any]] = {}
    tokens = 0
    cost = 0.0
    ok_steps = 0
    local_ok = 0
    try:
        records, _corrupt = await _read_records(root, run_id)
    except Exception as exc:  # noqa: BLE001 - stats must not sink a finished run
        logger.warning("[awflow.ultracode] journal unreadable: %s", exc)
        for step in fallback:
            lane = step.get("lane") or "none"
            entry = lanes_used.setdefault(lane, {"attempts": 0, "ok": 0,
                                                 "failures": 0, "served_by": {}})
            entry["attempts"] += 1
            entry["ok" if step.get("ok") else "failures"] += 1
            tokens += int(step.get("tokens") or 0)
            cost += float(step.get("cost_usd") or 0)
            if step.get("ok"):
                ok_steps += 1
                local_ok += 1 if lane == "local" else 0
        return {"lanes_used": lanes_used, "tokens": tokens, "cost_usd": cost,
                "local_share": (local_ok / ok_steps) if ok_steps else 0.0}
    for record in records:
        if record.get("type") != "LANE_DISPATCH":
            continue
        lane = str(record.get("lane") or "none")
        entry = lanes_used.setdefault(lane, {"attempts": 0, "ok": 0,
                                             "failures": 0, "served_by": {}})
        entry["attempts"] += 1
        tokens += int(record.get("tokens") or 0)
        cost += float(record.get("cost_usd") or 0)
        if record.get("ok"):
            entry["ok"] += 1
            ok_steps += 1
            if lane == "local":
                local_ok += 1
            served = str(record.get("served_by") or lane)
            entry["served_by"][served] = entry["served_by"].get(served, 0) + 1
        else:
            entry["failures"] += 1
            err = str(record.get("error") or "")
            if err and "error_sample" not in entry:
                entry["error_sample"] = err[:300]
    return {"lanes_used": lanes_used, "tokens": tokens, "cost_usd": cost,
            "local_share": (local_ok / ok_steps) if ok_steps else 0.0}


async def _read_records(root: Path, run_id: str):
    journal = _journal.Journal(run_id=run_id, root=root)
    return await journal.read_records()


def _write_notebook(plan: WorkflowPlan, result: UltraResult,
                    events: Sequence[Dict[str, Any]]):
    """Persist the run as an Agent Notebook. Returns (notebook_id, note)."""
    try:
        from lib.orchestration.NotebookStore import NotebookStore
    except Exception as exc:  # noqa: BLE001 - a host without the store still runs
        return None, f"notebook not written: NotebookStore unavailable ({exc})"
    notebook_id = f"nb_ultra_{result.run_id}"
    cells: List[Dict[str, Any]] = [{
        "id": "plan",
        "type": "markdown",
        "title": "Plan",
        "content": json.dumps(plan.to_dict(), indent=2),
    }]
    for outcome in result.outcomes:
        cells.append({
            "id": str(outcome.get("id") or "item"),
            "type": "markdown",
            "title": str(outcome.get("title") or "item"),
            "content": json.dumps({
                "work": outcome.get("work"),
                "verify": outcome.get("verify"),
            }, indent=2, default=str),
        })
    cells.append({
        "id": "result",
        "type": "markdown",
        "title": "Result",
        "content": json.dumps({
            "synthesis": result.synthesis,
            "complete": result.complete,
            "local_share": result.local_share,
            "lanes_used": result.lanes_used,
            "journal": result.journal_path,
            "events": list(events)[-50:],
        }, indent=2, default=str),
    })
    definition = json.dumps({"metadata": {"id": notebook_id,
                                          "name": f"Ultracode: {plan.goal[:60]}",
                                          "run_id": result.run_id},
                             "cells": cells}, default=str)
    try:
        NotebookStore().save_notebook(
            notebook_id, f"Ultracode: {plan.goal[:60]}",
            f"awflow ultracode run {result.run_id}", definition,
            status="draft", tags=["ultracode", "awflow"],
            created_by="awflow.ultracode")
        return notebook_id, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"notebook write failed ({type(exc).__name__}: {exc})"


async def run_ultracode(
    goal: str,
    items: Sequence[WorkItem | Dict[str, Any]] = (),
    **plan_kwargs: Any,
) -> UltraResult:
    """One-call convenience: plan, then run. Everything else lives in the two
    halves so a caller can show the plan before it spends."""
    plan = plan_workflow(goal, items, **plan_kwargs)
    return await run_plan(plan)


__all__ = [
    "DEFAULT_SKEPTICS", "MAX_SKEPTICS", "MAX_WORK_ITEMS",
    "DEFAULT_BUDGET_TOKENS",
    "VERDICT_CONFIRMED", "VERDICT_REFUTED",
    "PlanError", "WorkItem", "PhasePlan", "WorkflowPlan", "UltraResult",
    "plan_workflow", "run_plan", "run_ultracode", "items_from_dicts",
    "scout_prompt", "work_prompt", "skeptic_prompt", "critic_prompt",
    "synthesis_prompt", "interpret_skeptic",
    "ultracode_journal_root", "ultra_status_path", "write_ultra_status",
    "read_ultra_status", "list_ultra_status",
]
