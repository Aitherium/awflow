"""
awflow.lanes — run each workflow step on the cheapest lane that can do it.

WHY THIS EXISTS
---------------
awflow could already call a model (``dispatch.py``) and journal the call, but
every step went to the SAME place. The box has three very different places to
run a step and no code chose between them:

* **local** — the Spark DeepSeek pool's non-interactive slot and Bonsai 2 on the
  5090, reached through the same MicroScheduler endpoint as everything else.
  Cheapest by far (no per-token bill) but the pool has ONE agent slot, and a
  large prompt cannot finish prefill inside a normal step deadline. Measured
  2026-11-22: an 8k-token probe got no first token in 180 s; a subagent turn's
  median prompt is ~244k tokens. So local is the RIGHT lane only for a small
  step, and it must REFUSE a large one rather than stall on it.
* **runner** — GitHub-hosted ``ubuntu-latest`` runners on the PUBLIC ``aw*``
  mirror repos. Free and unlimited for public repositories, and already proven
  by ``awrun/.github/workflows/bonsai-swarm.yml`` (recent green CI, measured
  2026-11-22). It is a COMPUTE lane: the jobs hold no secret and never call
  back. A model call from a runner only exists as the in-job CPU model that
  workflow already loads, never as a gateway credential (see the credential
  note below).
* **cloud** — DeepSeek through the router, the burst/overflow lane. Costs money
  per token and is the only lane that can take a too-large-for-local prompt.

A step DECLARES what it needs; the selector returns the ordered lanes that can
serve it, cheapest first, and dispatch fails over along that order. Every
attempt is journaled with ``served_by``, latency, tokens and cost, so "where did
this step actually run, and what did it cost" is a recorded fact rather than a
guess.

CREDENTIAL VERDICT (owner rule: NO NEW CI SECRETS)
--------------------------------------------------
A GitHub-hosted runner is given NO credential by this module. The existing ways
to mint one all fail the rule:

* ``provision_gateway_key`` (``apps/awnode/tools/mcp/mcp_gateway_provision.py``)
  mints ``aither_sk_live_*`` from ACTA and needs ``AITHER_INTERNAL_SECRET`` plus
  an authenticated caller — putting that in a public repo's Actions is a new CI
  secret and a public leak.
* ``awpermit`` issues Ed25519 permits a door verifies offline, but no gateway
  door verifies them today, and a permit handed to a public workflow_dispatch is
  visible in the run.
* ``tokengate`` is a local model-tier router, not a credential.

So the runner lane is NON-LLM COMPUTE ONLY. A caller that asks for an LLM step
on the runner gets a refusal naming the reason, never a silent cloud call.

Design: pure selection, injected transports.
Every network edge is an injectable callable so a test exercises the whole
matrix — selection, failover, journaling — with fake lanes and no spend.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

KIND_LLM = "llm"
KIND_COMPUTE = "compute"
VALID_KINDS = frozenset({KIND_LLM, KIND_COMPUTE})

#: Local model ids as the router/policy names them (2026-11-22). Cheap first:
#: the pool's one non-interactive slot, then the 5090's Bonsai 2.
DEFAULT_LOCAL_MODELS: Tuple[str, ...] = ("deepseek-v4-flash-pool", "bonsai2-5090")
#: Cloud overflow: DeepSeek Flash THROUGH THE AWRouter (127.0.0.1:8286), the same
#: cloud path every Claude session uses. The paid api.deepseek.com account is
#: dark by owner order (2026-11-12), and MicroScheduler STEERS a cloud DeepSeek
#: model back to the local pool, so a ``cloud`` lane pointed at the scheduler
#: would silently be local. The router's cloud lane is the real overflow.
DEFAULT_CLOUD_MODEL = "deepseek-v4-flash"
DEFAULT_CLOUD_URL = "http://127.0.0.1:8286"
CLOUD_URL_ENV = "AWFLOW_CLOUD_URL"

#: DeepSeek list price, USD per 1M tokens. A step reports only a total token
#: count, so cost is a documented UPPER BOUND priced at the output rate.
CLOUD_USD_PER_MTOK = 0.42

#: A local lane refuses a step whose input exceeds this many characters. Matches
#: the subagent gate the router agent set on 2026-11-22: a bigger prompt cannot
#: finish prefill here inside the step deadline.
DEFAULT_LOCAL_MAX_INPUT_CHARS = 48_000


class NeedsError(ValueError):
    """A step declaration that cannot be acted on (a refusal, not a crash)."""


class NoLaneError(RuntimeError):
    """No lane can serve the declared needs; the message names why each refused."""

    def __init__(self, message: str, refusals: Optional[Dict[str, str]] = None):
        super().__init__(message)
        self.refusals = refusals or {}


@dataclass(frozen=True)
class StepNeeds:
    """What a step needs. ``public_safe`` defaults to False: fail closed.

    A step is public-safe only when its inputs AND outputs are already public
    (public repositories, public data). Assuming it is safe is how private work
    would end up on a public CI log, so the default refuses the free lane.
    """

    kind: str = KIND_LLM
    public_safe: bool = False
    input_chars: int = 0
    max_tokens: int = 2048
    deadline_s: float = 120.0
    require_local: bool = False
    allow_cloud: bool = True
    model: Optional[str] = None
    job: Optional[str] = None  # runner workflow/job name, when kind == compute

    def __post_init__(self) -> None:
        if self.kind not in VALID_KINDS:
            raise NeedsError(f"kind must be one of {sorted(VALID_KINDS)}, got {self.kind!r}")
        if self.input_chars < 0:
            raise NeedsError("input_chars must be >= 0")
        if self.max_tokens <= 0:
            raise NeedsError("max_tokens must be > 0")
        if self.deadline_s <= 0:
            raise NeedsError("deadline_s must be > 0")
        if self.require_local and not self.allow_cloud:
            # Both are fine together; nothing to refuse. Kept for symmetry docs.
            pass


@dataclass
class LaneResult:
    """Outcome of one lane attempt. ``ok=False`` carries the reason, never raises."""

    ok: bool
    lane: str
    served_by: str = ""
    result: Any = None
    tokens: Optional[int] = None
    latency_ms: int = 0
    cost_usd: float = 0.0
    error: str = ""

    def to_journal_fields(self) -> Dict[str, Any]:
        return {
            "lane": self.lane,
            "served_by": self.served_by or self.lane,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "cost_usd": round(self.cost_usd, 6),
            "error": self.error or None,
        }


# A dispatch edge: (payload, needs) -> LaneResult. Injected so tests never
# touch the network and a host can point a lane at its own transport.
DispatchFn = Callable[[str, StepNeeds], Awaitable[LaneResult]]


class Lane:
    """Base lane. Subclasses declare limits and an injected ``dispatch_fn``."""

    name = "lane"
    #: The free public lane refuses anything not declared public-safe.
    requires_public_safe = False
    #: Which step kinds this lane can run. The runner lane is compute-only; the
    #: cloud LLM lane is llm-only; local can do either (when a host wires it).
    serves_llm = True
    serves_compute = True
    #: Max input characters, or None for unlimited.
    max_context_chars: Optional[int] = None
    #: How many steps this lane will run at once (the swarm bound).
    max_concurrency: int = 1

    def __init__(self, dispatch_fn: Optional[DispatchFn] = None):
        self._dispatch_fn = dispatch_fn

    # -- capability -------------------------------------------------------
    def eligible(self, needs: StepNeeds) -> Tuple[bool, str]:
        """Return (can_serve, reason). Pure; subclasses add their own limits."""
        if needs.kind == KIND_LLM and not self.serves_llm:
            return False, f"{self.name}: does not serve llm steps"
        if needs.kind == KIND_COMPUTE and not self.serves_compute:
            return False, f"{self.name}: does not serve compute steps"
        return True, "ok"

    def price(self, tokens: Optional[int]) -> float:
        return 0.0

    # -- execution --------------------------------------------------------
    async def dispatch(self, payload: str, needs: StepNeeds) -> LaneResult:
        """Run one step. Never raises: a failure is a LaneResult(ok=False)."""
        can, reason = self.eligible(needs)
        if not can:
            return LaneResult(ok=False, lane=self.name, served_by=self.name, error=reason)
        if self._dispatch_fn is None:  # pragma: no cover - guarded per subclass
            return LaneResult(ok=False, lane=self.name, served_by=self.name,
                              error=f"{self.name}: no dispatch transport configured")
        start = time.time()
        try:
            res = await self._dispatch_fn(payload, needs)
        except Exception as exc:  # noqa: BLE001 - a lane failure is data, not a crash
            logger.warning("[awflow.lanes] %s dispatch raised: %s", self.name, exc)
            return LaneResult(ok=False, lane=self.name, served_by=self.name,
                              latency_ms=int((time.time() - start) * 1000),
                              error=f"{type(exc).__name__}: {exc}")
        if res.latency_ms == 0:
            res.latency_ms = int((time.time() - start) * 1000)
        if res.served_by == "":
            res.served_by = self.name
        if res.tokens is not None and not res.cost_usd:
            res.cost_usd = self.price(res.tokens)
        return res


class LocalLane(Lane):
    """The box's own models. Cheapest; refuses a step it cannot finish in time.

    ``capacity`` is an optional callable returning ``(free_slots, detail)``.
    ``None`` means "not measured here": the lane stays eligible, because a
    missing probe must not read as "no capacity" (that would retire the local
    lane on every host that did not wire one).
    """

    name = "local"
    serves_llm = True
    serves_compute = True  # only when a host wires a compute transport
    max_concurrency = 1  # one non-interactive slot on the pool

    def __init__(
        self,
        dispatch_fn: Optional[DispatchFn] = None,
        models: Sequence[str] = DEFAULT_LOCAL_MODELS,
        max_context_chars: int = DEFAULT_LOCAL_MAX_INPUT_CHARS,
        capacity: Optional[Callable[[], Tuple[int, str]]] = None,
    ):
        super().__init__(dispatch_fn)
        self.models = tuple(models)
        self.max_context_chars = int(max_context_chars)
        self._capacity = capacity

    def eligible(self, needs: StepNeeds) -> Tuple[bool, str]:
        can, reason = super().eligible(needs)
        if not can:
            return can, reason
        if needs.input_chars > self.max_context_chars:
            return False, (
                f"local: too-large ({needs.input_chars} chars > "
                f"{self.max_context_chars}; prefill cannot finish in the deadline)"
            )
        if self._capacity is not None:
            try:
                free, detail = self._capacity()
            except Exception as exc:  # noqa: BLE001
                return False, f"local: capacity probe failed ({exc})"
            if free <= 0:
                return False, f"local: saturated ({detail})"
        return True, "ok"

    def price(self, tokens: Optional[int]) -> float:
        return 0.0  # amortised electricity; no per-token bill


class CloudLane(Lane):
    """DeepSeek through the router: the burst/overflow lane, priced per token."""

    name = "cloud"
    serves_llm = True
    serves_compute = False  # a model call is not a compute executor
    max_concurrency = 8  # cloud rate limits are the real bound

    def __init__(self, dispatch_fn: Optional[DispatchFn] = None,
                 model: str = DEFAULT_CLOUD_MODEL):
        super().__init__(dispatch_fn)
        self.model = model

    def eligible(self, needs: StepNeeds) -> Tuple[bool, str]:
        if needs.require_local:
            return False, "cloud: step requires local (never leaves the box)"
        if not needs.allow_cloud:
            return False, "cloud: step forbids cloud"
        return super().eligible(needs)

    def price(self, tokens: Optional[int]) -> float:
        return (float(tokens or 0) / 1_000_000.0) * CLOUD_USD_PER_MTOK


class RunnerLane(Lane):
    """GitHub-hosted runners on a public mirror: FREE, and compute-only.

    It refuses an LLM step outright (see the module's credential note) and
    refuses any step not declared public-safe. Both refusals are the security
    boundary; neither is a policy a caller can opt out of by accident.
    """

    name = "runner"
    requires_public_safe = True
    serves_llm = False
    serves_compute = True
    max_concurrency = 32  # the documented bonsai-swarm bound

    def __init__(self, dispatch_fn: Optional[DispatchFn] = None,
                 repos: Sequence[str] = ()):
        super().__init__(dispatch_fn)
        self.repos = tuple(repos)

    def eligible(self, needs: StepNeeds) -> Tuple[bool, str]:
        if needs.kind == KIND_LLM:
            return False, (
                "runner: llm steps need a credential and NO NEW CI SECRETS "
                "allows none; the runner lane is non-llm compute only"
            )
        if not needs.public_safe:
            return False, "runner: step is not declared public-safe"
        if needs.require_local:
            return False, "runner: step requires local"
        return True, "ok"

    def price(self, tokens: Optional[int]) -> float:
        return 0.0  # free minutes on public repositories


#: A ``workflow_dispatch`` input is capped near 65 KB. A public-safe compute
#: payload larger than this must be fetched by the job, not passed inline.
RUNNER_MAX_INLINE_CHARS = 60_000


def github_actions_argv(repo: str, workflow: str, ref: str,
                        inputs: Dict[str, str]) -> List[str]:
    """The ``gh workflow run`` argv for one dispatch. PURE — the testable half.

    ``repo`` is ``owner/name`` of a PUBLIC mirror, ``workflow`` a file (e.g.
    ``swarm.yml``) or name, ``ref`` a branch/tag. Every ``inputs`` pair becomes
    a ``-f key=value``; values are passed as single argv items, never through a
    shell, so a payload cannot become a command.
    """
    if "/" not in repo:
        raise NeedsError(f"repo must be owner/name, got {repo!r}")
    if not workflow or not ref:
        raise NeedsError("workflow and ref are required")
    argv = ["gh", "workflow", "run", workflow, "--repo", repo, "--ref", ref]
    for key, value in inputs.items():
        argv += ["-f", f"{key}={value}"]
    return argv


def github_actions_transport(
    repo: str,
    workflow: str,
    *,
    ref: str = "main",
    input_key: str = "tasks_json",
    runner: Optional[Callable[[List[str]], str]] = None,
) -> DispatchFn:
    """A dispatch transport that TRIGGERS a public workflow; it does not wait.

    The job's result is collected by whoever dispatched (``awrun swarm`` already
    does exactly this), so this returns the run pointer as the step result and
    records it in the journal. ``runner(argv) -> stdout`` is injectable: the
    default shells out with no shell, so a payload is never interpreted.
    """
    async def dispatch_fn(payload: str, needs: StepNeeds) -> LaneResult:
        if len(payload) > RUNNER_MAX_INLINE_CHARS:
            return LaneResult(
                ok=False, lane="runner",
                error=(f"runner: payload too large for a workflow_dispatch input "
                       f"({len(payload)} > {RUNNER_MAX_INLINE_CHARS}); have the job "
                       "fetch it instead"))
        argv = github_actions_argv(repo, workflow, ref, {input_key: payload})
        runner_fn = runner or _gh_runner
        try:
            # subprocess is blocking; run it off the event loop so a fan-out of
            # dispatches does not serialise on process spawn.
            out = await asyncio.to_thread(runner_fn, argv)
        except Exception as exc:  # noqa: BLE001 - a dispatch failure fails over
            return LaneResult(ok=False, lane="runner",
                              error=f"runner: gh workflow run failed: {exc}")
        return LaneResult(ok=True, lane="runner",
                          served_by=f"github:{repo}/{workflow}",
                          result=(out or "").strip()[-2000:])

    return dispatch_fn


def _gh_runner(argv: List[str]) -> str:
    import subprocess
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=60,
                          encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "gh failed").strip()[:400])
    return proc.stdout


def parse_anthropic_reply(data: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    """Pull ``(text, total_tokens)`` out of an Anthropic Messages reply.

    The awrouter speaks the Anthropic shape: ``content`` is a list of blocks and
    ``usage`` carries ``input_tokens``/``output_tokens``. A reply with no text
    block is ``("")`` -- the caller treats an empty answer as a failure.
    """
    blocks = data.get("content") or []
    text = "".join(str(b.get("text") or "") for b in blocks
                   if isinstance(b, dict) and b.get("type") == "text")
    usage = data.get("usage") or {}
    total = (int(usage.get("input_tokens") or 0)
             + int(usage.get("output_tokens") or 0))
    return text, (total or None)


def default_lanes() -> List[Lane]:
    """The standard lane ladder, cheapest first.

    Transports are wired only when their environment is present; a lane with no
    transport still takes part in SELECTION (so the plan is honest) but reports
    "no dispatch transport configured" if chosen -- which fails over cleanly.
    """
    from .dispatch import dispatch_agent_call

    async def _local_dispatch(payload: str, needs: StepNeeds) -> LaneResult:
        if needs.kind == KIND_COMPUTE:
            # No local command runner is wired by default: running an arbitrary
            # payload as a shell command is a capability a HOST must grant, not
            # something a workflow runtime does on its own. Refuse and fail over.
            return LaneResult(
                ok=False, lane="local",
                error="local: no compute transport configured (a host must "
                      "supply one); refusing to treat a payload as a command")
        errs: List[str] = []
        for model in DEFAULT_LOCAL_MODELS:
            result, tokens, err = await dispatch_agent_call(
                payload, model=needs.model or model,
                max_tokens=needs.max_tokens)
            if err is None and result is not None:
                return LaneResult(ok=True, lane="local", served_by=model,
                                  result=result, tokens=tokens)
            errs.append(f"{model}: {getattr(err, 'message', err) or 'no result'}")
        return LaneResult(ok=False, lane="local", error="; ".join(errs))

    async def _cloud_dispatch(payload: str, needs: StepNeeds) -> LaneResult:
        # The awrouter's Anthropic-compatible door. NOT the paid DeepSeek API
        # (dark by owner order) and NOT MicroScheduler (which steers a cloud
        # DeepSeek model back to the local pool -> a false "cloud" lane).
        base = (os.environ.get(CLOUD_URL_ENV) or DEFAULT_CLOUD_URL).rstrip("/")
        model = needs.model or DEFAULT_CLOUD_MODEL
        body = {
            "model": model,
            "max_tokens": needs.max_tokens,
            "messages": [{"role": "user", "content": payload}],
        }
        try:
            import httpx
            async with httpx.AsyncClient(timeout=needs.deadline_s) as client:
                resp = await client.post(
                    f"{base}/v1/messages", json=body,
                    headers={"anthropic-version": "2023-06-01",
                             "content-type": "application/json"})
        except Exception as exc:  # noqa: BLE001 - transport failure fails over
            return LaneResult(ok=False, lane="cloud",
                              error=f"cloud: {base} unreachable ({exc})")
        if resp.status_code != 200:
            return LaneResult(ok=False, lane="cloud",
                              error=f"cloud: HTTP {resp.status_code}: "
                                    f"{(resp.text or '')[:200]}")
        try:
            text, tokens = parse_anthropic_reply(resp.json())
        except Exception as exc:  # noqa: BLE001
            return LaneResult(ok=False, lane="cloud",
                              error=f"cloud: unparseable reply ({exc})")
        if not text:
            return LaneResult(ok=False, lane="cloud",
                              error="cloud: empty answer")
        return LaneResult(ok=True, lane="cloud", served_by=model, result=text,
                          tokens=tokens,
                          cost_usd=(float(tokens or 0) / 1e6) * CLOUD_USD_PER_MTOK)

    return [
        LocalLane(dispatch_fn=_local_dispatch),
        CloudLane(dispatch_fn=_cloud_dispatch),
        RunnerLane(),  # transport is host-supplied; absent means "selectable, not runnable"
    ]


def plan(needs: StepNeeds, lanes: Sequence[Lane]) -> List[Tuple[str, str]]:
    """Ordered ``(lane_name, reason)`` the lanes that can serve ``needs``.

    Cheapest first. This is the whole selection policy and it is PURE:
    ``local`` before ``cloud`` for an LLM step (free before paid), ``runner``
    before ``local`` for public-safe compute (free before our own hardware).
    Each lane's own limits (size, saturation, secrets) decide whether it is in
    the list at all, so an oversized prompt naturally lands on cloud and a
    private step never lands on the free lane.
    """
    candidates = list(lanes)
    if needs.kind == KIND_COMPUTE and needs.public_safe and not needs.require_local:
        # Free public compute outranks our own hardware.
        candidates.sort(key=lambda lane: 0 if lane.name == "runner" else 1)
    elif needs.kind == KIND_LLM:
        # Free local model outranks the paid cloud lane.
        candidates.sort(key=lambda lane: 0 if lane.name == "local" else 1)

    ordered: List[Tuple[str, str]] = []
    for lane in candidates:
        can, reason = lane.eligible(needs)
        if can:
            ordered.append((lane.name, reason))
    return ordered


def refusals(needs: StepNeeds, lanes: Sequence[Lane]) -> Dict[str, str]:
    """Why each lane refused, for the error a caller sees when none can serve."""
    out: Dict[str, str] = {}
    for lane in lanes:
        can, reason = lane.eligible(needs)
        if not can:
            out[lane.name] = reason
    return out


@dataclass
class LaneDispatcher:
    """Select a lane per step, dispatch with failover, journal every attempt.

    ``journal`` is any object with ``await write_record(dict)`` — awflow's
    Journal, or a test stub. ``lanes`` defaults to the standard ladder.
    """

    lanes: List[Lane] = field(default_factory=default_lanes)
    journal: Any = None
    run_id: str = ""
    sequence: int = 0
    #: per-lane concurrency gates, created lazily so a dispatcher can be built
    #: outside an event loop.
    _gates: Dict[str, asyncio.Semaphore] = field(default_factory=dict, repr=False)

    def lane(self, name: str) -> Optional[Lane]:
        return next((lane for lane in self.lanes if lane.name == name), None)

    def plan(self, needs: StepNeeds) -> List[Tuple[str, str]]:
        return plan(needs, self.lanes)

    def _gate(self, lane: Lane) -> asyncio.Semaphore:
        gate = self._gates.get(lane.name)
        if gate is None:
            gate = asyncio.Semaphore(max(1, lane.max_concurrency))
            self._gates[lane.name] = gate
        return gate

    async def dispatch(self, payload: str, needs: StepNeeds,
                       *, label: Optional[str] = None) -> LaneResult:
        """Run one step on the first lane that can serve it; fail over on error.

        Raises NoLaneError only when NO lane can serve the declaration at all
        (a refusal naming every reason). A lane that was selectable but failed
        at runtime is a recorded attempt, and the next lane is tried.
        """
        ordered = self.plan(needs)
        if not ordered:
            raise NoLaneError(
                f"no lane can serve {needs.kind} step "
                f"(public_safe={needs.public_safe}, input_chars={needs.input_chars})",
                refusals(needs, self.lanes),
            )

        attempts: List[LaneResult] = []
        for name, _reason in ordered:
            lane = self.lane(name)
            if lane is None:  # pragma: no cover - plan() only returns real lanes
                continue
            async with self._gate(lane):
                res = await lane.dispatch(payload, needs)
            attempts.append(res)
            await self._journal_attempt(res, needs, label, lane_order=[n for n, _ in ordered])
            if res.ok:
                return res

        if not attempts:  # pragma: no cover - plan() guarantees at least one
            return LaneResult(ok=False, lane="none", served_by="none",
                              error="no lane attempted")
        return LaneResult(
            ok=False, lane="none", served_by="none",
            latency_ms=sum(a.latency_ms for a in attempts),
            error="all lanes failed: " + "; ".join(
                f"{a.lane}={a.error}" for a in attempts),
        )

    async def fan_out(self, items: Sequence[Any],
                      needs_fn: Callable[[Any], StepNeeds],
                      payload_fn: Callable[[Any], str],
                      *, label: Optional[str] = None) -> List[LaneResult]:
        """Run many steps concurrently — the swarm.

        Each item is routed independently (so a big one lands on cloud while a
        small one lands on local), and each lane's own gate bounds how many run
        at once. Results keep input order; a failed step is a LaneResult, not an
        exception, so one bad item never sinks the batch.
        """
        async def one(item: Any) -> LaneResult:
            try:
                return await self.dispatch(payload_fn(item), needs_fn(item), label=label)
            except NoLaneError as exc:
                return LaneResult(ok=False, lane="none", served_by="none", error=str(exc))

        return list(await asyncio.gather(*(one(item) for item in items)))

    async def _journal_attempt(self, res: LaneResult, needs: StepNeeds,
                               label: Optional[str], *,
                               lane_order: Sequence[str]) -> None:
        if self.journal is None:
            return
        record = {
            "type": "LANE_DISPATCH",
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence": self.sequence,
            "version": "1",
            "label": label,
            "need_kind": needs.kind,
            "public_safe": needs.public_safe,
            "input_chars": needs.input_chars,
            "lane_order": list(lane_order),
            "ok": res.ok,
            **res.to_journal_fields(),
        }
        self.sequence += 1
        try:
            await self.journal.write_record(record)
        except Exception as exc:  # noqa: BLE001 - a journal miss must not lose the step
            logger.error("[awflow.lanes] JOURNAL WRITE FAILED (step result kept): %s", exc)


__all__ = [
    "KIND_LLM", "KIND_COMPUTE", "VALID_KINDS",
    "DEFAULT_LOCAL_MODELS", "DEFAULT_CLOUD_MODEL", "CLOUD_USD_PER_MTOK",
    "DEFAULT_CLOUD_URL", "CLOUD_URL_ENV",
    "DEFAULT_LOCAL_MAX_INPUT_CHARS",
    "NeedsError", "NoLaneError",
    "StepNeeds", "LaneResult", "Lane",
    "LocalLane", "CloudLane", "RunnerLane",
    "LaneDispatcher", "default_lanes", "plan", "refusals",
    "github_actions_argv", "github_actions_transport", "RUNNER_MAX_INLINE_CHARS",
    "parse_anthropic_reply",
]
