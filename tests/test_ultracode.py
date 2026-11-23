"""Tests for awflow.ultracode — plan, fan out, refute, synthesize.

Same shape as test_lanes.py: plain asyncio.run, injected fake lanes, NO live
call. These tests prove the properties the capability is sold on:

* a plan is deterministic and refuses an ill-sized request instead of
  truncating it;
* work items flow through work->verify with no barrier and every item is
  accounted for;
* verification is refute-by-default (a failed or ambiguous skeptic refutes);
* the run reports where each step actually ran (local_share from the journal);
* the token budget is a hard ceiling.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from awflow import lanes, runtime, ultracode

L = lanes  # plain assignments, not import aliases (N812)
U = ultracode


class FakeLimits(L.Lane):
    """A real lane subclass with a fake transport; limits stay real."""

    name = "local"
    serves_llm = True
    serves_compute = False
    max_concurrency = 4
    max_context_chars = L.DEFAULT_LOCAL_MAX_INPUT_CHARS

    def __init__(self, name, dispatch_fn, cls=None):
        inner = (cls or L.LocalLane)(dispatch_fn=dispatch_fn)
        self.name = name
        self._dispatch_fn = dispatch_fn
        self.max_context_chars = getattr(inner, "max_context_chars", None)
        self.max_concurrency = getattr(inner, "max_concurrency", 4)
        self.serves_llm = inner.serves_llm
        self.serves_compute = inner.serves_compute
        self.requires_public_safe = inner.requires_public_safe

    def eligible(self, needs):
        if needs.kind == L.KIND_LLM and not self.serves_llm:
            return False, f"{self.name}: no llm"
        if needs.kind == L.KIND_COMPUTE and not self.serves_compute:
            return False, f"{self.name}: no compute"
        if (self.name == "local" and self.max_context_chars
                and needs.input_chars > self.max_context_chars):
            return False, f"local: too-large ({needs.input_chars})"
        return True, "ok"


def _reply(payload):
    low = payload.lower()
    if "you are skeptic" in low:
        return "NOT-REFUTED"
    if "completeness critic" in low:
        return "DONE"
    if "write the final answer" in low:
        return "SYNTH: confirmed"
    if "scouting a fan-out" in low:
        return "scout: cite ids"
    return "FINDING: evidence id=42"


def _lane(name, *, reply=_reply, fail=False, tokens=10, cls=None):
    async def dispatch_fn(payload, needs):
        return L.LaneResult(ok=not fail, lane=name, served_by=f"{name}:fake",
                            result=(None if fail else
                                    (reply(payload) if callable(reply) else reply)),
                            error=("fake failure" if fail else ""),
                            tokens=tokens)
    return FakeLimits(name, dispatch_fn, cls=cls)


def _run(plan, lanes, root=None, **kwargs):
    if root is None:
        with tempfile.TemporaryDirectory() as tmp:
            return asyncio.run(U.run_plan(
                plan, lanes=lanes, journal_root=tmp, notebook=False,
                mirror=False, **kwargs))
    return asyncio.run(U.run_plan(
        plan, lanes=lanes, journal_root=root, notebook=False,
        mirror=False, **kwargs))


def _small_plan(**overrides):
    kwargs = dict(items=[{"id": "a", "title": "Item A", "payload": "p"}],
                  verify_skeptics=2, scout=False, synthesize=False,
                  completeness_critic=False)
    kwargs.update(overrides)
    return U.plan_workflow("audit the thing", **kwargs)


# ---------------------------------------------------------------- planning


def test_plan_is_deterministic_and_names_lanes():
    first = _small_plan().to_dict()
    second = _small_plan().to_dict()
    assert first == second
    assert first["phases"][0]["name"] == "work"
    assert first["phases"][0]["pattern"] == "pipeline"
    assert first["phases"][1]["name"] == "verify"
    assert first["refute_default"] is True
    assert first["phases"][0]["lane_order"][0] == "local"


def test_plan_refuses_oversized_list_instead_of_truncating():
    items = [{"id": f"i{i}", "title": f"T{i}"} for i in range(6)]
    with pytest.raises(U.PlanError, match="cap is 5"):
        U.plan_workflow("goal", items, max_items=5)


def test_plan_refuses_budget_below_estimate():
    plan = _small_plan()
    with pytest.raises(U.PlanError, match="below the plan's estimate"):
        U.plan_workflow("goal", plan.items, budget_tokens=10)


def test_plan_refuses_empty_goal_and_bad_skeptics():
    with pytest.raises(U.PlanError):
        U.plan_workflow("   ")
    with pytest.raises(U.PlanError):
        _small_plan(verify_skeptics=U.MAX_SKEPTICS + 1)


def test_goal_with_no_items_is_still_a_workflow():
    plan = U.plan_workflow("do the thing", scout=False, synthesize=False,
                           completeness_critic=False)
    assert [item.id for item in plan.items] == ["goal"]


def test_plan_threads_the_step_deadline_into_every_phase():
    plan = _small_plan(deadline_s=30)
    assert plan.deadline_s == 30
    assert plan.to_dict()["deadline_s"] == 30


def test_plan_sizes_the_pipeline_stage_budget_for_a_serialized_lane():
    # One local slot: item 10's stage waits behind 9 earlier stages. A fixed
    # 300 s cap dropped all 10 items on 2026-11-22, so the plan sizes it.
    items = [{"id": f"i{i}", "title": f"T{i}"} for i in range(10)]
    plan = _small_plan(items=items, deadline_s=45)
    assert plan.stage_timeout_s > 300
    assert plan.to_dict()["stage_timeout_s"] == plan.stage_timeout_s


def test_pipeline_stage_timeout_is_configurable():
    class _J:
        async def write_record(self, record):
            return None

    async def run_with(timeout):
        rt = runtime.WorkflowRuntime(journal=_J(), concurrency_cap=4,
                                     stage_timeout_s=timeout)

        async def slow(_context, _stage):
            await asyncio.sleep(0.3)
            return "ok"

        return await rt.pipeline(["x"], slow), rt.stage_timeout_s

    dropped, applied = asyncio.run(run_with(0.1))
    assert dropped == [None] and applied == 0.1
    kept, _ = asyncio.run(run_with(2.0))
    assert kept == ["ok"]


def test_plan_notes_zero_skeptics_cannot_confirm():
    plan = _small_plan(verify_skeptics=0)
    assert any("nothing can be CONFIRMED" in n for n in plan.notes)


def test_cloud_lane_returns_text_and_names_the_model(monkeypatch):
    import httpx

    seen = {}

    class _Resp:
        status_code = 200

        def json(self):
            return {"content": [{"type": "text", "text": "ok"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1}}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, json=None, headers=None):
            seen["url"] = url
            seen["body"] = json
            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    cloud = next(lane for lane in L.default_lanes() if lane.name == "cloud")
    result = asyncio.run(cloud.dispatch(
        "hi", L.StepNeeds(kind="llm", input_chars=2, max_tokens=8)))
    assert result.ok is True
    assert result.served_by == L.DEFAULT_CLOUD_MODEL
    assert seen["body"]["thinking"] == {"type": "disabled"}


def test_cloud_body_disables_thinking_so_a_step_gets_text():
    # Measured 2026-11-22: with thinking on, DeepSeek spends max_tokens on
    # thinking and returns no text block, so every cloud fallback failed.
    needs = L.StepNeeds(kind="llm", input_chars=10, max_tokens=64)
    body = L.cloud_request_body("hi", needs)
    assert body["thinking"] == {"type": "disabled"}
    assert body["model"] == L.DEFAULT_CLOUD_MODEL
    assert body["max_tokens"] == 64


# ----------------------------------------------------------- verification


def test_skeptic_parsing_refutes_by_default():
    assert U.interpret_skeptic("NOT-REFUTED", ok=True) == (False, "not refuted")
    refuted, why = U.interpret_skeptic("REFUTED: the id does not exist", ok=True)
    assert refuted and "REFUTED" in why
    assert U.interpret_skeptic("", ok=True)[0] is True
    assert U.interpret_skeptic("maybe?", ok=True)[0] is True
    assert U.interpret_skeptic(None, ok=False)[0] is True


# ------------------------------------------------------------------- run


def test_run_confirms_synthesizes_and_reports_local_share(tmp_path):
    plan = _small_plan(scout=True, synthesize=True)
    result = _run(plan, [_lane("local")], root=tmp_path)
    assert result.complete is True
    assert result.synthesis == "SYNTH: confirmed"
    assert result.outcomes[0]["verify"]["confirmed"] is True
    assert result.outcomes[0]["verify"]["skeptic_votes"] == 2
    assert result.local_share == 1.0
    assert result.lanes_used["local"]["ok"] >= 1
    assert Path(result.journal_path).exists()


def test_one_refuting_skeptic_refutes_the_claim():
    def skeptic(payload):
        if "You are skeptic 2" in payload:
            return "REFUTED: the evidence contradicts the claim"
        return _reply(payload)

    plan = _small_plan()
    result = _run(plan, [_lane("local", reply=skeptic)])
    verdict = result.outcomes[0]["verify"]
    assert verdict["confirmed"] is False
    assert any("REFUTED" in reason for reason in verdict["refutations"])
    # A refuted claim is a finished audit, not an unfinished run.
    assert result.complete is True


def test_an_unjudged_item_makes_the_run_incomplete():
    plan = _small_plan()
    # Every lane fails, so work produces no claim and verify has no votes.
    result = _run(plan, [_lane("local", fail=True), _lane("cloud", fail=True)])
    assert result.outcomes[0]["work"]["ok"] is False
    assert result.complete is False


def test_an_item_that_never_got_a_claim_makes_the_run_incomplete():
    items = [{"id": "a", "title": "A"}, {"id": "b", "title": "B"}]
    plan = _small_plan(items=items, verify_skeptics=1)

    async def dispatch_fn(payload, needs):
        if "ITEM b" in payload:
            return L.LaneResult(ok=False, lane="local", error="fake")
        return L.LaneResult(ok=True, lane="local", served_by="local:fake",
                            result=_reply(payload), tokens=5)

    result = _run(plan, [FakeLimits("local", dispatch_fn)])
    assert result.outcomes[1]["work"]["ok"] is False
    assert result.complete is False


def test_failed_skeptic_counts_as_refutation__fail_closed():
    seen = {"skeptics": 0}

    def reply(payload):
        if "You are skeptic" in payload:
            seen["skeptics"] += 1
            return ""          # empty answer is not an abstention
        return _reply(payload)

    plan = _small_plan(verify_skeptics=1)
    result = _run(plan, [_lane("local", reply=reply)])
    assert seen["skeptics"] == 1
    assert result.outcomes[0]["verify"]["confirmed"] is False


def test_lane_failover_is_recorded_and_local_share_is_honest():
    plan = _small_plan()
    result = _run(plan, [_lane("local", fail=True), _lane("cloud")])
    assert result.complete is True
    assert result.local_share == 0.0
    assert result.lanes_used["local"]["failures"] == 3
    assert result.lanes_used["cloud"]["ok"] >= 1
    assert "error_sample" in result.lanes_used["local"]


def test_every_item_is_accounted_for():
    items = [{"id": f"i{i}", "title": f"T{i}"} for i in range(4)]
    plan = _small_plan(items=items, verify_skeptics=1)
    result = _run(plan, [_lane("local")])
    assert [o["id"] for o in result.outcomes] == ["i0", "i1", "i2", "i3"]
    assert all(o["verify"]["confirmed"] for o in result.outcomes)


def test_critic_adds_missing_work_until_dry():
    rounds = {"critic": 0}

    def reply(payload):
        if "completeness critic" in payload:
            rounds["critic"] += 1
            return ("MISSING: check the licence headers"
                    if rounds["critic"] == 1 else "DONE")
        return _reply(payload)

    plan = U.plan_workflow("audit", items=[{"id": "a", "title": "A"}],
                           scout=False, synthesize=False, verify_skeptics=1,
                           critic_rounds=2)
    result = _run(plan, [_lane("local", reply=reply)])
    assert rounds["critic"] == 2
    assert [o["id"] for o in result.outcomes] == ["a", "extra-1-1"]
    assert all(o["verify"]["confirmed"] for o in result.outcomes)


def test_critic_still_demanding_marks_run_incomplete_not_silently_closed():
    plan = U.plan_workflow("audit", items=[{"id": "a", "title": "A"}],
                           scout=False, synthesize=False, verify_skeptics=1,
                           critic_rounds=1)

    def reply(payload):
        if "completeness critic" in payload:
            return "MISSING: yet another check"
        return _reply(payload)

    result = _run(plan, [_lane("local", reply=reply)])
    assert result.complete is False
    assert any("still listing missing" in note for note in result.notes)


def test_budget_is_a_hard_ceiling():
    plan = _small_plan()
    # Each fake step reports more tokens than the plan can afford after the
    # first reservation, so the ceiling must raise rather than overspend.
    with pytest.raises(Exception) as excinfo:
        _run(plan, [_lane("local", tokens=plan.budget_tokens)])
    name = type(excinfo.value).__name__
    assert name == "BudgetExhausted"


def test_progress_events_stream_and_are_json_safe():
    events = []

    def progress(event):
        events.append(event)
        json.dumps(event)  # must be serialisable for a relay/notebook

    plan = _small_plan(scout=True)
    _run(plan, [_lane("local")], progress=progress)
    kinds = {event["type"] for event in events}
    assert {"workflow_started", "step", "item_work", "item_verified",
            "workflow_finished"} <= kinds
    assert events[-1]["type"] == "workflow_finished"


def test_run_names_the_journal_by_run_id():
    plan = _small_plan()
    result = _run(plan, [_lane("local")], run_id="run-abc")
    assert result.run_id == "run-abc"
    assert Path(result.journal_path).parts[-2:] == ("run-abc", "journal.jsonl")


def test_status_roundtrip_uses_the_state_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("ULTRACODE_STATE_DIR", str(tmp_path))
    U.write_ultra_status("r1", {"run_id": "r1", "status": "running"})
    assert U.read_ultra_status("r1")["status"] == "running"
    assert U.list_ultra_status()[0]["run_id"] == "r1"
