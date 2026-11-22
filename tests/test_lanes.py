"""Tests for awflow.lanes — per-step lane selection, failover, journaling.

Plain synchronous tests driving the async API with ``asyncio.run`` on purpose:
the brick publish lane installs ``pytest`` and nothing else, so a
``pytest.mark.asyncio`` test fails collection there (same reason as
``test_runtime.py``). NO live call: every lane transport is a fake, so these
tests prove the selection matrix and the failover path without spending.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from awflow import lanes

L = lanes  # short local name; a plain assignment, not an import alias


# --------------------------------------------------------------------------
# fake lanes
# --------------------------------------------------------------------------


class _Journal:
    def __init__(self):
        self.records = []

    async def write_record(self, record):
        self.records.append(record)


def _lane(name, *, kind_ok=True, fail=False, tokens=10, delay=0.0,
          max_concurrency=4, serves_llm=True, max_context_chars=None):
    """Build a real Lane of the right class with a FAKE transport."""
    async def dispatch_fn(payload, needs):
        if delay:
            await asyncio.sleep(delay)
        if fail:
            return L.LaneResult(ok=False, lane=name, error="fake failure")
        return L.LaneResult(ok=True, lane=name, served_by=f"{name}:fake",
                            result=f"{name}:{payload[:20]}", tokens=tokens)

    if name == "local":
        lane = L.LocalLane(dispatch_fn=dispatch_fn)
        if max_context_chars is not None:
            lane.max_context_chars = max_context_chars
    elif name == "cloud":
        lane = L.CloudLane(dispatch_fn=dispatch_fn)
    elif name == "runner":
        lane = L.RunnerLane(dispatch_fn=dispatch_fn)
    else:  # pragma: no cover
        raise AssertionError(name)
    lane.max_concurrency = max_concurrency
    return lane


def _ladder(**overrides):
    """local + cloud + runner, each overridable by keyword (fail=True etc.)."""
    def make(name, **kw):
        opts = dict(overrides.get(name, {}))
        opts.update(kw)
        return _lane(name, **opts)
    return [make("local"), make("cloud"), make("runner")]


# --------------------------------------------------------------------------
# selection matrix
# --------------------------------------------------------------------------


class TestSelection:
    def test_llm_small_prefers_free_local_then_cloud(self):
        needs = L.StepNeeds(kind="llm", input_chars=1000)
        assert [n for n, _ in L.plan(needs, _ladder())] == ["local", "cloud"]

    def test_llm_too_large_for_local_falls_to_cloud(self):
        needs = L.StepNeeds(kind="llm", input_chars=L.DEFAULT_LOCAL_MAX_INPUT_CHARS + 1)
        assert [n for n, _ in L.plan(needs, _ladder())] == ["cloud"]

    def test_llm_require_local_never_touches_cloud(self):
        needs = L.StepNeeds(kind="llm", require_local=True)
        assert [n for n, _ in L.plan(needs, _ladder())] == ["local"]

    def test_llm_no_cloud_and_too_large_has_no_lane(self):
        needs = L.StepNeeds(kind="llm", input_chars=10_000_000, allow_cloud=False)
        assert L.plan(needs, _ladder()) == []

    def test_compute_public_safe_prefers_free_runner_then_local(self):
        needs = L.StepNeeds(kind="compute", public_safe=True)
        assert [n for n, _ in L.plan(needs, _ladder())] == ["runner", "local"]

    def test_compute_private_never_uses_public_runner(self):
        needs = L.StepNeeds(kind="compute", public_safe=False)
        assert [n for n, _ in L.plan(needs, _ladder())] == ["local"]

    def test_compute_require_local_is_local_only(self):
        needs = L.StepNeeds(kind="compute", public_safe=True, require_local=True)
        assert [n for n, _ in L.plan(needs, _ladder())] == ["local"]

    def test_compute_never_uses_cloud_llm_lane(self):
        needs = L.StepNeeds(kind="compute", public_safe=False)
        assert "cloud" not in [n for n, _ in L.plan(needs, _ladder())]


class TestRunnerCredentialVerdict:
    def test_runner_refuses_an_llm_step_and_says_why(self):
        lane = _lane("runner")
        ok, reason = lane.eligible(L.StepNeeds(kind="llm", public_safe=True))
        assert not ok
        assert "NO NEW CI SECRETS" in reason
        assert "non-llm compute only" in reason

    def test_runner_refuses_a_step_not_declared_public_safe(self):
        lane = _lane("runner")
        ok, reason = lane.eligible(L.StepNeeds(kind="compute", public_safe=False))
        assert not ok and "public-safe" in reason

    def test_a_step_must_declare_public_safe_explicitly(self):
        assert L.StepNeeds(kind="compute").public_safe is False


# --------------------------------------------------------------------------
# failover + cost + journal
# --------------------------------------------------------------------------


class TestDispatch:
    def test_failover_local_fails_then_cloud_serves(self):
        disp = L.LaneDispatcher(lanes=_ladder(local={"fail": True}), journal=_Journal(),
                                run_id="r1")
        res = asyncio.run(disp.dispatch("hello", L.StepNeeds(kind="llm", input_chars=100)))
        assert res.ok and res.lane == "cloud" and res.served_by == "cloud:fake"

    def test_all_lanes_fail_reports_every_reason(self):
        disp = L.LaneDispatcher(
            lanes=_ladder(local={"fail": True}, cloud={"fail": True}),
            journal=_Journal(), run_id="r1")
        res = asyncio.run(disp.dispatch("x", L.StepNeeds(kind="llm")))
        assert not res.ok and res.lane == "none"
        assert "local=fake failure" in res.error and "cloud=fake failure" in res.error

    def test_no_lane_raises_with_per_lane_refusals(self):
        disp = L.LaneDispatcher(lanes=_ladder(), journal=_Journal(), run_id="r1")
        needs = L.StepNeeds(kind="llm", input_chars=10_000_000, allow_cloud=False)
        try:
            asyncio.run(disp.dispatch("x", needs))
        except L.NoLaneError as exc:
            assert "local" in exc.refusals and "cloud" in exc.refusals
        else:  # pragma: no cover
            raise AssertionError("expected NoLaneError")

    def test_cloud_cost_is_priced_and_free_lanes_are_zero(self):
        disp = L.LaneDispatcher(lanes=_ladder(), journal=_Journal(), run_id="r1")
        res = asyncio.run(disp.dispatch(
            "x", L.StepNeeds(kind="llm", input_chars=10_000_000,
                             allow_cloud=True)))
        assert res.lane == "cloud"
        assert res.cost_usd > 0 and res.tokens == 10

        disp2 = L.LaneDispatcher(lanes=_ladder(), journal=_Journal(), run_id="r1")
        local = asyncio.run(disp2.dispatch("x", L.StepNeeds(kind="llm", input_chars=10)))
        assert local.lane == "local" and local.cost_usd == 0.0

    def test_journal_records_served_by_latency_tokens_cost(self):
        j = _Journal()
        disp = L.LaneDispatcher(lanes=_ladder(local={"fail": True}), journal=j, run_id="r9")
        asyncio.run(disp.dispatch("hi", L.StepNeeds(kind="llm", input_chars=100),
                                  label="step-a"))
        recs = [r for r in j.records if r["type"] == "LANE_DISPATCH"]
        assert len(recs) == 2  # local attempt (failed) + cloud attempt (served)
        assert recs[0]["ok"] is False and recs[0]["lane"] == "local"
        assert recs[1]["ok"] is True and recs[1]["served_by"] == "cloud:fake"
        for r in recs:
            assert r["run_id"] == "r9" and r["label"] == "step-a"
            assert "latency_ms" in r and "cost_usd" in r and "tokens" in r
            assert r["lane_order"] == ["local", "cloud"]
            assert r["public_safe"] is False

    def test_journal_write_failure_does_not_lose_the_step(self):
        class Boom:
            async def write_record(self, record):
                raise OSError("disk full")

        disp = L.LaneDispatcher(lanes=_ladder(), journal=Boom(), run_id="r")
        res = asyncio.run(disp.dispatch("x", L.StepNeeds(kind="llm", input_chars=10)))
        assert res.ok and res.lane == "local"


# --------------------------------------------------------------------------
# swarm fan-out
# --------------------------------------------------------------------------


class TestFanOut:
    def test_fan_out_routes_each_item_independently_and_keeps_order(self):
        disp = L.LaneDispatcher(lanes=_ladder(), journal=_Journal(), run_id="r")
        items = [
            ("small", L.StepNeeds(kind="llm", input_chars=100)),
            ("big", L.StepNeeds(kind="llm", input_chars=10_000_000)),
            ("pub-compute", L.StepNeeds(kind="compute", public_safe=True)),
        ]
        res = asyncio.run(disp.fan_out(
            items, needs_fn=lambda it: it[1], payload_fn=lambda it: it[0]))
        assert [r.lane for r in res] == ["local", "cloud", "runner"]
        assert [r.result for r in res] == [
            "local:small", "cloud:big", "runner:pub-compute"]

    def test_fan_out_caps_concurrency_per_lane(self):
        """Two local steps never run at once (the pool has one agent slot)."""
        active = {"n": 0, "peak": 0}

        async def slow(payload, needs):
            active["n"] += 1
            active["peak"] = max(active["peak"], active["n"])
            await asyncio.sleep(0.02)
            active["n"] -= 1
            return L.LaneResult(ok=True, lane="local", served_by="local:fake",
                                result=payload, tokens=1)

        local = L.LocalLane(dispatch_fn=slow)
        local.max_concurrency = 1
        disp = L.LaneDispatcher(lanes=[local], journal=_Journal(), run_id="r")
        needs = L.StepNeeds(kind="llm", input_chars=10)
        asyncio.run(disp.fan_out(
            ["a", "b", "c"], needs_fn=lambda _: needs, payload_fn=lambda p: p))
        assert active["peak"] == 1


# --------------------------------------------------------------------------
# runtime integration
# --------------------------------------------------------------------------


class TestRuntimeIntegration:
    def test_step_inside_run_workflow_journals_lane_dispatch(self):
        from awflow import run_workflow

        async def workflow():
            return await runtime_step_call()

        with tempfile.TemporaryDirectory() as tmp:
            res = asyncio.run(run_workflow(
                workflow, journal_path=tmp, budget_tokens=1000,
                mirror=False, lanes=_ladder()))
            assert res["lane"] == "local"
            journal_files = list(Path(tmp).rglob("journal.jsonl"))
            assert len(journal_files) == 1
            records = [json.loads(line) for line in
                       journal_files[0].read_text().splitlines() if line.strip()]
            lane_recs = [r for r in records if r.get("type") == "LANE_DISPATCH"]
            assert len(lane_recs) == 1
            assert lane_recs[0]["served_by"] == "local:fake"
            assert lane_recs[0]["ok"] is True


async def runtime_step_call():
    from awflow import step
    res = await step("payload", {"kind": "llm", "input_chars": 10}, label="s1")
    return {"lane": res.lane, "ok": res.ok, "served_by": res.served_by}


# --------------------------------------------------------------------------
# declaration validation
# --------------------------------------------------------------------------


class TestRunnerTransport:
    """The GitHub-Actions dispatch edge is argv-built and never shelled."""

    def test_argv_is_exact_and_payload_is_one_item(self):
        argv = L.github_actions_argv(
            "Aitherium/awrun", "swarm.yml", "main", {"tasks_json": '{"a":1}'})
        assert argv == [
            "gh", "workflow", "run", "swarm.yml",
            "--repo", "Aitherium/awrun", "--ref", "main",
            "-f", 'tasks_json={"a":1}',
        ]

    def test_argv_refuses_a_repo_that_is_not_owner_name(self):
        try:
            L.github_actions_argv("awrun", "w.yml", "main", {})
        except L.NeedsError as exc:
            assert "owner/name" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected NeedsError")

    def test_transport_triggers_and_returns_run_pointer(self):
        calls = []

        def fake_runner(argv):
            calls.append(argv)
            return "https://github.com/Aitherium/awrun/actions/runs/123"

        lane = L.RunnerLane(dispatch_fn=L.github_actions_transport(
            "Aitherium/awrun", "swarm.yml", runner=fake_runner))
        res = asyncio.run(lane.dispatch(
            "do the thing", L.StepNeeds(kind="compute", public_safe=True)))
        assert res.ok and res.lane == "runner"
        assert res.served_by == "github:Aitherium/awrun/swarm.yml"
        assert "runs/123" in res.result
        assert calls and calls[0][:4] == ["gh", "workflow", "run", "swarm.yml"]

    def test_transport_refuses_an_oversized_payload(self):
        lane = L.RunnerLane(dispatch_fn=L.github_actions_transport(
            "Aitherium/awrun", "swarm.yml", runner=lambda argv: "x"))
        res = asyncio.run(lane.dispatch(
            "x" * (L.RUNNER_MAX_INLINE_CHARS + 1),
            L.StepNeeds(kind="compute", public_safe=True)))
        assert not res.ok and "too large" in res.error

    def test_transport_failure_fails_over_not_crashes(self):
        def boom(argv):
            raise RuntimeError("gh: not authenticated")

        disp = L.LaneDispatcher(
            lanes=[L.RunnerLane(dispatch_fn=L.github_actions_transport(
                       "Aitherium/awrun", "swarm.yml", runner=boom)),
                   _lane("local")],
            journal=_Journal(), run_id="r")
        res = asyncio.run(disp.dispatch(
            "x", L.StepNeeds(kind="compute", public_safe=True)))
        assert res.ok and res.lane == "local"


class TestCloudReply:
    def test_anthropic_reply_parses_text_and_tokens(self):
        data = {
            "content": [
                {"type": "thinking", "thinking": "..."},
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }
        text, tokens = L.parse_anthropic_reply(data)
        assert text == "hello world"
        assert tokens == 15

    def test_reasoning_only_reply_is_empty_text_so_it_fails_over(self):
        data = {"content": [{"type": "thinking", "thinking": "hmm"}],
                "usage": {"input_tokens": 5, "output_tokens": 9}}
        text, tokens = L.parse_anthropic_reply(data)
        assert text == "" and tokens == 14


class TestStepNeeds:
    def test_bad_kind_is_refused(self):
        try:
            L.StepNeeds(kind="gpu")
        except L.NeedsError as exc:
            assert "kind" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected NeedsError")

    def test_negative_input_chars_is_refused(self):
        try:
            L.StepNeeds(kind="llm", input_chars=-1)
        except L.NeedsError as exc:
            assert "input_chars" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected NeedsError")
