"""A resumed run replays what it already did instead of doing it again.

The journal is the checkpoint. These tests kill a workflow partway, start a NEW
runtime on the same run id, and count the calls that reach the dispatcher -- the
only number that says whether resume is real. `mirror=False` keeps them offline.
"""

import asyncio

import pytest
from awflow import agent, run_workflow

STEPS = ["alpha", "beta", "gamma", "delta"]


class CrashError(RuntimeError):
    pass


def _counting_dispatcher(calls):
    async def dispatch(prompt, *, model=None, schema=None, temperature=0.7, seed=None,
                       max_tokens=2048, effort=None):
        calls.append(prompt)
        return f"answer:{prompt}", 7, None
    return dispatch


def _flow(die_after=None):
    async def flow():
        out = []
        for index, step in enumerate(STEPS):
            if die_after is not None and index == die_after:
                raise CrashError(f"killed before step {index}")
            out.append(await agent(f"do {step}", label=step))
        return out
    return flow


def _run(flow, tmp_path, calls, run_id):
    return asyncio.run(run_workflow(flow, journal_path=tmp_path, resume_from=run_id,
                                    mirror=False, dispatcher=_counting_dispatcher(calls)))


def test_resume_replays_finished_calls_and_only_pays_for_the_rest(tmp_path):
    first, second = [], []
    with pytest.raises(CrashError):
        _run(_flow(die_after=2), tmp_path, first, "run-a")
    assert first == ["do alpha", "do beta"]

    result = _run(_flow(), tmp_path, second, "run-a")
    assert second == ["do gamma", "do delta"], "finished calls were made a second time"
    assert result == [f"answer:do {s}" for s in STEPS]


def test_a_different_run_id_replays_nothing(tmp_path):
    first, second = [], []
    with pytest.raises(CrashError):
        _run(_flow(die_after=2), tmp_path, first, "run-a")
    _run(_flow(), tmp_path, second, "run-b")
    assert len(second) == len(STEPS)


def test_a_call_that_failed_is_retried_live_on_resume(tmp_path):
    attempts = []

    async def flaky(prompt, *, model=None, schema=None, temperature=0.7, seed=None,
                    max_tokens=2048, effort=None):
        attempts.append(prompt)
        if len(attempts) <= 3:          # the runtime's three tries, all failing
            return None, None, "backend down"
        return "recovered", 7, None

    async def flow():
        return await agent("do alpha", label="alpha")

    def run():
        return asyncio.run(run_workflow(flow, journal_path=tmp_path, resume_from="run-f",
                                        mirror=False, dispatcher=flaky))

    assert run() is None
    assert run() == "recovered", "a journaled FAILURE was replayed as the answer"


def test_a_queued_run_tells_its_expedition_why_it_exists(monkeypatch):
    from awflow import runtime
    for key in ("AWRUN_RUN_ID", "AWRUN_LINEAGE_GOAL", "AWRUN_LINEAGE_INTENT"):
        monkeypatch.delenv(key, raising=False)
    assert runtime._mirror_description() == "awflow execution"
    monkeypatch.setenv("AWRUN_RUN_ID", "r-2345abcd")
    monkeypatch.setenv("AWRUN_LINEAGE_GOAL", "G-42")
    monkeypatch.setenv("AWRUN_LINEAGE_INTENT", "keep-the-index-fresh")
    assert runtime._mirror_description() == \
        "awflow execution [run=r-2345abcd goal=G-42 intent=keep-the-index-fresh]"
