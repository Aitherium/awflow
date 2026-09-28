"""`awflow --help` prints usage and never runs the self-test workflow."""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_DIR = str(Path(__file__).resolve().parents[1])
if sys.path[:1] != [_PKG_DIR]:
    sys.path.insert(0, _PKG_DIR)

import pytest  # noqa: E402

from awflow import __main__ as cli  # noqa: E402


@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_help_exits_zero_without_running_workflow(flag, capsys, monkeypatch):
    def _boom():  # the self-test must not start
        raise AssertionError("--help ran the self-test")

    monkeypatch.setattr(cli, "run_test_workflow", _boom)
    assert cli.main([flag]) == 0
    assert "--self-test" in capsys.readouterr().out


def test_unknown_argument_exits_two(capsys, monkeypatch):
    monkeypatch.setattr(cli, "run_test_workflow", lambda: (_ for _ in ()).throw(AssertionError("ran")))
    assert cli.main(["--bogus"]) == 2
    assert "unrecognized" in capsys.readouterr().err
