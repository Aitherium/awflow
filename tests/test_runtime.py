"""Tests for awflow runtime.

Plain synchronous tests driving the loop with ``asyncio.run`` on purpose: the
brick publish lane installs ``build twine pytest .`` and nothing else, so a
``@pytest.mark.asyncio`` test is collected and FAILS there ("async def functions
are not natively supported") -- measured 2026-09-13 on the first
aitherium-awflow publish run. No plugin, no marker, no skip.
"""

import asyncio


class TestRuntime:
    """Test the WorkflowRuntime."""

    def test_budget_initialization(self):
        """A trivial workflow runs under a token budget and returns its value."""
        from awflow import run_workflow

        async def workflow():
            return "test"

        result = asyncio.run(run_workflow(workflow, budget_tokens=1000))
        assert result == "test"
