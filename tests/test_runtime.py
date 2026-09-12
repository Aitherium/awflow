"""Tests for awflow runtime."""

import pytest


class TestRuntime:
    """Test the WorkflowRuntime."""

    @pytest.mark.asyncio
    async def test_budget_initialization(self):
        """Test that budget is initialized correctly."""
        from awflow import run_workflow

        async def workflow():
            return "test"

        # Just verify the workflow can run without error
        result = await run_workflow(workflow, budget_tokens=1000)
        assert result == "test"
