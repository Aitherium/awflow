"""Self-test for awflow package.

Proves the package imports correctly and can execute a trivial script offline.
"""

import asyncio
import sys


async def self_test():
    """Minimal workflow for offline testing (no LLM calls needed)."""
    # Import here to catch any import-time errors
    try:
        from . import agent, get_budget, log, parallel, phase, pipeline  # noqa: F401 - injected into the script's namespace
    except ImportError as e:
        print(f"FAIL: Import error: {e}")
        return False

    try:
        # Test phase marker
        await phase("Test Phase")

        # Test log
        await log("Test log message")

        # Test get_budget
        budget = get_budget()
        if budget.total <= 0:
            print(f"FAIL: Budget total is {budget.total}")
            return False

        # Test parallel (with no-op coroutines)
        async def noop1():
            await log("Noop 1")
            return "result1"

        async def noop2():
            await log("Noop 2")
            return "result2"

        results = await parallel([noop1(), noop2()])
        if results != ["result1", "result2"]:
            print(f"FAIL: Parallel results incorrect: {results}")
            return False

        # Test pipeline (with no-op stages)
        async def stage1(context, stage_idx):
            prev_result, item, index = context
            await log(f"Stage 1: {item}")
            return f"stage1_{item}"

        async def stage2(context, stage_idx):
            prev_result, item, index = context
            await log(f"Stage 2: {prev_result}")
            return f"stage2_{prev_result}"

        items = ["a", "b"]
        pipeline_results = await pipeline(items, stage1, stage2)
        if len(pipeline_results) != 2:
            print(f"FAIL: Pipeline results incorrect: {pipeline_results}")
            return False

        spent = budget.spent()
        remaining = budget.remaining()
        if spent < 0 or remaining < 0:
            print(f"FAIL: Budget tracking broken: spent={spent}, remaining={remaining}")
            return False

        return True

    except Exception as e:
        print(f"FAIL: Test execution error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_test_workflow():
    """Run self-test via run_workflow()."""
    try:
        from . import run_workflow
    except ImportError as e:
        print(f"FAIL: Cannot import run_workflow: {e}")
        return False

    try:
        result = await run_workflow(
            self_test,
            journal_path=None,  # Use default
            budget_tokens=10000,
        )
        return result
    except Exception as e:
        print(f"FAIL: run_workflow error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for --self-test."""
    try:
        success = asyncio.run(run_test_workflow())
        if success:
            print("\nPASS: awflow package is functional")
            return 0
        else:
            print("\nFAIL: awflow self-test failed")
            return 1
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"\nFAIL: Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        sys.exit(main())
    else:
        print("Usage: python -m AitherOS.lib.orchestration.awflow --self-test")
        sys.exit(1)
