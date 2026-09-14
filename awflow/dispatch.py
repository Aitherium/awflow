"""
Single agent call dispatch through MicroScheduler at https://localhost:8150.

Provides:
- dispatch_agent_call: async function for one LLM agent call with schema validation
- parse_response_format: convert JSON Schema to OpenAI response_format
- validate_against_schema: validate response against schema

Handles:
- Request formatting (prompt, model, schema → response_format, temperature, etc.)
- MicroScheduler HTTPS round-trip with internal CA trust
- Response parsing (content + usage tokens)
- Validation with jsonschema (or fallback structural check)
- Bounded retry on parse/validation failure with repair instructions
- Token extraction (real or None, never 0)
- Error classification (distinguished error types for caller to handle)

Self-test: offline validation + optional live probe if MicroScheduler is reachable.
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

#: Standalone fallbacks for the two monorepo helpers this module needs.
#:
#: AWFL002 fires on the shipped path: `from lib.core...` is a HARD ImportError
#: for anyone who `pip install awflow`, because `lib/` is not on their path --
#: not a cosmetic leak, and the whole cloud-dispatch path would be dead for
#: every installer while every check inside the monorepo reads green. The fleet
#: keeps the real helpers (the try wins whenever `lib` is importable, so
#: in-fleet behaviour is unchanged); everyone else gets a faithful, smaller
#: implementation of the same contract.
try:                                    # fleet: the shared resolver + client
    from lib.core.AitherHttp import AsyncClient  # noqa: F401
    from lib.core.AitherPorts import get_service_url  # noqa: F401
except ImportError:                     # pip-installed brick: resolve it here
    import httpx  # declared dependency of this package

    def get_service_url(name: str) -> str:
        """Env-first resolution, the same contract the fleet resolver honours.

        A brick that cannot answer this for itself is not installable, and one
        that GUESSES a URL is worse: it fails at the network layer with a name
        nobody can act on. So an unset variable is a refusal that names the
        variable to set, never a default endpoint.
        """
        key = "AITHER_" + name.upper() + "_URL"
        url = os.environ.get(key)
        if not url:
            raise RuntimeError(
                f"{key} is not set, and this is not an AitherOS checkout (the "
                f"fleet resolver lives in lib.core.AitherPorts). Point {key} at "
                f"your {name} endpoint, e.g. {key}=https://api.aitherium.com"
            )
        return url

    class AsyncClient:                  # noqa: D101 - mirrors the fleet class
        """The one method this module uses, over the declared httpx dependency.

        TLS is VERIFIED here (httpx default). The fleet client disables
        verification because it speaks to in-network services over a private
        CA; an off-fleet caller reaches a public endpoint and must not inherit
        that exception.
        """

        async def post(self, url, **kwargs):
            async with httpx.AsyncClient() as client:
                return await client.post(url, **kwargs)

try:
    import jsonschema
except ImportError:
    jsonschema = None

logger = logging.getLogger(__name__)

# Ensure we're using the right level of logging
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


@dataclass
class DispatchError:
    """Distinguished error types from agent dispatch.

    Allows callers to distinguish between:
    - MODEL_NOT_FOUND: model name not recognized by MicroScheduler
    - TIMEOUT: request timed out
    - NETWORK_ERROR: connection/transport error
    - NON_200_ERROR: HTTP error response (status != 200)
    - EMPTY_CONTENT_ERROR: response has no choices/content
    - PARSE_ERROR: response content is not valid JSON (when schema given)
    - VALIDATION_ERROR: parsed JSON fails schema validation
    """
    code: str
    message: str
    retriable: bool = False


def parse_response_format(schema: Optional[dict]) -> Optional[dict]:
    """Convert a JSON Schema to OpenAI response_format.

    Args:
        schema: JSON Schema dict or None

    Returns:
        None if schema is None
        OpenAI json_schema format wrapper if schema is provided

    Raises:
        ValueError: if schema is not a valid dict
    """
    if schema is None:
        return None

    if not isinstance(schema, dict):
        raise ValueError(f"schema must be a dict, got {type(schema)}")

    # If already in OpenAI json_schema format, return as-is
    if schema.get("type") == "json_schema":
        return schema

    # Wrap in OpenAI json_schema format
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.get("title", "response"),
            "schema": schema,
            "strict": True
        }
    }


def validate_against_schema(data: Any, schema: dict) -> tuple[bool, Optional[str]]:
    """Validate data against a JSON Schema.

    Args:
        data: data to validate
        schema: JSON Schema dict or None

    Returns:
        Tuple of (valid: bool, error_message: str or None)
        - (True, None) if valid or schema is None
        - (False, error_str) if invalid

    Uses jsonschema if available, falls back to minimal structural check.
    """
    if schema is None:
        return True, None

    if jsonschema:
        try:
            jsonschema.validate(data, schema)
            return True, None
        except jsonschema.ValidationError as e:
            return False, str(e)
        except jsonschema.SchemaError as e:
            logger.warning(f"Invalid schema: {e}")
            return False, f"Schema error: {str(e)}"
        except Exception as e:
            logger.warning(f"jsonschema validation error: {e}")
            return False, str(e)

    # Fallback: minimal structural check (check required fields only)
    if isinstance(schema, dict) and isinstance(data, dict):
        required = schema.get("required", [])
        if required and isinstance(required, list):
            missing = [k for k in required if k not in data]
            if missing:
                return False, f"Missing required fields: {missing}"

    return True, None


async def dispatch_agent_call(
    prompt: str,
    *,
    model: Optional[str] = None,
    schema: Optional[dict] = None,
    temperature: float = 0.7,
    seed: Optional[int] = None,
    max_tokens: int = 2048,
    effort: Optional[int] = None,
) -> tuple[Optional[str | dict], Optional[int], Optional[DispatchError]]:
    """Call MicroScheduler with one agent prompt.

    Args:
        prompt: the prompt text to send to the LLM
        model: model name (default: qwen3.6-27b if not specified)
        schema: JSON Schema for constrained output (optional)
        temperature: sampling temperature (default: 0.7)
        seed: random seed for determinism (optional, included in hash if set)
        max_tokens: maximum tokens in response (default: 2048)
        effort: effort tier for routing (optional, passed through to MicroScheduler)

    Returns:
        Tuple of (result, tokens_used, error):
        - result: str (no schema) or validated dict (schema given), or None on final failure
        - tokens_used: int from response usage.total_tokens, or None if absent (never 0)
        - error: DispatchError on any failure, None on success

    Behavior:
        - If schema is given and response is not valid JSON or fails validation,
          retries up to 3 times with a repair instruction appended to prompt
        - Returns None result on final failure (after retries exhausted)
        - Each retry is journaled separately (if caller journals)
        - Never raises; all errors are returned in the error field

    Error types (from DispatchError.code):
        - MODEL_NOT_FOUND: model name not recognized
        - TIMEOUT: request timed out after 120s
        - NETWORK_ERROR: connection/transport error
        - NON_200_ERROR: HTTP status != 200
        - EMPTY_CONTENT_ERROR: no content in response
        - PARSE_ERROR: invalid JSON (when schema given)
        - VALIDATION_ERROR: JSON valid but fails schema validation
    """
    # AsyncClient / get_service_url come from the module-level block above:
    # the fleet helpers when `lib` is importable, the standalone pair otherwise.
    # Resolve MicroScheduler endpoint
    try:
        microscheduler_base = get_service_url("MicroScheduler")
    except Exception as e:
        return None, None, DispatchError(
            code="NETWORK_ERROR",
            message=f"Could not resolve MicroScheduler: {str(e)[:100]}",
            retriable=False
        )

    # Ensure HTTPS (MicroScheduler must use TLS)
    if microscheduler_base.startswith("http://"):
        microscheduler_base = microscheduler_base.replace("http://", "https://")
    elif not microscheduler_base.startswith("https://"):
        microscheduler_base = f"https://{microscheduler_base}"

    url = f"{microscheduler_base}/v1/chat/completions"

    # Build initial request body
    request_body = {
        "model": model or "qwen3.6-27b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    if seed is not None:
        request_body["seed"] = seed

    if max_tokens:
        request_body["max_tokens"] = max_tokens

    if schema:
        try:
            # 🚨 MEASURED 2026-09-05 on the live fleet: `response_format`
            # (json_schema) is FORWARDED by MicroScheduler and then DISCARDED
            # by the backend. Two different backends returned HTTP 200 with
            # plain prose for a strict json_schema request -- the silent-no-op
            # class: a caller that asked for an object receives confident text,
            # with every signal green.
            #
            # A synthetic tool plus `tool_choice="required"` IS honoured by
            # those same backends, first try, schema-conformant. The router
            # already forwards both fields (AitherMicroScheduler.py:6530,
            # :10171). So structure is forced through the TOOL door and
            # `response_format` rides along only as a hint for any backend that
            # does honour it. Local validation still runs regardless -- see the
            # retry loop below; neither door is trusted on its own.
            request_body["response_format"] = parse_response_format(schema)
            request_body["tools"] = [{
                "type": "function",
                "function": {
                    "name": "emit_result",
                    "description": "Emit the result as structured data.",
                    "parameters": schema,
                },
            }]
            request_body["tool_choice"] = "required"
        except ValueError as e:
            return None, None, DispatchError(
                code="VALIDATION_ERROR",
                message=f"Invalid schema: {str(e)[:100]}",
                retriable=False
            )

    if effort is not None:
        request_body["effort"] = effort

    # Retry loop: up to 3 retries on JSON parse or validation failure
    max_retries = 3
    attempt = 0
    client = AsyncClient()

    while attempt <= max_retries:
        try:
            # Call MicroScheduler
            response = await client.post(url, json=request_body, timeout=120)

            # Check HTTP status
            if response.status_code != 200:
                return None, None, DispatchError(
                    code="NON_200_ERROR",
                    message=f"MicroScheduler returned {response.status_code}: {response.text[:200]}",
                    retriable=response.status_code >= 500
                )

            # Parse response JSON
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                return None, None, DispatchError(
                    code="PARSE_ERROR",
                    message=f"Response is not valid JSON: {str(e)[:100]}",
                    retriable=False
                )

            # Extract response content
            if not data.get("choices"):
                return None, None, DispatchError(
                    code="EMPTY_CONTENT_ERROR",
                    message="Response has no choices",
                    retriable=False
                )

            _msg = data["choices"][0].get("message", {}) or {}

            # A schema call is forced through a TOOL, so the answer arrives in
            # tool_calls[0].function.arguments, NOT in content -- reading only
            # content would see an empty string and report EMPTY_CONTENT_ERROR
            # on a perfectly good structured reply.
            content = ""
            _tool_calls = _msg.get("tool_calls") or []
            if _tool_calls:
                _fn = (_tool_calls[0] or {}).get("function", {}) or {}
                content = (_fn.get("arguments") or "").strip()

            if not content:
                content = (_msg.get("content") or "").strip()

            # A REASONING model leaves content null while it thinks and puts the
            # chain of thought in `reasoning`/`reasoning_content`. That
            # is thinking, NOT an answer: using it as the result renders the
            # model's monologue as its reply. Report the empty answer honestly
            # and let the caller retry.
            if not content:
                _reasoning = _msg.get("reasoning") or _msg.get("reasoning_content")
                if _reasoning:
                    return None, None, DispatchError(
                        code="REASONING_ONLY_ERROR",
                        message=(
                            "Model returned reasoning but no answer "
                            f"({len(str(_reasoning))} chars of chain-of-thought)"
                        ),
                        retriable=True,
                    )
                return None, None, DispatchError(
                    code="EMPTY_CONTENT_ERROR",
                    message="Response content is empty",
                    retriable=True
                )

            # Extract token count (real value or None, never 0)
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens")
            if tokens_used == 0:
                tokens_used = None

            # If no schema, return text response as-is
            if schema is None:
                return content, tokens_used, None

            # Schema was given: parse JSON and validate
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    # Retry with repair instruction
                    repair = f"\n\nPrevious attempt returned invalid JSON: {str(e)[:50]}. Return ONLY valid JSON matching the schema."
                    request_body["messages"][0]["content"] = prompt + repair
                    attempt += 1
                    continue

                return None, tokens_used, DispatchError(
                    code="PARSE_ERROR",
                    message=f"Invalid JSON after {max_retries} retries: {str(e)[:100]}",
                    retriable=False
                )

            # Validate against schema
            valid, validation_error = validate_against_schema(parsed, schema)
            if not valid:
                if attempt < max_retries:
                    # Retry with repair instruction
                    repair = f"\n\nPrevious JSON did not match schema: {validation_error[:50]}. Return ONLY valid JSON matching the schema."
                    request_body["messages"][0]["content"] = prompt + repair
                    attempt += 1
                    continue

                return None, tokens_used, DispatchError(
                    code="VALIDATION_ERROR",
                    message=f"Validation failed after {max_retries} retries: {validation_error[:200]}",
                    retriable=False
                )

            # Success
            return parsed, tokens_used, None

        except asyncio.TimeoutError:
            return None, None, DispatchError(
                code="TIMEOUT",
                message="MicroScheduler request timed out",
                retriable=True
            )

        except Exception as e:
            error_str = str(e).lower()
            if "not found" in error_str or "model" in error_str or "unknown model" in error_str:
                return None, None, DispatchError(
                    code="MODEL_NOT_FOUND",
                    message=f"Model not found or unknown: {str(e)[:100]}",
                    retriable=False
                )

            return None, None, DispatchError(
                code="NETWORK_ERROR",
                message=f"Network error: {str(e)[:100]}",
                retriable=True
            )

    # Should not reach here, but guard against logic error
    return None, None, DispatchError(
        code="VALIDATION_ERROR",
        message="Failed to get valid response after retries",
        retriable=False
    )


# Self-test
if __name__ == "__main__":
    import sys

    def test_utility_functions():
        """Test parse_response_format and validate_against_schema offline."""
        logger.info("Testing utility functions...")

        # Test 1: parse_response_format with None
        result = parse_response_format(None)
        assert result is None, "parse_response_format(None) should return None"
        logger.info("✓ parse_response_format(None) returns None")

        # Test 2: parse_response_format with schema
        schema = {"type": "object", "title": "Test"}
        result = parse_response_format(schema)
        assert result["type"] == "json_schema", "Should wrap in json_schema"
        assert result["json_schema"]["schema"] == schema, "Should preserve schema"
        logger.info("✓ parse_response_format wraps schema correctly")

        # Test 3: validate_against_schema with None
        valid, err = validate_against_schema({"key": "value"}, None)
        assert valid and err is None, "Should validate when schema is None"
        logger.info("✓ validate_against_schema(data, None) returns (True, None)")

        # Test 4: validate_against_schema with required fields
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"]
        }
        valid, err = validate_against_schema({"name": "test"}, schema)
        assert valid and err is None, "Should validate data with required field"
        logger.info("✓ validate_against_schema accepts valid data")

        # Test 5: validate_against_schema with missing required field
        valid, err = validate_against_schema({}, schema)
        assert not valid, "Should reject data missing required field"
        assert "name" in str(err).lower() or "required" in str(err).lower(), "Error should mention the missing field"
        logger.info(f"✓ validate_against_schema rejects invalid data: {err[:50]}")

        logger.info("\nAll utility tests passed.\n")
        return 0

    async def test_live_call():
        """Try one live call if MicroScheduler is reachable."""
        logger.info("Attempting live test (MicroScheduler must be reachable)...")

        try:
            ms_url = get_service_url("MicroScheduler")
            logger.info(f"MicroScheduler endpoint resolved: {ms_url}")

            result, tokens, error = await dispatch_agent_call("Say the word 'awflow' and nothing else")

            if error:
                logger.info(f"Live test returned error: {error.code}: {error.message}")
                return 1

            logger.info("Live test succeeded:")
            logger.info(f"  Result: {str(result)[:80]}...")
            logger.info(f"  Tokens: {tokens}")
            return 0

        except Exception as e:
            logger.info(f"Live test skipped (MicroScheduler not reachable): {e}")
            return None

    async def main():
        """Run self-tests: offline required, live optional."""
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

        # Run offline tests (required)
        code = test_utility_functions()
        if code != 0:
            return code

        # The LIVE arm is OPT-IN (--live), not opportunistic.
        #
        # It used to run whenever the fleet happened to be reachable, and that
        # made the self-test FLAKY: gate AWF006 shells this with a 30s deadline,
        # and a live call under fleet load blows it. Measured 2026-09-05 -- the
        # same command exited 0 in a quiet minute and ERRORed on a timeout while
        # another workflow was running, so the gate's verdict tracked GPU
        # contention rather than the code. A flaky gate teaches people to re-run
        # until green, which trains away the signal (the lesson gate 1k is built
        # on). The offline arms are the contract; the live arm is a probe.
        if "--live" in sys.argv:
            live_code = await test_live_call()
            if live_code is not None:
                return live_code
        else:
            print("live arm SKIPPED (pass --live to run it against the fleet)")

        return 0

    sys.exit(asyncio.run(main()))
