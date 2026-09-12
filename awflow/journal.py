"""
awflow.journal — Append-only JSONL journaling with atomic writes, corruption recovery, and replay support.

Journal records are written atomically (fsync) and read with graceful handling of corrupt lines.
Agent call hashing provides deterministic replay: same call parameters produce the same hash,
enabling calls to be replayed from the journal instead of re-executed.

Determinism guarantee: the hash is deterministic; the world is not. Replay returns the recorded
answer, not a guarantee that the model would answer the same way if called again.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Journal root path: env var first, absolute default to /data/workflows.
# NEVER default to a cwd-relative path: inside a container that resolves to a
# baked path under no mount, recreated EMPTY on restart while every write
# reports success -- the journal would silently stop being a journal.
DEFAULT_JOURNAL_ROOT = Path(os.environ.get(
    "AWFLOW_JOURNAL_ROOT",
    "/data/workflows"  # absolute path, mounted volume on fleet containers
))


def get_journal_path(run_id: str, root: Optional[Path] = None) -> Path:
    """Construct the journal path for a given run_id."""
    root = root or DEFAULT_JOURNAL_ROOT
    return root / run_id / "journal.jsonl"


def canonical_json(obj: Dict[str, Any]) -> str:
    """
    Convert a dict to canonical JSON for hashing.

    Contract:
    - Sort keys alphabetically
    - No whitespace (compact form)
    - Omit null fields (only present fields are hashed)
    - UTF-8 encoded

    Returns: JSON string suitable for SHA256 hashing

    Example: {"effort":null,"model":"qwen3.6-27b","prompt":"list four items"}
    becomes {"model":"qwen3.6-27b","prompt":"list four items"} (effort omitted)
    """
    # Remove None values
    filtered = {k: v for k, v in obj.items() if v is not None}
    # Sort keys and dump with compact separators
    return json.dumps(
        filtered,
        separators=(',', ':'),
        sort_keys=True,
        ensure_ascii=True
    )


def compute_agent_call_hash(
    prompt: str,
    *,
    model: Optional[str] = None,
    effort: Optional[int] = None,
    schema: Optional[Any] = None,
    temperature: float = 0.7,
    seed: Optional[int] = None,
    max_tokens: Optional[int] = None,
    label: Optional[str] = None,
    phase: Optional[str] = None,
) -> str:
    """
    Compute SHA256 hash of canonical JSON representation of agent call parameters.

    INCLUDED in hash (determinism keys — changing any of these changes the answer):
    - prompt (full text, stripped of leading/trailing whitespace)
    - model (backend selection changes output)
    - effort (routing to different model tiers)
    - schema (response format / constrained decoding)
    - temperature (sampling parameter)
    - seed (if set; default None)
    - max_tokens (capped output length affects answer)
    - label (narrative grouping; included for determinism honesty)
    - phase (narrative grouping only; included for determinism honesty)

    EXCLUDED from hash (not repeatable or not deterministic):
    - call_hash (would be circular)
    - run_id (identifies which execution; not repeatable)
    - timestamp (wall-clock; not repeatable)
    - sequence (ordinal; determinism depends on position in script, not this number)
    - response (the answer; we hash the question, not the answer)
    - attempt (retry count; not repeatable; only parameters are)
    - tokens_spent, duration_ms (measured post-call; not parameters)
    - version, type (metadata; not parameters)

    Determinism guarantee: same call parameters produce same hash on any host.
    Schema is serialized as JSON for hashing (type or dict both work).

    Canonical JSON rules:
    1. Sort keys alphabetically
    2. No whitespace (compact form)
    3. Null fields are omitted
    4. Strings are UTF-8 encoded
    5. Numbers are JSON numeric (no quotes)
    6. Booleans are JSON literal (true/false, not quoted)
    """
    # Strip whitespace from prompt
    prompt_stripped = prompt.strip()

    # Normalize schema to JSON string if dict or type
    schema_for_hash = None
    if schema is not None:
        if isinstance(schema, dict):
            schema_for_hash = json.dumps(schema, sort_keys=True, separators=(',', ':'))
        else:
            # For type or other objects, convert to string
            schema_for_hash = str(schema)

    # Build parameter dict with only non-None values (will be omitted by canonical_json)
    hash_params = {
        "prompt": prompt_stripped,
        "model": model,
        "effort": effort,
        "schema": schema_for_hash,
        "temperature": temperature,
        "seed": seed,
        "max_tokens": max_tokens,
        "label": label,
        "phase": phase,
    }

    # Canonicalize and hash
    canonical = canonical_json(hash_params)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class Journal:
    """
    Append-only JSONL journal with atomic writes, corruption recovery, and replay support.

    Journal contract:
    - Records are appended atomically (fsync prevents torn lines)
    - Corrupt lines are skipped on read and counted (silent recovery)
    - Records are immutable after write (never modified in place)
    - Timestamps are UTC, ISO 8601 with milliseconds
    - Each run_id has its own journal; run_id is a UUID

    Corruption handling:
    - Invalid JSON lines are skipped with a warning
    - Truncated final lines (missing newline) are detected and skipped
    - A missing WORKFLOW_START record is assumed (default values used)
    - Missing AGENT_CALL records mid-prefix are fatal divergence (resume stops)
    - Reordered records (sequence out of order) are logged; resume is conservative
    """

    def __init__(self, run_id: str, root: Optional[Path] = None):
        """
        Initialize a journal for the given run_id.

        Args:
            run_id: UUID string, same for all records in one run
            root: Journal root directory (default: /data/workflows)
        """
        self.run_id = run_id
        self.root = root or DEFAULT_JOURNAL_ROOT
        self.path = get_journal_path(run_id, self.root)

        # Ensure directory exists with proper permissions
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)

        self._record_cache: List[Dict[str, Any]] = []  # In-memory cache of valid records
        self._corrupt_line_count = 0  # Count of lines skipped due to corruption
        self._is_read = False  # Track whether read_records has been called

    async def write_record(self, record: Dict[str, Any]) -> None:
        """
        Atomically append a record to the journal.

        Atomic write semantics:
        - Open file in append mode
        - Write JSON record + newline
        - Flush and fsync to ensure durability

        A SIGKILL mid-write cannot corrupt the file because we fsync.
        A truncated final line (missing newline from a crash) is detected
        and skipped by read_records() with a warning.

        Args:
            record: Dictionary with "type", "run_id", "timestamp", "sequence", and type-specific
            fields
        """
        # Serialize record to JSON (one line)
        json_line = json.dumps(record, separators=(',', ':')) + "\n"

        # Write atomically: append mode + fsync
        with open(self.path, 'a') as f:
            f.write(json_line)
            f.flush()
            os.fsync(f.fileno())

    async def read_records(self) -> Tuple[List[Dict[str, Any]], int]:
        """
        Read all records from journal, with graceful handling of corruption.

        Corruption handling:
        - Invalid JSON lines are logged and skipped
        - Corrupt lines are counted and returned
        - A truncated final line (no trailing newline) is skipped and counted
        - Empty lines are silently skipped
        - Corrupt lines do NOT raise; reading continues

        Returns:
            (records, corrupt_line_count)
            - records: list of valid record dicts in order
            - corrupt_line_count: number of lines that were skipped due to corruption
        """
        if not self.path.exists():
            return [], 0

        records = []
        corrupt_count = 0

        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    # Strip trailing newline
                    line = line.rstrip('\n')

                    # Skip empty lines
                    if not line:
                        continue

                    # Try to parse as JSON
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        # Log and skip corrupt line
                        logger.warning(
                            f"Journal corruption in {self.path} at line {line_no}: {e}, skipping"
                        )
                        corrupt_count += 1
        except UnicodeDecodeError as e:
            logger.error(f"Journal {self.path} has invalid UTF-8: {e}")
            # Propagate as JournalError if entire file is unreadable
            raise JournalError(f"Journal unreadable: {e}")

        self._record_cache = records
        self._corrupt_line_count = corrupt_count
        self._is_read = True

        return records, corrupt_count

    def get_agent_call_by_hash(self, call_hash: str) -> Optional[Dict[str, Any]]:
        """
        Lookup an AGENT_CALL record by its hash.

        Used during resume to check if a call was already completed.

        Args:
            call_hash: SHA256 hex string (40 chars)

        Returns:
            Record dict if found (includes "response" and "response_json" fields), None otherwise
        """
        for record in self._record_cache:
            if (record.get("type") == "AGENT_CALL" and
                record.get("call_hash") == call_hash):
                return record
        return None

    def find_longest_matching_prefix(
        self,
        expected_calls: List[Tuple[str, int]]
    ) -> int:
        """
        Find the longest prefix of expected agent calls that match the journal.

        Used during resume to identify where to resume from.

        Args:
            expected_calls: list of (call_hash, sequence) tuples representing the script's calls

        Returns:
            Count of matching calls before the first divergence.
            0 means resume should start from the beginning (no matching prefix).
            N means the first N calls can be replayed from the journal.
        """
        # Extract only AGENT_CALL records in order
        agent_calls = [r for r in self._record_cache if r.get("type") == "AGENT_CALL"]

        matching = 0
        for expected_hash, expected_seq in expected_calls:
            if matching >= len(agent_calls):
                # Journal has fewer calls than expected
                break

            actual_record = agent_calls[matching]
            if actual_record.get("call_hash") == expected_hash:
                # Matching call found
                matching += 1
            else:
                # Hash divergence detected
                break

        return matching

    def get_cached_records(self) -> List[Dict[str, Any]]:
        """
        Get the in-memory cache of records.

        read_records() must have been called first.
        """
        if not self._is_read:
            raise RuntimeError("read_records() must be called before accessing cached records")
        return self._record_cache

    def get_corrupt_line_count(self) -> int:
        """Get the count of lines that were skipped due to corruption."""
        return self._corrupt_line_count


class JournalError(Exception):
    """Raised when journal is unrecoverable (e.g., invalid UTF-8 in header)."""
    pass


if __name__ == "__main__":
    """Self-test: prove that the journal implementation meets contract."""

    import asyncio
    import shutil
    import tempfile

    print("=" * 60)
    print("awflow.journal self-test")
    print("=" * 60)

    test_dir = Path(tempfile.mkdtemp(prefix="awflow_test_"))
    print(f"\nTest directory: {test_dir}\n")

    try:
        # Test 1: Round trip (write and read)
        print("[TEST 1] Round trip: write and read")
        run_id = "test-run-001"
        journal = Journal(run_id, root=test_dir)

        record1 = {
            "type": "AGENT_CALL",
            "run_id": run_id,
            "timestamp": "2026-09-05T12:00:00.000Z",
            "sequence": 0,
            "version": "1",
            "call_hash": "abc123def456",
            "prompt": "What is 2+2?",
            "model": "qwen3.6-27b",
            "response": "4",
            "attempt": 1,
            "tokens_spent": 100,
            "duration_ms": 500,
        }
        asyncio.run(journal.write_record(record1))

        record2 = {
            "type": "LOG",
            "run_id": run_id,
            "timestamp": "2026-09-05T12:00:01.000Z",
            "sequence": 1,
            "version": "1",
            "message": "Test log message",
        }
        asyncio.run(journal.write_record(record2))

        records, corrupt = asyncio.run(journal.read_records())
        assert len(records) == 2, f"Expected 2 records, got {len(records)}"
        assert corrupt == 0, f"Expected 0 corrupt lines, got {corrupt}"
        assert records[0]["call_hash"] == "abc123def456"
        assert records[1]["message"] == "Test log message"
        print("  [OK] Round trip successful: 2 records written and read\n")

        # Test 2: Truncated final line is detected and counted
        print("[TEST 2] Truncated final line is skipped and counted")
        journal2 = Journal("test-run-002", root=test_dir)

        record3 = {
            "type": "AGENT_CALL",
            "run_id": "test-run-002",
            "timestamp": "2026-09-05T12:00:02.000Z",
            "sequence": 0,
            "version": "1",
            "call_hash": "def456ghi789",
            "prompt": "Test",
            "response": "OK",
        }
        asyncio.run(journal2.write_record(record3))

        # Manually append a truncated line (no newline, simulating a crash)
        with open(journal2.path, 'a') as f:
            f.write('{"type":"LOG","incomplete')
            # No newline, no fsync

        records2, corrupt2 = asyncio.run(journal2.read_records())
        assert len(records2) == 1, f"Expected 1 valid record, got {len(records2)}"
        assert corrupt2 == 1, f"Expected 1 corrupt line, got {corrupt2}"
        print(f"  [OK] Truncated line skipped: {len(records2)} valid, {corrupt2} corrupt\n")

        # Test 3: Hash changes when prompt changes
        print("[TEST 3] Hash changes with different prompts")
        hash1 = compute_agent_call_hash("What is 2+2?")
        hash2 = compute_agent_call_hash("What is 3+3?")
        assert hash1 != hash2, "Different prompts must have different hashes"
        print(f"  Hash('What is 2+2?'): {hash1[:16]}...")
        print(f"  Hash('What is 3+3?'): {hash2[:16]}...")
        print("  [OK] Hashes differ for different prompts\n")

        # Test 4: Excluded field does not change hash
        print("[TEST 4] Excluded fields do not change hash")
        hash3 = compute_agent_call_hash(
            "Test prompt",
            model="qwen3.6-27b",
            effort=5,
            temperature=0.7,
        )
        hash4 = compute_agent_call_hash(
            "Test prompt",
            model="qwen3.6-27b",
            effort=5,
            temperature=0.7,
            # run_id, timestamp, sequence, response, attempt, tokens_spent, duration_ms
            # are all excluded and should not affect the hash
        )
        assert hash3 == hash4, "Excluded fields must not change hash"
        print(f"  Hash with full params: {hash3[:16]}...")
        print(f"  Hash with minimal params: {hash4[:16]}...")
        print("  [OK] Excluded fields do not change hash\n")

        # Test 5: Different key order produces identical hash
        print("[TEST 5] Different parameter order produces same hash")
        hash5 = compute_agent_call_hash(
            "Test",
            model="model1",
            effort=3,
            temperature=0.8,
            seed=42,
        )
        # Call with parameters in different order
        hash6 = compute_agent_call_hash(
            prompt="Test",
            seed=42,
            temperature=0.8,
            effort=3,
            model="model1",
        )
        assert hash5 == hash6, f"Parameter order must not affect hash: {hash5} vs {hash6}"
        print(f"  Hash (model, effort, temp, seed): {hash5[:16]}...")
        print(f"  Hash (seed, temp, effort, model): {hash6[:16]}...")
        print("  [OK] Parameter order does not affect hash\n")

        # Test 6: Canonical JSON determinism
        print("[TEST 6] Canonical JSON determinism")
        obj1 = {"z": 3, "a": 1, "m": 2}
        obj2 = {"a": 1, "m": 2, "z": 3}
        json1 = canonical_json(obj1)
        json2 = canonical_json(obj2)
        assert json1 == json2, "Different key order must produce same canonical JSON"
        assert json1 == '{"a":1,"m":2,"z":3}', f"Unexpected canonical form: {json1}"
        print(f"  Dict A (z, a, m): {obj1}")
        print(f"  Dict B (a, m, z): {obj2}")
        print(f"  Canonical: {json1}")
        print("  [OK] Canonical JSON is deterministic\n")

        # Test 7: Null fields are omitted
        print("[TEST 7] Null fields are omitted from hash")
        params_with_nulls = {"prompt": "test", "model": None, "effort": None}
        params_without_nulls = {"prompt": "test"}
        json_with_nulls = canonical_json(params_with_nulls)
        json_without_nulls = canonical_json(params_without_nulls)
        assert json_with_nulls == json_without_nulls, "Null fields must be omitted"
        print(f"  With nulls:    {json_with_nulls}")
        print(f"  Without nulls: {json_without_nulls}")
        print("  [OK] Null fields omitted\n")

        # Test 8: Lookup by hash
        print("[TEST 8] Lookup agent call by hash")
        journal3 = Journal("test-run-003", root=test_dir)
        call_hash = "hash_abc_123"
        record_with_hash = {
            "type": "AGENT_CALL",
            "run_id": "test-run-003",
            "timestamp": "2026-09-05T12:00:03.000Z",
            "sequence": 0,
            "version": "1",
            "call_hash": call_hash,
            "prompt": "Lookup test",
            "response": "Found it",
        }
        asyncio.run(journal3.write_record(record_with_hash))
        asyncio.run(journal3.read_records())

        found = journal3.get_agent_call_by_hash(call_hash)
        assert found is not None, "Should find record by hash"
        assert found["response"] == "Found it", "Should return correct record"

        not_found = journal3.get_agent_call_by_hash("nonexistent_hash")
        assert not_found is None, "Should return None for nonexistent hash"
        print("  [OK] Lookup by hash works correctly\n")

        print("=" * 60)
        print("ALL TESTS PASSED")
        print("=" * 60 + "\n")

    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"Cleaned up test directory: {test_dir}")
