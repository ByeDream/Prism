"""Unit tests for the usage recorder and rate limiter."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from prism.ratelimit import RateLimiter
from prism.usage import UsageRecorder


# ---------------------------------------------------------------------------
# UsageRecorder
# ---------------------------------------------------------------------------

class TestUsageRecorder:
    @pytest.fixture
    def recorder(self, tmp_path: Path) -> UsageRecorder:
        return UsageRecorder(tmp_path / "test.db")

    def test_record_and_query(self, recorder: UsageRecorder):
        asyncio.get_event_loop().run_until_complete(
            recorder.record("pip", "gpt-4", False, 100, 50, 200, 200)
        )
        rows, total = asyncio.get_event_loop().run_until_complete(recorder.query())
        assert total == 1
        assert rows[0]["client"] == "pip"
        assert rows[0]["model"] == "gpt-4"
        assert rows[0]["input_tokens"] == 100
        assert rows[0]["output_tokens"] == 50

    def test_query_filter_by_client(self, recorder: UsageRecorder):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(recorder.record("pip", "gpt-4", False, 10, 5, 100, 200))
        loop.run_until_complete(recorder.record("bot", "gpt-4", False, 20, 10, 100, 200))

        rows, total = loop.run_until_complete(recorder.query(client="pip"))
        assert total == 1
        assert rows[0]["client"] == "pip"

    def test_query_filter_by_model(self, recorder: UsageRecorder):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(recorder.record("pip", "gpt-4", False, 10, 5, 100, 200))
        loop.run_until_complete(recorder.record("pip", "gpt-3.5", False, 20, 10, 100, 200))

        rows, total = loop.run_until_complete(recorder.query(model="gpt-3.5"))
        assert total == 1
        assert rows[0]["model"] == "gpt-3.5"

    def test_query_pagination(self, recorder: UsageRecorder):
        loop = asyncio.get_event_loop()
        for i in range(5):
            loop.run_until_complete(recorder.record("pip", "m", False, i, 0, 0, 200))

        rows, total = loop.run_until_complete(recorder.query(limit=2, offset=0))
        assert total == 5
        assert len(rows) == 2

        rows2, _ = loop.run_until_complete(recorder.query(limit=2, offset=2))
        assert len(rows2) == 2

    def test_summary_totals(self, recorder: UsageRecorder):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(recorder.record("pip", "gpt-4", False, 100, 50, 200, 200))
        loop.run_until_complete(recorder.record("pip", "gpt-4", False, 200, 100, 300, 200))

        s = loop.run_until_complete(recorder.summary())
        assert s["total_requests"] == 2
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 150

    def test_summary_group_by_client(self, recorder: UsageRecorder):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(recorder.record("pip", "m", False, 10, 5, 100, 200))
        loop.run_until_complete(recorder.record("bot", "m", False, 20, 10, 100, 200))

        s = loop.run_until_complete(recorder.summary(group_by="client"))
        assert "breakdown" in s
        names = {entry["client"] for entry in s["breakdown"]}
        assert names == {"pip", "bot"}

    def test_summary_group_by_model(self, recorder: UsageRecorder):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(recorder.record("pip", "gpt-4", False, 10, 5, 100, 200))
        loop.run_until_complete(recorder.record("pip", "gpt-3.5", False, 20, 10, 100, 200))

        s = loop.run_until_complete(recorder.summary(group_by="model"))
        assert "breakdown" in s
        models = {entry["model"] for entry in s["breakdown"]}
        assert models == {"gpt-4", "gpt-3.5"}

    def test_empty_db_returns_zeros(self, recorder: UsageRecorder):
        loop = asyncio.get_event_loop()
        rows, total = loop.run_until_complete(recorder.query())
        assert total == 0
        assert rows == []

        s = loop.run_until_complete(recorder.summary())
        assert s["total_requests"] == 0
        assert s["total_input_tokens"] == 0


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter()
        for _ in range(5):
            assert rl.check("pip", 5) is True

    def test_blocks_over_limit(self):
        rl = RateLimiter()
        for _ in range(3):
            assert rl.check("pip", 3) is True
        assert rl.check("pip", 3) is False

    def test_zero_limit_means_unlimited(self):
        rl = RateLimiter()
        for _ in range(100):
            assert rl.check("pip", 0) is True

    def test_different_clients_independent(self):
        rl = RateLimiter()
        for _ in range(2):
            rl.check("a", 2)
        assert rl.check("a", 2) is False
        assert rl.check("b", 2) is True

    def test_window_expires(self):
        rl = RateLimiter()

        with patch("prism.ratelimit.time.monotonic") as mock_time:
            mock_time.return_value = 1000.0
            for _ in range(3):
                rl.check("pip", 3)
            assert rl.check("pip", 3) is False

            mock_time.return_value = 1061.0
            assert rl.check("pip", 3) is True
