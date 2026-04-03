"""End-to-end integration tests against a running Prism server.

Prerequisites:
    1. Configure .env with valid upstream credentials
    2. Start the server:  $env:PYTHONPATH="src"; python -m prism
    3. Run:  pytest tests/test_e2e.py -v -s
"""

from __future__ import annotations

import os
import sys

import httpx
import pytest

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

BASE_URL = os.getenv("PRISM_TEST_URL", "http://localhost:8080")
API_KEY = os.getenv("PRISM_TEST_KEY", "test-key-123")
MODEL = os.getenv("PRISM_TEST_MODEL", "qwen-14b-chat")

MESSAGES_URL = f"{BASE_URL}/v1/messages"
HEALTH_URL = f"{BASE_URL}/health"
TIMEOUT = 180.0


def _headers(key: str | None = API_KEY) -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    if key:
        h["x-api-key"] = key
    return h


def _assert_anthropic_response(data: dict) -> None:
    """Validate that a response conforms to Anthropic Messages API shape."""
    assert data["type"] == "message", f"Expected type 'message', got {data.get('type')}"
    assert data["role"] == "assistant"
    assert data["id"].startswith("msg_")
    assert isinstance(data["content"], list) and len(data["content"]) > 0
    assert data["content"][0]["type"] == "text"
    assert len(data["content"][0]["text"]) > 0
    assert data["stop_reason"] is not None
    assert data["usage"]["input_tokens"] >= 0
    assert data["usage"]["output_tokens"] >= 0


def _parse_sse_events(text: str) -> list[dict]:
    """Parse SSE text into a list of {event, data} dicts."""
    import json

    events: list[dict] = []
    current_event = ""
    current_data = ""

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data = line[len("data: "):]
        elif line == "":
            if current_event and current_data:
                try:
                    events.append({"event": current_event, "data": json.loads(current_data)})
                except json.JSONDecodeError:
                    events.append({"event": current_event, "data": current_data})
                current_event = ""
                current_data = ""

    return events


# ── T1: Simple single-turn conversation ─────────────────────────────────

class TestNonStreaming:
    def test_t1_simple_message(self):
        resp = httpx.post(
            MESSAGES_URL,
            headers=_headers(),
            json={
                "model": MODEL,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            },
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
        data = resp.json()
        print(f"\n[T1] Response: {data['content'][0]['text'][:100]}")
        _assert_anthropic_response(data)

    # ── T2: With system prompt ──────────────────────────────────────────

    def test_t2_system_prompt(self):
        resp = httpx.post(
            MESSAGES_URL,
            headers=_headers(),
            json={
                "model": MODEL,
                "max_tokens": 256,
                "system": "You must reply in exactly 3 words, no more, no less.",
                "messages": [{"role": "user", "content": "How are you?"}],
            },
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
        data = resp.json()
        print(f"\n[T2] Response: {data['content'][0]['text'][:100]}")
        _assert_anthropic_response(data)

    # ── T3: Multi-turn conversation ─────────────────────────────────────

    def test_t3_multi_turn(self):
        resp = httpx.post(
            MESSAGES_URL,
            headers=_headers(),
            json={
                "model": MODEL,
                "max_tokens": 256,
                "messages": [
                    {"role": "user", "content": "My name is Prism."},
                    {"role": "assistant", "content": "Nice to meet you, Prism!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            },
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
        data = resp.json()
        text = data["content"][0]["text"]
        print(f"\n[T3] Response: {text[:100]}")
        _assert_anthropic_response(data)
        assert "prism" in text.lower(), "Model should remember the name from context"

    # ── T4: Content blocks format ───────────────────────────────────────

    def test_t4_content_blocks(self):
        resp = httpx.post(
            MESSAGES_URL,
            headers=_headers(),
            json={
                "model": MODEL,
                "max_tokens": 256,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is 2+2?"},
                            {"type": "text", "text": "Reply with just the number."},
                        ],
                    }
                ],
            },
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
        data = resp.json()
        print(f"\n[T4] Response: {data['content'][0]['text'][:100]}")
        _assert_anthropic_response(data)

    # ── T5: Parameter pass-through ──────────────────────────────────────

    def test_t5_parameters(self):
        resp = httpx.post(
            MESSAGES_URL,
            headers=_headers(),
            json={
                "model": MODEL,
                "max_tokens": 50,
                "temperature": 0.1,
                "stop_sequences": ["10"],
                "messages": [{"role": "user", "content": "Count from 1 to 100, one number per line."}],
            },
            timeout=TIMEOUT,
        )
        assert resp.status_code == 200, f"Status {resp.status_code}: {resp.text}"
        data = resp.json()
        text = data["content"][0]["text"]
        print(f"\n[T5] Response ({len(text)} chars): {text[:200]}")
        _assert_anthropic_response(data)


# ── Streaming tests ─────────────────────────────────────────────────────

class TestStreaming:
    def test_t6_basic_streaming(self):
        with httpx.stream(
            "POST",
            MESSAGES_URL,
            headers=_headers(),
            json={
                "model": MODEL,
                "max_tokens": 256,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            },
            timeout=TIMEOUT,
        ) as resp:
            assert resp.status_code == 200, f"Status {resp.status_code}"
            body = resp.read().decode("utf-8")

        events = _parse_sse_events(body)
        event_types = [e["event"] for e in events]
        print(f"\n[T6] SSE events: {event_types}")

        assert event_types[0] == "message_start"
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert event_types[-1] == "message_stop"

        deltas = [e for e in events if e["event"] == "content_block_delta"]
        full_text = "".join(e["data"]["delta"]["text"] for e in deltas)
        print(f"[T6] Streamed text: {full_text[:100]}")
        assert len(full_text) > 0

    def test_t7_streaming_with_system(self):
        with httpx.stream(
            "POST",
            MESSAGES_URL,
            headers=_headers(),
            json={
                "model": MODEL,
                "max_tokens": 256,
                "stream": True,
                "system": "Always reply in English.",
                "messages": [{"role": "user", "content": "Greet me."}],
            },
            timeout=TIMEOUT,
        ) as resp:
            assert resp.status_code == 200, f"Status {resp.status_code}"
            body = resp.read().decode("utf-8")

        events = _parse_sse_events(body)
        event_types = [e["event"] for e in events]
        print(f"\n[T7] SSE events: {event_types}")

        assert event_types[0] == "message_start"
        assert event_types[-1] == "message_stop"

        deltas = [e for e in events if e["event"] == "content_block_delta"]
        full_text = "".join(e["data"]["delta"]["text"] for e in deltas)
        print(f"[T7] Streamed text: {full_text[:100]}")
        assert len(full_text) > 0


# ── Auth tests ──────────────────────────────────────────────────────────

class TestAuth:
    def test_t8_no_api_key(self):
        resp = httpx.post(
            MESSAGES_URL,
            headers=_headers(key=None),
            json={
                "model": MODEL,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=TIMEOUT,
        )
        print(f"\n[T8] Status: {resp.status_code}")
        assert resp.status_code == 401

    def test_t9_wrong_api_key(self):
        resp = httpx.post(
            MESSAGES_URL,
            headers=_headers(key="wrong-key-999"),
            json={
                "model": MODEL,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            timeout=TIMEOUT,
        )
        print(f"\n[T9] Status: {resp.status_code}")
        assert resp.status_code == 401


# ── Health check ────────────────────────────────────────────────────────

class TestHealth:
    def test_t10_health_check(self):
        resp = httpx.get(HEALTH_URL, timeout=10.0)
        print(f"\n[T10] Health: {resp.json()}")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
