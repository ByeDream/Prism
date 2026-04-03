"""End-to-end tests using the official Anthropic Python SDK.

Verifies that Prism is fully compatible with the anthropic library,
which is the intended client interface for downstream projects.

Prerequisites:
    1. Configure .env with valid upstream credentials
    2. Start the server:  $env:PYTHONPATH="src"; python -m prism
    3. Run:  pytest tests/test_sdk.py -v -s
"""

from __future__ import annotations

import os
import sys

import anthropic
import pytest

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

BASE_URL = os.getenv("PRISM_TEST_URL", "http://localhost:9877")
API_KEY = os.getenv("PRISM_TEST_KEY", "test-key-123")
MODEL = os.getenv("PRISM_TEST_MODEL", "claude-sonnet-4-6")


@pytest.fixture()
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=API_KEY, base_url=BASE_URL)


class TestSDKNonStreaming:
    def test_simple_message(self, client: anthropic.Anthropic):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say hi in one word."}],
        )
        assert isinstance(resp, anthropic.types.Message)
        assert resp.id.startswith("msg_")
        assert resp.role == "assistant"
        assert resp.type == "message"
        assert resp.stop_reason == "end_turn"
        assert len(resp.content) > 0
        assert resp.content[0].type == "text"
        assert len(resp.content[0].text) > 0
        assert resp.usage.input_tokens >= 0
        assert resp.usage.output_tokens >= 0
        print(f"\n  Text: {resp.content[0].text}")

    def test_system_prompt(self, client: anthropic.Anthropic):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=64,
            system="You must reply with exactly one word.",
            messages=[{"role": "user", "content": "What color is the sky?"}],
        )
        assert isinstance(resp, anthropic.types.Message)
        print(f"\n  Text: {resp.content[0].text}")

    def test_multi_turn(self, client: anthropic.Anthropic):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=128,
            messages=[
                {"role": "user", "content": "Remember this number: 42"},
                {"role": "assistant", "content": "Got it, I will remember 42."},
                {"role": "user", "content": "What number did I tell you?"},
            ],
        )
        assert "42" in resp.content[0].text
        print(f"\n  Text: {resp.content[0].text}")

    def test_content_blocks(self, client: anthropic.Anthropic):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is 3+7?"},
                        {"type": "text", "text": "Reply with just the number."},
                    ],
                }
            ],
        )
        assert "10" in resp.content[0].text
        print(f"\n  Text: {resp.content[0].text}")


class TestSDKStreaming:
    def test_basic_streaming(self, client: anthropic.Anthropic):
        collected_text = ""
        event_types: list[str] = []

        with client.messages.stream(
            model=MODEL,
            max_tokens=128,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
        ) as stream:
            for event in stream:
                event_types.append(type(event).__name__)
                if hasattr(event, "text"):
                    collected_text += event.text

            final = stream.get_final_message()

        assert len(collected_text) > 0
        assert final.stop_reason == "end_turn"
        assert final.content[0].text == collected_text
        print(f"\n  Events: {event_types}")
        print(f"  Text: {collected_text}")

    def test_streaming_with_system(self, client: anthropic.Anthropic):
        with client.messages.stream(
            model=MODEL,
            max_tokens=128,
            system="Always reply in English.",
            messages=[{"role": "user", "content": "Greet me briefly."}],
        ) as stream:
            final = stream.get_final_message()

        assert final.stop_reason == "end_turn"
        assert len(final.content[0].text) > 0
        print(f"\n  Text: {final.content[0].text}")
