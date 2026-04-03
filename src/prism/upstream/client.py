"""Async HTTP client for forwarding requests to the upstream OpenAI-compatible endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from prism.config import settings


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


def _completions_url() -> str:
    base = settings.upstream_base_url.rstrip("/")
    return f"{base}/chat/completions"


async def send_request(payload: dict[str, Any]) -> httpx.Response:
    """Send a non-streaming request to the upstream endpoint and return the response."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        return await client.post(
            _completions_url(),
            json=payload,
            headers=_build_headers(),
        )


async def send_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    """Send a streaming request and yield raw byte chunks from the SSE response."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        async with client.stream(
            "POST",
            _completions_url(),
            json=payload,
            headers=_build_headers(),
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk
