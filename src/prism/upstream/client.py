"""Async HTTP client for forwarding requests to the upstream OpenAI-compatible endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from prism.config import settings

_client: httpx.AsyncClient | None = None


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if settings.upstream_api_key:
        headers["Authorization"] = f"Bearer {settings.upstream_api_key}"
    return headers


def _completions_url() -> str:
    base = settings.upstream_base_url.rstrip("/")
    return f"{base}/chat/completions"


async def startup() -> None:
    """Create the shared httpx client.  Called once during app lifespan."""
    global _client  # noqa: PLW0603
    timeout = httpx.Timeout(settings.upstream_timeout, connect=10.0)
    _client = httpx.AsyncClient(timeout=timeout)


async def shutdown() -> None:
    """Close the shared httpx client.  Called once during app lifespan."""
    global _client  # noqa: PLW0603
    if _client is not None:
        await _client.aclose()
        _client = None


def _get_client() -> httpx.AsyncClient:
    if _client is None:
        raise RuntimeError("upstream client not initialized — call startup() first")
    return _client


async def send_request(payload: dict[str, Any]) -> httpx.Response:
    """Send a non-streaming request to the upstream endpoint and return the response."""
    return await _get_client().post(
        _completions_url(),
        json=payload,
        headers=_build_headers(),
    )


async def send_stream(payload: dict[str, Any]) -> AsyncIterator[bytes]:
    """Send a streaming request and yield raw byte chunks from the SSE response."""
    client = _get_client()
    async with client.stream(
        "POST",
        _completions_url(),
        json=payload,
        headers=_build_headers(),
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk
