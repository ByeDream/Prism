"""Inbound request authentication for Prism."""

from __future__ import annotations

from fastapi import HTTPException, Request

from prism.config import settings


async def verify_api_key(request: Request) -> None:
    """Validate the inbound API key.

    The Anthropic SDK sends the key in the ``x-api-key`` header.
    We also accept ``Authorization: Bearer <key>`` for flexibility.
    If ``PRISM_API_KEY`` is empty, authentication is disabled.
    """
    expected = settings.prism_api_key
    if not expected:
        return

    api_key = request.headers.get("x-api-key", "")
    if api_key == expected:
        return

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer ") and auth_header[7:] == expected:
        return

    raise HTTPException(status_code=401, detail="Invalid API key")
