"""Inbound request authentication for Prism.

Supports two modes:
1. Multi-client registry (clients.json) — each client has its own key.
2. Legacy single-key (PRISM_API_KEY) — backward-compatible fallback.

If neither is configured, authentication is disabled.
"""

from __future__ import annotations

from fastapi import HTTPException, Request

from prism.clients import ClientInfo, ClientRegistry
from prism.config import settings

_registry: ClientRegistry | None = None
_legacy_client: ClientInfo | None = None


def _init_auth() -> None:
    """Lazy initialisation — called on first request."""
    global _registry, _legacy_client  # noqa: PLW0603

    _registry = ClientRegistry(settings.clients_file)

    if not _registry.loaded and settings.prism_api_key:
        _legacy_client = ClientInfo(name="default", key=settings.prism_api_key)


def get_registry() -> ClientRegistry | None:
    if _registry is None:
        _init_auth()
    return _registry


def _extract_key(request: Request) -> str:
    """Pull the API key from ``x-api-key`` or ``Authorization: Bearer``."""
    key = request.headers.get("x-api-key", "")
    if key:
        return key
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return ""


async def verify_api_key(request: Request) -> None:
    """Validate the inbound API key and attach client info to the request."""
    if _registry is None:
        _init_auth()

    key = _extract_key(request)

    # Multi-client registry takes precedence
    if _registry and _registry.loaded:
        if not key:
            raise HTTPException(status_code=401, detail="Missing API key")
        client = _registry.lookup(key)
        if client is None:
            raise HTTPException(status_code=401, detail="Invalid API key")
        request.state.client = client
        return

    # Legacy single-key fallback
    if _legacy_client:
        if not key or key != _legacy_client.key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        request.state.client = _legacy_client
        return

    # No auth configured — allow anonymous access
    request.state.client = ClientInfo(name="anonymous", key="")
