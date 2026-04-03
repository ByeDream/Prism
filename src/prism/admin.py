"""Admin API router — client management and usage queries.

Mounted at ``/admin`` and protected by ``PRISM_ADMIN_KEY``.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from prism.auth import get_registry
from prism.clients import ClientInfo, generate_key
from prism.config import settings

admin_router = APIRouter(prefix="/admin", tags=["admin"])


# -- admin auth dependency -------------------------------------------------

async def _verify_admin_key(request: Request) -> None:
    expected = settings.prism_admin_key
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")

    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer ") and auth[7:] == expected:
        return
    raise HTTPException(status_code=401, detail="Invalid admin key")


# -- request / response models --------------------------------------------

class CreateClientRequest(BaseModel):
    name: str
    allowed_models: list[str] = Field(default_factory=lambda: ["*"])
    rate_limit: int = 0


class ClientResponse(BaseModel):
    name: str
    key: str
    allowed_models: list[str]
    rate_limit: int


class ClientListItem(BaseModel):
    name: str
    key_preview: str
    allowed_models: list[str]
    rate_limit: int


def _mask_key(key: str) -> str:
    if len(key) <= 10:
        return key[:4] + "****"
    return key[:10] + "****" + key[-4:]


# -- usage endpoints -------------------------------------------------------

@admin_router.get("/usage", dependencies=[Depends(_verify_admin_key)])
async def get_usage(
    client: str | None = None,
    model: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
):
    from prism.app import usage_recorder

    rows, total = await usage_recorder.query(
        client=client, model=model, start=start, end=end, limit=limit, offset=offset
    )
    return {"requests": rows, "total": total}


@admin_router.get("/usage/summary", dependencies=[Depends(_verify_admin_key)])
async def get_usage_summary(
    client: str | None = None,
    model: str | None = None,
    start: str | None = None,
    end: str | None = None,
    group_by: str | None = Query(default=None, pattern="^(client|model|day)$"),
):
    from prism.app import usage_recorder

    return await usage_recorder.summary(
        client=client, model=model, start=start, end=end, group_by=group_by
    )


# -- client management endpoints -------------------------------------------

@admin_router.get("/clients", dependencies=[Depends(_verify_admin_key)])
async def list_clients():
    registry = get_registry()
    if registry is None:
        return {"clients": []}
    return {
        "clients": [
            ClientListItem(
                name=c.name,
                key_preview=_mask_key(c.key),
                allowed_models=c.allowed_models,
                rate_limit=c.rate_limit,
            ).model_dump()
            for c in registry.list_all()
        ]
    }


@admin_router.post("/clients", dependencies=[Depends(_verify_admin_key)], status_code=201)
async def create_client(body: CreateClientRequest):
    registry = get_registry()
    if registry is None:
        raise HTTPException(status_code=500, detail="Client registry not initialized")

    if registry.get(body.name) is not None:
        raise HTTPException(status_code=409, detail=f"Client '{body.name}' already exists")

    key = generate_key(body.name)
    client = ClientInfo(
        name=body.name,
        key=key,
        allowed_models=body.allowed_models,
        rate_limit=body.rate_limit,
    )
    registry.add(client)
    return ClientResponse(
        name=client.name,
        key=client.key,
        allowed_models=client.allowed_models,
        rate_limit=client.rate_limit,
    ).model_dump()


@admin_router.delete("/clients/{name}", dependencies=[Depends(_verify_admin_key)])
async def delete_client(name: str):
    registry = get_registry()
    if registry is None:
        raise HTTPException(status_code=500, detail="Client registry not initialized")

    if not registry.remove(name):
        raise HTTPException(status_code=404, detail=f"Client '{name}' not found")
    return {"status": "deleted", "name": name}


@admin_router.post("/clients/{name}/rotate-key", dependencies=[Depends(_verify_admin_key)])
async def rotate_client_key(name: str):
    registry = get_registry()
    if registry is None:
        raise HTTPException(status_code=500, detail="Client registry not initialized")

    client = registry.rotate_key(name)
    if client is None:
        raise HTTPException(status_code=404, detail=f"Client '{name}' not found")
    return ClientResponse(
        name=client.name,
        key=client.key,
        allowed_models=client.allowed_models,
        rate_limit=client.rate_limit,
    ).model_dump()


@admin_router.post("/clients/reload", dependencies=[Depends(_verify_admin_key)])
async def reload_clients():
    registry = get_registry()
    if registry is None:
        raise HTTPException(status_code=500, detail="Client registry not initialized")

    count = registry.reload()
    return {"status": "reloaded", "clients_loaded": count}
