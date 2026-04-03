"""Integration tests for the admin API endpoints.

Uses FastAPI's TestClient so no running server is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def _setup_env(tmp_path: Path):
    """Patch settings so the app uses temp files for clients and usage db."""
    clients_file = tmp_path / "clients.json"
    clients_file.write_text(json.dumps({
        "clients": [
            {"name": "pip", "key": "pk-pip-testkey", "allowed_models": ["*"], "rate_limit": 0},
        ]
    }))
    usage_db = str(tmp_path / "test_usage.db")
    admin_key = "admin-secret"

    with patch("prism.config.settings") as mock_settings:
        mock_settings.prism_api_key = ""
        mock_settings.clients_file = str(clients_file)
        mock_settings.usage_db = usage_db
        mock_settings.prism_admin_key = admin_key
        mock_settings.upstream_base_url = "http://fake"
        mock_settings.upstream_api_key = ""
        mock_settings.default_model = ""
        mock_settings.prism_host = "0.0.0.0"
        mock_settings.prism_port = 9877

        import importlib
        import prism.auth
        import prism.app

        prism.auth._registry = None
        prism.auth._legacy_client = None
        importlib.reload(prism.app)

        from prism.app import app
        from prism.admin import admin_router

        if not any(r.path.startswith("/admin") for r in app.routes):
            app.include_router(admin_router)

        yield {
            "app": app,
            "admin_key": admin_key,
            "clients_file": clients_file,
            "usage_db": usage_db,
        }


@pytest.fixture
def ctx(_setup_env):
    return _setup_env


@pytest.fixture
def client(ctx):
    return TestClient(ctx["app"])


def _admin_headers(admin_key: str) -> dict:
    return {"Authorization": f"Bearer {admin_key}"}


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------

class TestAdminAuth:
    def test_no_auth_returns_401(self, client):
        resp = client.get("/admin/clients")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, client):
        resp = client.get("/admin/clients", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_valid_key_works(self, client, ctx):
        resp = client.get("/admin/clients", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Client CRUD
# ---------------------------------------------------------------------------

class TestClientCRUD:
    def test_list_clients(self, client, ctx):
        resp = client.get("/admin/clients", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["clients"]) >= 1
        assert data["clients"][0]["name"] == "pip"
        assert "****" in data["clients"][0]["key_preview"]

    def test_create_client(self, client, ctx):
        resp = client.post(
            "/admin/clients",
            headers=_admin_headers(ctx["admin_key"]),
            json={"name": "new-agent", "allowed_models": ["gpt-4"], "rate_limit": 10},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "new-agent"
        assert data["key"].startswith("pk-new-agent-")
        assert data["allowed_models"] == ["gpt-4"]
        assert data["rate_limit"] == 10

    def test_create_duplicate_returns_409(self, client, ctx):
        resp = client.post(
            "/admin/clients",
            headers=_admin_headers(ctx["admin_key"]),
            json={"name": "pip"},
        )
        assert resp.status_code == 409

    def test_delete_client(self, client, ctx):
        client.post(
            "/admin/clients",
            headers=_admin_headers(ctx["admin_key"]),
            json={"name": "to-delete"},
        )
        resp = client.delete("/admin/clients/to-delete", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_returns_404(self, client, ctx):
        resp = client.delete("/admin/clients/ghost", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 404

    def test_rotate_key(self, client, ctx):
        resp = client.post(
            "/admin/clients/pip/rotate-key",
            headers=_admin_headers(ctx["admin_key"]),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "pip"
        assert data["key"].startswith("pk-pip-")
        assert data["key"] != "pk-pip-testkey"

    def test_rotate_key_nonexistent_returns_404(self, client, ctx):
        resp = client.post(
            "/admin/clients/ghost/rotate-key",
            headers=_admin_headers(ctx["admin_key"]),
        )
        assert resp.status_code == 404

    def test_reload_clients(self, client, ctx):
        resp = client.post("/admin/clients/reload", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 200
        assert "clients_loaded" in resp.json()


# ---------------------------------------------------------------------------
# Usage endpoints
# ---------------------------------------------------------------------------

class TestUsageEndpoints:
    def test_usage_empty(self, client, ctx):
        resp = client.get("/admin/usage", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["requests"] == []

    def test_usage_summary_empty(self, client, ctx):
        resp = client.get("/admin/usage/summary", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 0

    def test_usage_after_recording(self, client, ctx):
        import asyncio
        from prism.app import usage_recorder

        asyncio.get_event_loop().run_until_complete(
            usage_recorder.record("pip", "gpt-4", False, 100, 50, 200, 200)
        )

        resp = client.get("/admin/usage", headers=_admin_headers(ctx["admin_key"]))
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["requests"][0]["client"] == "pip"

    def test_usage_summary_with_group_by(self, client, ctx):
        import asyncio
        from prism.app import usage_recorder

        asyncio.get_event_loop().run_until_complete(
            usage_recorder.record("pip", "gpt-4", False, 100, 50, 200, 200)
        )
        asyncio.get_event_loop().run_until_complete(
            usage_recorder.record("bot", "gpt-4", False, 200, 100, 300, 200)
        )

        resp = client.get(
            "/admin/usage/summary",
            headers=_admin_headers(ctx["admin_key"]),
            params={"group_by": "client"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requests"] == 2
        assert "breakdown" in data
