"""Unit tests for the client registry and auth module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prism.clients import ClientInfo, ClientRegistry, generate_key


# ---------------------------------------------------------------------------
# ClientInfo
# ---------------------------------------------------------------------------

class TestClientInfo:
    def test_wildcard_allows_any_model(self):
        c = ClientInfo(name="a", key="k", allowed_models=["*"])
        assert c.model_allowed("anything") is True

    def test_explicit_list_allows_listed_model(self):
        c = ClientInfo(name="a", key="k", allowed_models=["gpt-4", "gpt-3.5"])
        assert c.model_allowed("gpt-4") is True
        assert c.model_allowed("gpt-3.5") is True

    def test_explicit_list_blocks_unlisted_model(self):
        c = ClientInfo(name="a", key="k", allowed_models=["gpt-4"])
        assert c.model_allowed("gpt-3.5") is False

    def test_default_allowed_models_is_wildcard(self):
        c = ClientInfo(name="a", key="k")
        assert c.model_allowed("any-model") is True


# ---------------------------------------------------------------------------
# generate_key
# ---------------------------------------------------------------------------

class TestGenerateKey:
    def test_format(self):
        key = generate_key("pip")
        assert key.startswith("pk-pip-")
        assert len(key) == len("pk-pip-") + 32  # 16 bytes hex = 32 chars

    def test_uniqueness(self):
        keys = {generate_key("x") for _ in range(50)}
        assert len(keys) == 50


# ---------------------------------------------------------------------------
# ClientRegistry — loading
# ---------------------------------------------------------------------------

class TestRegistryLoad:
    def test_load_from_json(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({
            "clients": [
                {"name": "pip", "key": "pk-pip-aaa", "allowed_models": ["*"], "rate_limit": 0},
                {"name": "bot", "key": "pk-bot-bbb", "allowed_models": ["gpt-4"], "rate_limit": 10},
            ]
        }))
        reg = ClientRegistry(f)
        assert reg.loaded is True
        assert len(reg.list_all()) == 2

    def test_lookup_by_key(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({
            "clients": [{"name": "pip", "key": "pk-pip-aaa"}]
        }))
        reg = ClientRegistry(f)
        client = reg.lookup("pk-pip-aaa")
        assert client is not None
        assert client.name == "pip"

    def test_lookup_missing_key_returns_none(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": [{"name": "a", "key": "k"}]}))
        reg = ClientRegistry(f)
        assert reg.lookup("nonexistent") is None

    def test_get_by_name(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": [{"name": "pip", "key": "k1"}]}))
        reg = ClientRegistry(f)
        assert reg.get("pip") is not None
        assert reg.get("pip").key == "k1"

    def test_missing_file_creates_empty_registry(self, tmp_path: Path):
        reg = ClientRegistry(tmp_path / "nonexistent.json")
        assert reg.loaded is False
        assert reg.list_all() == []

    def test_malformed_json_creates_empty_registry(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("not json!!!")
        reg = ClientRegistry(f)
        assert reg.loaded is False

    def test_defaults_for_optional_fields(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": [{"name": "x", "key": "k"}]}))
        reg = ClientRegistry(f)
        client = reg.lookup("k")
        assert client.allowed_models == ["*"]
        assert client.rate_limit == 0


# ---------------------------------------------------------------------------
# ClientRegistry — mutation
# ---------------------------------------------------------------------------

class TestRegistryMutation:
    def test_add_persists(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": []}))
        reg = ClientRegistry(f)

        reg.add(ClientInfo(name="new", key="pk-new-xxx"))
        assert reg.lookup("pk-new-xxx") is not None

        # Verify persisted to disk
        reg2 = ClientRegistry(f)
        assert reg2.lookup("pk-new-xxx") is not None

    def test_remove_persists(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": [{"name": "a", "key": "k1"}]}))
        reg = ClientRegistry(f)

        assert reg.remove("a") is True
        assert reg.lookup("k1") is None
        assert reg.remove("a") is False  # already gone

        reg2 = ClientRegistry(f)
        assert reg2.lookup("k1") is None

    def test_rotate_key(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": [{"name": "a", "key": "old-key"}]}))
        reg = ClientRegistry(f)

        client = reg.rotate_key("a")
        assert client is not None
        assert client.key != "old-key"
        assert client.key.startswith("pk-a-")
        assert reg.lookup("old-key") is None
        assert reg.lookup(client.key) is not None

    def test_rotate_key_nonexistent(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": []}))
        reg = ClientRegistry(f)
        assert reg.rotate_key("nope") is None

    def test_reload(self, tmp_path: Path):
        f = tmp_path / "clients.json"
        f.write_text(json.dumps({"clients": [{"name": "a", "key": "k1"}]}))
        reg = ClientRegistry(f)
        assert len(reg.list_all()) == 1

        f.write_text(json.dumps({
            "clients": [
                {"name": "a", "key": "k1"},
                {"name": "b", "key": "k2"},
            ]
        }))
        count = reg.reload()
        assert count == 2
        assert reg.lookup("k2") is not None
