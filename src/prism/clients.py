"""Client registry — loads API keys and metadata from a JSON file."""

from __future__ import annotations

import json
import logging
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("prism")


@dataclass
class ClientInfo:
    name: str
    key: str
    allowed_models: list[str] = field(default_factory=lambda: ["*"])
    rate_limit: int = 0

    def model_allowed(self, model: str) -> bool:
        return "*" in self.allowed_models or model in self.allowed_models


def generate_key(name: str) -> str:
    """Generate a random client key in the form ``pk-<name>-<hex>``."""
    return f"pk-{name}-{secrets.token_hex(16)}"


class ClientRegistry:
    """Thread-safe, file-backed client registry with O(1) key lookup."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._by_key: dict[str, ClientInfo] = {}
        self._by_name: dict[str, ClientInfo] = {}
        if self._path.exists():
            self._load()

    # -- public query API --------------------------------------------------

    def lookup(self, key: str) -> ClientInfo | None:
        return self._by_key.get(key)

    def get(self, name: str) -> ClientInfo | None:
        return self._by_name.get(name)

    def list_all(self) -> list[ClientInfo]:
        return list(self._by_name.values())

    @property
    def loaded(self) -> bool:
        return len(self._by_key) > 0

    # -- mutation API (used by admin endpoints) ----------------------------

    def add(self, client: ClientInfo) -> None:
        with self._lock:
            self._by_key[client.key] = client
            self._by_name[client.name] = client
            self._save()

    def remove(self, name: str) -> bool:
        with self._lock:
            client = self._by_name.pop(name, None)
            if client is None:
                return False
            self._by_key.pop(client.key, None)
            self._save()
            return True

    def rotate_key(self, name: str) -> ClientInfo | None:
        with self._lock:
            client = self._by_name.get(name)
            if client is None:
                return None
            self._by_key.pop(client.key, None)
            client.key = generate_key(name)
            self._by_key[client.key] = client
            self._save()
            return client

    def reload(self) -> int:
        """Re-read the JSON file. Returns the number of clients loaded."""
        with self._lock:
            self._by_key.clear()
            self._by_name.clear()
            self._load()
            return len(self._by_name)

    # -- persistence -------------------------------------------------------

    def _load(self) -> None:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load client registry from %s: %s", self._path, exc)
            return

        for entry in data.get("clients", []):
            client = ClientInfo(
                name=entry["name"],
                key=entry["key"],
                allowed_models=entry.get("allowed_models", ["*"]),
                rate_limit=entry.get("rate_limit", 0),
            )
            self._by_key[client.key] = client
            self._by_name[client.name] = client

        logger.info("Loaded %d client(s) from %s", len(self._by_name), self._path)

    def _save(self) -> None:
        data = {
            "clients": [
                {
                    "name": c.name,
                    "key": c.key,
                    "allowed_models": c.allowed_models,
                    "rate_limit": c.rate_limit,
                }
                for c in self._by_name.values()
            ]
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(self._path)
