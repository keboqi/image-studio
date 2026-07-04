"""Atomic JSON caches and prompt sidecars."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any


class JsonFileCache:
    def __init__(self, path: str, *, schema_version: int = 1) -> None:
        self.path = path
        self.schema_version = schema_version
        self._data: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            if self._data is not None:
                return dict(self._data)
            try:
                with open(self.path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if payload.get("schema_version") != self.schema_version:
                    payload = {}
            except (OSError, json.JSONDecodeError, AttributeError):
                payload = {}
            self._data = dict(payload.get("entries", {}))
            return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        return self.load().get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            data = self.load()
            data[key] = value
            payload = {"schema_version": self.schema_version, "entries": data}
            temporary = f"{self.path}.tmp"
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(temporary, self.path)
            self._data = data

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        with self._lock:
            value = self.get(key)
            if value is not None:
                return value
            value = factory()
            self.set(key, value)
            return value


class PromptMetadataStore:
    def __init__(self, contained_path: Callable[[str], str | None], *, suffix: str, schema_version: int = 1):
        self._contained_path = contained_path
        self.suffix = suffix
        self.schema_version = schema_version

    def sidecar_path(self, raw_path: str) -> str | None:
        safe = self._contained_path(raw_path)
        return f"{safe}{self.suffix}" if safe else None

    def write(self, raw_path: str, metadata: dict[str, Any]) -> None:
        sidecar = self.sidecar_path(raw_path)
        if not sidecar:
            return
        payload = {
            "schema_version": self.schema_version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **metadata,
        }
        temporary = f"{sidecar}.tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, sidecar)

    def read(self, raw_path: str) -> dict[str, Any] | None:
        sidecar = self.sidecar_path(raw_path)
        if not sidecar or not os.path.isfile(sidecar):
            return None
        try:
            with open(sidecar, encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if payload.get("schema_version") == self.schema_version else None
        except (OSError, json.JSONDecodeError, AttributeError):
            return None
