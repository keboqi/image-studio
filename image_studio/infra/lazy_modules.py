"""Thread-safe lazy import groups."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from typing import Any

from ..errors import BackendUnavailableError


class LazyModuleGroup:
    def __init__(
        self,
        name: str,
        ensure: Callable[[], bool],
        importer: Callable[[], Mapping[str, Any]],
    ) -> None:
        self.name = name
        self._ensure = ensure
        self._importer = importer
        self._cache: dict[str, Any] | None = None
        self._lock = threading.Lock()

    def get(self) -> dict[str, Any]:
        if self._cache is not None:
            return self._cache
        with self._lock:
            if self._cache is not None:
                return self._cache
            if not self._ensure():
                raise BackendUnavailableError(f"{self.name} is unavailable.")
            try:
                modules = dict(self._importer())
            except BackendUnavailableError:
                raise
            except Exception as exc:
                raise BackendUnavailableError(f"Could not import {self.name}: {exc}") from exc
            self._cache = modules
            return modules

    def clear(self) -> None:
        with self._lock:
            self._cache = None
