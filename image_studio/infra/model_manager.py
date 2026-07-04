"""GPU model lifecycle and LRU eviction without UI dependencies."""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    name: str
    pipeline: Any
    vram_mb: float
    unload_fn: Callable[[], None]
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


@dataclass(frozen=True)
class ManagedModelSpec:
    key: str
    display_name: str
    vram_mb: int
    exclusive_group: str | None = None


class ModelManager:
    """Central registry for GPU-resident models with LRU eviction.

    Lock order for callers is: application model-load lock, application GPU
    execution lock, then this manager's registry lock. Unload callbacks must not
    initiate another managed load.
    """

    def __init__(self, memory_info: Callable[[], tuple[int, int]] | None = None):
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.RLock()
        self._memory_info = memory_info or self._default_memory_info

    @staticmethod
    def _default_memory_info() -> tuple[int, int]:
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.mem_get_info()
        except Exception:
            pass
        return 0, 0

    def _gpu_free_mb(self) -> float:
        free, _ = self._memory_info()
        return free / (1024 * 1024)

    def _gpu_total_mb(self) -> float:
        _, total = self._memory_info()
        return total / (1024 * 1024)

    def _gpu_used_mb(self) -> float:
        free, total = self._memory_info()
        return (total - free) / (1024 * 1024)

    def register(self, name: str, pipeline: Any, vram_mb: float, unload_fn: Callable[[], None]) -> None:
        with self._lock:
            if name in self._models:
                entry = self._models[name]
                entry.pipeline = pipeline
                entry.vram_mb = vram_mb
                entry.unload_fn = unload_fn
                entry.last_used = time.time()
            else:
                self._models[name] = LoadedModel(name, pipeline, vram_mb, unload_fn)
            self._models.move_to_end(name)
        log.info("ModelManager: registered '%s' (~%.0f MiB)", name, vram_mb)

    def touch(self, name: str) -> None:
        with self._lock:
            if name in self._models:
                self._models[name].last_used = time.time()
                self._models.move_to_end(name)

    def get(self, name: str) -> Any | None:
        with self._lock:
            entry = self._models.get(name)
            if entry is None:
                return None
            self.touch(name)
            return entry.pipeline

    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._models

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._models)

    def ensure_vram(self, need_mb: float, exclude: str | None = None) -> None:
        with self._lock:
            if self._gpu_free_mb() >= need_mb:
                return
            for key in list(self._models):
                if key != exclude and self._gpu_free_mb() < need_mb:
                    self.unload(key)

    def unload(self, name: str) -> None:
        with self._lock:
            entry = self._models.pop(name, None)
            if entry is None:
                return
            try:
                entry.unload_fn()
            except Exception as exc:
                log.warning("ModelManager: unload callback for '%s' failed: %s", name, exc)
            entry.pipeline = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def unload_all(self) -> None:
        for name in self.keys():
            self.unload(name)

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                {
                    "name": item.name,
                    "vram_mb": round(item.vram_mb),
                    "loaded_at": datetime.fromtimestamp(item.loaded_at).strftime("%H:%M:%S"),
                    "last_used": datetime.fromtimestamp(item.last_used).strftime("%H:%M:%S"),
                }
                for item in self._models.values()
            ]

    def gpu_summary(self) -> dict[str, int]:
        with self._lock:
            return {
                "total_mb": round(self._gpu_total_mb()),
                "used_mb": round(self._gpu_used_mb()),
                "free_mb": round(self._gpu_free_mb()),
                "tracked_mb": round(sum(item.vram_mb for item in self._models.values())),
                "model_count": len(self._models),
            }

    def get_or_load(
        self,
        spec: ManagedModelSpec,
        factory: Callable[[], Any],
        unload_fn_factory: Callable[[Any], Callable[[], None]],
        *,
        specs: dict[str, ManagedModelSpec] | None = None,
    ) -> Any:
        existing = self.get(spec.key)
        if existing is not None:
            return existing
        if spec.exclusive_group and specs:
            for other in specs.values():
                if other.key != spec.key and other.exclusive_group == spec.exclusive_group:
                    self.unload(other.key)
        self.ensure_vram(spec.vram_mb, exclude=spec.key)
        loaded = factory()
        self.register(spec.key, loaded, spec.vram_mb, unload_fn_factory(loaded))
        return loaded
