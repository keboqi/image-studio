"""Explicit application-owned services shared with UI wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .infra.model_manager import ModelManager
from .storage.output_store import OutputStore


@dataclass(frozen=True)
class AppContext:
    config: AppConfig
    model_manager: ModelManager
    output_store: OutputStore
    ltx_video: Any
    diffusiongemma: Any
    krea2: Any
    chat_selector: Any
    model_load_lock: Any
    gpu_lock: Any
