"""Explicit application-owned services shared with UI wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from .config import AppConfig

if TYPE_CHECKING:
    from .core.executor import ModelExecutor
    from .infra.model_manager import ModelManager
    from .infra.model_storage import ModelStorageCatalog
    from .storage.output_store import OutputStore
    from .ui.actions import UiActions


class ManagedBackend(Protocol):
    def ensure_running(self) -> Any: ...

    def stop(self) -> Any: ...


class VideoBackend(Protocol):
    def start(self) -> bool: ...

    def stop(self) -> None: ...

    def health(self) -> tuple[bool, bool | None]: ...


class ChatSelector(Protocol):
    choice: str
    service: Any


class Lock(Protocol):
    def __enter__(self) -> Any: ...

    def __exit__(self, *args: Any) -> Any: ...


@dataclass(frozen=True)
class AppContext:
    config: AppConfig
    model_manager: ModelManager
    output_store: OutputStore
    ltx_video: VideoBackend
    diffusiongemma: ManagedBackend
    krea2: ManagedBackend
    chat_selector: ChatSelector
    model_load_lock: Lock
    gpu_lock: Lock
    image_executor: ModelExecutor | None = None
    model_storage: ModelStorageCatalog | None = None
    ui_actions: UiActions | None = None
