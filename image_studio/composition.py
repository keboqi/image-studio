"""Application composition boundary.

Importing this module is intentionally lightweight.  Heavy GPU dependencies
are loaded only when ``create_application`` is called.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .config import AppConfig
from .context import AppContext
from .core.executor import ModelExecutor


class RuntimeFacade(Protocol):
    APP_CONFIG: AppConfig
    APP_CONTEXT: AppContext
    IMAGE_MODEL_EXECUTOR: ModelExecutor
    CSS: str
    THEME: Any
    log: Any

    def build_ui(self, context: AppContext) -> Any: ...

    def attach_app_routes(self, app: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ImageStudioApplication:
    """The composed application and its public integration points."""

    context: AppContext
    model_executor: ModelExecutor
    runtime: RuntimeFacade

    def build_ui(self) -> Any:
        return self.runtime.build_ui(self.context)

    def attach_routes(self, app: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("model_catalog_provider", self.model_executor.catalog)
        return self.runtime.attach_app_routes(app, **kwargs)


_application: ImageStudioApplication | None = None
_application_lock = threading.RLock()


def create_application(config: AppConfig | None = None) -> ImageStudioApplication:
    """Compose Image Studio once, deferring heavyweight imports until now.

    ``config`` is accepted for dependency-injected callers.  During the
    compatibility migration the legacy runtime still owns its environment
    snapshot, so a non-identical snapshot is rejected rather than silently
    ignored.
    """
    global _application
    with _application_lock:
        if _application is None:
            runtime = cast(RuntimeFacade, importlib.import_module("image_studio.runtime"))
            if config is not None and config != runtime.APP_CONFIG:
                raise ValueError(
                    "The compatibility runtime was initialized with a different AppConfig."
                )
            _application = ImageStudioApplication(
                context=runtime.APP_CONTEXT,
                model_executor=runtime.IMAGE_MODEL_EXECUTOR,
                runtime=runtime,
            )
        elif config is not None and config != _application.context.config:
            raise ValueError("Image Studio is already composed with a different AppConfig.")
        return _application


def reset_application_for_tests() -> None:
    """Clear the process-local composition cache for isolated tests."""
    global _application
    with _application_lock:
        _application = None
