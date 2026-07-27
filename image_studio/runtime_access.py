"""Temporary access to the legacy runtime composition namespace.

This module deliberately does not copy names into feature modules.  Extracted
compatibility functions resolve the few application-owned objects they still
need at call time.  New code must receive dependencies through ``AppContext``
or a constructor and must not import this module.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

_runtime: ModuleType | None = None


def install_runtime_namespace(module: ModuleType) -> None:
    """Install the active compatibility runtime exactly once."""
    global _runtime
    if _runtime is not None and _runtime is not module:
        raise RuntimeError("A different Image Studio runtime is already installed.")
    _runtime = module


def runtime_namespace() -> Any:
    """Return the active compatibility runtime.

    The explicit error makes independently imported adapters fail predictably
    instead of carrying a partially copied module namespace.
    """
    if _runtime is None:
        raise RuntimeError(
            "Image Studio runtime services are not initialized. "
            "Create the application through image_studio.composition first."
        )
    return _runtime


def export_public(module: ModuleType, destination: dict[str, Any]) -> None:
    """Re-export a compatibility module's declared public API.

    This only maintains the historical ``image_studio.app``/``runtime`` export
    surface.  It never writes runtime values back into the source module.
    """
    for name in getattr(module, "__all__", ()):
        destination[name] = getattr(module, name)

