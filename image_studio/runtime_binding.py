"""Compatibility binding for mechanically extracted runtime modules.

The original application relied on one module-wide namespace. During the
refactor, extracted functions keep resolving the same shared objects through a
single startup binding instead of importing the application (which would cause
circular imports). New code should prefer explicit arguments and AppConfig.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

_INTERNAL = {"_runtime_protected", "_runtime_bound"}


def bind_module(module_globals: dict[str, Any], source: dict[str, Any]) -> None:
    protected = module_globals.setdefault("_runtime_protected", set(module_globals))
    bound = module_globals.setdefault("_runtime_bound", set())
    for name, value in source.items():
        if name.startswith("__") or name in _INTERNAL or name in protected:
            continue
        module_globals[name] = value
        bound.add(name)


def seal_module(module_globals: dict[str, Any]) -> None:
    module_globals.setdefault("_runtime_protected", set()).update(module_globals)


def export_module(module: ModuleType, destination: dict[str, Any]) -> None:
    bind_module(vars(module), destination)
    for name in getattr(module, "__all__", ()):
        destination[name] = getattr(module, name)


def rebind_modules(modules: list[ModuleType], source: dict[str, Any]) -> None:
    for module in modules:
        namespace = vars(module)
        protected = namespace.get("_runtime_protected", set())
        bound = namespace.get("_runtime_bound", set())
        for name, value in source.items():
            if name.startswith("__") or name in _INTERNAL:
                continue
            if name in bound or name not in protected:
                namespace[name] = value
                bound.add(name)
