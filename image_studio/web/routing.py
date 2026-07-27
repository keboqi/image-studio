"""Framework-neutral route ordering helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def promote_routes_before_fallback(app: Any, names: Iterable[str]) -> None:
    routes = getattr(getattr(app, "router", None), "routes", None)
    if not isinstance(routes, list):
        return
    names = set(names)
    promoted = [route for route in routes if getattr(route, "name", "") in names]
    if not promoted:
        return
    remaining = [route for route in routes if getattr(route, "name", "") not in names]
    insertion = len(remaining)
    for index, route in enumerate(remaining):
        path = str(getattr(route, "path", ""))
        if path in {"/{path:path}", "/{full_path:path}"} or path.endswith("{path:path}"):
            insertion = index
            break
    routes[:] = remaining[:insertion] + promoted + remaining[insertion:]

