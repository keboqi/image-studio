"""Typed environment parsing.

The helpers accept an explicit mapping so :class:`AppConfig` can read one
environment snapshot and tests do not need to mutate process state.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

log = logging.getLogger(__name__)


def env_str(env: Mapping[str, str], name: str, default: str = "") -> str:
    value = env.get(name)
    return default if value is None else str(value).strip()


def env_int(
    env: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    try:
        value = int(env.get(name, str(default)) or default)
    except (TypeError, ValueError):
        log.warning("Invalid %s; using default %s.", name, default)
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_float(
    env: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        value = float(env.get(name, str(default)) or default)
    except (TypeError, ValueError):
        log.warning("Invalid %s; using default %s.", name, default)
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def env_bool(env: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    log.warning("Invalid %s=%r; using default %s.", name, value, default)
    return default
