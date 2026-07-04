"""Typed component bundles with temporary mapping compatibility."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import fields
from typing import Any


class ComponentSet(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        if key not in {item.name for item in fields(self)}:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return (item.name for item in fields(self))

    def __len__(self) -> int:
        return len(fields(self))
