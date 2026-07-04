"""Presentation-neutral progress callback used by domain operations."""

from __future__ import annotations

from typing import Any


class NullProgress:
    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def tqdm(self, iterable, *args: Any, **kwargs: Any):
        return iterable


NO_PROGRESS = NullProgress()
