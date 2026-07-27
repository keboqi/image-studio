"""Explicit generator registry."""

from __future__ import annotations

from typing import Any

from .base import Generator


class GeneratorRegistry:
    def __init__(self) -> None:
        self._generators: dict[str, Generator] = {}

    def register(self, generator: Generator) -> None:
        if generator.mode in self._generators:
            raise ValueError(f"Generator mode already registered: {generator.mode}")
        self._generators[generator.mode] = generator

    def get(self, mode: str) -> Generator:
        try:
            return self._generators[mode]
        except KeyError:
            raise KeyError(f"Unknown generator mode: {mode}") from None

    def modes(self) -> tuple[str, ...]:
        return tuple(self._generators)

    def generate(self, mode: str, request: Any, progress: Any = None) -> Any:
        return self.get(mode).generate(request, progress)
