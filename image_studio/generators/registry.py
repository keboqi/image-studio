"""Explicit generator registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from .base import Generator

RequestT = TypeVar("RequestT")


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


class RequestHandlerRegistry(Generic[RequestT]):
    """Compatibility registry for the existing flat Gradio request adapters."""

    def __init__(self, default: Callable[..., Any]) -> None:
        self.default = default
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(self, mode: str, handler: Callable[..., Any]) -> None:
        if mode in self._handlers:
            raise ValueError(f"Request handler already registered: {mode}")
        self._handlers[mode] = handler

    def modes(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def dispatch(self, mode: str, request: RequestT, *, progress: Any = None) -> Any:
        return self._handlers.get(mode, self.default)(request, progress=progress)
