"""Translate domain exceptions at the Gradio boundary."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from ..errors import AppError

F = TypeVar("F", bound=Callable[..., Any])


def ui_endpoint(fn: F) -> F:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except AppError as exc:
            import gradio as gr

            raise gr.Error(str(exc)) from exc

    return wrapper  # type: ignore[return-value]
