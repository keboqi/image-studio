"""Translate domain exceptions at the Gradio boundary."""

from __future__ import annotations

import functools
from collections.abc import Callable

from ..errors import AppError


def ui_endpoint[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except AppError as exc:
            import gradio as gr

            raise gr.Error(str(exc)) from exc

    return wrapper
