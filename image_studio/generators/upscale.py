"""Upscale generator entry points."""

from __future__ import annotations

from typing import Any


def run_upscale(*args: Any, **kwargs: Any):
    from image_studio.pipelines.seedvr2 import run_upscale as implementation

    return implementation(*args, **kwargs)


def run_video_upscale(*args: Any, **kwargs: Any):
    from image_studio.pipelines.seedvr2 import run_video_upscale as implementation

    return implementation(*args, **kwargs)
