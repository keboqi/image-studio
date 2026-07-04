"""Pure request validation shared by UI and API entry points."""

from __future__ import annotations

import math

from .errors import UserInputError

MAX_PIXELS = 4096 * 4096


def validate_dims(width: int | None, height: int | None) -> tuple[int, int]:
    try:
        width, height = int(width), int(height)
    except (TypeError, ValueError) as exc:
        raise UserInputError("Width and height must be integers.") from exc
    if width % 16 or height % 16:
        raise UserInputError("Width/height must be multiples of 16")
    if width * height > MAX_PIXELS:
        raise UserInputError("Max total pixels: 16M")
    return width, height


def validate_ideogram_dims(width: int | None, height: int | None) -> tuple[int, int]:
    width, height = validate_dims(width, height)
    if not (256 <= width <= 4096 and 256 <= height <= 4096):
        raise UserInputError("Ideogram 4 width/height must be between 256 and 4096.")
    if max(width / height, height / width) > 6:
        raise UserInputError("Ideogram 4 supports aspect ratios up to 6:1 or 1:6.")
    return width, height


def validate_boogu_dims(width: int | None, height: int | None) -> tuple[int, int]:
    width, height = validate_dims(width, height)
    if max(width, height) > 2048 or width * height > 2048 * 2048:
        raise UserInputError("Boogu-Image supports up to 2K native output (max 2048x2048).")
    return width, height


def snap_ltx_audio_video_frames(frames: int, *, max_frames: int = 1201) -> int:
    maximum = ((max_frames - 9) // 16) * 16 + 9
    snapped = math.floor(((int(frames) - 9) / 16) + 0.5) * 16 + 9
    return max(9, min(maximum, snapped))


def is_ltx_audio_video_frame_count(frames: int, *, max_frames: int = 1201) -> bool:
    frames = int(frames)
    return 9 <= frames <= max_frames and (frames - 9) % 16 == 0
