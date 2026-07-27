"""Lightweight value objects shared by composition and GPU adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HiDreamSpec:
    model_id: str
    label: str
    short_label: str
    steps: int
    guidance_scale: float
    shift: float
    scheduler_name: str
    use_default_timesteps: bool
    noise_scale_start: float
    noise_scale_end: float
    noise_clip_std: float


@dataclass(frozen=True)
class PIDCheckpointSpec:
    registry_key: str
    experiment: str
    relative_checkpoint_path: str
    label: str


__all__ = ("HiDreamSpec", "PIDCheckpointSpec")
