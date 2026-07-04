"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.progress import NO_PROGRESS

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def run_zimage(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    pid_enabled: bool,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress=NO_PROGRESS,
) -> tuple[Image.Image, str]:
    prompt = require_prompt(prompt)
    width, height = validate_dims(width, height)
    seed = normalize_seed(seed)
    precision = _PRECISION
    rank = 128
    progress(0.1, desc="Loading model...")
    pipe = get_zimage_pipe()
    gen = make_cuda_generator(seed)

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
    )
    if gen is not None:
        kwargs["generator"] = gen

    if pid_enabled:
        result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h = _decode_zimage_family_with_pid(
            pipe, prompt, kwargs, width, height,
            pid_ckpt, pid_steps, pid_cfg, seed, progress,
        )
        status = ok_status(
            elapsed,
            f"{width}x{height} -> {pid_out_w}x{pid_out_h}",
            f"steps {steps}",
            f"g {guidance}",
            f"rank {rank}",
            precision,
            f"{_pid_checkpoint_label(PID_BACKBONE_ZIMAGE, pid_ckpt_type)} 4x",
            f"PiD steps {pid_steps}",
            f"PiD cfg {pid_cfg}",
        )
        return finalize_image_result("zimage_pid", result, status, seed)

    progress(0.3, desc="Generating...")
    result, elapsed = timed_result(lambda: pipe(**kwargs).images[0])
    status = ok_status(
        elapsed,
        f"{width}x{height}",
        f"steps {steps}",
        f"g {guidance}",
        f"rank {rank}",
        precision,
    )
    return finalize_image_result("zimage", result, status, seed)

def run_zimage_full(
    prompt: str,
    neg_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    pid_enabled: bool,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress=NO_PROGRESS,
) -> tuple[Image.Image, str]:
    """Run the full (non-distilled) Z-Image pipeline for best quality."""
    prompt = require_prompt(prompt)
    width, height = validate_dims(width, height)
    seed = normalize_seed(seed)
    progress(0.1, desc="Loading model...")
    pipe = get_zimage_full_pipe()
    gen = make_cuda_generator(seed)

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        cfg_normalization=False,
    )
    if neg_prompt and neg_prompt.strip():
        kwargs["negative_prompt"] = neg_prompt
    if gen is not None:
        kwargs["generator"] = gen

    if not pid_enabled:
        progress(0.3, desc="Generating...")
        result, elapsed = timed_result(lambda: pipe(**kwargs).images[0])
        status = ok_status(
            elapsed,
            f"{width}x{height}",
            f"steps {steps}",
            f"guidance {guidance}",
            "bfloat16 (full)",
        )
        return finalize_image_result("zimage_full", result, status, seed)

    result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h = _decode_zimage_family_with_pid(
        pipe, prompt, kwargs, width, height,
        pid_ckpt, pid_steps, pid_cfg, seed, progress,
    )
    status = ok_status(
        elapsed,
        f"{width}x{height} -> {pid_out_w}x{pid_out_h}",
        f"steps {steps}",
        f"guidance {guidance}",
        f"{_pid_checkpoint_label(PID_BACKBONE_ZIMAGE, pid_ckpt_type)} 4x",
        f"PiD steps {pid_steps}",
        f"PiD cfg {pid_cfg}",
    )
    return finalize_image_result("zimage_full_pid", result, status, seed)

__all__ = (
    'run_zimage',
    'run_zimage_full',
)
_seal_runtime_module(globals())
