"""Extracted runtime implementation."""

from __future__ import annotations

from PIL import Image

# --- extracted runtime implementation ---
from image_studio.progress import NO_PROGRESS
from image_studio.runtime_access import runtime_namespace as _runtime


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
    prompt = _runtime().require_prompt(prompt)
    width, height = _runtime().validate_dims(width, height)
    seed = _runtime().normalize_seed(seed)
    precision = _runtime()._PRECISION
    rank = 128
    progress(0.1, desc="Loading model...")
    pipe = _runtime().get_zimage_pipe()
    gen = _runtime().make_cuda_generator(seed)

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
        result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h = _runtime()._decode_zimage_family_with_pid(
            pipe, prompt, kwargs, width, height,
            pid_ckpt, pid_steps, pid_cfg, seed, progress,
        )
        status = _runtime().ok_status(
            elapsed,
            f"{width}x{height} -> {pid_out_w}x{pid_out_h}",
            f"steps {steps}",
            f"g {guidance}",
            f"rank {rank}",
            precision,
            f"{_runtime()._pid_checkpoint_label(_runtime().PID_BACKBONE_ZIMAGE, pid_ckpt_type)} 4x",
            f"PiD steps {pid_steps}",
            f"PiD cfg {pid_cfg}",
        )
        return _runtime().finalize_image_result("zimage_pid", result, status, seed)

    progress(0.3, desc="Generating...")
    result, elapsed = _runtime().timed_result(lambda: pipe(**kwargs).images[0])
    status = _runtime().ok_status(
        elapsed,
        f"{width}x{height}",
        f"steps {steps}",
        f"g {guidance}",
        f"rank {rank}",
        precision,
    )
    return _runtime().finalize_image_result("zimage", result, status, seed)

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
    prompt = _runtime().require_prompt(prompt)
    width, height = _runtime().validate_dims(width, height)
    seed = _runtime().normalize_seed(seed)
    progress(0.1, desc="Loading model...")
    pipe = _runtime().get_zimage_full_pipe()
    gen = _runtime().make_cuda_generator(seed)

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
        result, elapsed = _runtime().timed_result(lambda: pipe(**kwargs).images[0])
        status = _runtime().ok_status(
            elapsed,
            f"{width}x{height}",
            f"steps {steps}",
            f"guidance {guidance}",
            "bfloat16 (full)",
        )
        return _runtime().finalize_image_result("zimage_full", result, status, seed)

    result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h = _runtime()._decode_zimage_family_with_pid(
        pipe, prompt, kwargs, width, height,
        pid_ckpt, pid_steps, pid_cfg, seed, progress,
    )
    status = _runtime().ok_status(
        elapsed,
        f"{width}x{height} -> {pid_out_w}x{pid_out_h}",
        f"steps {steps}",
        f"guidance {guidance}",
        f"{_runtime()._pid_checkpoint_label(_runtime().PID_BACKBONE_ZIMAGE, pid_ckpt_type)} 4x",
        f"PiD steps {pid_steps}",
        f"PiD cfg {pid_cfg}",
    )
    return _runtime().finalize_image_result("zimage_full_pid", result, status, seed)

__all__ = (
    'run_zimage',
    'run_zimage_full',
)
