"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError
from image_studio.progress import NO_PROGRESS

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def run_generate(
    prompt: str,
    neg_prompt: str,
    width: int,
    height: int,
    cfg: float,
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
    progress(0.1, desc="Loading model...")
    pipe = get_gen_pipe()
    gen = make_cuda_generator(seed)
    kwargs = dict(prompt=prompt, width=width, height=height,
                  num_inference_steps=LIGHTNING_STEPS, true_cfg_scale=cfg)
    if neg_prompt and neg_prompt.strip():
        kwargs["negative_prompt"] = neg_prompt
    if gen:
        kwargs["generator"] = gen

    if pid_enabled:
        result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h = _decode_qwen_with_pid(
            pipe, prompt, kwargs, width, height,
            pid_ckpt, pid_steps, pid_cfg, seed, progress,
        )
        status = ok_status(
            elapsed,
            f"{width}x{height} -> {pid_out_w}x{pid_out_h}",
            f"CFG {cfg}",
            f"{_pid_checkpoint_label(PID_BACKBONE_QWEN, pid_ckpt_type)} 4x",
            f"PiD steps {pid_steps}",
            f"PiD cfg {pid_cfg}",
        )
        return finalize_image_result("gen_pid", result, status, seed)

    progress(0.3, desc="Generating...")
    result, elapsed = timed_result(lambda: pipe(**kwargs).images[0])
    status = ok_status(elapsed, f"{width}x{height}", f"CFG {cfg}")
    return finalize_image_result("gen", result, status, seed)

def run_edit(img1: Any, img2: Any, img3: Any, prompt: str, neg_prompt: str, cfg: float, seed: int, progress=NO_PROGRESS) -> tuple[Image.Image, str]:
    prompt = require_prompt(prompt)
    seed = normalize_seed(seed)
    images = collect_rgb_images(img1, img2, img3)
    if not images:
        raise UserInputError("Upload at least one source image.")
    progress(0.1, desc="Loading model...")
    pipe = get_edit_pipe()
    gen = make_cuda_generator(seed)
    kwargs = dict(image=images if len(images) > 1 else images[0], prompt=prompt,
                  negative_prompt=neg_prompt.strip() if neg_prompt and neg_prompt.strip() else " ",
                  num_inference_steps=LIGHTNING_STEPS, true_cfg_scale=cfg)
    if gen:
        kwargs["generator"] = gen

    progress(0.3, desc="Generating...")
    result, elapsed = timed_result(lambda: pipe(**kwargs).images[0])
    status = ok_status(elapsed, f"{len(images)} image(s)", f"CFG {cfg}")
    return finalize_image_result("edit", result, status, seed)

__all__ = (
    'run_generate',
    'run_edit',
)
_seal_runtime_module(globals())
