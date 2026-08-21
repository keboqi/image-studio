"""SenseNova U1.5 text-to-image and instruction-edit generation."""

from __future__ import annotations

import math
from typing import Any

from image_studio.errors import UserInputError
from image_studio.progress import NO_PROGRESS
from image_studio.runtime_access import runtime_namespace as _runtime


def _sensenova_aspect_size(source_width: int, source_height: int, target_pixels: int) -> tuple[int, int]:
    ratio = source_width / source_height
    height = math.sqrt(target_pixels / ratio)
    width = height * ratio
    return max(256, round(width / 32) * 32), max(256, round(height / 32) * 32)


def run_sensenova_generate(
    prompt: str,
    width: int,
    height: int,
    quality: str,
    seed: int,
    progress=NO_PROGRESS,
):
    prompt = _runtime().require_prompt(prompt)
    width, height = _runtime().validate_sensenova_dims(width, height)
    seed = _runtime().resolve_seed(seed)
    model_key = _runtime().SENSENOVA_QUALITY_KEYS.get(
        quality, _runtime().MODEL_SENSENOVA_FAST
    )
    progress(0.1, desc="Loading SenseNova U1.5...")
    bundle = _runtime().get_sensenova_pipe(model_key)
    progress(0.3, desc=f"Generating with SenseNova U1.5 {bundle['label']}...")

    def infer():
        with _runtime().torch.inference_mode():
            output = bundle["model"].t2i_generate(
                bundle["tokenizer"],
                prompt,
                image_size=(width, height),
                cfg_scale=bundle["cfg_scale"],
                cfg_norm="none",
                timestep_shift=3.0,
                cfg_interval=(0.0, 1.0),
                num_steps=bundle["steps"],
                batch_size=1,
                seed=seed,
                think_mode=False,
            )
        return _runtime().sensenova_output_to_pil(output)

    image, elapsed = _runtime().timed_result(infer)
    status = _runtime().ok_status(
        elapsed, f"{image.width}x{image.height}", bundle["label"], f"CFG {bundle['cfg_scale']:g}"
    )
    return _runtime().finalize_image_result(
        "sensenova_gen", image, status, seed, always_seed=True
    )


def run_sensenova_edit(
    img1: Any,
    img2: Any,
    img3: Any,
    prompt: str,
    width: int,
    height: int,
    keep_original_aspect: bool,
    seed: int,
    progress=NO_PROGRESS,
):
    prompt = _runtime().require_prompt(prompt)
    images = [_runtime().coerce_rgb_image(image) for image in (img1, img2, img3) if image is not None]
    if not images:
        raise UserInputError("Upload at least one source image.")
    if keep_original_aspect and len(images) == 1:
        width, height = _sensenova_aspect_size(images[0].width, images[0].height, int(width) * int(height))
    width, height = _runtime().validate_sensenova_dims(width, height)
    seed = _runtime().resolve_seed(seed)
    progress(0.1, desc="Loading SenseNova U1.5 50-step base...")
    bundle = _runtime().get_sensenova_pipe(_runtime().MODEL_SENSENOVA_BASE)
    progress(0.3, desc="Editing with SenseNova U1.5...")

    def infer():
        with _runtime().torch.inference_mode():
            output = bundle["model"].it2i_generate(
                bundle["tokenizer"],
                prompt,
                images,
                image_size=(width, height),
                cfg_scale=4.0,
                img_cfg_scale=1.0,
                cfg_norm="none",
                timestep_shift=3.0,
                cfg_interval=(0.0, 1.0),
                num_steps=50,
                batch_size=1,
                seed=seed,
                think_mode=False,
            )
        return _runtime().sensenova_output_to_pil(output)

    image, elapsed = _runtime().timed_result(infer)
    status = _runtime().ok_status(
        elapsed, f"{len(images)} ref(s)", f"{image.width}x{image.height}", "50-step base"
    )
    return _runtime().finalize_image_result(
        "sensenova_edit", image, status, seed, always_seed=True
    )


__all__ = ("_sensenova_aspect_size", "run_sensenova_edit", "run_sensenova_generate")
