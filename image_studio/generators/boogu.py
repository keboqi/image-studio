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

def run_boogu_generate(
    prompt: str,
    neg_prompt: str,
    width: int,
    height: int,
    version: str,
    steps: int,
    base_guidance: float,
    seed: int,
    progress=NO_PROGRESS,
) -> tuple[str, str, str]:
    prompt = require_prompt(prompt)
    width, height = validate_boogu_dims(width, height)
    seed = resolve_seed(seed)
    version = _normalize_boogu_generation_version(version)
    model_key = BOOGU_IMAGE_VERSION_KEYS[version]
    max_pixels, max_side = _boogu_image_limits(width, height)

    progress(0.1, desc=f"Loading Boogu-Image {version}...")
    pipe = get_boogu_image_pipe(model_key)
    generator = _boogu_generator(seed)

    if version == BOOGU_IMAGE_VERSION_TURBO:
        steps = max(1, int(steps))
        kwargs = dict(
            instruction=[prompt],
            negative_instruction="",
            empty_instruction="",
            height=height,
            width=width,
            max_input_image_pixels=max_pixels,
            max_input_image_side_length=max_side,
            num_inference_steps=steps,
            **_boogu_text_encoder_kwargs(),
            text_guidance_scale=1.0,
            image_guidance_scale=1.0,
            empty_instruction_guidance_scale=0.0,
            use_dmd_student_inference=True,
            dmd_conditioning_sigma=0.001,
            generator=generator,
        )
        status_parts = [f"{width}x{height}", "Boogu-Image Turbo", f"steps {steps}"]
    else:
        steps = max(1, int(steps))
        guidance = float(base_guidance)
        kwargs = dict(
            instruction=prompt,
            negative_instruction=(neg_prompt or "").strip(),
            height=height,
            width=width,
            max_input_image_pixels=max_pixels,
            max_input_image_side_length=max_side,
            num_inference_steps=steps,
            **_boogu_text_encoder_kwargs(),
            text_guidance_scale=guidance,
            image_guidance_scale=1.0,
            generator=generator,
        )
        status_parts = [f"{width}x{height}", "Boogu-Image Base", f"steps {steps}", f"guidance {guidance}"]

    progress(0.3, desc=f"Generating with Boogu-Image {version}...")
    result, elapsed = _run_boogu_pipeline(pipe, kwargs)
    return finalize_image_result("boogu_gen", result, ok_status(elapsed, *status_parts), seed, always_seed=True)

def run_boogu_edit(
    img1: Any,
    img2: Any,
    img3: Any,
    prompt: str,
    neg_prompt: str,
    version: str,
    width: int,
    height: int,
    keep_original_aspect: bool,
    steps: int,
    text_guidance: float,
    image_guidance: float,
    seed: int,
    progress=NO_PROGRESS,
) -> tuple[str, str, str]:
    prompt = require_prompt(prompt)
    images = collect_rgb_images(img1, img2, img3)
    if not images:
        raise UserInputError("Upload one source image for Boogu-Image Edit.")
    if len(images) > 1:
        raise UserInputError("Boogu-Image Edit currently supports one reference image per edit.")

    seed = resolve_seed(seed)
    version = _normalize_boogu_edit_version(version)
    model_key = BOOGU_IMAGE_EDIT_VERSION_KEYS[version]
    output_width = None
    output_height = None
    if not keep_original_aspect:
        output_width, output_height = validate_boogu_dims(width, height)

    progress(0.1, desc=f"Loading Boogu-Image Edit {version}...")
    pipe = get_boogu_image_pipe(model_key)
    generator = _boogu_generator(seed)
    max_pixels = BOOGU_IMAGE_MAX_PIXELS if keep_original_aspect else output_width * output_height
    max_side = BOOGU_IMAGE_MAX_SIDE * 2 if keep_original_aspect else 2 * max(output_width, output_height)
    steps = max(1, int(steps))

    kwargs = dict(
        instruction=[prompt],
        input_images=[images],
        height=output_height,
        width=output_width,
        max_input_image_pixels=max_pixels,
        max_input_image_side_length=max_side,
        align_res=bool(keep_original_aspect),
        num_inference_steps=steps,
        **_boogu_text_encoder_kwargs(),
        generator=generator,
    )

    if version == BOOGU_IMAGE_VERSION_TURBO:
        kwargs.update(
            negative_instruction="",
            empty_instruction="",
            text_guidance_scale=1.0,
            image_guidance_scale=1.0,
            empty_instruction_guidance_scale=0.0,
            use_dmd_student_inference=True,
            dmd_conditioning_sigma=0.0,
        )
        status_parts = [f"steps {steps}"]
    else:
        text_guidance = float(text_guidance)
        image_guidance = float(image_guidance)
        kwargs.update(
            negative_instruction=(neg_prompt or "").strip(),
            text_guidance_scale=text_guidance,
            image_guidance_scale=image_guidance,
        )
        status_parts = [
            f"steps {steps}",
            f"text guidance {text_guidance}",
            f"image guidance {image_guidance}",
        ]

    progress(0.3, desc=f"Editing with Boogu-Image {version}...")
    result, elapsed = _run_boogu_pipeline(pipe, kwargs)
    size_part = f"{result.width}x{result.height}" if keep_original_aspect else f"{output_width}x{output_height}"
    status = ok_status(
        elapsed,
        size_part,
        f"Boogu-Image Edit {version}",
        *status_parts,
    )
    return finalize_image_result("boogu_edit", result, status, seed, always_seed=True)

__all__ = (
    'run_boogu_generate',
    'run_boogu_edit',
)
_seal_runtime_module(globals())
