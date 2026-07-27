"""Extracted runtime implementation."""

from __future__ import annotations

from typing import Any

from PIL import Image

# --- extracted runtime implementation ---
from image_studio.errors import UserInputError
from image_studio.progress import NO_PROGRESS
from image_studio.runtime_access import runtime_namespace as _runtime


def _hidream_seed(seed: int) -> int:
    return _runtime().resolve_seed(seed)

def _save_hidream_ref_images(images: list[Any]) -> list[str]:
    temp_dir = _runtime().os.path.join(_runtime().OUTPUT_DIR, ".hidream_refs")
    _runtime().os.makedirs(temp_dir, exist_ok=True)
    _runtime().cleanup_old_files(temp_dir, 3600)

    paths = []
    for image in images:
        pil = _runtime().coerce_rgb_image(image)
        fd, path = _runtime().tempfile.mkstemp(prefix="hidream_ref_", suffix=".png", dir=temp_dir)
        _runtime().os.close(fd)
        pil.save(path)
        paths.append(path)
    return paths

def _cleanup_hidream_ref_images(paths: list[str]):
    for path in paths:
        try:
            _runtime().os.remove(path)
        except OSError:
            pass

def _run_hidream_o1(
    model_key: str,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    ref_image_paths: list[str] | None = None,
    keep_original_aspect: bool = False,
    progress=NO_PROGRESS,
) -> Image.Image:
    width, height = _runtime().validate_dims(width, height)
    seed = _hidream_seed(seed)
    spec = _runtime()._get_hidream_o1_spec(model_key)
    progress(0.1, desc=f"Loading {spec.short_label}...")
    bundle = _runtime().get_hidream_o1_pipe(model_key)
    _runtime()._repair_hidream_rope_buffers(bundle["model"])
    progress(0.3, desc=f"Generating with {spec.short_label}...")
    return bundle["generate_image"](
        model=bundle["model"],
        processor=bundle["processor"],
        prompt=prompt,
        ref_image_paths=ref_image_paths,
        height=height,
        width=width,
        num_inference_steps=bundle["steps"],
        guidance_scale=bundle["guidance_scale"],
        shift=bundle["shift"],
        timesteps_list=bundle["timesteps_list"],
        scheduler_name=bundle["scheduler_name"],
        seed=seed,
        noise_scale_start=bundle["noise_scale_start"],
        noise_scale_end=bundle["noise_scale_end"],
        noise_clip_std=bundle["noise_clip_std"],
        keep_original_aspect=keep_original_aspect,
    )

def run_hidream_generate(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    model_key: str | None = None,
    progress=NO_PROGRESS,
) -> tuple[Image.Image, str]:
    model_key = model_key or _runtime().MODEL_HIDREAM_O1_DEV
    prompt = _runtime().require_prompt(prompt)
    seed = _hidream_seed(seed)
    spec = _runtime()._get_hidream_o1_spec(model_key)
    result, elapsed = _runtime().timed_result(
        lambda: _run_hidream_o1(model_key, prompt, width, height, seed, progress=progress)
    )
    status = _runtime().ok_status(
        elapsed,
        f"{result.width}x{result.height}",
        spec.short_label,
        f"steps {spec.steps}",
    )
    return _runtime().finalize_image_result("hidream_gen", result, status, seed, always_seed=True)

def run_hidream_edit(
    img1: Any,
    img2: Any,
    img3: Any,
    prompt: str,
    width: int,
    height: int,
    keep_original_aspect: bool,
    seed: int,
    model_key: str | None = None,
    progress=NO_PROGRESS,
) -> tuple[Image.Image, str]:
    model_key = model_key or _runtime().MODEL_HIDREAM_O1_DEV
    prompt = _runtime().require_prompt(prompt)
    images = [i for i in [img1, img2, img3] if i is not None]
    if not images:
        raise UserInputError("Upload at least one source image.")

    seed = _hidream_seed(seed)
    spec = _runtime()._get_hidream_o1_spec(model_key)
    ref_paths = _save_hidream_ref_images(images)
    try:
        result, elapsed = _runtime().timed_result(
            lambda: _run_hidream_o1(
                model_key,
                prompt,
                width,
                height,
                seed,
                ref_image_paths=ref_paths,
                keep_original_aspect=bool(keep_original_aspect and len(ref_paths) == 1),
                progress=progress,
            )
        )
    finally:
        _cleanup_hidream_ref_images(ref_paths)

    status = _runtime().ok_status(
        elapsed,
        f"{len(images)} ref(s)",
        f"{result.width}x{result.height}",
        spec.short_label,
        f"steps {spec.steps}",
    )
    return _runtime().finalize_image_result("hidream_edit", result, status, seed, always_seed=True)

__all__ = (
    '_hidream_seed',
    '_save_hidream_ref_images',
    '_cleanup_hidream_ref_images',
    '_run_hidream_o1',
    'run_hidream_generate',
    'run_hidream_edit',
)
