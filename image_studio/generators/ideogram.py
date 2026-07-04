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

def _ideogram4_seed(seed: int | float | str) -> int:
    return resolve_seed(seed)

def _ideogram4_guidance_schedule(preset, cfg_one_final_steps: int = 0) -> tuple[float, ...]:
    """Return preset guidance with optional final CFG=1.0 branch-skipping steps."""
    schedule = list(preset.guidance_schedule)
    final_steps = max(0, min(int(cfg_one_final_steps or 0), len(schedule)))
    for idx in range(final_steps):
        schedule[idx] = 1.0
    return tuple(schedule)

def run_ideogram4_generate(
    prompt: str,
    width: int,
    height: int,
    pipeline: str,
    sampler_preset: str,
    upsampler: str,
    strip_prompt: bool,
    reuse_upsample_cache: bool,
    gemma_max_new_tokens: int,
    gemma_enable_thinking: bool,
    cfg_one_final_steps: int,
    lora_mode: str,
    lora_weight: str,
    lora_cond_strength: float,
    lora_uncond_strength: float,
    pid_enabled: bool,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    ideogram_api_key: str,
    seed: int,
    progress=NO_PROGRESS,
) -> tuple[Image.Image, str]:
    prompt = require_prompt(prompt)
    width, height = validate_ideogram4_dims(width, height)
    pipeline = _normalize_ideogram4_pipeline(pipeline)
    lora_mode = _normalize_ideogram4_lora_mode(lora_mode)
    lora_weight = _normalize_ideogram4_lora_weight(lora_weight)
    lora_cond_strength = float(lora_cond_strength or 0.0)
    lora_uncond_strength = float(lora_uncond_strength or 0.0)
    _ideogram4_unload_if_fused_lora_changed(
        pipeline,
        lora_mode,
        lora_weight,
        lora_cond_strength,
        lora_uncond_strength,
    )
    seed = _ideogram4_seed(seed)
    pid_enabled = bool(pid_enabled)
    if pid_enabled:
        _validate_pid_dims(width, height)

    source_prompt = prompt
    upsampled_prompt = prompt
    if upsampler != IDEOGRAM4_UPSAMPLE_NONE:
        progress(0.05, desc=f"Upsampling prompt ({upsampler})...")
        upsampled_prompt = _ideogram4_upsample_prompt(
            prompt=source_prompt,
            upsampler=upsampler,
            width=width,
            height=height,
            gemma_max_new_tokens=int(gemma_max_new_tokens or 2048),
            gemma_enable_thinking=bool(gemma_enable_thinking),
            reuse_cache=bool(reuse_upsample_cache),
            api_key=ideogram_api_key,
        )
    final_prompt = _ideogram4_normalize_caption_for_model(
        upsampled_prompt,
        width,
        height,
        strip_prompt=bool(strip_prompt),
    )

    progress(0.2, desc=f"Loading Ideogram 4 {_ideogram4_pipeline_label(pipeline)}...")
    pipe = get_ideogram4_pipe(pipeline)
    lora_status = _apply_ideogram4_lora(
        pipe,
        lora_mode,
        lora_weight,
        lora_cond_strength,
        lora_uncond_strength,
        progress=progress,
    )
    mods = _get_ideogram4()
    preset_key = IDEOGRAM4_SAMPLER_PRESETS.get(
        sampler_preset or "",
        IDEOGRAM4_SAMPLER_PRESETS["Turbo - 12 steps"],
    )
    preset = mods["PRESETS"][preset_key]
    guidance_schedule = _ideogram4_guidance_schedule(preset, cfg_one_final_steps)

    progress(0.35, desc="Generating with Ideogram 4...")
    t_diff = time.perf_counter()
    step_state = {"last_time": t_diff}

    def step_callback(step: int, total_steps: int):
        now = time.perf_counter()
        elapsed = now - t_diff
        step_time = now - step_state["last_time"]
        step_state["last_time"] = now
        avg_step_time = elapsed / max(1, step)
        remaining = avg_step_time * (total_steps - step)
        desc = (
            f"Generating image (Step {step}/{total_steps}) | "
            f"Step: {step_time:.2f}s | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {remaining:.1f}s"
        )
        progress(0.35 + 0.45 * (step / max(1, total_steps)), desc=desc)

    pipe_kwargs = dict(
        prompts=final_prompt,
        width=width,
        height=height,
        num_steps=preset.num_steps,
        guidance_schedule=guidance_schedule,
        mu=preset.mu,
        std=preset.std,
        seed=seed,
        raise_on_caption_issues=upsampler != IDEOGRAM4_UPSAMPLE_NONE,
        callback_on_step_end=step_callback,
    )
    if pid_enabled:
        pipe_kwargs["output_type"] = "latent"

    with make_ideogram4_sdpa_context():
        try:
            image_or_latents = pipe(**pipe_kwargs)
        except TypeError as exc:
            if "callback_on_step_end" not in str(exc):
                raise
            pipe_kwargs.pop("callback_on_step_end", None)
            image_or_latents = pipe(**pipe_kwargs)

    diffusion_elapsed = time.perf_counter() - t_diff
    log.info("Ideogram diffusion (%s) complete in %.2fs", sampler_preset, diffusion_elapsed)

    pid_parts: list[str] = []
    if pid_enabled:
        progress(0.82, desc="Preparing Ideogram PiD 4x decode...")
        result, pid_elapsed, pid_ckpt_type, pid_out_w, pid_out_h = _decode_ideogram4_with_pid(
            image_or_latents,
            final_prompt,
            source_prompt,
            width,
            height,
            pid_ckpt,
            pid_steps,
            pid_cfg,
            seed,
            progress,
        )
        elapsed = diffusion_elapsed + pid_elapsed
        pid_parts = [
            f"{width}x{height} -> {pid_out_w}x{pid_out_h}",
            f"{_pid_checkpoint_label(PID_BACKBONE_IDEOGRAM4, pid_ckpt_type)} 4x",
            f"PiD steps {pid_steps}",
            f"PiD cfg {pid_cfg}",
        ]
    else:
        result = image_or_latents[0]
        elapsed = diffusion_elapsed

    cfg_parts = []
    cfg_one_final_steps = int(cfg_one_final_steps or 0)
    if cfg_one_final_steps:
        cfg_parts.append(f"CFG=1 final {cfg_one_final_steps}")
    if lora_status:
        cfg_parts.append(lora_status)
    status = ok_status(
        elapsed,
        *(pid_parts or [f"{width}x{height}"]),
        f"Ideogram 4 {_ideogram4_pipeline_label(pipeline)}",
        sampler_preset,
        *cfg_parts,
        f"upsampler {upsampler}",
    )
    preview_path, status, raw_path = finalize_image_result("ideogram4", result, status, seed, always_seed=True)
    _write_ideogram4_prompt_metadata(
        raw_path,
        {
            "source_prompt": source_prompt,
            "upsampled_prompt": upsampled_prompt,
            "model_prompt": final_prompt,
            "editor_prompt": _ideogram4_editor_prompt_from_candidates(
                upsampled_prompt,
                final_prompt,
                source_prompt,
            ),
            "upsampler": upsampler,
            "pipeline": pipeline,
            "sampler_preset": sampler_preset,
            "width": width,
            "height": height,
            "strip_prompt": bool(strip_prompt),
            "reuse_upsample_cache": bool(reuse_upsample_cache),
            "gemma_max_new_tokens": int(gemma_max_new_tokens or 2048),
            "gemma_enable_thinking": bool(gemma_enable_thinking),
        },
    )
    return preview_path, status, raw_path

__all__ = (
    '_ideogram4_seed',
    '_ideogram4_guidance_schedule',
    'run_ideogram4_generate',
)
_seal_runtime_module(globals())
