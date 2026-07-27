"""Extracted runtime implementation."""

from __future__ import annotations

from PIL import Image

# --- extracted runtime implementation ---
from image_studio.errors import UserInputError
from image_studio.progress import NO_PROGRESS
from image_studio.runtime_access import runtime_namespace as _runtime


def validate_krea2_dims(w: int, h: int) -> tuple[int, int]:
    width, height = _runtime().validate_dims(w, h)
    if max(width, height) > 2048:
        raise UserInputError("Krea2 Turbo is tuned for 1K-2K output; use a max side up to 2048.")
    if width * height > 2048 * 2048:
        raise UserInputError("Krea2 Turbo supports up to roughly 2048x2048 total pixels.")
    return width, height

def _krea2_guidance_scale(cfg: float) -> float:
    cfg = float(cfg)
    return 1.0 if cfg <= 0.0 else cfg

def _decode_krea2_with_pid(
    image: Image.Image,
    latent: _runtime().torch.Tensor,
    prompt: str,
    width: int,
    height: int,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress,
) -> tuple[Image.Image, float, str, int, int]:
    """PiD 4x decode a Krea2 result using the Qwen Image VAE backbone.

    *latent* is the raw KSampler output from ComfyUI (saved by SaveLatent).
    *image* is the VAE-decoded baseline (saved by SaveImage), used as the
    low-resolution conditioning input for PiD.
    """
    if not _runtime().torch.cuda.is_available():
        raise UserInputError("PiD decoding requires CUDA.")
    pid_out_w, pid_out_h = _runtime()._validate_pid_dims(width, height)
    pid_ckpt_type = _runtime()._resolve_pid_ckpt_type(_runtime().PID_BACKBONE_QWEN, pid_ckpt, width, height)

    # Convert PIL baseline image to [-1, 1] tensor matching PiD expectations.
    baseline_np = _runtime().np.array(image.convert("RGB")).astype(_runtime().np.float32) / 255.0
    baseline_01 = _runtime().torch.from_numpy(baseline_np).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    baseline_neg1_1 = (baseline_01 * 2.0 - 1.0).to(dtype=_runtime().torch.bfloat16, device="cuda")

    # Ensure latent is on cuda in the right dtype.
    # ComfyUI SaveLatent preserves the Qwen VAE's CogVideoX 5D format
    # (B, C, T, H, W) with T=1; squeeze it to the 4D (B, C, H, W) PiD expects.
    latent = latent.to(dtype=_runtime().torch.bfloat16, device="cuda")
    if latent.ndim == 5:
        if latent.shape[2] == 1:
            latent = latent.squeeze(2)
        else:
            raise UserInputError(
                f"Krea2 latent has unexpected temporal dim: {tuple(latent.shape)}"
            )
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)

    def decode_with_pid() -> Image.Image:
        progress(0.68, desc="Loading Krea2 Qwen PiD decoder...")
        pid_model = _runtime().get_qwen_pid_decoder(pid_ckpt_type)
        lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
        data_batch = _runtime()._pid_data_batch(
            pid_model,
            prompt,
            latent,
            0.0,  # sigma=0: ComfyUI KSampler outputs denoised latents
            baseline_neg1_1,
        )
        progress(0.78, desc="PiD 4x decoding Krea2 latent...")
        return _runtime()._pid_generate_image(
            pid_model,
            data_batch,
            pid_cfg=pid_cfg,
            pid_steps=pid_steps,
            seed=seed,
            image_size=(lq_h * _runtime().PID_SCALE, lq_w * _runtime().PID_SCALE),
        )

    result, elapsed = _runtime().timed_result(decode_with_pid)
    return result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h

def run_krea2_generate(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    pid_enabled: bool,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress=NO_PROGRESS,
) -> tuple[str, str, str]:
    prompt = _runtime().require_prompt(prompt)
    width, height = validate_krea2_dims(width, height)
    seed = _runtime().resolve_seed(seed)
    steps = max(1, int(steps or _runtime().KREA2_DEFAULT_STEPS))
    guidance_scale = _krea2_guidance_scale(cfg)
    pid_enabled = bool(pid_enabled)
    if pid_enabled:
        _runtime()._validate_pid_dims(width, height)

    progress(0.1, desc="Starting Krea2 ComfyUI backend...")
    _runtime()._krea2_comfy_service.ensure_running()

    if pid_enabled:
        progress(0.25, desc=f"Generating Krea2 + latent ({steps} steps)...")
        image, latent, gen_elapsed = _runtime()._krea2_comfy_service.generate_image_with_latent(
            prompt, width, height, steps, guidance_scale, seed,
        )
        progress(0.60, desc="Preparing Krea2 PiD 4x decode...")
        result, pid_elapsed, pid_ckpt_type, pid_out_w, pid_out_h = _decode_krea2_with_pid(
            image, latent, prompt, width, height,
            pid_ckpt, pid_steps, pid_cfg, seed, progress,
        )
        elapsed = gen_elapsed + pid_elapsed
        cfg_label = "CFG disabled" if guidance_scale == 1.0 else f"CFG {guidance_scale:g}"
        status = _runtime().ok_status(
            elapsed,
            f"{width}x{height} -> {pid_out_w}x{pid_out_h}",
            "Krea2 Turbo ComfyUI",
            f"steps {steps}",
            cfg_label,
            f"{_runtime()._pid_checkpoint_label(_runtime().PID_BACKBONE_QWEN, pid_ckpt_type)} 4x",
            f"PiD steps {pid_steps}",
            f"PiD cfg {pid_cfg}",
        )
        return _runtime().finalize_image_result("krea2_pid", result, status, seed, always_seed=True)

    progress(0.35, desc=f"Generating Krea2 ({steps} steps)...")
    result, elapsed = _runtime()._krea2_comfy_service.generate_image(
        prompt, width, height, steps, guidance_scale, seed,
    )
    cfg_label = "CFG disabled" if guidance_scale == 1.0 else f"CFG {guidance_scale:g}"
    status = _runtime().ok_status(
        elapsed,
        f"{width}x{height}",
        "Krea2 Turbo ComfyUI",
        f"steps {steps}",
        cfg_label,
    )
    return _runtime().finalize_image_result("krea2", result, status, seed, always_seed=True)

__all__ = (
    'validate_krea2_dims',
    '_krea2_guidance_scale',
    '_decode_krea2_with_pid',
    'run_krea2_generate',
)
