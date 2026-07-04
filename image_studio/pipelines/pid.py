"""Pure PiD latent helpers shared by pipeline adapters."""

from __future__ import annotations
from image_studio.errors import UserInputError
from image_studio.infra.lazy_modules import LazyModuleGroup

from typing import Any
import threading

from ..errors import UserInputError


def validate_dims(width: int, height: int, *, scale: int = 4, max_low_side: int = 1024) -> tuple[int, int]:
    if max(width, height) > max_low_side:
        raise UserInputError(
            f"PiD {scale}x decode supports low-res sides up to {max_low_side} px."
        )
    return width * scale, height * scale


def patchify_flux2_raw_latents(raw_latents: Any) -> Any:
    """Pack ``(B, 32, H, W)`` Flux2 VAE latents into ``(B, 128, H/2, W/2)``."""
    if getattr(raw_latents, "ndim", None) != 4:
        raise UserInputError(
            f"Expected raw Flux2 latents with shape (B,C,H,W), got {getattr(raw_latents, 'shape', None)}"
        )
    batch, channels, height, width = raw_latents.shape
    if channels != 32:
        raise UserInputError(f"Flux2 PiD expects 32-channel raw VAE latents, got {channels}.")
    if height % 2 or width % 2:
        raise UserInputError(f"Flux2 raw latent height/width must be even, got {height}x{width}.")
    return (
        raw_latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(batch, channels * 4, height // 2, width // 2)
        .contiguous()
    )


# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

_pid_load_lock = threading.Lock()

def _ensure_pid():
    """Clone the PiD reference repo if its inference code is missing."""
    return _git_bootstrap.ensure(PID_REPO_SPEC)

def _import_pid_modules():
    """Lazy-import PiD helpers from the reference repo."""
    if not _ensure_pid():
        raise UserInputError(
            "PiD is unavailable. Clone the PiD repo into "
            f"{PID_DIR} or set PID_DIR to an existing checkout."
        )
    if PID_DIR not in sys.path:
        sys.path.insert(0, PID_DIR)

    try:
        from huggingface_hub import hf_hub_download
        from pid._src.utils.model_loader import load_model_from_checkpoint
    except ImportError as exc:
        raise UserInputError(
            "PiD dependencies are missing. Install the PiD requirements from "
            "the quick-start block at the top of this file. "
            f"Import error: {exc}"
        ) from exc

    return {
        "hf_hub_download": hf_hub_download,
        "load_model_from_checkpoint": load_model_from_checkpoint,
    }


_pid_module_group = LazyModuleGroup("PiD", lambda: True, _import_pid_modules)


def _get_pid_modules():
    return _pid_module_group.get()

def _pid_checkpoint_specs(backbone: str) -> dict[str, PIDCheckpointSpec]:
    try:
        return PID_CHECKPOINTS[backbone]
    except KeyError:
        raise UserInputError(f"Unknown PiD backbone: {backbone}") from None

def _pid_checkpoint_label(backbone: str, ckpt_type: str) -> str:
    return _pid_checkpoint_specs(backbone)[ckpt_type].label

def _resolve_pid_ckpt_type(backbone: str, requested: str, width: int, height: int) -> str:
    specs = _pid_checkpoint_specs(backbone)
    requested = (requested or PID_CKPT_AUTO).strip()
    if requested == PID_CKPT_AUTO:
        if PID_CKPT_2K in specs and max(width, height) <= 512:
            return PID_CKPT_2K
        if PID_CKPT_2KTO4K in specs:
            return PID_CKPT_2KTO4K
        return next(iter(specs))
    if requested not in specs:
        valid = ", ".join([PID_CKPT_AUTO, *specs.keys()])
        raise UserInputError(f"Unknown PiD checkpoint for {backbone}: {requested}. Valid: {valid}")
    return requested

def _validate_pid_dims(width: int, height: int) -> tuple[int, int]:
    out_w, out_h = width * PID_SCALE, height * PID_SCALE
    if max(width, height) > PID_MAX_LOW_SIDE:
        raise UserInputError(
            "PiD 4x decode supports low-res sides up to 1024 px "
            "(outputs up to 4096 px per side). Reduce width/height or disable PiD."
        )
    return out_w, out_h

def _ensure_pid_checkpoint(backbone: str, ckpt_type: str, force_download: bool = False) -> str:
    modules = _get_pid_modules()
    spec = _pid_checkpoint_specs(backbone)[ckpt_type]
    checkpoint_path = os.path.join(PID_DIR, *spec.relative_checkpoint_path.split("/"))
    if os.path.isfile(checkpoint_path) and not force_download:
        return checkpoint_path
    if force_download and os.path.isfile(checkpoint_path):
        try:
            os.remove(checkpoint_path)
        except OSError:
            pass

    log.info("Downloading %s checkpoint from %s...", spec.label, PID_HF_REPO)
    try:
        modules["hf_hub_download"](
            repo_id=PID_HF_REPO,
            filename=spec.relative_checkpoint_path,
            local_dir=PID_DIR,
            force_download=force_download,
        )
    except Exception as exc:
        raise UserInputError(
            f"Failed to download {spec.label} checkpoint from {PID_HF_REPO}. "
            "Check network access and Hugging Face permissions."
        ) from exc
    if not os.path.isfile(checkpoint_path):
        raise UserInputError(f"PiD checkpoint download did not create {checkpoint_path}")
    return checkpoint_path

def _ensure_pid_vae_asset(backbone: str) -> str:
    """Download the VAE asset used by a PiD latent-space checkpoint."""
    modules = _get_pid_modules()
    try:
        relative_path = PID_VAE_ASSETS[backbone]
    except KeyError:
        raise UserInputError(f"No PiD VAE asset is registered for {backbone}") from None

    vae_path = os.path.join(PID_DIR, *relative_path.split("/"))
    if os.path.isfile(vae_path):
        return vae_path

    log.info("Downloading PiD VAE asset %s from %s...", relative_path, PID_HF_REPO)
    try:
        modules["hf_hub_download"](
            repo_id=PID_HF_REPO,
            filename=relative_path,
            local_dir=PID_DIR,
        )
    except Exception as exc:
        raise UserInputError(
            f"Failed to download {relative_path} from {PID_HF_REPO}. "
            "PiD needs this latent-space VAE file."
        ) from exc
    if not os.path.isfile(vae_path):
        raise UserInputError(f"PiD VAE download did not create {vae_path}")
    return vae_path

def _ensure_pid_experiment_available(backbone: str, spec: PIDCheckpointSpec):
    if backbone != PID_BACKBONE_QWEN:
        return

    config_root = os.path.join(PID_DIR, "pid", "_src", "configs", "pid")
    for root, _dirs, files in os.walk(config_root):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    if spec.experiment in fh.read():
                        return
            except OSError:
                continue

    raise UserInputError(
        "Your PiD checkout does not include the Qwen Image PiD experiment "
        f"({spec.experiment}). Update or reclone {PID_DIR} from {PID_REPO}."
    )

def _pipeline_execution_device(pipe) -> torch.device:
    target_device = getattr(pipe, "_execution_device", None)
    if target_device is None or getattr(target_device, "type", None) == "meta":
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return target_device

def _zimage_decode_latent_01(pipe, latent: torch.Tensor) -> torch.Tensor:
    """VAE-decode normalized Z-Image latents to an RGB tensor in [0, 1]."""
    scale = float(pipe.vae.config.scaling_factor)
    shift = float(getattr(pipe.vae.config, "shift_factor", None) or 0.0)
    target_device = _pipeline_execution_device(pipe)
    raw_latent = (latent.to(target_device) / scale + shift).to(pipe.vae.dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(raw_latent, return_dict=False)[0]
    return (decoded * 0.5 + 0.5).clamp(0, 1)

def _qwen_extract_latent(pipe, raw_output, width: int, height: int) -> torch.Tensor:
    """Normalize Qwen Image output_type='latent' into (B, C, H, W)."""
    latent = raw_output.images
    if not isinstance(latent, torch.Tensor):
        raise UserInputError(f"Qwen Image returned an unexpected latent value: {type(latent)!r}")
    if latent.ndim == 3:
        latent = QwenImagePipeline._unpack_latents(
            latent,
            height=height,
            width=width,
            vae_scale_factor=pipe.vae_scale_factor,
        )
    if latent.ndim == 5:
        if latent.shape[2] != 1:
            raise UserInputError(f"Qwen Image returned an unexpected temporal latent shape: {tuple(latent.shape)}")
        latent = latent.squeeze(2)
    if latent.ndim != 4:
        raise UserInputError(f"Qwen Image returned an unexpected latent shape: {tuple(latent.shape)}")
    return latent

def _qwen_decode_latent_01(pipe, latent: torch.Tensor) -> torch.Tensor:
    """VAE-decode normalized Qwen Image latents to an RGB tensor in [0, 1]."""
    config = pipe.vae.config
    if not hasattr(config, "latents_mean") or not hasattr(config, "latents_std"):
        raise UserInputError("Qwen Image VAE config is missing latents_mean/latents_std.")

    target_device = _pipeline_execution_device(pipe)
    latent = latent.to(target_device)
    latents_mean = torch.tensor(config.latents_mean).view(1, -1, 1, 1).to(latent.device, latent.dtype)
    latents_std = torch.tensor(config.latents_std).view(1, -1, 1, 1).to(latent.device, latent.dtype)
    raw_latent = (latent * latents_std + latents_mean).unsqueeze(2).to(pipe.vae.dtype)
    with torch.no_grad():
        decoded = pipe.vae.decode(raw_latent, return_dict=False)[0]
    if decoded.ndim == 5:
        decoded = decoded[:, :, 0]
    return (decoded * 0.5 + 0.5).clamp(0, 1)

def _neg1_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a PiD output tensor in [-1, 1] to PIL."""
    while tensor.dim() > 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        elif tensor.shape[0] in (3, 4):
            tensor = tensor[:, 0]
        else:
            tensor = tensor[0]
    if tensor.dim() != 3:
        raise UserInputError(f"PiD returned an unsupported sample shape: {tuple(tensor.shape)}")
    if tensor.shape[0] not in (3, 4) and tensor.shape[-1] in (3, 4):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] not in (3, 4):
        raise UserInputError(f"PiD returned an unsupported channel layout: {tuple(tensor.shape)}")
    if tensor.shape[0] == 4:
        tensor = tensor[:3]
    array = ((tensor.float().clamp(-1, 1) + 1.0) * 127.5)
    array = array.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(array)

def resolve_seed(seed: int | float | str) -> int:
    seed = int(seed)
    if seed >= 0:
        return seed
    return int(torch.randint(0, 2**31 - 1, (1,), device="cpu").item())

def _scheduler_sigma(pipe) -> float:
    try:
        return float(pipe.scheduler.sigmas[-1].detach().float().cpu().item())
    except Exception:
        return 0.0

def _pid_data_batch(
    pid_model,
    prompt: str,
    latent: torch.Tensor,
    sigma: float,
    baseline_neg1_1: torch.Tensor | None = None,
) -> dict[str, Any]:
    data_batch = {
        pid_model.config.input_caption_key: [prompt],
        "LQ_latent": latent.to(dtype=torch.bfloat16, device="cuda"),
        "degrade_sigma": torch.tensor([sigma], device="cuda", dtype=torch.float32),
    }
    if baseline_neg1_1 is not None:
        data_batch["LQ_video_or_image"] = baseline_neg1_1.to(dtype=torch.bfloat16, device="cuda")
    return data_batch

def _pid_generate_image(
    pid_model,
    data_batch: dict[str, Any],
    *,
    pid_cfg: float,
    pid_steps: int,
    seed: int,
    image_size: tuple[int, int],
) -> Image.Image:
    with torch.no_grad():
        samples = pid_model.generate_samples_from_batch(
            data_batch,
            cfg_scale=float(pid_cfg),
            num_steps=int(pid_steps),
            seed=resolve_seed(seed),
            shift=None,
            image_size=image_size,
        )
    return _neg1_tensor_to_pil(samples[0])

def _zimage_extract_latent(pipe, raw_output, width: int, height: int) -> torch.Tensor:
    latent = raw_output.images
    if not isinstance(latent, torch.Tensor) or latent.ndim != 4:
        raise UserInputError(f"Z-Image returned an unexpected latent shape: {getattr(latent, 'shape', None)}")
    return latent

@dataclass(frozen=True)
class PiDBackboneAdapter:
    backbone: str
    load_decoder: Callable[[str], Any]
    extract_latent: Callable[[Any, Any, int, int], torch.Tensor]
    decode_baseline_01: Callable[[Any, torch.Tensor], torch.Tensor]
    lowres_desc: str
    vae_desc: str
    decoder_desc: str
    decode_desc: str


def _pid_backbone_adapter(backbone: str) -> PiDBackboneAdapter:
    adapters = {
        PID_BACKBONE_ZIMAGE: PiDBackboneAdapter(
            backbone=PID_BACKBONE_ZIMAGE,
            load_decoder=get_zimage_pid_decoder,
            extract_latent=_zimage_extract_latent,
            decode_baseline_01=_zimage_decode_latent_01,
            lowres_desc="Generating low-res latent...",
            vae_desc="VAE decoding low-res conditioning image...",
            decoder_desc="Loading PiD decoder...",
            decode_desc="PiD 4x decoding...",
        ),
        PID_BACKBONE_QWEN: PiDBackboneAdapter(
            backbone=PID_BACKBONE_QWEN,
            load_decoder=get_qwen_pid_decoder,
            extract_latent=_qwen_extract_latent,
            decode_baseline_01=_qwen_decode_latent_01,
            lowres_desc="Generating low-res Qwen latent...",
            vae_desc="VAE decoding low-res conditioning image...",
            decoder_desc="Loading Qwen PiD decoder...",
            decode_desc="PiD 4x decoding...",
        ),
    }
    try:
        return adapters[backbone]
    except KeyError:
        raise UserInputError(f"Unknown PiD backbone adapter: {backbone}") from None

def _decode_pipeline_with_pid(
    adapter: PiDBackboneAdapter,
    pipe,
    prompt: str,
    kwargs: dict,
    width: int,
    height: int,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress,
) -> tuple[Image.Image, float, str, int, int]:
    if not torch.cuda.is_available():
        raise UserInputError("PiD decoding requires CUDA.")
    pid_out_w, pid_out_h = _validate_pid_dims(width, height)
    pid_ckpt_type = _resolve_pid_ckpt_type(adapter.backbone, pid_ckpt, width, height)
    pid_kwargs = dict(kwargs)
    pid_kwargs["output_type"] = "latent"

    def generate_with_pid() -> Image.Image:
        progress(0.3, desc=adapter.lowres_desc)
        raw_output = pipe(**pid_kwargs)
        latent = adapter.extract_latent(pipe, raw_output, width, height)

        progress(0.55, desc=adapter.vae_desc)
        with torch.no_grad():
            baseline_01 = adapter.decode_baseline_01(pipe, latent)
        baseline_neg1_1 = baseline_01 * 2.0 - 1.0

        progress(0.68, desc=adapter.decoder_desc)
        pid_model = adapter.load_decoder(pid_ckpt_type)
        lq_h, lq_w = baseline_01.shape[-2], baseline_01.shape[-1]
        data_batch = _pid_data_batch(
            pid_model,
            prompt,
            latent,
            _scheduler_sigma(pipe),
            baseline_neg1_1,
        )

        progress(0.78, desc=adapter.decode_desc)
        return _pid_generate_image(
            pid_model,
            data_batch,
            pid_cfg=pid_cfg,
            pid_steps=pid_steps,
            seed=seed,
            image_size=(lq_h * PID_SCALE, lq_w * PID_SCALE),
        )

    result, elapsed = timed_result(generate_with_pid)
    return result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h

def _decode_zimage_family_with_pid(
    pipe,
    prompt: str,
    kwargs: dict,
    width: int,
    height: int,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress,
) -> tuple[Image.Image, float, str, int, int]:
    """Run a Z-Image-family pipeline to latent, then PiD-decode it at 4x."""
    adapter = _pid_backbone_adapter(PID_BACKBONE_ZIMAGE)
    return _decode_pipeline_with_pid(
        adapter, pipe, prompt, kwargs, width, height,
        pid_ckpt, pid_steps, pid_cfg, seed, progress,
    )

def _decode_qwen_with_pid(
    pipe,
    prompt: str,
    kwargs: dict,
    width: int,
    height: int,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress,
) -> tuple[Image.Image, float, str, int, int]:
    """Run Qwen Image to latent, then PiD-decode it at 4x."""
    adapter = _pid_backbone_adapter(PID_BACKBONE_QWEN)
    return _decode_pipeline_with_pid(
        adapter, pipe, prompt, kwargs, width, height,
        pid_ckpt, pid_steps, pid_cfg, seed, progress,
    )

def _ideogram4_caption_text_for_pid(final_prompt: str, source_prompt: str) -> str:
    """Convert Ideogram's JSON caption into a compact PiD text condition."""
    try:
        caption = json.loads(final_prompt)
    except Exception:
        text = final_prompt or source_prompt or ""
        return text[:4000]

    pieces: list[str] = []
    high_level = caption.get("high_level_description")
    if isinstance(high_level, str):
        pieces.append(high_level)

    style = caption.get("style_description")
    if isinstance(style, dict):
        pieces.extend(value for value in style.values() if isinstance(value, str))
    elif isinstance(style, str):
        pieces.append(style)

    cd = caption.get("compositional_deconstruction")
    elements = cd.get("elements") if isinstance(cd, dict) else caption.get("elements")
    if isinstance(elements, list):
        for element in elements:
            if not isinstance(element, dict):
                continue
            for key in ("desc", "description", "visual_description", "content", "text"):
                value = element.get(key)
                if isinstance(value, str):
                    pieces.append(value)

    text = " ".join(piece.strip() for piece in pieces if piece and piece.strip())
    return (text or source_prompt or final_prompt or "")[:4000]

def _ideogram4_patchify_flux2_raw_latents(raw_latents: torch.Tensor) -> torch.Tensor:
    """Flux2/PiD patchify: (B, 32, H/8, W/8) -> (B, 128, H/16, W/16)."""
    return patchify_flux2_raw_latents(raw_latents)

def _ideogram4_normalize_flux2_packed_latents(pid_model, packed_latents: torch.Tensor) -> torch.Tensor:
    """Apply PiD's Flux2 VAE BatchNorm normalization to packed raw latents."""
    vae_interface = getattr(pid_model, "vae_encoder", None)
    vae_wrapper = getattr(vae_interface, "model", None)
    autoencoder = getattr(vae_wrapper, "model", None)
    bn = getattr(autoencoder, "bn", None)
    if bn is None:
        raise UserInputError("Loaded Ideogram PiD model does not expose a Flux2 VAE BatchNorm normalizer.")

    bn.eval()
    eps = float(getattr(autoencoder, "bn_eps", getattr(bn, "eps", 1e-4)))
    mean = bn.running_mean.view(1, -1, 1, 1).to(packed_latents.device, packed_latents.dtype)
    std = torch.sqrt(bn.running_var.view(1, -1, 1, 1).to(packed_latents.device, packed_latents.dtype) + eps)
    if mean.shape[1] != packed_latents.shape[1]:
        raise UserInputError(
            f"PiD Flux2 VAE BN has {mean.shape[1]} channels, but Ideogram latents have "
            f"{packed_latents.shape[1]} channels."
        )
    return (packed_latents - mean) / std

def _decode_ideogram4_with_pid(
    latent_output: Any,
    final_prompt: str,
    source_prompt: str,
    width: int,
    height: int,
    pid_ckpt: str,
    pid_steps: int,
    pid_cfg: float,
    seed: int,
    progress,
) -> tuple[Image.Image, float, str, int, int]:
    """Run Ideogram 4 to raw Flux2 latents, then PiD-decode at 4x."""
    if not torch.cuda.is_available():
        raise UserInputError("PiD decoding requires CUDA.")
    pid_out_w, pid_out_h = _validate_pid_dims(width, height)
    pid_ckpt_type = _resolve_pid_ckpt_type(PID_BACKBONE_IDEOGRAM4, pid_ckpt, width, height)

    if not isinstance(latent_output, dict):
        raise UserInputError(f"Ideogram returned an unexpected latent output type: {type(latent_output)!r}")
    raw_latents = latent_output.get("latents")
    baseline = latent_output.get("decoded")
    if not isinstance(raw_latents, torch.Tensor) or raw_latents.ndim != 4:
        raise UserInputError(f"Ideogram returned an unexpected latent shape: {getattr(raw_latents, 'shape', None)}")
    if raw_latents.shape[1] != 32:
        raise UserInputError(f"Flux2 PiD expects 32-channel Ideogram latents, got {tuple(raw_latents.shape)}.")
    if not isinstance(baseline, torch.Tensor) or baseline.ndim != 4:
        raise UserInputError(f"Ideogram returned an unexpected decoded tensor shape: {getattr(baseline, 'shape', None)}")

    def decode_with_pid() -> Image.Image:
        progress(0.86, desc="Loading Ideogram PiD decoder...")
        pid_model = get_pid_decoder(PID_BACKBONE_IDEOGRAM4, pid_ckpt_type)
        packed = _ideogram4_patchify_flux2_raw_latents(
            raw_latents.detach().to(dtype=torch.bfloat16, device="cuda").contiguous()
        )
        latents = _ideogram4_normalize_flux2_packed_latents(pid_model, packed).to(
            dtype=torch.bfloat16,
            device="cuda",
        )
        pid_prompt = _ideogram4_caption_text_for_pid(final_prompt, source_prompt)
        lq_h, lq_w = baseline.shape[-2], baseline.shape[-1]
        data_batch = _pid_data_batch(pid_model, pid_prompt, latents, 0.0)
        progress(0.92, desc="PiD 4x decoding Ideogram latent...")
        return _pid_generate_image(
            pid_model,
            data_batch,
            pid_cfg=pid_cfg,
            pid_steps=pid_steps,
            seed=seed,
            image_size=(lq_h * PID_SCALE, lq_w * PID_SCALE),
        )

    result, elapsed = timed_result(decode_with_pid)
    return result, elapsed, pid_ckpt_type, pid_out_w, pid_out_h

def get_pid_decoder(backbone: str, ckpt_type: str):
    """Load a PiD decoder checkpoint for the requested latent-space backbone."""
    spec = _pid_checkpoint_specs(backbone)[ckpt_type]

    def factory():
        modules = _get_pid_modules()
        checkpoint_path = _ensure_pid_checkpoint(backbone, ckpt_type)
        _ensure_pid_vae_asset(backbone)
        _ensure_pid_experiment_available(backbone, spec)

        def load_from_path(path: str):
            log.info("Loading %s decoder from %s...", spec.label, path)
            with _pid_load_lock:
                cwd = os.getcwd()
                try:
                    os.chdir(PID_DIR)
                    return modules["load_model_from_checkpoint"](
                        experiment_name=spec.experiment,
                        checkpoint_path=path,
                        config_file="pid/_src/configs/pid/config.py",
                        enable_fsdp=False,
                        experiment_opts=[],
                        strict=False,
                        load_ema_to_reg=False,
                    )
                finally:
                    os.chdir(cwd)

        try:
            model, _config = load_from_path(checkpoint_path)
        except RuntimeError as exc:
            message = str(exc)
            if "PytorchStreamReader failed reading zip archive" not in message:
                raise
            log.warning(
                "%s checkpoint appears corrupt or incomplete; re-downloading from %s.",
                spec.label,
                PID_HF_REPO,
            )
            checkpoint_path = _ensure_pid_checkpoint(backbone, ckpt_type, force_download=True)
            model, _config = load_from_path(checkpoint_path)
        model.eval()
        log.info("%s decoder ready.", spec.label)
        return model

    return _load_managed_model(spec.registry_key, factory)

def get_zimage_pid_decoder(ckpt_type: str):
    """Load a PiD decoder checkpoint for Z-Image's Flux-compatible VAE latent."""
    return get_pid_decoder(PID_BACKBONE_ZIMAGE, ckpt_type)

def get_qwen_pid_decoder(ckpt_type: str):
    """Load a PiD decoder checkpoint for Qwen-Image's VAE latent."""
    return get_pid_decoder(PID_BACKBONE_QWEN, ckpt_type)

__all__ = (
    '_ensure_pid',
    '_get_pid_modules',
    '_pid_checkpoint_specs',
    '_pid_checkpoint_label',
    '_resolve_pid_ckpt_type',
    '_validate_pid_dims',
    '_ensure_pid_checkpoint',
    '_ensure_pid_vae_asset',
    '_ensure_pid_experiment_available',
    '_pipeline_execution_device',
    '_zimage_decode_latent_01',
    '_qwen_extract_latent',
    '_qwen_decode_latent_01',
    '_neg1_tensor_to_pil',
    'resolve_seed',
    '_scheduler_sigma',
    '_pid_data_batch',
    '_pid_generate_image',
    '_zimage_extract_latent',
    'PiDBackboneAdapter',
    '_decode_pipeline_with_pid',
    '_decode_zimage_family_with_pid',
    '_decode_qwen_with_pid',
    '_ideogram4_caption_text_for_pid',
    '_ideogram4_patchify_flux2_raw_latents',
    '_ideogram4_normalize_flux2_packed_latents',
    '_decode_ideogram4_with_pid',
    'get_pid_decoder',
    'get_zimage_pid_decoder',
    'get_qwen_pid_decoder',
)
_seal_runtime_module(globals())
