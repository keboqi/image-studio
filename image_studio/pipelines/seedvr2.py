"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError
from image_studio.infra.lazy_modules import LazyModuleGroup
from image_studio.progress import NO_PROGRESS

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

class SeedVR2Runner:
    """Own SeedVR2 imports, model cache, and currently loaded model."""

    def __init__(self):
        self.available = None
        self.modules = LazyModuleGroup("SeedVR2", lambda: True, _import_seedvr2)
        self.cache = {}
        self.loaded_model = None


def _ensure_seedvr2():
    """Clone the SeedVR2 repo if not present and install its dependencies."""
    return _git_bootstrap.ensure(SEEDVR2_REPO_SPEC)

def is_seedvr2_available() -> bool:
    """Return whether SeedVR2 can be imported, bootstrapping it lazily once."""
    if _seedvr2_state.available is None:
        _seedvr2_state.available = _ensure_seedvr2()
    return _seedvr2_state.available

def _import_seedvr2():
    """Lazy-import SeedVR2 modules; returns a dict of useful objects."""
    if not is_seedvr2_available():
        raise UserInputError(
            "SeedVR2 upscaler is unavailable. Check git/network access "
            f"or clone {SEEDVR2_REPO} into {SEEDVR2_DIR}."
        )
    if SEEDVR2_DIR not in sys.path:
        sys.path.insert(0, SEEDVR2_DIR)

    from src.utils.downloads import download_weight
    from src.utils.model_registry import (
        get_available_dit_models,
        DEFAULT_DIT,
        DEFAULT_VAE,
    )
    from src.utils.constants import SEEDVR2_FOLDER_NAME
    from src.core.generation_utils import (
        setup_generation_context,
        prepare_runner,
        compute_generation_info,
        log_generation_start,
        load_text_embeddings,
        script_directory as seedvr2_script_dir,
    )
    from src.core.generation_phases import (
        encode_all_batches,
        upscale_all_batches,
        decode_all_batches,
        postprocess_all_batches,
    )
    from src.utils.debug import Debug as SeedVR2Debug
    from src.optimization.memory_manager import (
        clear_memory,
        get_gpu_backend,
    )

    return {
        "download_weight": download_weight,
        "get_available_dit_models": get_available_dit_models,
        "DEFAULT_DIT": DEFAULT_DIT,
        "DEFAULT_VAE": DEFAULT_VAE,
        "SEEDVR2_FOLDER_NAME": SEEDVR2_FOLDER_NAME,
        "setup_generation_context": setup_generation_context,
        "prepare_runner": prepare_runner,
        "compute_generation_info": compute_generation_info,
        "log_generation_start": log_generation_start,
        "load_text_embeddings": load_text_embeddings,
        "seedvr2_script_dir": seedvr2_script_dir,
        "encode_all_batches": encode_all_batches,
        "upscale_all_batches": upscale_all_batches,
        "decode_all_batches": decode_all_batches,
        "postprocess_all_batches": postprocess_all_batches,
        "debug": SeedVR2Debug(enabled=False),
        "clear_memory": clear_memory,
        "get_gpu_backend": get_gpu_backend,
    }


_seedvr2_state = SeedVR2Runner()


def _get_seedvr2():
    return _seedvr2_state.modules.get()

def _seedvr2_device_name(device_id: str, backend: str) -> str:
    if backend == "mps":
        return "mps"
    return f"{backend}:{device_id}"

def _seedvr2_offload(arg: str, backend: str, cache_on: bool = False):
    if arg == "none":
        return "cpu" if cache_on else None
    if arg == "cpu":
        return "cpu"
    if ":" in arg:
        return arg
    return _seedvr2_device_name(arg, backend)

@dataclass(frozen=True)
class SeedVR2DevicePlan:
    backend: str
    inference_device: str
    dit_offload: str | None
    vae_offload: str | None
    tensor_offload: str | None

def _prepare_seedvr2_source_image(image) -> tuple[np.ndarray, torch.Tensor]:
    if image is None:
        raise UserInputError("Please upload an image to upscale.")
    img = coerce_rgb_image(image)
    img_np = np.array(img, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(img_np[None, ...]).to(torch.float16)
    return img_np, image_tensor

def _resolve_video_path(video: Any) -> str | None:
    """Extract a filesystem path from a Gradio Video value."""
    if video is None:
        return None
    if isinstance(video, str):
        return video
    if isinstance(video, dict):
        for key in ("video", "path", "name"):
            value = video.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                nested = value.get("path") or value.get("name")
                if isinstance(nested, str):
                    return nested
    if isinstance(video, (list, tuple)) and video:
        return _resolve_video_path(video[0])
    return None

def _require_seedvr2_video_path(video: Any) -> str:
    path = _resolve_video_path(video)
    if not path:
        raise UserInputError("No video available to upscale.")
    if not os.path.isfile(path):
        raise UserInputError(f"Video file not found: {path}")
    if not path.lower().endswith(SEEDVR2_VIDEO_EXTENSIONS):
        raise UserInputError("Please provide a supported video file.")
    return path

def _read_seedvr2_video_frames(cap, max_frames: int) -> torch.Tensor | None:
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        frames.append(frame)
    if not frames:
        return None
    return torch.from_numpy(np.stack(frames)).to(torch.float32)

def _open_seedvr2_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise UserInputError(f"Cannot open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise UserInputError("Could not read video dimensions.")
    return cap, fps, total_frames, width, height

def _seedvr2_video_output_path(source_path: str) -> str:
    stem = os.path.splitext(os.path.basename(source_path))[0]
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._") or "video"
    return os.path.join(OUTPUT_DIR, f"video_upscale_{safe_stem}_{datetime.now():%Y%m%d_%H%M%S}.mp4")

def _ensure_seedvr2_weight(s: dict, dit_model: str) -> str:
    model_dir = os.path.join(SEEDVR2_DIR, "models", s["SEEDVR2_FOLDER_NAME"])
    if not s["download_weight"](
        dit_model=dit_model, vae_model=s["DEFAULT_VAE"], model_dir=model_dir, debug=s["debug"]
    ):
        raise UserInputError("SeedVR2 model download failed - check console logs.")
    return model_dir

def _clear_seedvr2_cache_on_model_change(s: dict, dit_model: str):
    if _seedvr2_state.loaded_model is None or _seedvr2_state.loaded_model == dit_model:
        return
    log.info("SeedVR2 model changed; clearing cache")
    _seedvr2_state.cache = {}
    s["clear_memory"](debug=s["debug"], deep=True, force=True)
    _seedvr2_state.loaded_model = None

def _seedvr2_device_plan(s: dict, blocks_to_swap: int, cache_models: bool) -> SeedVR2DevicePlan:
    backend = s["get_gpu_backend"]()
    inference_device = _seedvr2_device_name("0", backend)
    if cache_models and blocks_to_swap == 0:
        dit_offload = None
        vae_offload = None
    else:
        dit_offload = _seedvr2_offload("cpu" if blocks_to_swap > 0 else "none", backend, cache_models)
        vae_offload = _seedvr2_offload("cpu" if cache_models else "none", backend, cache_models)
    tensor_offload = _seedvr2_offload("cpu", backend, False)
    return SeedVR2DevicePlan(
        backend=backend,
        inference_device=inference_device,
        dit_offload=dit_offload,
        vae_offload=vae_offload,
        tensor_offload=tensor_offload,
    )

def _get_seedvr2_context(s: dict, plan: SeedVR2DevicePlan, cache_models: bool) -> dict:
    if cache_models and "ctx" in _seedvr2_state.cache:
        ctx = _seedvr2_state.cache["ctx"]
        keep = {
            "dit_device",
            "vae_device",
            "dit_offload_device",
            "vae_offload_device",
            "tensor_offload_device",
            "compute_dtype",
        }
        for key in list(ctx.keys()):
            if key not in keep:
                del ctx[key]
        return ctx

    ctx = s["setup_generation_context"](
        dit_device=plan.inference_device,
        vae_device=plan.inference_device,
        dit_offload_device=plan.dit_offload,
        vae_offload_device=plan.vae_offload,
        tensor_offload_device=plan.tensor_offload,
        debug=s["debug"],
    )
    if cache_models:
        _seedvr2_state.cache["ctx"] = ctx
    return ctx

def _prepare_seedvr2_runner(
    s: dict,
    ctx: dict,
    model_dir: str,
    dit_model: str,
    blocks_to_swap: int,
    vae_tiling: bool,
    vae_tile_size: int,
    plan: SeedVR2DevicePlan,
    cache_models: bool,
):
    dit_id = "webui_dit" if cache_models else None
    vae_id = "webui_vae" if cache_models else None
    runner, cache_context = s["prepare_runner"](
        dit_model=dit_model,
        vae_model=s["DEFAULT_VAE"],
        model_dir=model_dir,
        debug=s["debug"],
        ctx=ctx,
        dit_cache=cache_models,
        vae_cache=cache_models,
        dit_id=dit_id,
        vae_id=vae_id,
        block_swap_config={
            "blocks_to_swap": blocks_to_swap,
            "swap_io_components": blocks_to_swap > 0,
            "offload_device": plan.dit_offload,
        } if blocks_to_swap > 0 else None,
        encode_tiled=vae_tiling,
        encode_tile_size=(vae_tile_size, vae_tile_size),
        encode_tile_overlap=(128, 128),
        decode_tiled=vae_tiling,
        decode_tile_size=(vae_tile_size, vae_tile_size),
        decode_tile_overlap=(128, 128),
        tile_debug="false",
        attention_mode="sdpa",
    )
    ctx["cache_context"] = cache_context
    if cache_models:
        _seedvr2_state.cache["runner"] = runner
        _seedvr2_state.loaded_model = dit_model
        model_mgr.register(
            MODEL_SEEDVR2, runner, MODEL_SPECS[MODEL_SEEDVR2].vram_mb,
            unload_fn=lambda: _evict_seedvr2(),
        )
    return runner

def _run_seedvr2_phases(
    s: dict,
    runner,
    ctx: dict,
    image_tensor: torch.Tensor,
    resolution: int,
    max_resolution: int,
    seed: int,
    color_correction: str,
    cache_models: bool,
    batch_size: int = 1,
    uniform_batch_size: bool = False,
    prepend_frames: int = 0,
    temporal_overlap: int = 0,
    input_noise_scale: float = 0.0,
    latent_noise_scale: float = 0.0,
    progress_callback=None,
) -> tuple[dict, float]:
    ctx["text_embeds"] = s["load_text_embeddings"](
        s["seedvr2_script_dir"], ctx["dit_device"], ctx["compute_dtype"], s["debug"],
    )
    image_tensor, gen_info = s["compute_generation_info"](
        ctx=ctx, images=image_tensor,
        resolution=resolution, max_resolution=max_resolution,
        batch_size=batch_size, uniform_batch_size=uniform_batch_size,
        seed=seed, prepend_frames=prepend_frames, temporal_overlap=temporal_overlap,
        debug=s["debug"],
    )
    s["log_generation_start"](gen_info, s["debug"])

    t0 = time.time()
    ctx = s["encode_all_batches"](
        runner, ctx=ctx, images=image_tensor, debug=s["debug"],
        batch_size=batch_size, uniform_batch_size=uniform_batch_size, seed=seed,
        progress_callback=progress_callback, temporal_overlap=temporal_overlap,
        resolution=resolution, max_resolution=max_resolution,
        input_noise_scale=input_noise_scale, color_correction=color_correction,
    )
    ctx = s["upscale_all_batches"](
        runner, ctx=ctx, debug=s["debug"], progress_callback=progress_callback,
        seed=seed, latent_noise_scale=latent_noise_scale, cache_model=cache_models,
    )
    ctx = s["decode_all_batches"](
        runner, ctx=ctx, debug=s["debug"], progress_callback=progress_callback,
        cache_model=cache_models,
    )
    ctx = s["postprocess_all_batches"](
        ctx=ctx, debug=s["debug"], progress_callback=progress_callback,
        color_correction=color_correction, prepend_frames=prepend_frames,
        temporal_overlap=temporal_overlap, batch_size=batch_size,
    )
    return ctx, time.time() - t0

def _seedvr2_result_to_frames(ctx: dict) -> np.ndarray:
    result_tensor = ctx["final_video"]
    if result_tensor.device.type != "cpu":
        result_tensor = result_tensor.cpu()
    if result_tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        result_tensor = result_tensor.to(torch.float32)
    return (result_tensor.numpy() * 255.0).clip(0, 255).astype(np.uint8)

def _seedvr2_result_to_image(ctx: dict) -> tuple[Image.Image, np.ndarray]:
    frames_uint8 = _seedvr2_result_to_frames(ctx)
    frame_uint8 = frames_uint8[0]
    return Image.fromarray(frame_uint8), frame_uint8

def _write_seedvr2_video_frames(
    frames_uint8: np.ndarray,
    output_path: str,
    fps: float,
    writer=None,
):
    if frames_uint8.size == 0:
        return writer
    if writer is None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        height, width = frames_uint8.shape[1], frames_uint8.shape[2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise UserInputError(f"Cannot create video writer for: {output_path}")
    for frame in frames_uint8:
        if frame.shape[-1] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    return writer

def run_upscale(
    image,
    resolution,
    max_resolution,
    dit_model,
    color_correction,
    vae_tiling,
    vae_tile_size,
    blocks_to_swap,
    seed,
    progress=NO_PROGRESS,
):
    """Upscale a single image via the SeedVR2 4-phase pipeline (runs locally)."""
    seed = normalize_seed(seed)
    progress(0.1, desc="Ensuring VRAM...")
    model_mgr.ensure_vram(MODEL_SPECS[MODEL_SEEDVR2].vram_mb, exclude=MODEL_SEEDVR2)

    progress(0.2, desc="Loading ModelManager...")
    s = _get_seedvr2()

    resolution = int(resolution)
    max_resolution = int(max_resolution)
    blocks_to_swap = int(blocks_to_swap)
    vae_tile_size = int(vae_tile_size)

    img_np, image_tensor = _prepare_seedvr2_source_image(image)
    model_dir = _ensure_seedvr2_weight(s, dit_model)
    _clear_seedvr2_cache_on_model_change(s, dit_model)
    cache_models = True
    plan = _seedvr2_device_plan(s, blocks_to_swap, cache_models)
    ctx = _get_seedvr2_context(s, plan, cache_models)
    runner = _prepare_seedvr2_runner(
        s, ctx, model_dir, dit_model, blocks_to_swap,
        vae_tiling, vae_tile_size, plan, cache_models,
    )
    ctx, elapsed = _run_seedvr2_phases(
        s, runner, ctx, image_tensor, resolution, max_resolution,
        seed, color_correction, cache_models,
    )
    result_pil, frame_uint8 = _seedvr2_result_to_image(ctx)
    preview_path, raw_path = save_output_image_pair("upscale", result_pil)

    in_h, in_w = img_np.shape[:2]
    out_h, out_w = frame_uint8.shape[:2]
    status = ok_status(
        elapsed,
        f"{in_w}x{in_h} -> {out_w}x{out_h}",
        f"model ...{dit_model[-30:]}",
        f"color {color_correction}",
        f"seed {seed}",
    )
    return preview_path, status, raw_path

def _validate_seedvr2_video_batching(batch_size: int, chunk_size: int, temporal_overlap: int):
    if batch_size < 1 or (batch_size - 1) % 4 != 0:
        raise UserInputError("SeedVR2 batch size must be 1, 5, 9, 13, ...")
    if chunk_size < 0:
        raise UserInputError("Chunk size must be 0 or greater.")
    if temporal_overlap < 0:
        raise UserInputError("Temporal overlap must be 0 or greater.")
    if temporal_overlap >= batch_size:
        raise UserInputError("Temporal overlap must be smaller than batch size.")
    if chunk_size > 0 and temporal_overlap >= chunk_size:
        raise UserInputError("Temporal overlap must be smaller than chunk size.")

def run_video_upscale(
    video,
    resolution,
    max_resolution,
    dit_model,
    color_correction,
    vae_tiling,
    vae_tile_size,
    blocks_to_swap,
    batch_size,
    chunk_size,
    temporal_overlap,
    seed,
    progress=NO_PROGRESS,
):
    """Upscale a video through SeedVR2, streaming frame chunks into a new MP4."""
    video_path = _require_seedvr2_video_path(video)
    seed = normalize_seed(seed)
    resolution = int(resolution)
    max_resolution = int(max_resolution)
    blocks_to_swap = int(blocks_to_swap)
    vae_tile_size = int(vae_tile_size)
    batch_size = int(batch_size)
    chunk_size = int(chunk_size)
    temporal_overlap = int(temporal_overlap)
    _validate_seedvr2_video_batching(batch_size, chunk_size, temporal_overlap)

    progress(0.05, desc="Reading source video...")
    cap, fps, total_frames, in_w, in_h = _open_seedvr2_video(video_path)
    if total_frames <= 0:
        cap.release()
        raise UserInputError("Could not read the video frame count.")
    frames_to_process = total_frames
    chunk_size = chunk_size or frames_to_process
    total_chunks = max(1, math.ceil(frames_to_process / chunk_size))

    progress(0.1, desc="Ensuring VRAM...")
    model_mgr.ensure_vram(MODEL_SPECS[MODEL_SEEDVR2].vram_mb, exclude=MODEL_SEEDVR2)

    progress(0.15, desc="Loading SeedVR2...")
    s = _get_seedvr2()
    model_dir = _ensure_seedvr2_weight(s, dit_model)
    _clear_seedvr2_cache_on_model_change(s, dit_model)
    cache_models = True
    plan = _seedvr2_device_plan(s, blocks_to_swap, cache_models)

    output_path = _seedvr2_video_output_path(video_path)
    writer = None
    frames_read = 0
    frames_written = 0
    chunk_idx = 0
    prev_raw_tail = None
    first_output_shape = None
    t0 = time.time()

    try:
        while frames_read < frames_to_process:
            read_count = min(chunk_size, frames_to_process - frames_read)
            new_frames = _read_seedvr2_video_frames(cap, read_count)
            if new_frames is None:
                break

            frames_read += int(new_frames.shape[0])
            chunk_idx += 1

            if prev_raw_tail is not None and temporal_overlap > 0:
                context_count = min(temporal_overlap, int(prev_raw_tail.shape[0]))
                chunk_tensor = torch.cat([prev_raw_tail[-context_count:], new_frames], dim=0)
            else:
                context_count = 0
                chunk_tensor = new_frames

            chunk_start = 0.2 + 0.7 * ((chunk_idx - 1) / total_chunks)
            progress(
                chunk_start,
                desc=f"Upscaling chunk {chunk_idx}/{total_chunks} ({new_frames.shape[0]} frames)...",
            )

            ctx = _get_seedvr2_context(s, plan, cache_models)
            runner = _prepare_seedvr2_runner(
                s, ctx, model_dir, dit_model, blocks_to_swap,
                vae_tiling, vae_tile_size, plan, cache_models,
            )

            def phase_progress(
                current, total, _frames, phase_name,
                _chunk_start=chunk_start, _chunk_idx=chunk_idx,
            ):
                if total <= 0:
                    return
                local = min(1.0, max(0.0, current / total))
                phase_text = str(phase_name)
                if "Phase 2" in phase_text:
                    phase_offset = 0.25
                elif "Phase 3" in phase_text:
                    phase_offset = 0.55
                elif "Phase 4" in phase_text:
                    phase_offset = 0.8
                else:
                    phase_offset = 0.0
                phase_width = 0.25 if phase_offset < 0.8 else 0.2
                progress(
                    min(0.9, _chunk_start + (0.7 / total_chunks) * (phase_offset + phase_width * local)),
                    desc=f"{phase_text} ({_chunk_idx}/{total_chunks})",
                )

            ctx, _chunk_elapsed = _run_seedvr2_phases(
                s, runner, ctx, chunk_tensor.to(torch.float16),
                resolution, max_resolution, seed,
                color_correction, cache_models,
                batch_size=batch_size,
                uniform_batch_size=False,
                prepend_frames=0,
                temporal_overlap=temporal_overlap,
                input_noise_scale=0.0,
                latent_noise_scale=0.0,
                progress_callback=phase_progress,
            )
            frames_uint8 = _seedvr2_result_to_frames(ctx)
            if context_count > 0:
                frames_uint8 = frames_uint8[context_count:]
            if frames_uint8.size:
                if first_output_shape is None:
                    first_output_shape = frames_uint8.shape[1:3]
                writer = _write_seedvr2_video_frames(frames_uint8, output_path, fps, writer)
                frames_written += int(frames_uint8.shape[0])

            prev_raw_tail = (
                new_frames[-temporal_overlap:].clone()
                if temporal_overlap > 0 else None
            )
            del chunk_tensor, new_frames, frames_uint8, ctx
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frames_written <= 0 or not os.path.isfile(output_path):
        raise UserInputError("Video upscale produced no frames.")

    elapsed = time.time() - t0
    out_h, out_w = first_output_shape if first_output_shape else (0, 0)
    progress(1.0, desc="Done")
    status = ok_status(
        elapsed,
        f"{frames_written} frames",
        f"{in_w}x{in_h} -> {out_w}x{out_h}",
        f"{fps:.2f} fps",
        f"model {dit_model[-30:]}",
        f"batch {batch_size}",
        f"chunk {chunk_size}",
        f"seed {seed}",
    )
    return output_path, status

def _evict_seedvr2():
    """Tear down the SeedVR2 upscaler state so VRAM can be reclaimed."""
    s = _get_seedvr2()
    _seedvr2_state.cache = {}
    _seedvr2_state.loaded_model = None
    s["clear_memory"](debug=s["debug"], deep=True, force=True)
    log.info("SeedVR2 upscaler evicted.")

__all__ = (
    'SeedVR2Runner',
    '_ensure_seedvr2',
    'is_seedvr2_available',
    '_get_seedvr2',
    '_seedvr2_device_name',
    '_seedvr2_offload',
    'SeedVR2DevicePlan',
    '_prepare_seedvr2_source_image',
    '_resolve_video_path',
    '_require_seedvr2_video_path',
    '_read_seedvr2_video_frames',
    '_open_seedvr2_video',
    '_seedvr2_video_output_path',
    '_ensure_seedvr2_weight',
    '_clear_seedvr2_cache_on_model_change',
    '_seedvr2_device_plan',
    '_get_seedvr2_context',
    '_prepare_seedvr2_runner',
    '_run_seedvr2_phases',
    '_seedvr2_result_to_frames',
    '_seedvr2_result_to_image',
    '_write_seedvr2_video_frames',
    'run_upscale',
    '_validate_seedvr2_video_batching',
    'run_video_upscale',
    '_evict_seedvr2',
)
_seal_runtime_module(globals())
