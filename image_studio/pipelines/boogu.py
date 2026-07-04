"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError
from image_studio.infra.lazy_modules import LazyModuleGroup

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _ensure_boogu_image() -> bool:
    """Clone Boogu-Image reference code if the local package is unavailable."""
    return _git_bootstrap.ensure(BOOGU_IMAGE_REPO_SPEC)

def _boogu_device() -> str:
    if APP_CONFIG.boogu.device:
        return APP_CONFIG.boogu.device
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def _prepare_boogu_import_path() -> None:
    os.environ["device"] = _boogu_device()
    if os.path.isdir(BOOGU_IMAGE_DIR):
        os.makedirs(os.path.join(BOOGU_IMAGE_DIR, ".hf_modules_cache"), exist_ok=True)
        os.environ.setdefault(
            "HF_MODULES_CACHE",
            os.path.join(BOOGU_IMAGE_DIR, ".hf_modules_cache"),
        )
        if BOOGU_IMAGE_DIR not in sys.path:
            sys.path.insert(0, BOOGU_IMAGE_DIR)

def _import_boogu_image():
    """Lazy-import Boogu-Image pipeline classes from an install or checkout."""
    _prepare_boogu_import_path()
    try:
        from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
        from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
    except ImportError as first_exc:
        if not _ensure_boogu_image():
            raise UserInputError(
                "Boogu-Image reference code is unavailable. Install boogu-image "
                f"or clone {BOOGU_IMAGE_REPO} into {BOOGU_IMAGE_DIR}. "
                f"Import error: {first_exc}"
            ) from first_exc
        _prepare_boogu_import_path()
        try:
            from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline
            from boogu.pipelines.boogu.pipeline_boogu_turbo import BooguImageTurboPipeline
        except ImportError as exc:
            raise UserInputError(
                "Boogu-Image dependencies are missing or outdated. Install/update "
                "cache-dit, webdataset, python-dotenv, and omegaconf while keeping "
                "the main WebUI pinned to diffusers==0.36.0 for Nunchaku Z-Image Turbo. "
                f"Import error: {exc}"
            ) from exc

    return {
        "BooguImagePipeline": BooguImagePipeline,
        "BooguImageTurboPipeline": BooguImageTurboPipeline,
    }


_boogu_modules = LazyModuleGroup("Boogu-Image", lambda: True, _import_boogu_image)


def _get_boogu_image():
    return _boogu_modules.get()

def _boogu_model_location(model_key: str) -> str:
    model_name = BOOGU_IMAGE_MODEL_NAMES[model_key]
    explicit = BOOGU_IMAGE_MODEL_LOCATIONS[model_key]
    if explicit:
        return explicit
    local_path = os.path.join(BASE_DIR, "models", model_name)
    if os.path.isdir(local_path):
        return local_path
    return f"Boogu/{model_name}"

def _place_boogu_pipe(pipe):
    device = _boogu_device()
    if device == "cpu" or not torch.cuda.is_available():
        pipe.to("cpu")
        return

    if get_gpu_memory() > 70:
        pipe.to(device)
        return

    try:
        pipe.enable_model_cpu_offload(device=device)
    except TypeError:
        pipe.enable_model_cpu_offload()
    except Exception as exc:
        log.warning("Boogu-Image CPU offload unavailable; moving to %s: %s", device, exc)
        pipe.to(device)

def get_boogu_image_pipe(model_key: str):
    if model_key not in BOOGU_IMAGE_MODEL_NAMES:
        raise ValueError(f"Unknown Boogu-Image model key: {model_key!r}")

    def factory():
        modules = _get_boogu_image()
        model_path = _boogu_model_location(model_key)
        pipeline_cls = (
            modules["BooguImageTurboPipeline"]
            if model_key in {MODEL_BOOGU_IMAGE_TURBO, MODEL_BOOGU_IMAGE_EDIT_TURBO}
            else modules["BooguImagePipeline"]
        )
        log.info("Loading %s from %s...", BOOGU_IMAGE_MODEL_NAMES[model_key], model_path)
        pipe = pipeline_cls.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        _place_boogu_pipe(pipe)
        log.info("%s pipeline ready.", BOOGU_IMAGE_MODEL_NAMES[model_key])
        return pipe

    return _load_managed_model(model_key, factory)

def _normalize_boogu_generation_version(version: str | None) -> str:
    version = (version or BOOGU_IMAGE_VERSION_TURBO).strip()
    if version in BOOGU_IMAGE_GENERATION_VERSIONS:
        return version
    return BOOGU_IMAGE_VERSION_TURBO

def _normalize_boogu_edit_version(version: str | None) -> str:
    version = (version or BOOGU_IMAGE_VERSION_TURBO).strip()
    if version in BOOGU_IMAGE_EDIT_VERSIONS:
        return version
    return BOOGU_IMAGE_VERSION_TURBO

def _boogu_image_limits(width: int, height: int) -> tuple[int, int]:
    return width * height, 2 * max(width, height)

def _boogu_generator(seed: int):
    return torch.Generator(_boogu_device()).manual_seed(seed)

def _boogu_text_encoder_kwargs() -> dict[str, Any]:
    return {
        "max_sequence_length": BOOGU_IMAGE_MAX_SEQUENCE_LENGTH,
        "truncate_instruction_sequence": True,
    }

def _run_boogu_pipeline(pipe, kwargs: dict[str, Any]):
    try:
        return timed_result(lambda: pipe(**kwargs).images[0])
    except Exception as exc:
        message = str(exc)
        if "device-side assert" in message or "scatter gather kernel index out of bounds" in message:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            raise UserInputError(
                "Boogu-Image hit a CUDA position-index limit. The prompt was likely too long; "
                "the app now truncates Boogu prompts to a safe token length, but this worker may "
                "need a restart after the CUDA assert."
            ) from exc
        raise

__all__ = (
    '_ensure_boogu_image',
    '_boogu_device',
    '_prepare_boogu_import_path',
    '_get_boogu_image',
    '_boogu_model_location',
    '_place_boogu_pipe',
    'get_boogu_image_pipe',
    '_normalize_boogu_generation_version',
    '_normalize_boogu_edit_version',
    '_boogu_image_limits',
    '_boogu_generator',
    '_boogu_text_encoder_kwargs',
    '_run_boogu_pipeline',
)
_seal_runtime_module(globals())
