"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError
from image_studio.infra.lazy_modules import LazyModuleGroup
from importlib.metadata import PackageNotFoundError, version

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _ensure_hidream_o1():
    """Clone the HiDream-O1 reference repo if its pipeline code is missing."""
    return _git_bootstrap.ensure(HIDREAM_O1_REPO_SPEC)

def _hidream_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
    """Default RoPE init for Transformers versions that no longer export it."""
    if config is not None and rope_kwargs:
        raise ValueError(
            "Unexpected RoPE keyword arguments when config is provided: "
            f"{sorted(rope_kwargs)}"
    )

    if config is not None:
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        partial_rotary_factor = rope_parameters.get(
            "partial_rotary_factor",
            getattr(config, "partial_rotary_factor", 1.0),
        )
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
    else:
        base = rope_kwargs.get("base", 10000.0)
        partial_rotary_factor = rope_kwargs.get("partial_rotary_factor", 1.0)
        head_dim = rope_kwargs["dim"]

    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, 1.0

def _patch_hidream_rope_registry():
    """Compatibility patch for Transformers RoPE registry drift.

    The patch is expected only for the current 4.x pin and should be reviewed
    once Transformers 5.x becomes the supported baseline.
    """
    try:
        transformers_major = int(version("transformers").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        transformers_major = 4
    if transformers_major >= 5:
        log.warning("Review the HiDream RoPE compatibility patch for Transformers %s.", version("transformers"))
    try:
        import transformers.modeling_rope_utils as rope_utils
    except ImportError as exc:
        raise UserInputError(f"Transformers RoPE utilities are unavailable: {exc}") from exc

    registry = getattr(rope_utils, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(registry, dict):
        raise UserInputError("Transformers ROPE_INIT_FUNCTIONS registry is unavailable.")
    if "default" in registry:
        return

    default_fn = getattr(rope_utils, "_compute_default_rope_parameters", None)
    if default_fn is None:
        default_fn = _hidream_default_rope_parameters

    registry["default"] = default_fn
    log.info("Patched Transformers ROPE_INIT_FUNCTIONS with default RoPE for HiDream-O1.")

def _patch_hidream_qwen3_vl_classes():
    """Patch methods expected by newer Transformers onto HiDream's vendored Qwen3-VL."""
    module = sys.modules.get("models.qwen3_vl_transformers")
    if module is None:
        return

    rotary_cls = getattr(module, "Qwen3VLTextRotaryEmbedding", None)
    if rotary_cls is not None and not hasattr(rotary_cls, "compute_default_rope_parameters"):
        rotary_cls.compute_default_rope_parameters = staticmethod(_hidream_default_rope_parameters)
        log.info("Patched HiDream Qwen3VLTextRotaryEmbedding default RoPE method.")

def _hidream_model_device(model):
    for tensor in list(model.parameters()) + list(model.buffers()):
        if torch.is_tensor(tensor) and tensor.device.type != "meta":
            return tensor.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _is_meta_tensor(value):
    return torch.is_tensor(value) and value.device.type == "meta"

def _repair_hidream_rope_buffers(model):
    """Recreate non-checkpoint RoPE buffers left on meta by newer Transformers."""
    device = _hidream_model_device(model)
    repaired = 0

    for module in model.modules():
        if module.__class__.__name__ != "Qwen3VLTextRotaryEmbedding":
            continue

        inv_freq = getattr(module, "inv_freq", None)
        original_inv_freq = getattr(module, "original_inv_freq", None)
        if (
            inv_freq is not None
            and not _is_meta_tensor(inv_freq)
            and original_inv_freq is not None
            and not _is_meta_tensor(original_inv_freq)
        ):
            continue

        rope_fn = getattr(
            module,
            "compute_default_rope_parameters",
            _hidream_default_rope_parameters,
        )
        new_inv_freq, attention_scaling = rope_fn(module.config, device=device)
        new_inv_freq = new_inv_freq.to(device=device)

        if "inv_freq" in module._buffers:
            module._buffers["inv_freq"] = new_inv_freq
        else:
            module.register_buffer("inv_freq", new_inv_freq, persistent=False)
        module.original_inv_freq = new_inv_freq.clone()
        module.attention_scaling = attention_scaling
        repaired += 1

    if repaired:
        log.info("Repaired %d HiDream Qwen3-VL RoPE buffer(s) on %s.", repaired, device)

def _import_hidream_o1():
    """Lazy-import HiDream-O1 helpers from the official reference repo."""
    if not _ensure_hidream_o1():
        raise UserInputError(
            "HiDream-O1 reference code is unavailable. Check git/network access "
            f"or clone {HIDREAM_O1_REPO} into {HIDREAM_O1_DIR}."
        )

    if HIDREAM_O1_DIR not in sys.path:
        sys.path.insert(0, HIDREAM_O1_DIR)

    try:
        _patch_hidream_rope_registry()
        from transformers import AutoProcessor, PreTrainedTokenizerBase
        from models.pipeline import DEFAULT_TIMESTEPS, generate_image
        from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
        _patch_hidream_qwen3_vl_classes()
    except ImportError as exc:
        raise UserInputError(
            "HiDream-O1 dependencies are missing or outdated. Install/update "
            "the requirements, including transformers>=4.57.1, torchvision, "
            f"einops, tqdm, and scipy. Import error: {exc}"
        ) from exc

    return {
        "AutoProcessor": AutoProcessor,
        "PreTrainedTokenizerBase": PreTrainedTokenizerBase,
        "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
        "DEFAULT_TIMESTEPS": DEFAULT_TIMESTEPS,
        "generate_image": generate_image,
    }


_hidream_modules = LazyModuleGroup("HiDream-O1", lambda: True, _import_hidream_o1)


def _get_hidream_o1():
    modules = _hidream_modules.get()
    _patch_hidream_rope_registry()
    _patch_hidream_qwen3_vl_classes()
    return modules

def _hidream_get_tokenizer(processor):
    mods = _get_hidream_o1()
    if isinstance(processor, mods["PreTrainedTokenizerBase"]):
        return processor
    return processor.tokenizer

def _hidream_add_special_tokens(tokenizer):
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"

def _unload_hidream_o1_bundle(bundle: dict):
    model = bundle.get("model") if isinstance(bundle, dict) else None
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    if isinstance(bundle, dict):
        bundle.clear()

def _get_hidream_o1_spec(model_key: str) -> HiDreamSpec:
    try:
        return HIDREAM_O1_SPECS[model_key]
    except KeyError:
        raise ValueError(f"Unknown HiDream-O1 model key: {model_key!r}") from None

def get_hidream_o1_pipe(model_key: str):
    spec = _get_hidream_o1_spec(model_key)
    
    other_key = MODEL_HIDREAM_O1_FULL if model_key == MODEL_HIDREAM_O1_DEV else MODEL_HIDREAM_O1_DEV
    if model_mgr.is_loaded(other_key):
        model_mgr.unload(other_key)

    def factory():
        mods = _get_hidream_o1()
        log.info("Loading %s pipeline from %s...", spec.label, spec.model_id)
        processor = mods["AutoProcessor"].from_pretrained(spec.model_id)
        _patch_hidream_rope_registry()
        model = mods["Qwen3VLForConditionalGeneration"].from_pretrained(
            spec.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
        ).eval()
        _repair_hidream_rope_buffers(model)
        _hidream_add_special_tokens(_hidream_get_tokenizer(processor))

        bundle = {
            "model": model,
            "processor": processor,
            "generate_image": mods["generate_image"],
            "timesteps_list": mods["DEFAULT_TIMESTEPS"] if spec.use_default_timesteps else None,
            "model_key": model_key,
            "label": spec.label,
            "short_label": spec.short_label,
            "steps": spec.steps,
            "guidance_scale": spec.guidance_scale,
            "shift": spec.shift,
            "scheduler_name": spec.scheduler_name,
            "noise_scale_start": spec.noise_scale_start,
            "noise_scale_end": spec.noise_scale_end,
            "noise_clip_std": spec.noise_clip_std,
        }
        log.info("%s pipeline ready.", spec.label)
        return bundle

    return _load_managed_model(
        model_key,
        factory,
        unload_fn_factory=lambda bundle: lambda: _unload_hidream_o1_bundle(bundle),
    )

def get_hidream_o1_dev_pipe():
    return get_hidream_o1_pipe(MODEL_HIDREAM_O1_DEV)

__all__ = (
    '_ensure_hidream_o1',
    '_hidream_default_rope_parameters',
    '_patch_hidream_rope_registry',
    '_patch_hidream_qwen3_vl_classes',
    '_hidream_model_device',
    '_is_meta_tensor',
    '_repair_hidream_rope_buffers',
    '_get_hidream_o1',
    '_hidream_get_tokenizer',
    '_hidream_add_special_tokens',
    '_unload_hidream_o1_bundle',
    '_get_hidream_o1_spec',
    'get_hidream_o1_pipe',
    'get_hidream_o1_dev_pipe',
)
_seal_runtime_module(globals())
