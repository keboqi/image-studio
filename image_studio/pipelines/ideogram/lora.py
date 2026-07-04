"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError
from image_studio.progress import NO_PROGRESS

# --- extracted runtime implementation ---
import sys as _runtime_sys
import threading
import torch
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

_ideogram4_lora_cache = {}
_ideogram4_lora_lock = threading.RLock()

@dataclass(frozen=True)
class Ideogram4LoRAGroup:
    module_name: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    alpha: float | None = None

    @property
    def rank(self) -> int:
        return int(self.lora_a.shape[0])

    @property
    def scale(self) -> float:
        if self.alpha is None:
            return 1.0
        return float(self.alpha) / max(1, self.rank)

class Ideogram4LoRALinear(torch.nn.Module):
    """Runtime LoRA wrapper for Ideogram 4 Linear-compatible modules."""

    def __init__(self, base: torch.nn.Module, group: Ideogram4LoRAGroup, strength: float):
        super().__init__()
        self.base = base
        self.module_name = group.module_name
        self.in_features = getattr(base, "in_features", int(group.lora_a.shape[1]))
        self.out_features = getattr(base, "out_features", int(group.lora_b.shape[0]))
        self.lora_scale = float(group.scale)
        device, dtype = _ideogram4_module_device_dtype(base)
        self.register_buffer(
            "lora_a",
            group.lora_a.to(device=device, dtype=dtype).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "lora_b",
            group.lora_b.to(device=device, dtype=dtype).contiguous(),
            persistent=False,
        )
        self.lora_strength = float(strength)

    @property
    def weight(self):
        return getattr(self.base, "weight", None)

    @property
    def bias(self):
        return getattr(self.base, "bias", None)

    def set_strength(self, strength: float) -> None:
        self.lora_strength = float(strength)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        strength = float(self.lora_strength)
        if strength == 0.0 or self.lora_scale == 0.0:
            return out
        dtype = x.dtype if x.is_floating_point() else self.lora_a.dtype
        down = torch.nn.functional.linear(x, self.lora_a.to(dtype=dtype))
        up = torch.nn.functional.linear(down, self.lora_b.to(dtype=dtype))
        return out + up * (strength * self.lora_scale)

def _normalize_ideogram4_lora_mode(mode: str | None) -> str:
    value = (mode or IDEOGRAM4_LORA_OFF).strip()
    if value in IDEOGRAM4_LORA_CHOICES:
        return value
    lower = value.lower()
    if lower in {"off", "none", "disabled", "disable"}:
        return IDEOGRAM4_LORA_OFF
    if lower in {"runtime", "adapter", "runtime adapter"}:
        return IDEOGRAM4_LORA_RUNTIME
    if lower in {"fused", "fuse", "fused in memory"}:
        return IDEOGRAM4_LORA_FUSED
    return IDEOGRAM4_LORA_OFF

def _normalize_ideogram4_lora_weight(weight_name: str | None) -> str:
    value = (weight_name or IDEOGRAM4_REALISM_LORA_DEFAULT).strip()
    return value or IDEOGRAM4_REALISM_LORA_DEFAULT

def _ideogram4_lora_short_name(weight_name: str) -> str:
    name = os.path.basename(weight_name)
    return os.path.splitext(name)[0].replace("Realism_Engine_Ideogram", "REI")

def _ideogram4_lora_signature(
    mode: str,
    weight_name: str,
    conditional_strength: float,
    unconditional_strength: float,
) -> tuple:
    if mode == IDEOGRAM4_LORA_FUSED:
        return (
            IDEOGRAM4_REALISM_LORA_REPO,
            weight_name,
            round(float(conditional_strength), 6),
            round(float(unconditional_strength), 6),
        )
    return (IDEOGRAM4_REALISM_LORA_REPO, weight_name)

def _ideogram4_unload_if_fused_lora_changed(
    pipeline: str,
    mode: str,
    weight_name: str,
    conditional_strength: float,
    unconditional_strength: float,
) -> None:
    model_key = IDEOGRAM4_PIPELINE_MODEL_KEYS[_normalize_ideogram4_pipeline(pipeline)]
    pipe = model_mgr.get(model_key)
    if pipe is None:
        return
    state = getattr(pipe, "_image_studio_ideogram4_lora_state", None) or {}
    if state.get("mode") != IDEOGRAM4_LORA_FUSED:
        return
    requested_sig = _ideogram4_lora_signature(
        mode, weight_name, conditional_strength, unconditional_strength
    )
    if mode != IDEOGRAM4_LORA_FUSED or state.get("signature") != requested_sig:
        log.info("Reloading Ideogram 4 pipeline to clear previous fused LoRA.")
        model_mgr.unload(model_key)

def _ideogram4_load_lora_groups(weight_name: str) -> dict[str, Ideogram4LoRAGroup]:
    weight_name = _normalize_ideogram4_lora_weight(weight_name)
    cache_key = (IDEOGRAM4_REALISM_LORA_REPO, weight_name)
    with _ideogram4_lora_lock:
        cached = _ideogram4_lora_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise UserInputError(
                "Realism Engine LoRA needs huggingface_hub and safetensors installed."
            ) from exc

        log.info(
            "Downloading/loading Ideogram Realism Engine LoRA %s from %s...",
            weight_name,
            IDEOGRAM4_REALISM_LORA_REPO,
        )
        try:
            path = hf_hub_download(
                repo_id=IDEOGRAM4_REALISM_LORA_REPO,
                filename=weight_name,
            )
            raw = load_file(path, device="cpu")
        except Exception as exc:
            raise UserInputError(
                f"Could not load Realism Engine LoRA {weight_name} from "
                f"{IDEOGRAM4_REALISM_LORA_REPO}: {exc}"
            ) from exc

        parts: dict[str, dict[str, torch.Tensor]] = {}
        alphas: dict[str, float] = {}
        weight_re = re.compile(r"^(?P<module>.+)\.(?P<part>lora_[AB])\.weight$")
        alpha_re = re.compile(r"^(?P<module>.+)\.(?:alpha|lora_alpha)$")
        for raw_key, tensor in raw.items():
            key = raw_key[len("diffusion_model."):] if raw_key.startswith("diffusion_model.") else raw_key
            weight_match = weight_re.match(key)
            if weight_match:
                module_name = weight_match.group("module")
                parts.setdefault(module_name, {})[weight_match.group("part")] = (
                    tensor.detach().cpu().contiguous()
                )
                continue
            alpha_match = alpha_re.match(key)
            if alpha_match and isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
                alphas[alpha_match.group("module")] = float(tensor.reshape(()).item())

        groups: dict[str, Ideogram4LoRAGroup] = {}
        for module_name, tensors in parts.items():
            lora_a = tensors.get("lora_A")
            lora_b = tensors.get("lora_B")
            if lora_a is None or lora_b is None:
                continue
            if lora_a.ndim != 2 or lora_b.ndim != 2 or lora_a.shape[0] != lora_b.shape[1]:
                log.warning("Skipping malformed LoRA tensors for %s", module_name)
                continue
            groups[module_name] = Ideogram4LoRAGroup(
                module_name=module_name,
                lora_a=lora_a,
                lora_b=lora_b,
                alpha=alphas.get(module_name),
            )
        if not groups:
            raise UserInputError(f"Realism Engine LoRA {weight_name} did not contain usable tensors.")
        _ideogram4_lora_cache[cache_key] = groups
        log.info("Loaded %d Ideogram Realism Engine LoRA target(s).", len(groups))
        return groups

def _ideogram4_matmul_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return dtype
    return torch.bfloat16

def _ideogram4_module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(module, "compute_dtype", None)
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        if tensor is None:
            continue
        device = tensor.device
        if dtype is None and tensor.is_floating_point():
            dtype = tensor.dtype
    return device, _ideogram4_matmul_dtype(dtype or torch.bfloat16)

def _ideogram4_lora_base_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.base if isinstance(module, Ideogram4LoRALinear) else module

def _ideogram4_lora_target_shape(module: torch.nn.Module) -> tuple[int | None, int | None]:
    module = _ideogram4_lora_base_module(module)
    out_features = getattr(module, "out_features", None)
    in_features = getattr(module, "in_features", None)
    if out_features is not None and in_features is not None:
        return int(out_features), int(in_features)
    weight = getattr(module, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[0]), int(weight.shape[1])
    return None, None

def _ideogram4_lora_group_matches(module: torch.nn.Module, group: Ideogram4LoRAGroup) -> bool:
    out_features, in_features = _ideogram4_lora_target_shape(module)
    return (
        out_features == int(group.lora_b.shape[0])
        and in_features == int(group.lora_a.shape[1])
        and int(group.lora_a.shape[0]) == int(group.lora_b.shape[1])
    )

def _ideogram4_replace_submodule(root: torch.nn.Module, module_name: str, new_module: torch.nn.Module):
    parent_path, _, child_name = module_name.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, child_name, new_module)

def _ideogram4_unwrap_lora_adapters(root: torch.nn.Module) -> None:
    for name, child in list(root.named_children()):
        if isinstance(child, Ideogram4LoRALinear):
            setattr(root, name, child.base)
        else:
            _ideogram4_unwrap_lora_adapters(child)

def _ideogram4_set_runtime_lora_strength(root: torch.nn.Module, strength: float) -> int:
    count = 0
    for module in root.modules():
        if isinstance(module, Ideogram4LoRALinear):
            module.set_strength(strength)
            count += 1
    return count

def _ideogram4_install_runtime_lora(
    transformer: torch.nn.Module,
    group: Ideogram4LoRAGroup,
    strength: float,
) -> bool:
    modules = dict(transformer.named_modules())
    module = modules.get(group.module_name)
    if module is None or not _ideogram4_lora_group_matches(module, group):
        return False
    if isinstance(module, Ideogram4LoRALinear):
        module.set_strength(strength)
        return True
    wrapper = Ideogram4LoRALinear(module, group, strength)
    _ideogram4_replace_submodule(transformer, group.module_name, wrapper)
    return True

def _ideogram4_lora_delta(
    module: torch.nn.Module,
    group: Ideogram4LoRAGroup,
    strength: float,
) -> torch.Tensor:
    device, dtype = _ideogram4_module_device_dtype(module)
    lora_a = group.lora_a.to(device=device, dtype=dtype)
    lora_b = group.lora_b.to(device=device, dtype=dtype)
    return torch.matmul(lora_b, lora_a).mul_(float(strength) * group.scale)

def _ideogram4_dequantized_weight(
    module: torch.nn.Module,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    class_name = module.__class__.__name__
    weight = getattr(module, "weight", None)
    if isinstance(module, torch.nn.Linear):
        return module.weight.detach().to(dtype=dtype)
    if class_name == "Fp8Linear" and isinstance(weight, torch.Tensor):
        scale = module.weight_scale.to(dtype=dtype)
        if scale.ndim == 0:
            return weight.detach().to(dtype=dtype) * scale
        return weight.detach().to(dtype=dtype) * scale.unsqueeze(1)
    if class_name == "ComfyQuantLinear" and isinstance(weight, torch.Tensor):
        try:
            if hasattr(weight, "dequantize"):
                return weight.detach().dequantize().to(dtype=dtype)
            return weight.detach().to(dtype=dtype)
        except Exception:
            return None
    return None

def _ideogram4_try_fuse_lora(
    module: torch.nn.Module,
    group: Ideogram4LoRAGroup,
    strength: float,
) -> bool:
    module = _ideogram4_lora_base_module(module)
    if not _ideogram4_lora_group_matches(module, group):
        return False
    class_name = module.__class__.__name__
    device, dtype = _ideogram4_module_device_dtype(module)
    with torch.no_grad():
        delta = _ideogram4_lora_delta(module, group, strength)
        base_weight = _ideogram4_dequantized_weight(module, dtype)
        if base_weight is None:
            return False
        merged = (base_weight.to(device=device, dtype=dtype) + delta).contiguous()

        if isinstance(module, torch.nn.Linear):
            module.weight.copy_(merged.to(dtype=module.weight.dtype))
            return True

        if class_name == "Fp8Linear":
            try:
                from ideogram4.quantized_loading import quantize_weight_to_fp8

                q_weight, scale = quantize_weight_to_fp8(merged)
                module.weight = q_weight.to(device=device)
                module.weight_scale = scale.to(device=device, dtype=torch.float32)
                return True
            except Exception as exc:
                log.warning("Could not fuse LoRA into FP8 layer %s: %s", group.module_name, exc)
                return False

        if class_name == "ComfyQuantLinear":
            try:
                from ideogram4.quantized_loading import _load_comfy_kitchen

                QuantizedTensor, _get_layout_class = _load_comfy_kitchen()
                q_weight = QuantizedTensor.from_float(merged, module.layout_type)
                module.weight = torch.nn.Parameter(q_weight, requires_grad=False)
                return True
            except Exception as exc:
                log.warning(
                    "Could not fuse LoRA into Comfy quant layer %s: %s",
                    group.module_name,
                    exc,
                )
                return False

    return False

def _ideogram4_apply_lora_to_transformer(
    transformer: torch.nn.Module,
    groups: dict[str, Ideogram4LoRAGroup],
    strength: float,
    mode: str,
    branch_name: str,
) -> tuple[int, int, int]:
    strength = float(strength)
    if strength == 0.0:
        return 0, 0, 0
    modules = dict(transformer.named_modules())
    matched = 0
    fused = 0
    runtime = 0
    for module_name, group in groups.items():
        module = modules.get(module_name)
        if module is None or not _ideogram4_lora_group_matches(module, group):
            continue
        matched += 1
        if mode == IDEOGRAM4_LORA_FUSED and _ideogram4_try_fuse_lora(module, group, strength):
            fused += 1
            continue
        if _ideogram4_install_runtime_lora(transformer, group, strength):
            runtime += 1
            modules = dict(transformer.named_modules())
    if matched == 0:
        raise UserInputError(
            f"Realism Engine LoRA did not match any Ideogram 4 {branch_name} modules."
        )
    return matched, fused, runtime

def _apply_ideogram4_lora(
    pipe,
    mode: str,
    weight_name: str,
    conditional_strength: float,
    unconditional_strength: float,
    progress=NO_PROGRESS,
) -> str:
    mode = _normalize_ideogram4_lora_mode(mode)
    weight_name = _normalize_ideogram4_lora_weight(weight_name)
    conditional_strength = float(conditional_strength or 0.0)
    unconditional_strength = float(unconditional_strength or 0.0)
    state = getattr(pipe, "_image_studio_ideogram4_lora_state", None) or {}

    if mode == IDEOGRAM4_LORA_OFF:
        if state.get("mode") != IDEOGRAM4_LORA_FUSED:
            for attr in ("conditional_transformer", "unconditional_transformer"):
                transformer = getattr(pipe, attr, None)
                if transformer is not None:
                    _ideogram4_unwrap_lora_adapters(transformer)
        pipe._image_studio_ideogram4_lora_state = {"mode": IDEOGRAM4_LORA_OFF}
        return ""
    if conditional_strength == 0.0 and unconditional_strength == 0.0:
        if state.get("mode") != IDEOGRAM4_LORA_FUSED:
            for attr in ("conditional_transformer", "unconditional_transformer"):
                transformer = getattr(pipe, attr, None)
                if transformer is not None:
                    _ideogram4_unwrap_lora_adapters(transformer)
        pipe._image_studio_ideogram4_lora_state = {"mode": IDEOGRAM4_LORA_OFF}
        return ""

    signature = _ideogram4_lora_signature(
        mode, weight_name, conditional_strength, unconditional_strength
    )
    if state.get("mode") == mode and state.get("signature") == signature:
        if mode == IDEOGRAM4_LORA_RUNTIME:
            _ideogram4_set_runtime_lora_strength(pipe.conditional_transformer, conditional_strength)
            _ideogram4_set_runtime_lora_strength(pipe.unconditional_transformer, unconditional_strength)
        return state.get("status", "")

    if state.get("mode") != IDEOGRAM4_LORA_FUSED:
        for attr in ("conditional_transformer", "unconditional_transformer"):
            transformer = getattr(pipe, attr, None)
            if transformer is not None:
                _ideogram4_unwrap_lora_adapters(transformer)

    progress(0.28, desc=f"Loading Realism Engine LoRA ({_ideogram4_lora_short_name(weight_name)})...")
    groups = _ideogram4_load_lora_groups(weight_name)
    cond_counts = _ideogram4_apply_lora_to_transformer(
        pipe.conditional_transformer,
        groups,
        conditional_strength,
        mode,
        "conditional",
    )
    uncond_counts = _ideogram4_apply_lora_to_transformer(
        pipe.unconditional_transformer,
        groups,
        unconditional_strength,
        mode,
        "unconditional",
    )

    fused_count = cond_counts[1] + uncond_counts[1]
    runtime_count = cond_counts[2] + uncond_counts[2]
    if mode == IDEOGRAM4_LORA_FUSED and runtime_count:
        log.warning(
            "Some Realism Engine LoRA layers could not be fused and are using "
            "runtime adapters instead."
        )
    status = (
        f"{_ideogram4_lora_short_name(weight_name)} {mode} "
        f"c{conditional_strength:.2g}/u{unconditional_strength:.2g}"
    )
    if mode == IDEOGRAM4_LORA_FUSED:
        status += f" ({fused_count} fused"
        if runtime_count:
            status += f", {runtime_count} runtime"
        status += ")"
    pipe._image_studio_ideogram4_lora_state = {
        "mode": mode,
        "signature": signature,
        "status": status,
    }
    log.info("Ideogram 4 LoRA ready: %s", status)
    return status

__all__ = (
    'Ideogram4LoRAGroup',
    'Ideogram4LoRALinear',
    '_normalize_ideogram4_lora_mode',
    '_normalize_ideogram4_lora_weight',
    '_ideogram4_lora_short_name',
    '_ideogram4_lora_signature',
    '_ideogram4_unload_if_fused_lora_changed',
    '_ideogram4_load_lora_groups',
    '_ideogram4_matmul_dtype',
    '_ideogram4_module_device_dtype',
    '_ideogram4_lora_base_module',
    '_ideogram4_lora_target_shape',
    '_ideogram4_lora_group_matches',
    '_ideogram4_replace_submodule',
    '_ideogram4_unwrap_lora_adapters',
    '_ideogram4_set_runtime_lora_strength',
    '_ideogram4_install_runtime_lora',
    '_ideogram4_lora_delta',
    '_ideogram4_dequantized_weight',
    '_ideogram4_try_fuse_lora',
    '_ideogram4_apply_lora_to_transformer',
    '_apply_ideogram4_lora',
)
_seal_runtime_module(globals())
