"""Lazy, managed access to the official SenseNova-U1.5 inference package."""

from __future__ import annotations

from image_studio.errors import UserInputError
from image_studio.runtime_access import runtime_namespace as _runtime


def _import_sensenova():
    try:
        import sensenova_u1
        from huggingface_hub import hf_hub_download
        from sensenova_u1.utils import (
            load_and_merge_lora_weight_from_safetensors,
            load_model_and_tokenizer,
        )
    except ImportError as exc:
        raise UserInputError(
            "SenseNova-U1 is not installed. Install the pinned official upstream "
            f"package and restart Image Studio. Import error: {exc}"
        ) from exc
    return {
        "set_attn_backend": sensenova_u1.set_attn_backend,
        "load_model_and_tokenizer": load_model_and_tokenizer,
        "load_lora": load_and_merge_lora_weight_from_safetensors,
        "hf_hub_download": hf_hub_download,
    }


def _unload_sensenova_bundle(bundle: dict) -> None:
    model = bundle.get("model") if isinstance(bundle, dict) else None
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    if isinstance(bundle, dict):
        bundle.clear()


def get_sensenova_pipe(model_key: str):
    if model_key not in _runtime().SENSENOVA_MODEL_KEYS:
        raise ValueError(f"Unknown SenseNova model key: {model_key!r}")

    def factory():
        mods = _import_sensenova()
        mods["set_attn_backend"]("auto")
        _runtime().log.info("Loading SenseNova U1.5 from %s...", _runtime().SENSENOVA_MODEL_ID)
        model, tokenizer = mods["load_model_and_tokenizer"](
            _runtime().SENSENOVA_MODEL_ID,
            dtype=_runtime().torch.bfloat16,
            device="cuda",
            device_map=None,
            max_memory=None,
            gguf_checkpoint=None,
            for_offload=False,
        )
        fast = model_key == _runtime().MODEL_SENSENOVA_FAST
        if fast:
            lora_path = mods["hf_hub_download"](
                repo_id=_runtime().SENSENOVA_LORA_REPO,
                filename=_runtime().SENSENOVA_LORA_FILENAME,
            )
            mods["load_lora"](model, lora_path)
        model.eval()
        return {
            "model": model,
            "tokenizer": tokenizer,
            "steps": 8 if fast else 50,
            "cfg_scale": 1.0 if fast else 4.0,
            "label": "8-step LoRA" if fast else "50-step base",
        }

    return _runtime()._load_managed_model(
        model_key,
        factory,
        unload_fn_factory=lambda bundle: lambda: _unload_sensenova_bundle(bundle),
    )


def sensenova_output_to_pil(output):
    """Convert the upstream normalized BCHW tensor (or PIL result) to PIL."""
    if isinstance(output, (list, tuple)):
        if not output:
            raise UserInputError("SenseNova returned no images.")
        output = output[0]
    if isinstance(output, _runtime().Image.Image):
        return output.convert("RGB")
    if getattr(output, "ndim", None) == 4:
        output = output[0]
    if getattr(output, "ndim", None) != 3:
        raise UserInputError("SenseNova returned an unsupported image result.")
    array = (
        (output.detach().float() * 0.5 + 0.5)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return _runtime().Image.fromarray((array * 255).round().astype("uint8"), mode="RGB")


__all__ = ("get_sensenova_pipe", "sensenova_output_to_pil")
