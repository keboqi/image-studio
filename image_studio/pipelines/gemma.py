"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module


class ChatModelSelector:
    """Own the selected chat model and active local service."""

    def __init__(self, choice: str):
        self.choice = choice
        self.service = None

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

@dataclass(frozen=True)
class GemmaModelSpec:
    key: str
    model_id: str
    label: str
    mgr_key: str
    vram_mb: float
    assistant_model_id: str = ""
    trust_remote_code: bool = False
    supports_audio: bool = True
    supports_visual: bool = True

class GemmaService:
    """Owns one selected Gemma-family model for prompt enhancement and chat."""

    def __init__(self, spec: GemmaModelSpec):
        self.spec = spec
        self.model_id = spec.model_id
        self.assistant_model_id = (spec.assistant_model_id or "").strip()
        self.mgr_key = spec.mgr_key
        self.vram_mb = spec.vram_mb
        self.trust_remote_code = spec.trust_remote_code
        self.label = spec.label
        self.model = None
        self.assistant_model = None
        self.processor = None

    def load(self):
        with _model_load_lock:
            if self.model is not None:
                return self.model, self.processor

            model_mgr.ensure_vram(self.vram_mb, exclude=self.mgr_key)
            log.info("Loading %s (%s) ...", self.label, self.model_id)
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForMultimodalLM

            processor_kwargs = {"padding_side": "left"}
            model_kwargs = {"torch_dtype": "auto", "device_map": "auto"}
            if self.trust_remote_code:
                processor_kwargs["trust_remote_code"] = True
                model_kwargs["trust_remote_code"] = True

            self.processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
            self.model = AutoModelForMultimodalLM.from_pretrained(self.model_id, **model_kwargs)

            if self.assistant_model_id:
                try:
                    log.info("Loading Gemma MTP assistant %s ...", self.assistant_model_id)
                    self.assistant_model = AutoModelForCausalLM.from_pretrained(
                        self.assistant_model_id,
                        torch_dtype="auto",
                        device_map="auto",
                        trust_remote_code=True,
                    )
                    log.info("Gemma MTP assistant ready.")
                except Exception as exc:
                    self.assistant_model = None
                    log.warning("Gemma MTP assistant unavailable; continuing without it: %s", exc)
            model_mgr.register(
                self.mgr_key, self.model, self.vram_mb,
                unload_fn=self.unload,
            )
            log.info("%s ready.", self.label)
            return self.model, self.processor

    def unload(self):
        if self.assistant_model is not None:
            try:
                self.assistant_model.to("cpu")
            except Exception:
                pass
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception:
                pass
        self.assistant_model = None
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("%s unloaded.", self.label)

    @staticmethod
    def _normalise_messages(messages) -> list[dict]:
        normalised = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalised.append({"role": msg["role"], "content": content})
        return normalised

    @staticmethod
    def _content_parts(content) -> list[dict]:
        if isinstance(content, list):
            return [part for part in content if isinstance(part, dict)]
        if isinstance(content, dict):
            return [content]
        return [{"type": "text", "text": str(content)}]

    @classmethod
    def _messages_have_non_text_modality(cls, messages) -> bool:
        for msg in messages:
            for part in cls._content_parts(msg.get("content")):
                if part.get("type", "text") != "text":
                    return True
        return False

    def _validate_messages(self, messages):
        for msg in messages:
            for part in self._content_parts(msg.get("content")):
                part_type = part.get("type", "text")
                if part_type == "audio" and not self.spec.supports_audio:
                    raise UserInputError(f"{self.label} does not support audio input.")
                if part_type in {"image", "video"} and not self.spec.supports_visual:
                    raise UserInputError(f"{self.label} does not support {part_type} input.")

    @staticmethod
    def _input_device(model) -> torch.device:
        device = getattr(model, "device", None)
        if isinstance(device, torch.device) and device.type != "meta":
            return device
        try:
            for param in model.parameters():
                if param.device.type != "meta":
                    return param.device
        except Exception:
            pass
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(
        self,
        messages,
        max_new_tokens=1024,
        enable_thinking=False,
        do_sample=True,
    ):
        with _inprocess_gpu_lock:
            return self._generate_locked(
                messages,
                max_new_tokens=max_new_tokens,
                enable_thinking=enable_thinking,
                do_sample=do_sample,
            )

    def _generate_locked(
        self,
        messages,
        max_new_tokens=1024,
        enable_thinking=False,
        do_sample=True,
    ):
        model, processor = self.load()
        model_mgr.touch(self.mgr_key)

        normalised = self._normalise_messages(messages)
        self._validate_messages(normalised)
        has_non_text = self._messages_have_non_text_modality(normalised)

        inputs = processor.apply_chat_template(
            normalised, tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        if isinstance(inputs, torch.Tensor):
            inputs = {"input_ids": inputs}
        target_device = self._input_device(model)
        if hasattr(inputs, "to"):
            inputs = inputs.to(target_device)
        else:
            inputs = {
                key: value.to(target_device) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
        input_len = inputs["input_ids"].shape[-1]
        if enable_thinking:
            do_sample = True

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generate_kwargs.update({"temperature": 1.0, "top_p": 0.95, "top_k": 64})

        assistant_allowed = (
            self.assistant_model is not None
            and not has_non_text
        )
        if assistant_allowed:
            generate_kwargs["assistant_model"] = self.assistant_model
            generate_kwargs["num_assistant_tokens"] = GEMMA_NUM_ASSISTANT_TOKENS
        t_generate = time.perf_counter()
        with torch.inference_mode():
            try:
                outputs = model.generate(**inputs, **generate_kwargs)
            except Exception as exc:
                assistant = generate_kwargs.pop("assistant_model", None)
                if assistant is None:
                    raise
                log.warning("Gemma MTP assistant failed during generate; retrying without it: %s", exc)
                try:
                    assistant.to("cpu")
                except Exception:
                    pass
                self.assistant_model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                outputs = model.generate(**inputs, **generate_kwargs)

        generated = outputs[0][input_len:]
        elapsed = max(1e-6, time.perf_counter() - t_generate)
        out_tokens = int(generated.numel())
        log.info(
            "Gemma generate | model=%s | assistant=%s | input_tokens=%d | output_tokens=%d | elapsed=%.2fs | tok/s=%.2f",
            self.label,
            bool(assistant_allowed),
            int(input_len),
            out_tokens,
            elapsed,
            out_tokens / elapsed,
        )
        raw = processor.decode(generated, skip_special_tokens=False)
        try:
            parsed = processor.parse_response(raw)
            if isinstance(parsed, dict) and "content" in parsed:
                return parsed["content"]
            return str(parsed)
        except Exception:
            return re.sub(r"<[^>]+>", "", raw).strip()

def _get_active_gemma_service() -> GemmaService:
    """Return whichever GemmaService the Chat tab currently has selected.

    Resolves the service based on the Chat tab's current choice (or the
    default if no explicit choice has been made yet).  This lazily creates
    the Huihui service on first access when it is the default.
    """
    if _normalize_chat_gemma_choice(_chat_selector.choice) == CHAT_DIFFUSIONGEMMA_VLLM:
        raise UserInputError("DiffusionGemma vLLM is managed through the backend script, not as a local Gemma model.")
    return _resolve_chat_gemma_service(_chat_selector.choice)

def _load_gemma():
    return _get_active_gemma_service().load()

def _unload_gemma():
    service = _get_active_gemma_service()
    if model_mgr.is_loaded(service.mgr_key):
        model_mgr.unload(service.mgr_key)
    else:
        service.unload()

def _gemma_generate(
    messages,
    max_new_tokens=1024,
    enable_thinking=False,
    do_sample=True,
    chat_model: str | None = None,
):
    """Low-level generate helper that reuses the Chat tab's selected Gemma model."""
    choice = _normalize_chat_gemma_choice(chat_model or _chat_selector.choice)
    if choice == CHAT_DIFFUSIONGEMMA_VLLM:
        return _diffusiongemma_vllm_service.generate(
            messages, max_new_tokens, enable_thinking, do_sample
        )
    return _resolve_chat_gemma_service(choice).generate(
        messages, max_new_tokens, enable_thinking, do_sample
    )

def _normalize_chat_gemma_choice(choice: str | None) -> str:
    choice = (choice or CHAT_GEMMA_DEFAULT).strip()
    if choice in CHAT_GEMMA_CHOICES:
        return choice
    return CHAT_GEMMA_LABEL_TO_KEY.get(choice, CHAT_GEMMA_DEFAULT)

def _resolve_chat_gemma_service(choice: str | None) -> GemmaService:
    """Return the Gemma service backing the Chat tab for *choice*."""
    choice = _normalize_chat_gemma_choice(choice)
    if choice == CHAT_DIFFUSIONGEMMA_VLLM:
        raise UserInputError("DiffusionGemma vLLM does not have a local Gemma service.")
    if _chat_selector.service is not None and _chat_selector.choice == choice:
        return _chat_selector.service

    if _chat_selector.service is not None:
        log.info(
            "Gemma selection changed: %s -> %s; unloading previous model.",
            _chat_selector.choice,
            choice,
        )
        if model_mgr.is_loaded(_chat_selector.service.mgr_key):
            model_mgr.unload(_chat_selector.service.mgr_key)
        else:
            _chat_selector.service.unload()

    _chat_selector.choice = choice
    _chat_selector.service = GemmaService(GEMMA_MODEL_SPECS[choice])
    return _chat_selector.service

def _set_chat_gemma_choice(choice: str | None):
    """Update the selected chat backend, unloading a previous local Gemma if needed."""
    choice = _normalize_chat_gemma_choice(choice)
    if choice == CHAT_DIFFUSIONGEMMA_VLLM:
        if _chat_selector.service is not None:
            log.info("Gemma selection changed: %s -> %s; unloading previous model.", _chat_selector.choice, choice)
            if model_mgr.is_loaded(_chat_selector.service.mgr_key):
                model_mgr.unload(_chat_selector.service.mgr_key)
            else:
                _chat_selector.service.unload()
        _chat_selector.service = None
        _chat_selector.choice = choice
        return

    _resolve_chat_gemma_service(choice)

def _chat_gemma_generate(
    messages,
    chat_model: str,
    max_new_tokens=1024,
    enable_thinking=False,
    do_sample=True,
):
    """Low-level generate helper for Chat tab (model-selectable)."""
    return _gemma_generate(
        messages,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
        do_sample=do_sample,
        chat_model=chat_model,
    )

def chat_model_changed(new_model: str, history: list):
    """Clear chat history when the user switches Gemma chat models."""
    with _inprocess_gpu_lock:
        _set_chat_gemma_choice(new_model)
    if history:
        log.info("Chat model changed; conversation cleared.")
    return [], "", None, None, _build_vram_widget_md()

__all__ = (
    'ChatModelSelector',
    'GemmaModelSpec',
    'GemmaService',
    '_get_active_gemma_service',
    '_load_gemma',
    '_unload_gemma',
    '_gemma_generate',
    '_normalize_chat_gemma_choice',
    '_resolve_chat_gemma_service',
    '_set_chat_gemma_choice',
    '_chat_gemma_generate',
    'chat_model_changed',
)
_seal_runtime_module(globals())
