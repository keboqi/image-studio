"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import AppError, BackendUnavailableError
from image_studio.services.managed_runtime import ManagedScriptConfig, ManagedScriptService

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

class DiffusionGemmaVllmService(ManagedScriptService):
    """OpenAI-compatible DiffusionGemma backend managed by deploy_diffusiongemma_vllm.sh."""

    def __init__(self):
        super().__init__(ManagedScriptConfig(
            label="DiffusionGemma vLLM",
            manager_key=MODEL_DIFFUSIONGEMMA_VLLM,
            vram_mb=MODEL_SPECS[MODEL_DIFFUSIONGEMMA_VLLM].vram_mb,
            script=DIFFUSIONGEMMA_VLLM_SCRIPT,
            shell=DIFFUSIONGEMMA_VLLM_BASH,
            shell_env_name="DIFFUSIONGEMMA_VLLM_BASH",
            ready_timeout=DIFFUSIONGEMMA_VLLM_READY_TIMEOUT,
            start_timeout=DIFFUSIONGEMMA_VLLM_START_TIMEOUT,
            request_timeout=DIFFUSIONGEMMA_VLLM_REQUEST_TIMEOUT,
        ))
        self.api_base = DIFFUSIONGEMMA_VLLM_API_BASE
        self.model = DIFFUSIONGEMMA_VLLM_MODEL

    def _script_env(self) -> dict[str, str]:
        env = super()._script_env()
        env["PORT"] = DIFFUSIONGEMMA_VLLM_PORT
        env["SERVED_MODEL_NAME"] = self.model
        env["READY_TIMEOUT"] = str(self.config.ready_timeout)
        env["REQUEST_TIMEOUT"] = str(self.config.request_timeout)
        env["RESTART_POLICY"] = DIFFUSIONGEMMA_VLLM_RESTART_POLICY
        env["WARMUP_ON_START"] = DIFFUSIONGEMMA_VLLM_WARMUP_ON_START
        env["DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL"] = DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL
        env["SLEEP_LEVEL"] = DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL
        if DIFFUSIONGEMMA_VLLM_HF_MODEL:
            env["MODEL"] = DIFFUSIONGEMMA_VLLM_HF_MODEL
        return env

    def is_healthy(self) -> bool:
        if _NO_BOOTSTRAP:
            return False
        try:
            req = urllib.request.Request(f"{self.api_base}/models", method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                return 200 <= res.status < 300
        except Exception:
            return False

    def _control_url(self, path: str) -> str:
        base = self.api_base[:-3] if self.api_base.endswith("/v1") else self.api_base
        return f"{base}{path}"

    def is_sleeping(self) -> bool:
        if _NO_BOOTSTRAP:
            return False
        try:
            req = urllib.request.Request(self._control_url("/is_sleeping"), method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                body = res.read().decode("utf-8", errors="replace").strip()
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    return bool(data.get("is_sleeping", data.get("sleeping", False)))
                if isinstance(data, bool):
                    return data
            except Exception:
                pass
            body_lc = body.lower()
            return "true" in body_lc and "false" not in body_lc
        except Exception:
            return False

    def is_ready(self) -> bool:
        return self.is_healthy() and not self.is_sleeping()

    def is_control_reachable(self) -> bool:
        if _NO_BOOTSTRAP:
            return False
        try:
            req = urllib.request.Request(self._control_url("/is_sleeping"), method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                return 200 <= res.status < 300
        except Exception:
            return False

    def wake(self):
        with self.lock:
            if self.is_ready():
                return
            if not self.is_healthy() and not self.is_control_reachable():
                return
            res = self._run_script("wake", DIFFUSIONGEMMA_VLLM_READY_TIMEOUT)
            if res.returncode != 0:
                raise BackendUnavailableError(
                    "Failed to wake DiffusionGemma vLLM backend.\n"
                    f"STDOUT:\n{self._tail(res.stdout)}\n\n"
                    f"STDERR:\n{self._tail(res.stderr)}"
                )

    def _wake_existing(self) -> bool:
        if not self.is_healthy() and not self.is_control_reachable():
            return False
        self.wake()
        return True

    def ensure_running(self):
        self._ensure_running(
            self.is_ready,
            self.api_base,
            prepare_existing=self._wake_existing,
        )

    def stop(self):
        with self.lock:
            action = "sleep" if DIFFUSIONGEMMA_VLLM_UNLOAD_MODE == "sleep" else "stop"
            fallback = "stop" if action == "sleep" else None
            self._stop_script(action, fallback_action=fallback)

    @staticmethod
    def _image_url_for_openai(part: dict[str, Any]) -> str:
        image_url = part.get("url") or part.get("image_url") or part.get("image")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        if not image_url:
            raise BackendUnavailableError("DiffusionGemma vLLM image input is missing an image URL/path.")
        image_url = str(image_url)
        if image_url.startswith(("data:", "http://", "https://")):
            return image_url
        if not os.path.isfile(image_url):
            raise BackendUnavailableError(f"DiffusionGemma vLLM image input not found: {image_url}")

        ext = os.path.splitext(image_url)[1].lower()
        mime = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(ext, "image/png")
        with open(image_url, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @classmethod
    def _messages_to_openai_messages(cls, messages) -> list[dict[str, Any]]:
        openai_messages = []
        for msg in GemmaService._normalise_messages(messages):
            content_parts = []
            has_image = False
            for part in GemmaService._content_parts(msg.get("content")):
                part_type = part.get("type", "text")
                if part_type == "text":
                    text = str(part.get("text", ""))
                    if text:
                        content_parts.append({"type": "text", "text": text})
                elif part_type in {"image", "image_url"}:
                    has_image = True
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": cls._image_url_for_openai(part)},
                    })
                elif part_type == "audio":
                    raise BackendUnavailableError(
                        "DiffusionGemma vLLM does not support audio input. "
                        "Use a local Gemma option for audio chat."
                    )
                elif part_type == "video":
                    raise BackendUnavailableError("DiffusionGemma vLLM does not support video input in this WebUI.")
                else:
                    raise BackendUnavailableError(f"DiffusionGemma vLLM does not support {part_type!r} input.")

            if has_image:
                content: str | list[dict[str, Any]] = content_parts
            else:
                content = "\n".join(
                    part["text"] for part in content_parts
                    if part.get("type") == "text" and part.get("text")
                ).strip()
            openai_messages.append({
                "role": msg.get("role", "user"),
                "content": content,
            })
        return openai_messages

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=DIFFUSIONGEMMA_VLLM_REQUEST_TIMEOUT) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise BackendUnavailableError(
                f"DiffusionGemma vLLM request failed ({exc.code}).\n{self._tail(detail)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailableError(f"DiffusionGemma vLLM request failed: {exc}") from exc

    @staticmethod
    def _message_content_text(message: dict[str, Any]) -> str:
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
                else:
                    parts.append(str(part))
            content = "\n".join(part for part in parts if part)
        if content:
            return str(content)
        return str(message.get("reasoning_content") or message.get("reasoning") or "").strip()

    def generate(
        self,
        messages,
        max_new_tokens=1024,
        enable_thinking=False,
        do_sample=True,
    ) -> str:
        openai_messages = self._messages_to_openai_messages(messages)
        self.ensure_running()
        model_mgr.touch(self.mgr_key)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": int(max_new_tokens or 1024),
            "temperature": 1.0 if do_sample else 0.0,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
        }
        if do_sample:
            payload["top_p"] = 0.95

        t_generate = time.perf_counter()
        try:
            data = self._post_chat(payload)
        except AppError as exc:
            payload.pop("chat_template_kwargs", None)
            log.warning(
                "DiffusionGemma vLLM request with chat_template_kwargs failed; retrying without it: %s",
                exc,
            )
            data = self._post_chat(payload)

        choices = data.get("choices") or []
        if not choices:
            raise BackendUnavailableError(f"DiffusionGemma vLLM returned no choices: {data}")
        message = choices[0].get("message") or {}
        text = self._message_content_text(message)
        elapsed = max(1e-6, time.perf_counter() - t_generate)
        usage = data.get("usage") or {}
        out_tokens = int(usage.get("completion_tokens") or 0)
        log.info(
            "DiffusionGemma vLLM generate | input_tokens=%s | output_tokens=%s | elapsed=%.2fs | tok/s=%.2f",
            usage.get("prompt_tokens", "?"),
            usage.get("completion_tokens", "?"),
            elapsed,
            (out_tokens / elapsed) if out_tokens else 0.0,
        )
        return text.strip()

__all__ = (
    'DiffusionGemmaVllmService',
)
_seal_runtime_module(globals())
