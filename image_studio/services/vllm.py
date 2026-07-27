"""OpenAI-compatible DiffusionGemma managed backend."""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any

from image_studio.config import VllmConfig
from image_studio.errors import AppError, BackendUnavailableError
from image_studio.infra.managed_service import ManagedScriptConfig, ManagedScriptService
from image_studio.infra.model_manager import ModelManager

log = logging.getLogger(__name__)


class DiffusionGemmaVllmService(ManagedScriptService):
    """OpenAI-compatible DiffusionGemma backend managed by deploy_diffusiongemma_vllm.sh."""

    def __init__(
        self,
        config: VllmConfig,
        *,
        model_manager: ModelManager | None = None,
        model_key: str = "diffusiongemma_vllm",
        vram_mb: int = 22_000,
        execution_lock: Any = None,
        bootstrap_allowed: bool = True,
    ) -> None:
        super().__init__(
            ManagedScriptConfig(
                label="DiffusionGemma vLLM",
                manager_key=model_key,
                vram_mb=vram_mb,
                script=str(config.script),
                shell=config.shell,
                shell_env_name="DIFFUSIONGEMMA_VLLM_BASH",
                ready_timeout=config.ready_timeout,
                start_timeout=config.start_timeout,
                request_timeout=config.request_timeout,
                working_dir=str(config.script.parent),
            ),
            model_manager=model_manager,
            execution_lock=execution_lock,
            bootstrap_allowed=bootstrap_allowed,
            logger=log,
        )
        self.api_base = config.api_base
        self.model = config.model
        self.port = str(config.port)
        self.hf_model = config.hf_model
        self.restart_policy = config.restart_policy
        self.warmup_on_start = config.warmup_on_start
        self.unload_mode = config.unload_mode
        self.sleep_level = config.sleep_level
        self.request_timeout = config.request_timeout

    def _script_env(self) -> dict[str, str]:
        env = super()._script_env()
        env["PORT"] = self.port
        env["SERVED_MODEL_NAME"] = self.model
        env["READY_TIMEOUT"] = str(self.config.ready_timeout)
        env["REQUEST_TIMEOUT"] = str(self.config.request_timeout)
        env["RESTART_POLICY"] = self.restart_policy
        env["WARMUP_ON_START"] = self.warmup_on_start
        env["DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL"] = self.sleep_level
        env["SLEEP_LEVEL"] = self.sleep_level
        if self.hf_model:
            env["MODEL"] = self.hf_model
        return env

    def is_healthy(self) -> bool:
        if not self.bootstrap_allowed:
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
        if not self.bootstrap_allowed:
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
        if not self.bootstrap_allowed:
            return False
        try:
            req = urllib.request.Request(self._control_url("/is_sleeping"), method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                return 200 <= res.status < 300
        except Exception:
            return False

    def _has_reachable_backend(self) -> bool:
        return self.is_healthy() or self.is_control_reachable()

    def wake(self) -> bool:
        with self.lock:
            if self.is_ready():
                return True
            if not self._has_reachable_backend():
                return False
            res = self._run_script("wake", self.config.ready_timeout)
            if res.returncode != 0:
                raise BackendUnavailableError(
                    "Failed to wake DiffusionGemma vLLM backend.\n"
                    f"STDOUT:\n{self._tail(res.stdout)}\n\n"
                    f"STDERR:\n{self._tail(res.stderr)}"
                )
            return True

    def _wake_existing(self) -> bool:
        return self.wake()

    def ensure_running(self):
        self._ensure_running(
            self.is_ready,
            self.api_base,
            prepare_existing=self._wake_existing,
        )

    def stop(self):
        with self.lock:
            action = "sleep" if self.unload_mode == "sleep" else "stop"
            fallback = "stop" if action == "sleep" else None
            self._stop_script(action, fallback_action=fallback)

    def stop_process(self):
        with self.lock:
            self._stop_script("stop")

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

    @staticmethod
    def _normalise_messages(messages: Any) -> list[dict[str, Any]]:
        normalised = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            normalised.append({"role": message["role"], "content": content})
        return normalised

    @staticmethod
    def _content_parts(content: Any) -> list[dict[str, Any]]:
        if isinstance(content, list):
            return [part for part in content if isinstance(part, dict)]
        if isinstance(content, dict):
            return [content]
        return [{"type": "text", "text": str(content)}]

    @classmethod
    def _messages_to_openai_messages(cls, messages: Any) -> list[dict[str, Any]]:
        openai_messages = []
        for msg in cls._normalise_messages(messages):
            content_parts: list[dict[str, Any]] = []
            has_image = False
            for part in cls._content_parts(msg.get("content")):
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
            with urllib.request.urlopen(req, timeout=self.request_timeout) as res:
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
        if self.model_manager is not None:
            self.model_manager.touch(self.mgr_key)

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
