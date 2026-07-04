"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import BackendUnavailableError
from image_studio.services.managed_runtime import ManagedScriptConfig, ManagedScriptService

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

class Krea2ComfyService(ManagedScriptService):
    """Krea-2 backend managed in an isolated ComfyUI venv."""

    def __init__(self):
        super().__init__(ManagedScriptConfig(
            label="Krea2 ComfyUI",
            manager_key=MODEL_KREA2_TURBO_NVFP4,
            vram_mb=MODEL_SPECS[MODEL_KREA2_TURBO_NVFP4].vram_mb,
            script=KREA2_COMFY_SCRIPT,
            shell=KREA2_COMFY_BASH,
            shell_env_name="KREA2_COMFY_BASH",
            ready_timeout=KREA2_COMFY_READY_TIMEOUT,
            start_timeout=KREA2_COMFY_START_TIMEOUT,
            request_timeout=KREA2_COMFY_REQUEST_TIMEOUT,
        ))
        self.server_base = KREA2_COMFY_SERVER_BASE
        self.comfy_dir = os.path.join(KREA2_COMFY_DIR, "ComfyUI")

    def _script_env(self) -> dict[str, str]:
        env = super()._script_env()
        env["PORT"] = KREA2_COMFY_PORT
        env["HOST"] = "0.0.0.0"
        env["KREA2_COMFY_PORT"] = KREA2_COMFY_PORT
        env["READY_TIMEOUT"] = str(self.config.ready_timeout)
        env["REQUEST_TIMEOUT"] = str(self.config.request_timeout)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        return env

    def is_healthy(self) -> bool:
        if _NO_BOOTSTRAP:
            return False
        try:
            req = urllib.request.Request(f"{self.server_base}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=2) as res:
                if 200 <= res.status < 300:
                    return True
        except Exception:
            pass
        return False

    def ensure_running(self):
        self._ensure_running(self.is_healthy, self.server_base)

    def stop(self):
        with self.lock:
            self._stop_script()

    def _comfy_request(self, path: str, payload: dict | None = None, timeout: int = 60) -> dict:
        """Send a request to the ComfyUI API and return parsed JSON."""
        url = f"{self.server_base}{path}"
        if payload is None:
            req = urllib.request.Request(url, method="GET")
        else:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                return json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise BackendUnavailableError(
                f"Krea2 ComfyUI request failed ({exc.code}).\n{self._tail(detail)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise BackendUnavailableError(f"Krea2 ComfyUI request failed: {exc}") from exc

    @staticmethod
    def _build_workflow(
        prompt: str, width: int, height: int, steps: int,
        cfg: float, seed: int, sampler: str = "euler",
        scheduler: str = "simple", denoise: float = 1.0,
        prefix: str = "Krea2_turbo",
    ) -> dict:
        """Build ComfyUI workflow JSON matching the Krea-2 Turbo core graph."""
        return {
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": "krea2_turbo_nvfp4.safetensors",
                    "weight_dtype": "default",
                },
            },
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": "qwen3vl_4b_fp8_scaled.safetensors",
                    "type": "krea2",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": "qwen_image_vae.safetensors",
                },
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "clip": ["2", 0],
                    "text": prompt,
                },
            },
            "5": {
                "class_type": "ConditioningZeroOut",
                "inputs": {
                    "conditioning": ["4", 0],
                },
            },
            "6": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1,
                },
            },
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["6", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": denoise,
                },
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["3", 0],
                },
            },
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["8", 0],
                    "filename_prefix": prefix,
                },
            },
        }

    @staticmethod
    def _build_workflow_with_latent(
        prompt: str, width: int, height: int, steps: int,
        cfg: float, seed: int, sampler: str = "euler",
        scheduler: str = "simple", denoise: float = 1.0,
        prefix: str = "Krea2_turbo",
        latent_prefix: str = "Krea2_latent",
    ) -> dict:
        """Build the standard graph and add its optional latent output."""
        workflow = Krea2ComfyService._build_workflow(
            prompt, width, height, steps, cfg, seed,
            sampler=sampler, scheduler=scheduler, denoise=denoise, prefix=prefix,
        )
        workflow["10"] = {
            "class_type": "SaveLatent",
            "inputs": {"samples": ["7", 0], "filename_prefix": latent_prefix},
        }
        return workflow

    def _poll_history_outputs(self, prompt_id: str, require_latent: bool = False):
        """Poll once for either image-only or image-plus-latent output."""
        deadline = time.time() + KREA2_COMFY_REQUEST_TIMEOUT
        while time.time() < deadline:
            history = self._comfy_request(
                f"/history/{urllib.parse.quote(prompt_id)}", timeout=60,
            )
            if prompt_id in history:
                item = history[prompt_id]
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise BackendUnavailableError(
                        f"Krea2 ComfyUI workflow failed:\n{json.dumps(status, indent=2)}"
                    )
                images, latents = [], []
                for node_output in item.get("outputs", {}).values():
                    images.extend(node_output.get("images", []))
                    latents.extend(node_output.get("latents", []))
                if images and (latents or not require_latent):
                    return (images[0], latents[0]) if require_latent else images[0]
            time.sleep(1)
        expected = "latent + image" if require_latent else "result"
        raise BackendUnavailableError(f"Krea2 ComfyUI workflow timed out waiting for {expected}.")

    def _poll_history(self, prompt_id: str) -> dict:
        return self._poll_history_outputs(prompt_id, require_latent=False)

    def _poll_history_with_latent(self, prompt_id: str) -> tuple[dict, dict]:
        return self._poll_history_outputs(prompt_id, require_latent=True)

    def _read_output_latent(self, latent_info: dict) -> torch.Tensor:
        """Read a saved latent tensor from ComfyUI's output directory."""
        from safetensors.torch import load_file as safetensors_load_file

        filename = latent_info["filename"]
        subfolder = latent_info.get("subfolder") or ""
        if subfolder:
            rel = os.path.join("output", subfolder, filename)
        else:
            rel = os.path.join("output", filename)
        full_path = os.path.join(self.comfy_dir, rel)

        if not os.path.isfile(full_path):
            raise BackendUnavailableError(
                f"Krea2 ComfyUI SaveLatent output not found: {full_path}"
            )

        tensors = safetensors_load_file(full_path, device="cpu")
        # ComfyUI SaveLatent stores the latent under the key "latent_tensor".
        latent = tensors.get("latent_tensor")
        if latent is None:
            available = list(tensors.keys())
            if len(available) == 1:
                latent = tensors[available[0]]
            else:
                raise BackendUnavailableError(
                    f"Krea2 ComfyUI saved latent file has unexpected keys: {available}"
                )
        return latent

    def _read_output_image(self, image_info: dict) -> Image.Image:
        """Read a generated image from ComfyUI's output directory."""
        filename = image_info["filename"]
        subfolder = image_info.get("subfolder") or ""
        if subfolder:
            rel = os.path.join("output", subfolder, filename)
        else:
            rel = os.path.join("output", filename)
        full_path = os.path.join(self.comfy_dir, rel)

        if os.path.isfile(full_path):
            image = Image.open(full_path)
            return image.convert("RGB")

        # Fallback: fetch via ComfyUI HTTP API
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": "output",
        })
        url = f"{self.server_base}/view?{params}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=KREA2_COMFY_REQUEST_TIMEOUT) as res:
            image_bytes = res.read()
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGB")

    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> tuple[Image.Image, float]:
        self.ensure_running()
        model_mgr.touch(self.mgr_key)
        workflow = self._build_workflow(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=guidance_scale,
            seed=seed,
        )

        t_generate = time.perf_counter()
        resp = self._comfy_request("/prompt", {"prompt": workflow}, timeout=60)
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise BackendUnavailableError(f"Krea2 ComfyUI did not return a prompt_id: {resp}")

        image_info = self._poll_history(prompt_id)
        image = self._read_output_image(image_info)
        elapsed = max(1e-6, time.perf_counter() - t_generate)
        log.info(
            "Krea2 ComfyUI generate | size=%sx%s | steps=%s | guidance=%s | elapsed=%.2fs",
            width,
            height,
            steps,
            guidance_scale,
            elapsed,
        )
        return image, elapsed

    def generate_image_with_latent(
        self,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        guidance_scale: float,
        seed: int,
    ) -> tuple[Image.Image, torch.Tensor, float]:
        """Generate an image and return both the decoded image and raw latent.

        Uses the dual-output workflow with SaveLatent + SaveImage so the
        PiD decoder can work directly on the native KSampler latents.
        """
        self.ensure_running()
        model_mgr.touch(self.mgr_key)
        workflow = self._build_workflow_with_latent(
            prompt=prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=guidance_scale,
            seed=seed,
        )

        t_generate = time.perf_counter()
        resp = self._comfy_request("/prompt", {"prompt": workflow}, timeout=60)
        prompt_id = resp.get("prompt_id")
        if not prompt_id:
            raise BackendUnavailableError(f"Krea2 ComfyUI did not return a prompt_id: {resp}")

        image_info, latent_info = self._poll_history_with_latent(prompt_id)
        image = self._read_output_image(image_info)
        latent = self._read_output_latent(latent_info)
        elapsed = max(1e-6, time.perf_counter() - t_generate)
        log.info(
            "Krea2 ComfyUI generate+latent | size=%sx%s | steps=%s | "
            "guidance=%s | latent=%s | elapsed=%.2fs",
            width, height, steps, guidance_scale,
            tuple(latent.shape), elapsed,
        )
        return image, latent, elapsed

__all__ = (
    'Krea2ComfyService',
)
_seal_runtime_module(globals())
