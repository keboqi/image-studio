"""LTX-Web process ownership and health client."""

from __future__ import annotations
from image_studio.errors import BackendUnavailableError, UserInputError
from image_studio.progress import NO_PROGRESS

import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.request

from ..errors import BackendUnavailableError
from ..infra.managed_service import ManagedService

log = logging.getLogger(__name__)

LTX_DISTILLED_STEPS = 8
LTX_VIDEO_MAX_FRAMES = 1201
LTX_AUDIO_VIDEO_RESOLUTION_MULTIPLE = 128


class LtxVideoService(ManagedService):
    def __init__(self, directory: str, api_base: str, *, ready_timeout: int = 30) -> None:
        self.directory = directory
        self.api_base = api_base.rstrip("/")
        self.ready_timeout = ready_timeout
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()

    @property
    def is_process_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        if self.health()[0]:
            return True
        with self._lock:
            if self.is_process_running:
                return True
            if not os.path.isdir(self.directory):
                raise BackendUnavailableError(
                    "ltx-web directory not found. Clone it according to scripts/quickstart.sh."
                )
            try:
                self._process = subprocess.Popen(
                    [sys.executable, "api.py"],
                    cwd=self.directory,
                    env=os.environ.copy(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise BackendUnavailableError(f"Failed to start LTX-Web video backend: {exc}") from exc

        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.health()[0]:
                log.info("LTX-Web video backend is ready.")
                return True
            if not self.is_process_running:
                raise BackendUnavailableError("LTX-Web exited before its health endpoint became ready.")
            time.sleep(1)
        log.warning("LTX-Web started but did not become healthy within %s seconds.", self.ready_timeout)
        return self.is_process_running

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            if process is None or process.poll() is not None:
                return
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def health(self) -> tuple[bool, bool | None]:
        try:
            request = urllib.request.Request(f"{self.api_base}/health", method="GET")
            with urllib.request.urlopen(request, timeout=2) as response:
                if response.status != 200:
                    return False, None
                payload = json.loads(response.read().decode())
                return payload.get("status") == "healthy", bool(payload.get("pipeline_loaded", False))
        except Exception:
            return False, None

    def unload_pipeline(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.api_base}/pipeline/unload", method="POST")
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _max_ltx_audio_video_frames() -> int:
    return ((LTX_VIDEO_MAX_FRAMES - 9) // 16) * 16 + 9

def _snap_ltx_audio_video_frames(frames: int) -> int:
    return snap_ltx_audio_video_frames(frames, max_frames=LTX_VIDEO_MAX_FRAMES)

def _is_ltx_audio_video_frame_count(frames: int) -> bool:
    frames = int(frames)
    return frames >= 9 and frames <= LTX_VIDEO_MAX_FRAMES and (frames - 9) % 16 == 0

def _encode_video_keyframes(*images: Any, frames: int) -> list[dict[str, Any]]:
    provided = [image for image in images if image is not None]
    if not provided:
        return []

    last_frame = max(0, int(frames) - 1)
    if len(provided) == 1:
        frame_indices = [0]
    elif len(provided) == 2:
        frame_indices = [0, last_frame]
    else:
        frame_indices = [0, last_frame // 2, last_frame]

    encoded = []
    for image, frame_index in zip(provided[:3], frame_indices):
        pil_img = coerce_rgb_image(image)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        encoded.append(
            {
                "image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
                "frame_index": int(frame_index),
                "strength": 1.0,
            }
        )
    return encoded

def _encode_video_audio(audio: Any) -> tuple[str | None, str | None]:
    audio_path = _resolve_video_path(audio)
    if not audio_path:
        return None, None
    if not os.path.isfile(audio_path):
        raise UserInputError(f"Audio file not found: {audio_path}")
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), os.path.basename(audio_path)

def _encode_video_conditioning(video: Any, strength: float = 1.0) -> list[dict[str, Any]]:
    video_path = _resolve_video_path(video)
    if not video_path:
        return []
    if not os.path.isfile(video_path):
        raise UserInputError(f"IC-LoRA reference video not found: {video_path}")
    with open(video_path, "rb") as f:
        video_base64 = base64.b64encode(f.read()).decode("utf-8")
    return [{"video_base64": video_base64, "strength": float(strength)}]

def _encode_ic_lora_reference_image(image: Any) -> tuple[str | None, str | None]:
    if image is None:
        return None, None
    pil_img = coerce_rgb_image(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8"), "ic_lora_reference.png"


def build_payload(
    *, prompt: str, negative_prompt: str, keyframes: list[dict[str, Any]],
    width: int, height: int, frames: int, fps: float, skip_memory_cleanup: bool,
    pipeline_type: str, ic_lora_key: str | None = None, ic_lora_strength: float = 1.0,
    ic_lora_attention_strength: float = 1.0, reference_image_base64: str | None = None,
    reference_image_filename: str | None = None, video_conditioning: list[dict[str, Any]] | None = None,
    audio_base64: str | None = None, audio_filename: str | None = None,
) -> dict[str, Any]:
    """Build and validate the LTX HTTP request independently of transport."""
    if audio_base64:
        if width % LTX_AUDIO_VIDEO_RESOLUTION_MULTIPLE or height % LTX_AUDIO_VIDEO_RESOLUTION_MULTIPLE:
            raise UserInputError("Audio-guided video width and height must be multiples of 128.")
        if not _is_ltx_audio_video_frame_count(frames):
            raise UserInputError("Audio-guided video frames must follow 16n+9.")
    payload: dict[str, Any] = {
        "preset_name": "default", "prompt": prompt, "negative_prompt": negative_prompt,
        "images": keyframes, "width": int(width), "height": int(height),
        "num_frames": int(frames), "frame_rate": float(fps),
        "num_inference_steps": LTX_DISTILLED_STEPS,
        "skip_memory_cleanup": bool(skip_memory_cleanup), "pipeline_type": pipeline_type,
    }
    if ic_lora_key:
        if bool(reference_image_base64) == bool(video_conditioning):
            raise UserInputError("IC-LoRA requires exactly one image or video reference.")
        payload.update({
            "ic_lora_model_key": ic_lora_key,
            "ic_lora_strength": float(ic_lora_strength),
            "ic_lora_attention_strength": float(ic_lora_attention_strength),
            "ic_lora_reference_strength": 1.0,
            "ic_lora_reference_image_base64": reference_image_base64,
            "ic_lora_reference_image_filename": reference_image_filename,
            "video_conditioning": video_conditioning or [],
        })
    if audio_base64:
        payload.update({
            "audio_base64": audio_base64, "audio_filename": audio_filename,
            "audio_start_time": 0.0, "audio_max_duration": int(frames) / float(fps),
        })
    return payload


class LtxClient:
    """HTTP/SSE client for one LTX-Web backend."""

    def __init__(self, api_base: str):
        self.api_base = api_base.rstrip("/")

    def generate(self, payload: dict[str, Any], progress=NO_PROGRESS) -> str:
        request = urllib.request.Request(
            f"{self.api_base}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        job_id, error_message = None, None
        with urllib.request.urlopen(request, timeout=3600) as response:
            if response.status != 200:
                raise BackendUnavailableError(f"Backend returned status {response.status}")
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                event_type = event.get("type")
                if event_type == "init":
                    job_id = event.get("job_id")
                    progress(0.45, desc=f"Job initialized: {job_id}")
                elif event_type == "stage":
                    progress(0.5, desc=f"Stage: {event.get('stage')}")
                elif event_type == "complete":
                    progress(0.8, desc="Generation complete on backend.")
                elif event_type == "error":
                    error_message = event.get("message", "Unknown error")
                    break
        if error_message:
            raise BackendUnavailableError(f"Backend error: {error_message}")
        if not job_id:
            raise BackendUnavailableError("No job ID returned by backend.")
        return str(job_id)

    def download(self, job_id: str, output_dir: str) -> str:
        request = urllib.request.Request(
            f"{self.api_base}/generate/{job_id}/download", method="GET"
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            if response.status != 200:
                raise BackendUnavailableError("Failed to download generated video.")
            output = os.path.join(output_dir, f"video_{job_id}.mp4")
            with open(output, "wb") as handle:
                handle.write(response.read())
        return output

def run_video_generation(
    image1, image2, image3, audio, ic_lora_name: str, ic_lora_reference_image,
    ic_lora_reference_video, ic_lora_reference_text: str, ic_lora_strength: float,
    ic_lora_attention_strength: float, prompt: str, neg_prompt: str, width: int, height: int,
    frames: int, fps: float, skip_memory_cleanup: bool,
    progress=NO_PROGRESS
):
    prompt = require_prompt(prompt, "Please enter a prompt for the video.")

    progress(0.1, desc="Starting video backend...")
    _start_ltx_video_backend()

    progress(0.2, desc="Preparing request...")
    ic_lora_key = LTX_IC_LORA_OPTIONS.get(ic_lora_name or LTX_IC_LORA_OFF)
    uses_ic_lora = bool(ic_lora_key)
    if uses_ic_lora and audio is not None:
        raise UserInputError("Generic IC-LoRA generation does not accept audio. Turn IC-LoRA off for audio-guided video.")

    keyframes = [] if uses_ic_lora else _encode_video_keyframes(image1, image2, image3, frames=frames)
    audio_base64, audio_filename = _encode_video_audio(audio)
    pipeline_type = "ic_lora" if uses_ic_lora else ("a2vid_two_stage" if audio_base64 and not keyframes else "distilled")

    reference_image_base64 = None
    reference_image_filename = None
    video_conditioning = []
    if uses_ic_lora:
        reference_image_base64, reference_image_filename = _encode_ic_lora_reference_image(ic_lora_reference_image)
        video_conditioning = _encode_video_conditioning(ic_lora_reference_video)
        if not reference_image_base64 and not video_conditioning:
            raise UserInputError("IC-LoRA needs a reference image/sheet or a reference/control video.")
        if reference_image_base64 and video_conditioning:
            raise UserInputError("Use either an IC-LoRA reference image or reference video for one generation, not both.")
        if ic_lora_key == "ltx-2.3-22b-ic-lora-ingredients-0.9" and int(frames) < 121:
            raise UserInputError("Ingredients IC-LoRA expects at least 121 frames.")
        if ic_lora_key == "ltx-2.3-22b-ic-lora-ingredients-0.9" and ic_lora_reference_text:
            prompt = f"Reference sheet: {ic_lora_reference_text.strip()}\n\nGenerated video: {prompt}"

    if audio_base64:
        if (
            int(width) % LTX_AUDIO_VIDEO_RESOLUTION_MULTIPLE != 0
            or int(height) % LTX_AUDIO_VIDEO_RESOLUTION_MULTIPLE != 0
        ):
            raise UserInputError(
                "Audio-guided video requires width and height to be multiples of 128. "
                "Use one of the Quick Ratio presets or set both sliders to 128-step values."
            )
        if not _is_ltx_audio_video_frame_count(frames):
            raise UserInputError(
                "Audio-guided video requires Frames to follow 16n+9 "
                f"(for example 73, 121, 169; max {_max_ltx_audio_video_frames()}). "
                "Use Quick Duration or set Frames to a compatible value."
            )

    payload = build_payload(
        prompt=prompt, negative_prompt=neg_prompt, keyframes=keyframes,
        width=int(width), height=int(height), frames=int(frames), fps=float(fps),
        skip_memory_cleanup=skip_memory_cleanup, pipeline_type=pipeline_type,
        ic_lora_key=ic_lora_key if uses_ic_lora else None,
        ic_lora_strength=ic_lora_strength,
        ic_lora_attention_strength=ic_lora_attention_strength,
        reference_image_base64=reference_image_base64,
        reference_image_filename=reference_image_filename,
        video_conditioning=video_conditioning,
        audio_base64=audio_base64, audio_filename=audio_filename,
    )

    progress(0.4, desc="Generating video... this may take a while")
    try:
        client = LtxClient(LTX_WEB_API)
        job_id = client.generate(payload, progress=progress)
        progress(0.9, desc="Downloading video result...")
        out_filename = client.download(job_id, OUTPUT_DIR)
        progress(1.0, desc="Done")
        mode_label = "Lipsync" if audio_base64 and keyframes else ("Audio-guided video" if audio_base64 else "Video")
        return out_filename, ok_status(0.0, f"{mode_label} generation complete.")
    except Exception as e:
        raise BackendUnavailableError(f"Video generation failed: {e}")

__all__ = (
    'build_payload',
    'LtxClient',
    '_max_ltx_audio_video_frames',
    '_snap_ltx_audio_video_frames',
    '_is_ltx_audio_video_frame_count',
    '_encode_video_keyframes',
    '_encode_video_audio',
    '_encode_video_conditioning',
    '_encode_ic_lora_reference_image',
    'run_video_generation',
)
_seal_runtime_module(globals())
