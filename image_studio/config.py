"""Immutable Image Studio configuration, read from the environment once."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .infra.env import env_bool, env_int, env_str


@dataclass(frozen=True)
class PathsConfig:
    base_dir: Path
    output_dir: Path
    remove_ai_watermarks_dir: Path


@dataclass(frozen=True)
class IdeogramConfig:
    directory: Path
    nvfp4_weights_repo: str
    nvfp4_config_repo: str
    fp8_weights_repo: str
    fp8_nvfp4_uncond_weights_repo: str
    realism_lora_repo: str
    realism_lora_weight: str
    api_key: str


@dataclass(frozen=True)
class PidConfig:
    directory: Path
    hf_repo: str = "nvidia/PiD"
    scale: int = 4
    max_low_side: int = 1024


@dataclass(frozen=True)
class BooguConfig:
    directory: Path
    device: str
    max_sequence_length: int
    turbo_model: str
    base_model: str
    edit_model: str
    edit_turbo_model: str


@dataclass(frozen=True)
class VllmConfig:
    script: Path
    shell: str
    port: int
    api_base: str
    model: str
    hf_model: str
    ready_timeout: int
    start_timeout: int
    request_timeout: int
    restart_policy: str
    warmup_on_start: str
    unload_mode: str
    sleep_level: str
    proxy_enabled: bool
    proxy_api_key: str


@dataclass(frozen=True)
class Krea2Config:
    script: Path
    shell: str
    port: int
    server_base: str
    directory: Path
    ready_timeout: int
    start_timeout: int
    request_timeout: int


@dataclass(frozen=True)
class ChatConfig:
    assistant_model: str
    nvfp4_assistant_model: str
    assistant_tokens: int
    default_model: str


@dataclass(frozen=True)
class SeedVR2Config:
    directory: Path


@dataclass(frozen=True)
class LtxConfig:
    directory: Path
    api_base: str


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    ideogram: IdeogramConfig
    pid: PidConfig
    boogu: BooguConfig
    vllm: VllmConfig
    krea2: Krea2Config
    chat: ChatConfig
    seedvr2: SeedVR2Config
    ltx: LtxConfig
    no_bootstrap: bool
    nunchaku_precision: str
    hidream_full_model: str
    hidream_dev_model: str
    pi_model: str

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        base_dir: str | os.PathLike[str] | None = None,
        argv: Sequence[str] = (),
    ) -> AppConfig:
        snapshot = dict(os.environ if env is None else env)
        root = Path(base_dir or Path(__file__).resolve().parents[1]).resolve()
        output_dir = root / "outputs"

        vllm_port = env_int(snapshot, "DIFFUSIONGEMMA_VLLM_PORT", 8001, minimum=1)
        vllm_ready = env_int(snapshot, "DIFFUSIONGEMMA_VLLM_READY_TIMEOUT", 900, minimum=1)
        vllm_api = env_str(
            snapshot,
            "DIFFUSIONGEMMA_VLLM_API_BASE",
            f"http://127.0.0.1:{vllm_port}/v1",
        ).rstrip("/")
        krea_port = env_int(snapshot, "KREA2_COMFY_PORT", 8188, minimum=1)
        krea_ready = env_int(snapshot, "KREA2_COMFY_READY_TIMEOUT", 300, minimum=1)
        fp8_repo = env_str(snapshot, "IDEOGRAM4_FP8_WEIGHTS_REPO", "ideogram-ai/ideogram-4-fp8")
        assistant = env_str(snapshot, "GEMMA_ASSISTANT_MODEL", "google/gemma-4-12B-it-assistant")

        paths = PathsConfig(
            base_dir=root,
            output_dir=output_dir,
            remove_ai_watermarks_dir=Path(
                env_str(snapshot, "REMOVE_AI_WATERMARKS_DIR", str(root / "remove-ai-watermarks"))
            ),
        )
        return cls(
            paths=paths,
            ideogram=IdeogramConfig(
                directory=Path(env_str(snapshot, "IDEOGRAM4_DIR", str(root / "ideogram4"))),
                nvfp4_weights_repo=env_str(snapshot, "IDEOGRAM4_NVFP4_WEIGHTS_REPO", "Comfy-Org/Ideogram-4"),
                nvfp4_config_repo=env_str(snapshot, "IDEOGRAM4_NVFP4_CONFIG_REPO", "Qwen/Qwen3-VL-8B-Instruct"),
                fp8_weights_repo=fp8_repo,
                fp8_nvfp4_uncond_weights_repo=env_str(
                    snapshot, "IDEOGRAM4_FP8_NVFP4_UNCOND_WEIGHTS_REPO", fp8_repo
                ),
                realism_lora_repo=env_str(snapshot, "IDEOGRAM4_REALISM_LORA_REPO", "RazzzHF/Realism_Engine_Ideogram_4"),
                realism_lora_weight=env_str(
                    snapshot,
                    "IDEOGRAM4_REALISM_LORA_WEIGHT",
                    "Realism_Engine_Ideogram_V2.safetensors",
                ),
                api_key=env_str(snapshot, "IDEOGRAM_API_KEY"),
            ),
            pid=PidConfig(directory=Path(env_str(snapshot, "PID_DIR", str(root / "PiD")))),
            boogu=BooguConfig(
                directory=Path(env_str(snapshot, "BOOGU_IMAGE_DIR", str(root / "boogu_image"))),
                device=env_str(snapshot, "BOOGU_IMAGE_DEVICE") or env_str(snapshot, "device"),
                max_sequence_length=env_int(
                    snapshot, "BOOGU_IMAGE_MAX_SEQUENCE_LENGTH", 256, minimum=1, maximum=256
                ),
                turbo_model=env_str(snapshot, "BOOGU_IMAGE_TURBO_MODEL"),
                base_model=env_str(snapshot, "BOOGU_IMAGE_BASE_MODEL"),
                edit_model=env_str(snapshot, "BOOGU_IMAGE_EDIT_MODEL"),
                edit_turbo_model=env_str(snapshot, "BOOGU_IMAGE_EDIT_TURBO_MODEL"),
            ),
            vllm=VllmConfig(
                script=Path(env_str(snapshot, "DIFFUSIONGEMMA_VLLM_SCRIPT", str(root / "deploy_diffusiongemma_vllm.sh"))),
                shell=env_str(snapshot, "DIFFUSIONGEMMA_VLLM_BASH", "bash") or "bash",
                port=vllm_port,
                api_base=vllm_api or f"http://127.0.0.1:{vllm_port}/v1",
                model=env_str(snapshot, "DIFFUSIONGEMMA_VLLM_MODEL", "diffusiongemma") or "diffusiongemma",
                hf_model=env_str(snapshot, "DIFFUSIONGEMMA_VLLM_HF_MODEL"),
                ready_timeout=vllm_ready,
                start_timeout=env_int(
                    snapshot, "DIFFUSIONGEMMA_VLLM_START_TIMEOUT", vllm_ready + 120, minimum=1
                ),
                request_timeout=env_int(snapshot, "DIFFUSIONGEMMA_VLLM_REQUEST_TIMEOUT", 600, minimum=1),
                restart_policy=env_str(snapshot, "DIFFUSIONGEMMA_VLLM_RESTART_POLICY", "unless-stopped") or "unless-stopped",
                warmup_on_start=env_str(
                    snapshot,
                    "DIFFUSIONGEMMA_VLLM_WARMUP_ON_START",
                    env_str(snapshot, "WARMUP_ON_START", "false"),
                ) or "false",
                unload_mode=env_str(snapshot, "DIFFUSIONGEMMA_VLLM_UNLOAD_MODE", "sleep").lower() or "sleep",
                sleep_level=env_str(snapshot, "DIFFUSIONGEMMA_VLLM_SLEEP_LEVEL", "1") or "1",
                proxy_enabled=env_bool(snapshot, "IMAGE_STUDIO_VLLM_PROXY", True),
                proxy_api_key=env_str(snapshot, "IMAGE_STUDIO_VLLM_PROXY_API_KEY"),
            ),
            krea2=Krea2Config(
                script=Path(env_str(snapshot, "KREA2_COMFY_SCRIPT", str(root / "deploy_krea2_comfy.sh"))),
                shell=env_str(snapshot, "KREA2_COMFY_BASH", "bash") or "bash",
                port=krea_port,
                server_base=env_str(snapshot, "KREA2_COMFY_SERVER_BASE", f"http://127.0.0.1:{krea_port}").rstrip("/"),
                directory=Path(env_str(snapshot, "KREA2_COMFY_DIR", str(root / "krea2-comfy"))),
                ready_timeout=krea_ready,
                start_timeout=env_int(snapshot, "KREA2_COMFY_START_TIMEOUT", krea_ready + 120, minimum=1),
                request_timeout=env_int(snapshot, "KREA2_COMFY_REQUEST_TIMEOUT", 900, minimum=1),
            ),
            chat=ChatConfig(
                assistant_model=assistant,
                nvfp4_assistant_model=env_str(snapshot, "GEMMA_NVFP4_ASSISTANT_MODEL", assistant),
                assistant_tokens=env_int(snapshot, "GEMMA_NUM_ASSISTANT_TOKENS", 4, minimum=1),
                default_model=env_str(snapshot, "IMAGE_STUDIO_CHAT_DEFAULT", "huihui") or "huihui",
            ),
            seedvr2=SeedVR2Config(directory=root / "seedvr2_upscaler"),
            ltx=LtxConfig(
                directory=Path(env_str(snapshot, "LTX_WEB_DIR", str(root / "ltx-web"))),
                api_base=env_str(snapshot, "LTX_WEB_API", "http://127.0.0.1:8000").rstrip("/"),
            ),
            no_bootstrap=env_bool(snapshot, "IMAGE_STUDIO_NO_BOOTSTRAP") or "--selftest" in argv,
            nunchaku_precision=env_str(snapshot, "IMAGE_STUDIO_NUNCHAKU_PRECISION").lower(),
            hidream_full_model=env_str(snapshot, "HIDREAM_O1_FULL_MODEL", "HiDream-ai/HiDream-O1-Image"),
            hidream_dev_model=env_str(snapshot, "HIDREAM_O1_DEV_MODEL", "HiDream-ai/HiDream-O1-Image-Dev"),
            pi_model=(
                env_str(snapshot, "PI_MODEL")
                or env_str(snapshot, "SERVED_MODEL_NAME")
                or env_str(snapshot, "DIFFUSIONGEMMA_VLLM_MODEL")
            ),
        )
