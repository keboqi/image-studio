"""Isolated remove-ai-watermarks workflow."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

from PIL import Image

from image_studio.errors import BackendUnavailableError, UserInputError
from image_studio.progress import NO_PROGRESS
from image_studio.storage.output_store import coerce_rgb_image, save_output_image_pair

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AiRemoverConfig:
    directory: str
    repository: str
    transformers_spec: str = "transformers>=4.57.1,<5"
    hf_transfer_spec: str = "hf_transfer>=0.1.9"


_config: AiRemoverConfig | None = None
_has_hf_transfer: bool | None = None


def configure_ai_remover(config: AiRemoverConfig) -> None:
    global _config, _has_hf_transfer
    _config = config
    _has_hf_transfer = None


def _settings() -> AiRemoverConfig:
    if _config is None:
        raise RuntimeError("AI remover is not configured.")
    return _config


def _remove_ai_watermarks_venv_paths() -> tuple[str, str]:
    config = _settings()
    if not os.path.isdir(config.directory):
        raise BackendUnavailableError(
            "remove-ai-watermarks is not installed. Run the quick start setup to clone "
            f"{config.repository} into {config.directory}."
        )
    exe_name = "remove-ai-watermarks.exe" if os.name == "nt" else "remove-ai-watermarks"
    python_name = "python.exe" if os.name == "nt" else "python"
    script_dir = "Scripts" if os.name == "nt" else "bin"
    remover_python = os.path.join(
        config.directory,
        ".venv",
        script_dir,
        python_name,
    )
    remover_exe = os.path.join(config.directory, ".venv", script_dir, exe_name)
    if not os.path.isfile(remover_exe):
        raise BackendUnavailableError(
            "remove-ai-watermarks venv is missing its CLI. Re-run setup inside "
            f"{config.directory}."
        )
    if not os.path.isfile(remover_python):
        raise BackendUnavailableError(
            "remove-ai-watermarks venv is missing Python. Re-run setup inside "
            f"{config.directory}."
        )
    return remover_python, remover_exe


def _remove_ai_watermarks_fix_command() -> str:
    config = _settings()
    python_path = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    return (
        f'cd "{config.directory}" && '
        f"uv pip install --python {python_path} -U "
        f'"{config.transformers_spec}" "{config.hf_transfer_spec}"'
    )


def _env_value_is_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _remove_ai_watermarks_has_hf_transfer() -> bool:
    global _has_hf_transfer
    if _has_hf_transfer is None:
        remover_python, _ = _remove_ai_watermarks_venv_paths()
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        result = subprocess.run(
            [remover_python, "-c", "import hf_transfer"],
            cwd=_settings().directory,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        _has_hf_transfer = result.returncode == 0
    return _has_hf_transfer


def _remove_ai_watermarks_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if (
        _env_value_is_enabled(env.get("HF_HUB_ENABLE_HF_TRANSFER"))
        and not _remove_ai_watermarks_has_hf_transfer()
    ):
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return env


def _is_qwen3_vl_transformers_mismatch(stderr: str) -> bool:
    return "Qwen3VLForConditionalGeneration" in stderr and "from 'transformers'" in stderr


def _is_transformers_tokenizers_mismatch(stderr: str) -> bool:
    return (
        "RobertaProcessing.__new__()" in stderr
        and "unexpected keyword argument 'cls'" in stderr
    )


def _is_hf_transfer_missing(stderr: str) -> bool:
    return (
        "HF_HUB_ENABLE_HF_TRANSFER=1" in stderr
        and "hf_transfer" in stderr
        and "not available" in stderr
    )


def _is_hf_model_unavailable(stderr: str) -> bool:
    return (
        "couldn't connect to 'https://huggingface.co'" in stderr
        and "couldn't find it in the cached files" in stderr
    )


def _tail_text(text: str, max_chars: int = 2000) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else "...\n" + text[-max_chars:]


def _raise_ai_remover_runtime_error(stderr: str) -> None:
    config = _settings()
    if _is_qwen3_vl_transformers_mismatch(stderr):
        message = (
            "AI Remover's isolated environment has an outdated transformers package. "
            f"It requires {config.transformers_spec}."
        )
    elif _is_transformers_tokenizers_mismatch(stderr):
        message = (
            "AI Remover's transformers/tokenizers versions are incompatible. "
            f"Use {config.transformers_spec}."
        )
    elif _is_hf_transfer_missing(stderr):
        message = (
            "AI Remover inherited HF_HUB_ENABLE_HF_TRANSFER=1 without hf_transfer."
        )
    elif _is_hf_model_unavailable(stderr):
        raise BackendUnavailableError(
            "AI Remover could not load its Hugging Face model files. Connect once or "
            "pre-cache the SDXL and ControlNet weights.\n\n"
            f"Original download error:\n{_tail_text(stderr)}"
        )
    else:
        raise BackendUnavailableError(
            "Watermark remover failed in its isolated environment.\n"
            f"STDERR:\n{stderr}"
        )
    raise BackendUnavailableError(
        f"{message}\n\nFix it with:\n{_remove_ai_watermarks_fix_command()}\n\n"
        f"Original error:\n{_tail_text(stderr)}"
    )


def _check_ai_remover_invisible_runtime() -> None:
    remover_python, _ = _remove_ai_watermarks_venv_paths()
    probe = (
        "from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [remover_python, "-c", probe],
        cwd=_settings().directory,
        capture_output=True,
        text=True,
        check=False,
        env=_remove_ai_watermarks_subprocess_env(),
    )
    if result.returncode != 0:
        _raise_ai_remover_runtime_error(result.stderr)


def _remove_ai_watermarks_cmd(*args: str) -> list[str]:
    _, remover_exe = _remove_ai_watermarks_venv_paths()
    return [remover_exe, *args]


def run_ai_remover(img, mode, humanize, progress=NO_PROGRESS):
    if img is None:
        raise UserInputError("Please upload or send an image first.")

    started = time.time()
    progress(0.1, desc="Preparing files...")
    with tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as source_file, tempfile.NamedTemporaryFile(
        suffix=".png",
        delete=False,
    ) as output_file:
        source_path = source_file.name
        output_path = output_file.name

    try:
        coerce_rgb_image(img).save(source_path, format="PNG")
        if mode == "metadata":
            shutil.copy(source_path, output_path)
            command = _remove_ai_watermarks_cmd("metadata", output_path, "--remove")
        else:
            if mode in ("all", "invisible"):
                _check_ai_remover_invisible_runtime()
            command = _remove_ai_watermarks_cmd(mode, source_path, "-o", output_path)
            if mode in ("all", "invisible") and humanize and humanize > 0:
                command.extend(["--humanize", str(humanize)])

        log.info("Running AI Remover command: %s", " ".join(command))
        progress(0.3, desc=f"Running watermark remover ({mode} mode)...")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            env=_remove_ai_watermarks_subprocess_env(),
        )
        if result.returncode != 0:
            log.warning("CLI execution failed: %s\n%s", result.stdout, result.stderr)
            _raise_ai_remover_runtime_error(result.stderr)

        progress(0.9, desc="Reading cleaned image...")
        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 0:
            raise BackendUnavailableError(
                f"Cleaned image file not generated. CLI output: {result.stderr}"
            )
        cleaned = Image.open(output_path).convert("RGB")
        preview_path, raw_path = save_output_image_pair("remover", cleaned)
        return (
            preview_path,
            f"Watermarks removed in **{time.time() - started:.2f}s**",
            raw_path,
        )
    finally:
        for path in (source_path, output_path):
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                pass


__all__ = (
    "AiRemoverConfig",
    "configure_ai_remover",
    "run_ai_remover",
)
