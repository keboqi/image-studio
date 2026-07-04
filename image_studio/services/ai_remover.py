"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import BackendUnavailableError, UserInputError
from image_studio.progress import NO_PROGRESS

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

_ai_remover_has_hf_transfer = None

def _remove_ai_watermarks_venv_paths() -> tuple[str, str]:
    if not os.path.isdir(REMOVE_AI_WATERMARKS_DIR):
        raise BackendUnavailableError(
            "remove-ai-watermarks is not installed. Run the quick start setup to clone "
            f"{REMOVE_AI_WATERMARKS_REPO} into {REMOVE_AI_WATERMARKS_DIR}."
        )
    exe_name = "remove-ai-watermarks.exe" if os.name == "nt" else "remove-ai-watermarks"
    python_name = "python.exe" if os.name == "nt" else "python"
    script_dir = "Scripts" if os.name == "nt" else "bin"
    remover_python = os.path.join(REMOVE_AI_WATERMARKS_DIR, ".venv", script_dir, python_name)
    remover_exe = os.path.join(REMOVE_AI_WATERMARKS_DIR, ".venv", script_dir, exe_name)
    if not os.path.isfile(remover_exe):
        raise BackendUnavailableError(
            "remove-ai-watermarks venv is missing its CLI. Re-run the AI remover quick start setup "
            f"inside {REMOVE_AI_WATERMARKS_DIR}."
        )
    if not os.path.isfile(remover_python):
        raise BackendUnavailableError(
            "remove-ai-watermarks venv is missing Python. Re-run the AI remover quick start setup "
            f"inside {REMOVE_AI_WATERMARKS_DIR}."
        )
    return remover_python, remover_exe

def _remove_ai_watermarks_fix_command() -> str:
    python_path = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    return (
        f'cd "{REMOVE_AI_WATERMARKS_DIR}" && '
        f'uv pip install --python {python_path} -U '
        f'"{REMOVE_AI_WATERMARKS_TRANSFORMERS_SPEC}" "{REMOVE_AI_WATERMARKS_HF_TRANSFER_SPEC}"'
    )

def _env_value_is_enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}

def _remove_ai_watermarks_has_hf_transfer() -> bool:
    global _ai_remover_has_hf_transfer

    if _ai_remover_has_hf_transfer is None:
        remover_python, _ = _remove_ai_watermarks_venv_paths()
        env = os.environ.copy()
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        res = subprocess.run(
            [remover_python, "-c", "import hf_transfer"],
            cwd=REMOVE_AI_WATERMARKS_DIR,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        _ai_remover_has_hf_transfer = res.returncode == 0
    return _ai_remover_has_hf_transfer

def _remove_ai_watermarks_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    if (
        _env_value_is_enabled(env.get("HF_HUB_ENABLE_HF_TRANSFER"))
        and not _remove_ai_watermarks_has_hf_transfer()
    ):
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return env

def _is_qwen3_vl_transformers_mismatch(stderr: str) -> bool:
    return (
        "Qwen3VLForConditionalGeneration" in stderr
        and "from 'transformers'" in stderr
    )

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
    if len(text) <= max_chars:
        return text
    return "...\n" + text[-max_chars:]

def _raise_ai_remover_runtime_error(stderr: str) -> None:
    if _is_qwen3_vl_transformers_mismatch(stderr):
        raise BackendUnavailableError(
            "AI Remover's isolated uv environment has an outdated transformers package. "
            "Its installed diffusers imports the Qwen3-VL pipeline, which requires "
            f"{REMOVE_AI_WATERMARKS_TRANSFORMERS_SPEC}.\n\n"
            "Fix it with:\n"
            f"{_remove_ai_watermarks_fix_command()}\n\n"
            f"Original import error:\n{_tail_text(stderr)}"
        )
    if _is_transformers_tokenizers_mismatch(stderr):
        raise BackendUnavailableError(
            "AI Remover's isolated uv environment has a transformers/tokenizers mismatch. "
            "The latest transformers 5.x release breaks SDXL's CLIP tokenizer loading here; "
            f"use {REMOVE_AI_WATERMARKS_TRANSFORMERS_SPEC} instead.\n\n"
            "Fix it with:\n"
            f"{_remove_ai_watermarks_fix_command()}\n\n"
            f"Original import error:\n{_tail_text(stderr)}"
        )
    if _is_hf_transfer_missing(stderr):
        raise BackendUnavailableError(
            "AI Remover inherited HF_HUB_ENABLE_HF_TRANSFER=1, but its isolated uv "
            "environment does not have hf_transfer installed. Restart this WebUI so it can "
            "disable fast transfer for that venv automatically, or install the missing package.\n\n"
            "Fix it with:\n"
            f"{_remove_ai_watermarks_fix_command()}\n\n"
            f"Original download error:\n{_tail_text(stderr)}"
        )
    if _is_hf_model_unavailable(stderr):
        raise BackendUnavailableError(
            "AI Remover could not load the Hugging Face model files needed for invisible "
            "watermark removal. Connect this environment to Hugging Face once, or pre-cache "
            "the required SDXL and ControlNet weights in the remove-ai-watermarks uv environment.\n\n"
            f"Original download error:\n{_tail_text(stderr)}"
        )
    raise BackendUnavailableError(
        "Watermark remover failed in its uv environment.\n"
        f"STDERR:\n{stderr}"
    )

def _check_ai_remover_invisible_runtime() -> None:
    remover_python, _ = _remove_ai_watermarks_venv_paths()
    probe = (
        "from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor\n"
        "print('ok')\n"
    )
    res = subprocess.run(
        [remover_python, "-c", probe],
        cwd=REMOVE_AI_WATERMARKS_DIR,
        capture_output=True,
        text=True,
        check=False,
        env=_remove_ai_watermarks_subprocess_env(),
    )
    if res.returncode != 0:
        _raise_ai_remover_runtime_error(res.stderr)

def _remove_ai_watermarks_cmd(*args: str) -> list[str]:
    """Run remove-ai-watermarks from its isolated project venv."""

    _, remover_exe = _remove_ai_watermarks_venv_paths()
    return [remover_exe, *args]

def run_ai_remover(img, mode, humanize, progress=NO_PROGRESS):
    if img is None:
        raise UserInputError("Please upload or send an image first.")

    t0 = time.time()
    progress(0.1, desc="Preparing files...")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_in, \
         tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out:
        tmp_in_path = f_in.name
        tmp_out_path = f_out.name

    try:
        # Save the raw source pixels to tmp_in_path.
        source_img = coerce_rgb_image(img)
        source_img.save(tmp_in_path, format="PNG")

        # Build remove-ai-watermarks command based on mode. It is run from
        # the cloned project's .venv so diffusers 0.38 stays out of the main
        # WebUI process, which remains on diffusers 0.36.
        if mode == "metadata":
            shutil.copy(tmp_in_path, tmp_out_path)
            cmd = _remove_ai_watermarks_cmd("metadata", tmp_out_path, "--remove")
        else:
            if mode in ("all", "invisible"):
                _check_ai_remover_invisible_runtime()

            cmd = _remove_ai_watermarks_cmd(mode, tmp_in_path, "-o", tmp_out_path)

            if mode in ("all", "invisible"):
                if humanize and humanize > 0:
                    cmd.extend(["--humanize", str(humanize)])

        log.info("Running AI Remover command: %s", " ".join(cmd))
        progress(0.3, desc=f"Running watermark remover ({mode} mode)...")

        # Run command
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=_remove_ai_watermarks_subprocess_env(),
        )

        if res.returncode != 0:
            log.warning("CLI execution failed: %s\n%s", res.stdout, res.stderr)
            _raise_ai_remover_runtime_error(res.stderr)

        progress(0.9, desc="Reading cleaned image...")

        if os.path.exists(tmp_out_path) and os.path.getsize(tmp_out_path) > 0:
            cleaned_img = Image.open(tmp_out_path).convert("RGB")
        else:
            raise BackendUnavailableError(f"Cleaned image file not generated. CLI output: {res.stderr}")

        elapsed = time.time() - t0
        preview_path, raw_path = save_output_image_pair("remover", cleaned_img)

        status_msg = f"Watermarks removed in **{elapsed:.2f}s**"
        return preview_path, status_msg, raw_path

    finally:
        for p in [tmp_in_path, tmp_out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

__all__ = (
    '_remove_ai_watermarks_venv_paths',
    '_remove_ai_watermarks_fix_command',
    '_env_value_is_enabled',
    '_remove_ai_watermarks_has_hf_transfer',
    '_remove_ai_watermarks_subprocess_env',
    '_is_qwen3_vl_transformers_mismatch',
    '_is_transformers_tokenizers_mismatch',
    '_is_hf_transfer_missing',
    '_is_hf_model_unavailable',
    '_tail_text',
    '_raise_ai_remover_runtime_error',
    '_check_ai_remover_invisible_runtime',
    '_remove_ai_watermarks_cmd',
    'run_ai_remover',
)
_seal_runtime_module(globals())
