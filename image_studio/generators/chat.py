"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import UserInputError

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _extract_json_block(text: str) -> str | None:
    return extract_json_object(text)

def _fix_unescaped_json_newlines(text: str) -> str:
    return fix_unescaped_json_newlines(text)

def _parse_enhance_json(text: str) -> dict:
    return parse_enhance_json(text)

def _fallback_enhanced_prompt(raw_prompt: str, raw_output: str = "") -> str:
    raw_prompt = raw_prompt.strip()
    if raw_output:
        cleaned = re.sub(r"<[^>]+>", "", raw_output).strip()
        if cleaned and len(cleaned.split()) >= 8 and not cleaned.startswith("{"):
            return cleaned
    return (
        f"{raw_prompt}, a clear self-contained image prompt with a specific main subject, "
        "precise composition, visible action, detailed setting, coherent lighting, material textures, "
        "sharp focus, high detail, and professional image-generation quality."
    )

def _detect_lang(text: str) -> str:
    """Simple heuristic: if text has CJK characters, return 'Chinese', else 'English'."""
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return "Chinese"
        if '\u3040' <= ch <= '\u30ff':
            return "Japanese"
        if '\uac00' <= ch <= '\ud7af':
            return "Korean"
    return "English"

def enhance_prompt(raw_prompt: str, image: Any = None, chat_model: str | None = None) -> str:
    """Use Gemma 4 12B-it to rewrite a rough prompt into a detailed one."""
    if not raw_prompt or not raw_prompt.strip():
        raise UserInputError("Enter a prompt to enhance.")
    lang = _detect_lang(raw_prompt)
    text_content = (
        f"Input language detected: {lang}.\n"
        "Rewrite this as an English image-generation prompt. "
        "If the user asks for visible non-English text, preserve that exact visible text inside quotes.\n\n"
        f"Raw prompt:\n{raw_prompt.strip()}"
    )
    
    if image is not None:
        tmp_path = save_chat_attachment(image, "enhance_img")
        user_content = [
            {"type": "image", "url": tmp_path},
            {"type": "text", "text": text_content}
        ]
    else:
        user_content = text_content

    messages = [
        {"role": "system", "content": _ENHANCE_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    t0 = time.time()
    raw_result = _gemma_generate(messages, max_new_tokens=1024, chat_model=chat_model)
    elapsed = time.time() - t0
    try:
        parsed = _parse_enhance_json(raw_result)
        prompt = parsed["prompt"].strip()
        log.info(
            "Prompt enhanced in %.1fs | resolved knowledge: %s",
            elapsed,
            str(parsed.get("resolved_knowledge", "none"))[:240],
        )
        return prompt
    except ValueError as e:
        log.warning("Prompt enhancer JSON parse failed: %s", e)
        log.info("Prompt enhanced in %.1fs with fallback", elapsed)
        return _fallback_enhanced_prompt(raw_prompt, raw_result)

def enhance_video_prompt(raw_prompt: str, image: Any = None, chat_model: str | None = None) -> str:
    """Use Gemma 4 12B-it to rewrite a rough prompt into a detailed LTX-2.3 video prompt."""
    if not raw_prompt or not raw_prompt.strip():
        raise UserInputError("Enter a prompt to enhance.")
    lang = _detect_lang(raw_prompt)
    text_content = (
        f"Input language detected: {lang}.\n"
        "Rewrite this as an English video-generation prompt following LTX-2.3 guidelines.\n\n"
        f"Raw prompt:\n{raw_prompt.strip()}"
    )
    
    if image is not None:
        tmp_path = save_chat_attachment(image, "enhance_vid_img")
        user_content = [
            {"type": "image", "url": tmp_path},
            {"type": "text", "text": text_content}
        ]
    else:
        user_content = text_content

    messages = [
        {"role": "system", "content": _ENHANCE_VIDEO_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    t0 = time.time()
    raw_result = _gemma_generate(messages, max_new_tokens=1024, chat_model=chat_model)
    elapsed = time.time() - t0
    try:
        parsed = _parse_enhance_json(raw_result)
        prompt = parsed["prompt"].strip()
        log.info(
            "Video prompt enhanced in %.1fs | resolved knowledge: %s",
            elapsed,
            str(parsed.get("resolved_knowledge", "none"))[:240],
        )
        return prompt
    except ValueError as e:
        log.warning("Video prompt enhancer JSON parse failed: %s", e)
        log.info("Video prompt enhanced in %.1fs with fallback", elapsed)
        return _fallback_enhanced_prompt(raw_prompt, raw_result)

def _pil_to_base64_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def chat_respond(
    user_msg: str,
    image: Any,
    audio: Any,
    history: list,
    system_prompt: str,
    enable_thinking: bool,
    chat_model: str = CHAT_GEMMA_DEFAULT,
    max_new_tokens: int = CHAT_MAX_TOKENS,
):
    """Process a user chat turn and return updated history.

    Supports text/image inputs on all chat backends; audio requires a local Gemma backend.
    Returns history in Gradio messages format: [{"role": ..., "content": ...}]
    """
    if not user_msg and image is None and audio is None:
        raise UserInputError("Send a message, image, or audio clip.")

    system = (system_prompt or _CHAT_SYSTEM).strip()
    history = list(history or [])

    # Rebuild LLM messages from Gradio history
    llm_messages = [{"role": "system", "content": system}]
    for h in history:
        llm_messages.append({"role": h["role"], "content": h["content"]})

    # Build current user content (multimodal for LLM)
    content_parts = []
    display_parts = []

    if image is not None:
        tmp_path = save_chat_attachment(image, "chat_img")
        content_parts.append({"type": "image", "url": tmp_path})
        display_parts.append("[image attached]")

    if audio is not None:
        content_parts.append({"type": "audio", "audio": audio})
        display_parts.append("[audio attached]")

    if user_msg and user_msg.strip():
        content_parts.append({"type": "text", "text": user_msg.strip()})
        display_parts.append(user_msg.strip())

    if len(content_parts) == 1 and content_parts[0].get("type") == "text":
        llm_messages.append({"role": "user", "content": user_msg.strip()})
    else:
        llm_messages.append({"role": "user", "content": content_parts})

    display_text = "\n".join(display_parts)

    # Generate
    try:
        max_new_tokens = int(max_new_tokens or CHAT_MAX_TOKENS)
    except (TypeError, ValueError):
        max_new_tokens = CHAT_MAX_TOKENS
    max_new_tokens = max(CHAT_MIN_TOKENS, min(CHAT_MAX_TOKEN_LIMIT, max_new_tokens))

    t0 = time.time()
    reply = _chat_gemma_generate(
        llm_messages,
        chat_model=chat_model,
        max_new_tokens=max_new_tokens,
        enable_thinking=enable_thinking,
    )
    elapsed = time.time() - t0
    reply = reply.strip()
    reply += f"\n\n*({elapsed:.1f}s)*"

    history.append({"role": "user", "content": display_text})
    history.append({"role": "assistant", "content": reply})
    return history, "", None, None

def _clean_cli_output(text: str | None) -> str:
    text = text or ""
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text).strip()

def _format_pi_result(completed: subprocess.CompletedProcess) -> str:
    stdout = _clean_cli_output(completed.stdout)
    stderr = _clean_cli_output(completed.stderr)
    if completed.returncode == 0:
        return stdout or "(Pi completed with no output.)"

    parts = [f"Pi exited with code {completed.returncode}."]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    return "\n\n".join(parts)

def pi_respond(user_msg: str, history: list):
    """Run the Pi CLI against the typed chat message and append its output."""
    if not user_msg or not user_msg.strip():
        raise UserInputError("Type a message for Pi.")

    prompt = user_msg.strip()
    history = list(history or [])
    history.append({"role": "user", "content": prompt})

    pi_bin = shutil.which("pi")
    t0 = time.time()
    if not pi_bin:
        reply = (
            "Pi command not found on PATH. Install it with the quick-start "
            "Node/npm/Pi setup block, then restart this WebUI shell."
        )
    else:
        try:
            pi_model = APP_CONFIG.pi_model
            command = [pi_bin]
            if pi_model:
                command.extend(["--model", pi_model])
            command.extend(["-p", prompt])
            completed = subprocess.run(
                command,
                cwd=BASE_DIR,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            reply = _format_pi_result(completed)
        except FileNotFoundError:
            reply = (
                "Pi command not found on PATH. Install it with the quick-start "
                "Node/npm/Pi setup block, then restart this WebUI shell."
            )
        except Exception as e:
            log.exception("Pi chat command failed")
            reply = f"Pi command failed: {e}"

    elapsed = time.time() - t0
    reply = reply.strip() + f"\n\n*Pi ({elapsed:.1f}s)*"
    history.append({"role": "assistant", "content": reply})
    return history, "", None, None

def chat_clear():
    return [], "", None, None

__all__ = (
    '_extract_json_block',
    '_fix_unescaped_json_newlines',
    '_parse_enhance_json',
    '_fallback_enhanced_prompt',
    '_detect_lang',
    'enhance_prompt',
    'enhance_video_prompt',
    '_pil_to_base64_url',
    'chat_respond',
    '_clean_cli_output',
    '_format_pi_result',
    'pi_respond',
    'chat_clear',
)
_seal_runtime_module(globals())
