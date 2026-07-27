"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from dataclasses import dataclass
from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.components.base import ComponentSet


@dataclass
class ChatTab(ComponentSet):
    box: Any
    message: Any
    send: Any
    pi: Any
    image: Any
    audio: Any
    model: Any
    system: Any
    thinking: Any
    max_tokens: Any
    clear: Any


def _build_chat_tab() -> ChatTab:
    with _runtime().gr.Tab("Chat", id=_runtime().TAB_CHAT):
        _runtime().gr.Markdown(
            "Chat with **Gemma 4** models - supports "
            "**text**, **image**, and **audio** input (max 30s).  \n"
            "DiffusionGemma vLLM supports text and image input, but not audio, and starts through `deploy_diffusiongemma_vllm.sh` on first use.  \n"
            "Choose the official Google model, the lighter "
            "[Huihui NVFP4](https://huggingface.co/sakamakismile/Huihui-gemma-4-12B-it-abliterated-NVFP4A16) "
            "variant, or the managed DiffusionGemma vLLM backend. Enhance Prompt and Gemma upsampling will reuse whichever model you select here."
        )
        with _runtime().gr.Row(equal_height=False):
            with _runtime().gr.Column(scale=7):
                chat_box = _runtime().gr.Chatbot(label="", elem_id="chat-box", height=480)
                with _runtime().gr.Row():
                    chat_msg = _runtime().gr.Textbox(
                        label="Message", lines=2,
                        placeholder="Type your message...",
                        scale=6, show_label=False,
                    )
                    chat_send = _runtime().gr.Button(
                        "Send", variant="primary",
                        elem_id="chat-send-btn", scale=1,
                        min_width=100,
                    )
                    chat_pi = _runtime().gr.Button(
                        "pi",
                        elem_id="chat-pi-btn", scale=1,
                        min_width=80,
                    )
                with _runtime().gr.Accordion("Attachments (Image / Audio)", open=False):
                    with _runtime().gr.Row():
                        chat_img = _runtime().gr.Image(label="Attach Image", type="pil", scale=1)
                        chat_audio = _runtime().gr.Audio(label="Attach Audio (max 30s)", type="filepath", scale=1)
            with _runtime().gr.Column(scale=3):
                chat_model = _runtime().gr.Dropdown(
                    choices=list(_runtime().CHAT_GEMMA_CHOICES.values()),
                    value=_runtime().CHAT_GEMMA_CHOICES[_runtime().CHAT_GEMMA_DEFAULT],
                    label="Chat Model",
                )
                chat_system = _runtime().gr.Textbox(label="System Prompt", lines=4, value=_runtime()._CHAT_SYSTEM)
                chat_thinking = _runtime().gr.Checkbox(False, label="Enable Thinking Mode")
                chat_max_tokens = _runtime().gr.Slider(
                    _runtime().CHAT_MIN_TOKENS,
                    _runtime().CHAT_MAX_TOKEN_LIMIT,
                    _runtime().CHAT_MAX_TOKENS,
                    step=64,
                    label="Max Output Tokens",
                )
                chat_clear_btn = _runtime().gr.Button("Clear Chat", size="sm", variant="stop")
                _runtime().gr.Markdown(
                    "---\n"
                    "**Tips:**\n"
                    "- Place images/audio **before** your question\n"
                    "- Audio max 30 seconds\n"
                    "- Thinking mode = step-by-step reasoning"
                )

    return ChatTab(**{
        "box": chat_box,
        "message": chat_msg,
        "send": chat_send,
        "pi": chat_pi,
        "image": chat_img,
        "audio": chat_audio,
        "model": chat_model,
        "system": chat_system,
        "thinking": chat_thinking,
        "max_tokens": chat_max_tokens,
        "clear": chat_clear_btn,
    })

__all__ = (
    '_build_chat_tab',
)
