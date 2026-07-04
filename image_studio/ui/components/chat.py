"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.ui.components.base import ComponentSet

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

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


def _build_chat_tab() -> dict[str, Any]:
    with gr.Tab("Chat", id=TAB_CHAT):
        gr.Markdown(
            "Chat with **Gemma 4** models - supports "
            "**text**, **image**, and **audio** input (max 30s).  \n"
            "DiffusionGemma vLLM supports text and image input, but not audio, and starts through `deploy_diffusiongemma_vllm.sh` on first use.  \n"
            "Choose the official Google model, the lighter "
            "[Huihui NVFP4](https://huggingface.co/sakamakismile/Huihui-gemma-4-12B-it-abliterated-NVFP4A16) "
            "variant, or the managed DiffusionGemma vLLM backend. Enhance Prompt and Gemma upsampling will reuse whichever model you select here."
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=7):
                chat_box = gr.Chatbot(label="", elem_id="chat-box", height=480)
                with gr.Row():
                    chat_msg = gr.Textbox(
                        label="Message", lines=2,
                        placeholder="Type your message...",
                        scale=6, show_label=False,
                    )
                    chat_send = gr.Button(
                        "Send", variant="primary",
                        elem_id="chat-send-btn", scale=1,
                        min_width=100,
                    )
                    chat_pi = gr.Button(
                        "pi",
                        elem_id="chat-pi-btn", scale=1,
                        min_width=80,
                    )
                with gr.Accordion("Attachments (Image / Audio)", open=False):
                    with gr.Row():
                        chat_img = gr.Image(label="Attach Image", type="pil", scale=1)
                        chat_audio = gr.Audio(label="Attach Audio (max 30s)", type="filepath", scale=1)
            with gr.Column(scale=3):
                chat_model = gr.Dropdown(
                    choices=list(CHAT_GEMMA_CHOICES.values()),
                    value=CHAT_GEMMA_CHOICES[CHAT_GEMMA_DEFAULT],
                    label="Chat Model",
                )
                chat_system = gr.Textbox(label="System Prompt", lines=4, value=_CHAT_SYSTEM)
                chat_thinking = gr.Checkbox(False, label="Enable Thinking Mode")
                chat_max_tokens = gr.Slider(
                    CHAT_MIN_TOKENS,
                    CHAT_MAX_TOKEN_LIMIT,
                    CHAT_MAX_TOKENS,
                    step=64,
                    label="Max Output Tokens",
                )
                chat_clear_btn = gr.Button("Clear Chat", size="sm", variant="stop")
                gr.Markdown(
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
_seal_runtime_module(globals())
