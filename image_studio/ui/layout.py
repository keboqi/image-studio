"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _build_header():
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=8):
            gr.Markdown(
                "# Image Studio WebUI\n"
                "**Text-to-Image** generation, **Multi-Image Editing**, **SeedVR2 Upscaling**, "
                "**HiDream-O1 Full/Dev** generation/editing, "
                "**Boogu-Image** generation/editing, "
                "**Ideogram 4** generation, "
                "**Krea2 Turbo (ComfyUI)** generation, "
                "and **Gemma 4 12B** multimodal chat & prompt enhancement "
                "- 4-step Qwen Image, FP4 rank-128."
            )
        with gr.Column(scale=2, min_width=200):
            vram_widget = gr.Markdown(_build_vram_widget_md(), elem_id="header-vram-widget")
    return vram_widget

def _build_api_tab():
    with gr.Tab("API"):
        gr.Markdown(API_DOCS)

def _build_footer():
    gr.Markdown(
        "Powered by FP4 Engine | Qwen Image | "
        "[**Boogu-Image**](https://github.com/boogu-project/Boogu-Image) | "
        "[**Ideogram 4**](https://github.com/keboqi/ideogram4) | "
        "[**SeedVR2**](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) Upscaler | "
        "[**HiDream-O1-Image**](https://huggingface.co/HiDream-ai/HiDream-O1-Image) | "
        "[**HiDream-O1-Image-Dev**](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev) | "
        f"[**Gemma 4 12B-it**]({GEMMA_MODEL_URL}) Chat & Prompt | Gradio",
        elem_id="footer",
    )

def build_ui(context=None):
    """Build the UI from an explicit AppContext (legacy globals remain bridged)."""
    context = context or APP_CONTEXT
    seedvr2_models, seedvr2_default, seedvr2_available = _get_seedvr2_model_options()
    with gr.Blocks(title="Image Studio WebUI") as app:
        vram_widget = _build_header()
        with gr.Tabs() as tabs:
            gen = _build_generate_tab()
            edit = _build_edit_tab()
            upscale = _build_upscale_tab(seedvr2_models, seedvr2_default, seedvr2_available)
            ai_remover = _build_ai_remover_tab()
            chat = _build_chat_tab()
            gallery = _build_gallery_tab()
            models = _build_models_tab()
            video = _build_video_tab()
            _build_api_tab()
        _wire_events(tabs, gen, edit, upscale, ai_remover, chat, gallery, models, video, vram_widget)
        attach_ideogram_json_designer_route(app)
        _build_footer()
    return app

__all__ = (
    '_build_header',
    '_build_api_tab',
    '_build_footer',
    'build_ui',
)
_seal_runtime_module(globals())
