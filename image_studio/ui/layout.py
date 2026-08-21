"""Gradio layout composition from explicit application dependencies."""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from image_studio.context import AppContext
from image_studio.generators.dispatch import _get_seedvr2_model_options
from image_studio.web.designer import attach_ideogram_json_designer_route

from .components.ai_remover import _build_ai_remover_tab
from .components.chat import _build_chat_tab
from .components.edit import _build_edit_tab
from .components.gallery import _build_gallery_tab
from .components.generate import _build_generate_tab
from .components.models import _build_models_tab
from .components.upscale import _build_upscale_tab
from .components.video import _build_video_tab
from .wiring import _wire_events

API_DOCS = (Path(__file__).parents[1] / "docs" / "api.md").read_text(encoding="utf-8")
GEMMA_MODEL_URL = "https://huggingface.co/google/gemma-4-12B-it"


def _build_header(context: AppContext):
    if context.ui_actions is None:
        raise RuntimeError("UI actions are not configured.")
    with gr.Row(elem_id="header-row"):
        with gr.Column(scale=8):
            gr.Markdown(
                "# Image Studio WebUI\n"
                "**Text-to-Image** generation, **Multi-Image Editing**, **SeedVR2 Upscaling**, "
                "**HiDream-O1 Full/Dev** generation/editing, "
                "**SenseNova U1.5** 50-step generation/editing and experimental 8-step editing, "
                "**Boogu-Image** generation/editing, "
                "**Ideogram 4** generation, "
                "**Krea2 Turbo (ComfyUI)** generation, "
                "and **Gemma 4 12B** multimodal chat & prompt enhancement "
                "- 4-step Qwen Image, FP4 rank-128."
            )
        with gr.Column(scale=2, min_width=200):
            widget = gr.Markdown(
                context.ui_actions.vram_markdown(),
                elem_id="header-vram-widget",
            )
    return widget


def _build_api_tab() -> None:
    with gr.Tab("API"):
        gr.Markdown(API_DOCS)


def _build_footer() -> None:
    gr.Markdown(
        "Powered by FP4 Engine | Qwen Image | "
        "[**Boogu-Image**](https://github.com/boogu-project/Boogu-Image) | "
        "[**Ideogram 4**](https://github.com/keboqi/ideogram4) | "
        "[**SeedVR2**](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler) Upscaler | "
        "[**HiDream-O1-Image**](https://huggingface.co/HiDream-ai/HiDream-O1-Image) | "
        "[**HiDream-O1-Image-Dev**](https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev) | "
        "[**SenseNova U1.5**](https://github.com/OpenSenseNova/SenseNova-U1) | "
        f"[**Gemma 4 12B-it**]({GEMMA_MODEL_URL}) Chat & Prompt | Gradio",
        elem_id="footer",
    )


def build_ui(context: AppContext):
    """Build the UI from a fully composed ``AppContext``."""
    if context.ui_actions is None:
        raise RuntimeError("UI actions are not configured.")
    seedvr2_models, seedvr2_default, seedvr2_available = _get_seedvr2_model_options()
    with gr.Blocks(title="Image Studio WebUI") as app:
        vram_widget = _build_header(context)
        with gr.Tabs() as tabs:
            generate = _build_generate_tab()
            edit = _build_edit_tab()
            upscale = _build_upscale_tab(
                seedvr2_models,
                seedvr2_default,
                seedvr2_available,
            )
            remover = _build_ai_remover_tab()
            chat = _build_chat_tab()
            gallery = _build_gallery_tab()
            models = _build_models_tab()
            video = _build_video_tab()
            _build_api_tab()
        _wire_events(
            tabs,
            generate,
            edit,
            upscale,
            remover,
            chat,
            gallery,
            models,
            video,
            vram_widget,
            context.ui_actions,
        )
        attach_ideogram_json_designer_route(app)
        _build_footer()
    return app


__all__ = ("build_ui",)
