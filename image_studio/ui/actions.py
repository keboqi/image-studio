"""Explicit callable dependencies used by Gradio event wiring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any

Action = Callable[..., Any]


@dataclass(frozen=True)
class UiActions:
    gpu_lock: Any
    designer_open_js: str
    vram_markdown: Action
    dispatch_generate: Action
    dispatch_edit: Action
    run_upscale: Action
    run_video_upscale: Action
    run_ai_remover: Action
    run_video_generation: Action
    chat_respond: Action
    pi_respond: Action
    enhance_prompt: Action
    enhance_video_prompt: Action
    prepare_ideogram_designer: Action
    build_ideogram_prompt: Action
    refresh_models: Action
    unload_model: Action
    unload_all: Action
    remove_model_files: Action
    remove_all_model_files: Action
    refresh_gallery: Action
    select_gallery: Action
    clear_gallery_selection: Action
    clear_gallery_download: Action
    delete_image: Action
    delete_all_images: Action
    delete_all_videos: Action
    extract_video_path: Action
    get_video_gallery: Action
    send_gallery_to_edit: Action
    selected_to_upscale: Action
    selected_to_ai_remover: Action
    selected_to_video: Action
    send_to_edit_slots: Action
    send_image_to_upscale: Action
    send_image_to_ai_remover: Action
    send_image_to_video: Action
    send_video_to_upscale: Action
    chat_clear: Action
    chat_model_changed: Action

    @classmethod
    def from_compatibility_runtime(cls, runtime: ModuleType) -> UiActions:
        """Build the typed boundary while the legacy composition module exists."""
        return cls(
            gpu_lock=runtime._inprocess_gpu_lock,
            designer_open_js=runtime.IDEOGRAM_JSON_DESIGNER_OPEN_JS,
            vram_markdown=runtime._build_vram_widget_md,
            dispatch_generate=runtime._dispatch_generate,
            dispatch_edit=runtime._dispatch_edit,
            run_upscale=runtime.run_upscale,
            run_video_upscale=runtime.run_video_upscale,
            run_ai_remover=runtime.run_ai_remover,
            run_video_generation=runtime.run_video_generation,
            chat_respond=runtime.chat_respond,
            pi_respond=runtime.pi_respond,
            enhance_prompt=runtime.enhance_prompt,
            enhance_video_prompt=runtime.enhance_video_prompt,
            prepare_ideogram_designer=runtime.prepare_ideogram_json_designer_payload,
            build_ideogram_prompt=runtime.build_ideogram_manual_upsampler_messages,
            refresh_models=runtime.refresh_models_tab,
            unload_model=runtime.unload_model_and_refresh,
            unload_all=runtime.unload_all_and_refresh,
            remove_model_files=runtime.remove_downloaded_model_files_and_refresh,
            remove_all_model_files=runtime.remove_all_downloaded_model_files_and_refresh,
            refresh_gallery=runtime.refresh_gallery_selection,
            select_gallery=runtime.select_gallery_path,
            clear_gallery_selection=runtime.clear_gallery_selection,
            clear_gallery_download=runtime.clear_gallery_download,
            delete_image=runtime.delete_image,
            delete_all_images=runtime.delete_all_images,
            delete_all_videos=runtime.delete_all_videos,
            extract_video_path=runtime.extract_video_path,
            get_video_gallery=runtime.get_video_gallery_images,
            send_gallery_to_edit=runtime.send_gallery_to_edit_slots,
            selected_to_upscale=runtime.require_selected_to_upscale,
            selected_to_ai_remover=runtime.require_selected_to_ai_remover,
            selected_to_video=runtime.require_selected_to_video,
            send_to_edit_slots=runtime.send_to_edit_slots,
            send_image_to_upscale=runtime.send_image_to_upscale,
            send_image_to_ai_remover=runtime.send_image_to_ai_remover,
            send_image_to_video=runtime.send_image_to_video,
            send_video_to_upscale=runtime.send_video_to_upscale,
            chat_clear=runtime.chat_clear,
            chat_model_changed=runtime.chat_model_changed,
        )
