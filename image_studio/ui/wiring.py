"""Typed Gradio event wiring."""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gradio as gr

from image_studio.errors import AppError
from image_studio.generators.base import EditRequest, GenerationRequest

from .actions import UiActions
from .components.ai_remover import AiRemoverTab
from .components.chat import ChatTab
from .components.edit import EditTab
from .components.gallery import GalleryTab
from .components.generate import GenerateTab
from .components.models import ModelsTab
from .components.upscale import UpscaleTab
from .components.video import VideoTab


def _as_tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


@dataclass(frozen=True)
class UiCallbacks:
    actions: UiActions

    def _with_vram_result(
        self,
        fn: Callable[..., Any],
        *args: Any,
        gpu_lock: bool = True,
    ) -> tuple[Any, ...]:
        lock = self.actions.gpu_lock if gpu_lock else contextlib.nullcontext()
        try:
            with lock:
                result = fn(*args)
        except AppError as exc:
            raise gr.Error(str(exc)) from exc
        return (*_as_tuple(result), self.actions.vram_markdown())

    def dispatch_generate(self, *args: Any):
        # ModelExecutor's backend lease is the safety boundary.
        return self._with_vram_result(
            self.actions.dispatch_generate,
            *args,
            gpu_lock=False,
        )

    def dispatch_edit(self, *args: Any):
        return self._with_vram_result(
            self.actions.dispatch_edit,
            *args,
            gpu_lock=False,
        )

    def run_upscale(self, *args: Any):
        return self._with_vram_result(self.actions.run_upscale, *args)

    def run_video_upscale(self, *args: Any):
        return self._with_vram_result(self.actions.run_video_upscale, *args)

    def run_ai_remover(self, *args: Any):
        return self._with_vram_result(self.actions.run_ai_remover, *args)

    def run_video(self, *args: Any):
        return self._with_vram_result(
            self.actions.run_video_generation,
            *args,
            gpu_lock=False,
        )

    def chat_respond(self, *args: Any):
        return self._with_vram_result(
            self.actions.chat_respond,
            *args,
            gpu_lock=False,
        )

    def pi_respond(self, *args: Any):
        return self._with_vram_result(
            self.actions.pi_respond,
            *args,
            gpu_lock=False,
        )

    def enhance_prompt(self, prompt: Any, chat_model: Any):
        return self._with_vram_result(
            self.actions.enhance_prompt,
            prompt,
            None,
            chat_model,
            gpu_lock=False,
        )

    def enhance_prompt_with_image(self, prompt: Any, image: Any, chat_model: Any):
        return self._with_vram_result(
            self.actions.enhance_prompt,
            prompt,
            image,
            chat_model,
            gpu_lock=False,
        )

    def enhance_video_prompt(self, prompt: Any, image: Any, chat_model: Any):
        return self._with_vram_result(
            self.actions.enhance_video_prompt,
            prompt,
            image,
            chat_model,
            gpu_lock=False,
        )

    def refresh_models(self):
        return self._with_vram_result(
            self.actions.refresh_models,
            gpu_lock=False,
        )

    def unload_model(self, key: Any):
        return self._with_vram_result(self.actions.unload_model, key)

    def unload_all(self):
        return self._with_vram_result(self.actions.unload_all)

    def remove_model_files(self, key: Any):
        return self._with_vram_result(self.actions.remove_model_files, key)

    def remove_all_model_files(self):
        return self._with_vram_result(self.actions.remove_all_model_files)


def _generation_event_inputs(tab: GenerateTab) -> list[Any]:
    return GenerationRequest.component_inputs(tab, {"neg_prompt": "negative"})


def _edit_event_inputs(tab: EditTab) -> list[Any]:
    return EditRequest.component_inputs(
        tab,
        {
            "model_name": "model",
            "neg_prompt": "negative",
            "qwen_seed": "seed",
            "keep_original_aspect": "keep_aspect",
        },
    )


def _upscale_image_event_inputs(tab: UpscaleTab) -> list[Any]:
    return [
        tab.image,
        tab.resolution,
        tab.max_resolution,
        tab.dit,
        tab.color,
        tab.tiling,
        tab.tile_size,
        tab.blocks,
        tab.seed,
    ]


def _ai_remover_event_inputs(tab: AiRemoverTab) -> list[Any]:
    return [tab.image, tab.mode, tab.humanize]


def _video_generation_event_inputs(tab: VideoTab) -> list[Any]:
    return [
        tab.image1,
        tab.image2,
        tab.image3,
        tab.audio,
        tab.ic_lora,
        tab.ic_ref_image,
        tab.ic_ref_video,
        tab.ic_ref_text,
        tab.ic_strength,
        tab.ic_attention,
        tab.prompt,
        tab.negative,
        tab.width,
        tab.height,
        tab.frames,
        tab.fps,
        tab.skip_cleanup,
    ]


def _video_upscale_event_inputs(tab: UpscaleTab) -> list[Any]:
    return [
        tab.video_input,
        tab.video_resolution,
        tab.video_max_resolution,
        tab.video_dit,
        tab.video_color,
        tab.video_tiling,
        tab.video_tile_size,
        tab.video_blocks,
        tab.video_batch,
        tab.video_chunk,
        tab.video_overlap,
        tab.video_seed,
    ]


def _wire_prompt_events(
    gen: GenerateTab,
    edit: EditTab,
    video: VideoTab,
    chat: ChatTab,
    vram_widget: Any,
    llm_queue: dict[str, Any],
    actions: UiActions,
    callbacks: UiCallbacks,
) -> None:
    gen.enhance_btn.click(
        callbacks.enhance_prompt,
        [gen.prompt, chat.model],
        [gen.prompt, vram_widget],
        **llm_queue,
    )
    edit.enhance_btn.click(
        callbacks.enhance_prompt_with_image,
        [edit.prompt, edit.img1, chat.model],
        [edit.prompt, vram_widget],
        **llm_queue,
    )
    video.enhance_btn.click(
        callbacks.enhance_video_prompt,
        [video.prompt, video.image1, chat.model],
        [video.prompt, vram_widget],
        **llm_queue,
    )
    gen.ideogram_open_designer_btn.click(
        fn=actions.prepare_ideogram_designer,
        inputs=[
            gen.prompt,
            gen.width,
            gen.height,
            gen.ideogram_upsampler,
            gen.raw,
        ],
        outputs=[gen.ideogram_designer_payload],
    ).then(
        fn=None,
        inputs=[gen.ideogram_designer_payload],
        outputs=None,
        js=actions.designer_open_js,
    )
    gen.ideogram_build_prompt_btn.click(
        fn=actions.build_ideogram_prompt,
        inputs=[gen.prompt, gen.width, gen.height],
        outputs=[gen.ideogram_raw_prompt],
    )


def _wire_image_events(
    gen: GenerateTab,
    edit: EditTab,
    upscale: UpscaleTab,
    remover: AiRemoverTab,
    vram_widget: Any,
    gpu_queue: dict[str, Any],
    callbacks: UiCallbacks,
) -> None:
    gen.button.click(
        callbacks.dispatch_generate,
        _generation_event_inputs(gen),
        [gen.output, gen.status, gen.raw, vram_widget],
        api_name="generate",
        **gpu_queue,
    )
    edit.button.click(
        callbacks.dispatch_edit,
        _edit_event_inputs(edit),
        [edit.output, edit.status, edit.raw, vram_widget],
        api_name="edit",
        **gpu_queue,
    )
    upscale.button.click(
        callbacks.run_upscale,
        _upscale_image_event_inputs(upscale),
        [upscale.output, upscale.status, upscale.raw, vram_widget],
        api_name="upscale",
        **gpu_queue,
    )
    remover.button.click(
        callbacks.run_ai_remover,
        _ai_remover_event_inputs(remover),
        [remover.output, remover.status, remover.raw, vram_widget],
        api_name="ai_remover",
        **gpu_queue,
    )


def _wire_gallery_events(
    tabs: Any,
    gallery: GalleryTab,
    edit: EditTab,
    upscale: UpscaleTab,
    remover: AiRemoverTab,
    video: VideoTab,
    actions: UiActions,
) -> None:
    outputs = [gallery.gallery, gallery.selected, gallery.download]
    gallery.tab.select(actions.refresh_gallery, None, outputs)
    gallery.gallery.select(
        actions.select_gallery,
        [gallery.gallery],
        [gallery.selected, gallery.download],
    )
    gallery.refresh.click(actions.refresh_gallery, None, outputs)
    gallery.delete.click(
        actions.delete_image,
        [gallery.selected],
        [gallery.gallery],
    ).then(actions.clear_gallery_selection, None, gallery.selected).then(
        actions.clear_gallery_download,
        None,
        gallery.download,
    )
    gallery.remove_all.click(
        actions.delete_all_images,
        None,
        [gallery.gallery],
    ).then(actions.clear_gallery_selection, None, gallery.selected).then(
        actions.clear_gallery_download,
        None,
        gallery.download,
    )
    gallery.to_edit.click(
        actions.send_gallery_to_edit,
        [gallery.selected, edit.img1, edit.img2, edit.img3],
        [edit.img1, edit.img2, edit.img3, tabs],
    )
    gallery.to_upscale.click(
        actions.selected_to_upscale,
        [gallery.selected],
        [upscale.image, tabs],
    )
    gallery.to_ai_remover.click(
        actions.selected_to_ai_remover,
        [gallery.selected],
        [remover.image, tabs],
    )
    gallery.to_video.click(
        actions.selected_to_video,
        [gallery.selected],
        [video.image1, tabs],
    )


def _wire_chat_events(
    chat: ChatTab,
    vram_widget: Any,
    llm_queue: dict[str, Any],
    actions: UiActions,
    callbacks: UiCallbacks,
) -> None:
    outputs = [chat.box, chat.message, chat.image, chat.audio, vram_widget]
    inputs = [
        chat.message,
        chat.image,
        chat.audio,
        chat.box,
        chat.system,
        chat.thinking,
        chat.model,
        chat.max_tokens,
    ]
    chat.send.click(callbacks.chat_respond, inputs, outputs, **llm_queue)
    chat.pi.click(callbacks.pi_respond, [chat.message, chat.box], outputs, **llm_queue)
    chat.message.submit(callbacks.chat_respond, inputs, outputs, **llm_queue)
    chat.clear.click(actions.chat_clear, inputs=None, outputs=outputs[:4])
    chat.model.change(
        actions.chat_model_changed,
        inputs=[chat.model, chat.box],
        outputs=outputs,
        **llm_queue,
    )


def _wire_send_events(
    tabs: Any,
    gen: GenerateTab,
    edit: EditTab,
    upscale: UpscaleTab,
    remover: AiRemoverTab,
    video: VideoTab,
    actions: UiActions,
) -> None:
    for button, source in ((gen.to_edit, gen.raw), (remover.to_edit, remover.raw)):
        button.click(
            actions.send_to_edit_slots,
            [source, edit.img1, edit.img2, edit.img3],
            [edit.img1, edit.img2, edit.img3, tabs],
        )

    for button, fn, source, target in (
        (gen.to_upscale, actions.send_image_to_upscale, gen.raw, upscale.image),
        (gen.to_ai_remover, actions.send_image_to_ai_remover, gen.raw, remover.image),
        (gen.to_video, actions.send_image_to_video, gen.raw, video.image1),
        (edit.to_upscale, actions.send_image_to_upscale, edit.raw, upscale.image),
        (edit.to_ai_remover, actions.send_image_to_ai_remover, edit.raw, remover.image),
        (edit.to_video, actions.send_image_to_video, edit.raw, video.image1),
        (
            upscale.to_ai_remover,
            actions.send_image_to_ai_remover,
            upscale.raw,
            remover.image,
        ),
        (
            remover.to_upscale,
            actions.send_image_to_upscale,
            remover.raw,
            upscale.image,
        ),
    ):
        button.click(fn, [source], [target, tabs])


def _wire_models_events(
    models: ModelsTab,
    vram_widget: Any,
    gpu_queue: dict[str, Any],
    callbacks: UiCallbacks,
) -> None:
    outputs = [models.status, models.picker, models.storage_picker, vram_widget]
    models.refresh.click(callbacks.refresh_models, None, outputs)
    models.unload.click(callbacks.unload_model, [models.picker], outputs, **gpu_queue)
    models.unload_all.click(callbacks.unload_all, None, outputs, **gpu_queue)
    models.remove_files.click(
        callbacks.remove_model_files,
        [models.storage_picker],
        outputs,
        **gpu_queue,
    )
    models.remove_all_files.click(
        callbacks.remove_all_model_files,
        None,
        outputs,
        **gpu_queue,
    )


def _wire_video_events(
    tabs: Any,
    video: VideoTab,
    upscale: UpscaleTab,
    vram_widget: Any,
    gpu_queue: dict[str, Any],
    actions: UiActions,
    callbacks: UiCallbacks,
) -> None:
    video.gallery_refresh.click(actions.get_video_gallery, None, video.gallery)
    video.gallery_remove_all.click(actions.delete_all_videos, None, video.gallery)
    video.gallery.select(actions.extract_video_path, None, video.output)
    video.to_upscale.click(
        actions.send_video_to_upscale,
        [video.output],
        [upscale.video_input, tabs],
    )
    video.button.click(
        callbacks.run_video,
        _video_generation_event_inputs(video),
        [video.output, video.status, vram_widget],
        api_name="generate_video",
        **gpu_queue,
    ).then(actions.get_video_gallery, None, video.gallery)
    upscale.video_button.click(
        callbacks.run_video_upscale,
        _video_upscale_event_inputs(upscale),
        [upscale.video_output, upscale.video_status, vram_widget],
        api_name="upscale_video",
        **gpu_queue,
    ).then(actions.get_video_gallery, None, video.gallery)


def _wire_events(
    tabs: Any,
    gen: GenerateTab,
    edit: EditTab,
    upscale: UpscaleTab,
    remover: AiRemoverTab,
    chat: ChatTab,
    gallery: GalleryTab,
    models: ModelsTab,
    video: VideoTab,
    vram_widget: Any,
    actions: UiActions,
) -> None:
    gpu_queue = {"concurrency_limit": 1, "concurrency_id": "gpu"}
    llm_queue = {"concurrency_limit": 1, "concurrency_id": "llm"}
    callbacks = UiCallbacks(actions)

    _wire_prompt_events(
        gen,
        edit,
        video,
        chat,
        vram_widget,
        llm_queue,
        actions,
        callbacks,
    )
    _wire_image_events(gen, edit, upscale, remover, vram_widget, gpu_queue, callbacks)
    _wire_gallery_events(tabs, gallery, edit, upscale, remover, video, actions)
    _wire_chat_events(chat, vram_widget, llm_queue, actions, callbacks)
    _wire_send_events(tabs, gen, edit, upscale, remover, video, actions)
    _wire_models_events(models, vram_widget, gpu_queue, callbacks)
    _wire_video_events(
        tabs,
        video,
        upscale,
        vram_widget,
        gpu_queue,
        actions,
        callbacks,
    )


__all__ = (
    "UiCallbacks",
    "_ai_remover_event_inputs",
    "_as_tuple",
    "_edit_event_inputs",
    "_generation_event_inputs",
    "_upscale_image_event_inputs",
    "_video_generation_event_inputs",
    "_video_upscale_event_inputs",
    "_wire_events",
)
