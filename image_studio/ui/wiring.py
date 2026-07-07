"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _as_tuple(value: Any) -> tuple:
    return value if isinstance(value, tuple) else (value,)

def _with_vram_result(fn: Callable, *args, gpu_lock: bool = True):
    ctx = _inprocess_gpu_lock if gpu_lock else contextlib.nullcontext()
    try:
        with ctx:
            result = fn(*args)
    except AppError as exc:
        raise gr.Error(str(exc)) from exc
    return (*_as_tuple(result), _build_vram_widget_md())

def _dispatch_generate_with_vram(*args):
    return _with_vram_result(_dispatch_generate, *args)

def _dispatch_edit_with_vram(*args):
    return _with_vram_result(_dispatch_edit, *args)

def run_upscale_with_vram(*args):
    return _with_vram_result(run_upscale, *args)

def run_video_upscale_with_vram(*args):
    return _with_vram_result(run_video_upscale, *args)

def run_ai_remover_with_vram(*args):
    return _with_vram_result(run_ai_remover, *args)

def run_video_with_vram(*args):
    return _with_vram_result(run_video_generation, *args, gpu_lock=False)

def chat_respond_with_vram(*args):
    return _with_vram_result(chat_respond, *args, gpu_lock=False)

def pi_respond_with_vram(*args):
    return _with_vram_result(pi_respond, *args, gpu_lock=False)

def enhance_prompt_and_vram(prompt, chat_model):
    return _with_vram_result(enhance_prompt, prompt, None, chat_model, gpu_lock=False)

def enhance_prompt_with_image_and_vram(prompt, image, chat_model):
    return _with_vram_result(enhance_prompt, prompt, image, chat_model, gpu_lock=False)

def enhance_video_prompt_and_vram(prompt, image, chat_model):
    return _with_vram_result(enhance_video_prompt, prompt, image, chat_model, gpu_lock=False)

def refresh_models_tab_with_vram():
    return _with_vram_result(refresh_models_tab, gpu_lock=False)

def unload_model_and_refresh_with_vram(key):
    return _with_vram_result(unload_model_and_refresh, key)

def unload_all_and_refresh_with_vram():
    return _with_vram_result(unload_all_and_refresh)

def remove_downloaded_model_files_and_refresh_with_vram(key):
    return _with_vram_result(remove_downloaded_model_files_and_refresh, key)

def remove_all_downloaded_model_files_and_refresh_with_vram():
    return _with_vram_result(remove_all_downloaded_model_files_and_refresh)

def _wire_prompt_events(gen: dict, edit: dict, video: dict, chat: dict, vram_widget, llm_queue: dict) -> None:
    gen["enhance_btn"].click(
        enhance_prompt_and_vram,
        [gen["prompt"], chat["model"]],
        [gen["prompt"], vram_widget],
        **llm_queue,
    )
    edit["enhance_btn"].click(
        enhance_prompt_with_image_and_vram,
        [edit["prompt"], edit["img1"], chat["model"]],
        [edit["prompt"], vram_widget],
        **llm_queue,
    )
    video["enhance_btn"].click(
        enhance_video_prompt_and_vram,
        [video["prompt"], video["image1"], chat["model"]],
        [video["prompt"], vram_widget],
        **llm_queue,
    )
    gen["ideogram_open_designer_btn"].click(
        fn=prepare_ideogram_json_designer_payload,
        inputs=[
            gen["prompt"],
            gen["width"],
            gen["height"],
            gen["ideogram_upsampler"],
            gen["raw"],
        ],
        outputs=[gen["ideogram_designer_payload"]],
    ).then(
        fn=None,
        inputs=[gen["ideogram_designer_payload"]],
        outputs=None,
        js=IDEOGRAM_JSON_DESIGNER_OPEN_JS,
    )
    gen["ideogram_build_prompt_btn"].click(
        fn=build_ideogram_manual_upsampler_messages,
        inputs=[gen["prompt"], gen["width"], gen["height"]],
        outputs=[gen["ideogram_raw_prompt"]],
    )

def _generation_event_inputs(gen: dict) -> list[Any]:
    return GenerationRequest.component_inputs(gen, {"neg_prompt": "negative"})

def _edit_event_inputs(edit: dict) -> list[Any]:
    return EditRequest.component_inputs(edit, {
        "model_name": "model",
        "neg_prompt": "negative",
        "qwen_seed": "seed",
        "keep_original_aspect": "keep_aspect",
    })

def _upscale_image_event_inputs(upscale: dict) -> list[Any]:
    return [
        upscale["image"], upscale["resolution"], upscale["max_resolution"],
        upscale["dit"], upscale["color"], upscale["tiling"],
        upscale["tile_size"], upscale["blocks"], upscale["seed"],
    ]

def _ai_remover_event_inputs(ai_remover: dict) -> list[Any]:
    return [ai_remover["image"], ai_remover["mode"], ai_remover["humanize"]]

def _video_generation_event_inputs(video: dict) -> list[Any]:
    return [
        video["image1"], video["image2"], video["image3"],
        video["audio"], video["ic_lora"], video["ic_ref_image"],
        video["ic_ref_video"], video["ic_ref_text"], video["ic_strength"],
        video["ic_attention"],
        video["prompt"], video["negative"],
        video["width"], video["height"], video["frames"],
        video["fps"], video["skip_cleanup"],
    ]

def _video_upscale_event_inputs(upscale: dict) -> list[Any]:
    return [
        upscale["video_input"], upscale["video_resolution"],
        upscale["video_max_resolution"], upscale["video_dit"],
        upscale["video_color"], upscale["video_tiling"],
        upscale["video_tile_size"], upscale["video_blocks"],
        upscale["video_batch"], upscale["video_chunk"],
        upscale["video_overlap"], upscale["video_seed"],
    ]

def _wire_image_generation_events(
    gen: dict,
    edit: dict,
    upscale: dict,
    ai_remover: dict,
    vram_widget,
    gpu_queue: dict,
) -> None:
    gen["button"].click(
        _dispatch_generate_with_vram,
        _generation_event_inputs(gen),
        [gen["output"], gen["status"], gen["raw"], vram_widget],
        api_name="generate",
        **gpu_queue,
    )
    edit["button"].click(
        _dispatch_edit_with_vram,
        _edit_event_inputs(edit),
        [edit["output"], edit["status"], edit["raw"], vram_widget],
        api_name="edit",
        **gpu_queue,
    )
    upscale["button"].click(
        run_upscale_with_vram,
        _upscale_image_event_inputs(upscale),
        [upscale["output"], upscale["status"], upscale["raw"], vram_widget],
        api_name="upscale",
        **gpu_queue,
    )
    ai_remover["button"].click(
        run_ai_remover_with_vram,
        _ai_remover_event_inputs(ai_remover),
        [ai_remover["output"], ai_remover["status"], ai_remover["raw"], vram_widget],
        api_name="ai_remover",
        **gpu_queue,
    )

def _wire_gallery_events(tabs, gallery: dict, edit: dict, upscale: dict, ai_remover: dict, video: dict) -> None:
    gallery["tab"].select(
        fn=refresh_gallery_selection,
        inputs=None,
        outputs=[gallery["gallery"], gallery["selected"], gallery["download"]],
    )
    gallery["gallery"].select(
        fn=select_gallery_path,
        inputs=[gallery["gallery"]],
        outputs=[gallery["selected"], gallery["download"]],
    )
    gallery["refresh"].click(
        fn=refresh_gallery_selection,
        inputs=None,
        outputs=[gallery["gallery"], gallery["selected"], gallery["download"]],
    )
    gallery["delete"].click(
        fn=delete_image,
        inputs=[gallery["selected"]],
        outputs=[gallery["gallery"]],
    ).then(clear_gallery_selection, None, gallery["selected"]).then(
        clear_gallery_download, None, gallery["download"],
    )
    gallery["remove_all"].click(
        fn=delete_all_images,
        inputs=None,
        outputs=[gallery["gallery"]],
    ).then(clear_gallery_selection, None, gallery["selected"]).then(
        clear_gallery_download, None, gallery["download"],
    )
    gallery["to_edit"].click(
        fn=send_gallery_to_edit_slots,
        inputs=[gallery["selected"], edit["img1"], edit["img2"], edit["img3"]],
        outputs=[edit["img1"], edit["img2"], edit["img3"], tabs],
    )
    gallery["to_upscale"].click(
        fn=require_selected_to_upscale,
        inputs=[gallery["selected"]],
        outputs=[upscale["image"], tabs],
    )
    gallery["to_ai_remover"].click(
        fn=require_selected_to_ai_remover,
        inputs=[gallery["selected"]],
        outputs=[ai_remover["image"], tabs],
    )
    gallery["to_video"].click(
        fn=require_selected_to_video,
        inputs=[gallery["selected"]],
        outputs=[video["image1"], tabs],
    )

def _wire_chat_events(chat: dict, vram_widget, llm_queue: dict) -> None:
    chat_outputs = [chat["box"], chat["message"], chat["image"], chat["audio"], vram_widget]
    chat_inputs = [
        chat["message"], chat["image"], chat["audio"],
        chat["box"], chat["system"], chat["thinking"], chat["model"], chat["max_tokens"],
    ]
    chat["send"].click(chat_respond_with_vram, chat_inputs, chat_outputs, **llm_queue)
    chat["pi"].click(
        pi_respond_with_vram,
        [chat["message"], chat["box"]],
        chat_outputs,
        **llm_queue,
    )
    chat["message"].submit(chat_respond_with_vram, chat_inputs, chat_outputs, **llm_queue)
    chat["clear"].click(chat_clear, inputs=None, outputs=chat_outputs[:4])
    chat["model"].change(
        chat_model_changed,
        inputs=[chat["model"], chat["box"]],
        outputs=chat_outputs,
        **llm_queue,
    )

def _wire_send_to_edit(button, source, edit: dict, tabs) -> None:
    button.click(
        fn=send_to_edit_slots,
        inputs=[source, edit["img1"], edit["img2"], edit["img3"]],
        outputs=[edit["img1"], edit["img2"], edit["img3"], tabs],
    )

def _wire_send_image(button, fn: Callable, source, target, tabs) -> None:
    button.click(fn=fn, inputs=[source], outputs=[target, tabs])

def _wire_send_to_events(tabs, gen: dict, edit: dict, upscale: dict, ai_remover: dict, video: dict) -> None:
    _wire_send_to_edit(gen["to_edit"], gen["raw"], edit, tabs)
    _wire_send_to_edit(ai_remover["to_edit"], ai_remover["raw"], edit, tabs)

    for button, fn, source, target in (
        (gen["to_upscale"], send_image_to_upscale, gen["raw"], upscale["image"]),
        (gen["to_ai_remover"], send_image_to_ai_remover, gen["raw"], ai_remover["image"]),
        (gen["to_video"], send_image_to_video, gen["raw"], video["image1"]),
        (edit["to_upscale"], send_image_to_upscale, edit["raw"], upscale["image"]),
        (edit["to_ai_remover"], send_image_to_ai_remover, edit["raw"], ai_remover["image"]),
        (edit["to_video"], send_image_to_video, edit["raw"], video["image1"]),
        (upscale["to_ai_remover"], send_image_to_ai_remover, upscale["raw"], ai_remover["image"]),
        (ai_remover["to_upscale"], send_image_to_upscale, ai_remover["raw"], upscale["image"]),
    ):
        _wire_send_image(button, fn, source, target, tabs)

def _wire_models_events(models: dict, vram_widget, gpu_queue: dict) -> None:
    models["refresh"].click(
        fn=refresh_models_tab_with_vram,
        inputs=None,
        outputs=[models["status"], models["picker"], models["storage_picker"], vram_widget],
    )
    models["unload"].click(
        fn=unload_model_and_refresh_with_vram,
        inputs=[models["picker"]],
        outputs=[models["status"], models["picker"], models["storage_picker"], vram_widget],
        **gpu_queue,
    )
    models["unload_all"].click(
        fn=unload_all_and_refresh_with_vram,
        inputs=None,
        outputs=[models["status"], models["picker"], models["storage_picker"], vram_widget],
        **gpu_queue,
    )
    models["remove_files"].click(
        fn=remove_downloaded_model_files_and_refresh_with_vram,
        inputs=[models["storage_picker"]],
        outputs=[models["status"], models["picker"], models["storage_picker"], vram_widget],
        **gpu_queue,
    )
    models["remove_all_files"].click(
        fn=remove_all_downloaded_model_files_and_refresh_with_vram,
        inputs=None,
        outputs=[models["status"], models["picker"], models["storage_picker"], vram_widget],
        **gpu_queue,
    )

def _wire_video_events(tabs, video: dict, upscale: dict, vram_widget, gpu_queue: dict) -> None:
    video["gallery_refresh"].click(fn=get_video_gallery_images, inputs=None, outputs=video["gallery"])
    video["gallery_remove_all"].click(fn=delete_all_videos, inputs=None, outputs=video["gallery"])
    video["gallery"].select(fn=extract_video_path, inputs=None, outputs=video["output"])
    video["to_upscale"].click(
        fn=send_video_to_upscale,
        inputs=[video["output"]],
        outputs=[upscale["video_input"], tabs],
    )
    video["button"].click(
        run_video_with_vram,
        _video_generation_event_inputs(video),
        [video["output"], video["status"], vram_widget],
        api_name="generate_video",
        **gpu_queue,
    ).then(
        fn=get_video_gallery_images,
        inputs=None,
        outputs=video["gallery"],
    )
    upscale["video_button"].click(
        run_video_upscale_with_vram,
        _video_upscale_event_inputs(upscale),
        [upscale["video_output"], upscale["video_status"], vram_widget],
        api_name="upscale_video",
        **gpu_queue,
    ).then(
        fn=get_video_gallery_images,
        inputs=None,
        outputs=video["gallery"],
    )

def _wire_events(tabs, gen: dict, edit: dict, upscale: dict, ai_remover: dict, chat: dict, gallery: dict, models: dict, video: dict, vram_widget):
    gpu_queue = {"concurrency_limit": 1, "concurrency_id": "gpu"}
    llm_queue = {"concurrency_limit": 1, "concurrency_id": "llm"}

    _wire_prompt_events(gen, edit, video, chat, vram_widget, llm_queue)
    _wire_image_generation_events(gen, edit, upscale, ai_remover, vram_widget, gpu_queue)
    _wire_gallery_events(tabs, gallery, edit, upscale, ai_remover, video)
    _wire_chat_events(chat, vram_widget, llm_queue)
    _wire_send_to_events(tabs, gen, edit, upscale, ai_remover, video)
    _wire_models_events(models, vram_widget, gpu_queue)
    _wire_video_events(tabs, video, upscale, vram_widget, gpu_queue)

__all__ = (
    '_as_tuple',
    '_with_vram_result',
    '_dispatch_generate_with_vram',
    '_dispatch_edit_with_vram',
    'run_upscale_with_vram',
    'run_video_upscale_with_vram',
    'run_ai_remover_with_vram',
    'run_video_with_vram',
    'chat_respond_with_vram',
    'pi_respond_with_vram',
    'enhance_prompt_and_vram',
    'enhance_prompt_with_image_and_vram',
    'enhance_video_prompt_and_vram',
    'refresh_models_tab_with_vram',
    'unload_model_and_refresh_with_vram',
    'unload_all_and_refresh_with_vram',
    'remove_downloaded_model_files_and_refresh_with_vram',
    'remove_all_downloaded_model_files_and_refresh_with_vram',
    '_wire_prompt_events',
    '_generation_event_inputs',
    '_edit_event_inputs',
    '_upscale_image_event_inputs',
    '_ai_remover_event_inputs',
    '_video_generation_event_inputs',
    '_video_upscale_event_inputs',
    '_wire_image_generation_events',
    '_wire_gallery_events',
    '_wire_chat_events',
    '_wire_send_to_edit',
    '_wire_send_image',
    '_wire_send_to_events',
    '_wire_models_events',
    '_wire_video_events',
    '_wire_events',
)
_seal_runtime_module(globals())
