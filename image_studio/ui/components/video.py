"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
from dataclasses import dataclass
from typing import Any

from image_studio.runtime_access import runtime_namespace as _runtime
from image_studio.ui.components.base import ComponentSet


def apply_quick_ratio(ratio, size="large"):
    presets = _runtime().VIDEO_QUICK_RATIO_PRESETS.get(size, _runtime().VIDEO_QUICK_RATIO_PRESETS["large"])
    ratio_key = ratio.split(" ")[0] if ratio else "3:2"
    next_height, next_width = presets.get(ratio_key, presets["3:2"])
    return _runtime().gr.update(value=next_width), _runtime().gr.update(value=next_height)

def apply_quick_ratio_small(ratio):
    return apply_quick_ratio(ratio, size="small")

def apply_quick_duration(duration_str, frame_rate_val):
    fps = float(frame_rate_val or 24)
    if isinstance(duration_str, str):
        duration_val = float(duration_str.replace("s", "")) if duration_str else 5.0
    else:
        duration_val = float(duration_str or 5.0)
    raw_frames = round(duration_val * fps) + 1
    clamped_frames = _runtime()._snap_ltx_audio_video_frames(raw_frames)
    return _runtime().gr.update(value=clamped_frames)

@dataclass
class VideoTab(ComponentSet):
    tab: Any
    image1: Any
    image2: Any
    image3: Any
    audio: Any
    ic_lora: Any
    ic_ref_image: Any
    ic_ref_video: Any
    ic_ref_text: Any
    ic_strength: Any
    ic_attention: Any
    prompt: Any
    enhance_btn: Any
    negative: Any
    width: Any
    height: Any
    frames: Any
    fps: Any
    skip_cleanup: Any
    button: Any
    output: Any
    to_upscale: Any
    status: Any
    gallery: Any
    gallery_refresh: Any
    gallery_remove_all: Any


def _build_video_tab() -> VideoTab:
    with _runtime().gr.Tab("Video", id=_runtime().TAB_VIDEO) as video_tab:
        with _runtime().gr.Row(equal_height=False):
            with _runtime().gr.Column(scale=6):
                v_prompt = _runtime().gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Describe the video",
                    elem_id="video-prompt",
                )
                with _runtime().gr.Row():
                    v_enhance_btn = _runtime().gr.Button(
                        "Enhance Prompt (Gemma 4)", size="sm",
                        elem_classes=["enhance-btn"],
                    )
                    v_btn = _runtime().gr.Button("Generate Video", variant="primary", elem_id="video-btn")
                v_neg_prompt = _runtime().gr.Textbox(label="Negative Prompt", lines=1, value="")

                with _runtime().gr.Tabs(elem_id="video-input-tabs"):
                    with _runtime().gr.Tab("Keyframes"):
                        with _runtime().gr.Row():
                            v_img1 = _runtime().gr.Image(label="Start", type="pil", height=180)
                            v_img2 = _runtime().gr.Image(label="Middle", type="pil", height=180)
                            v_img3 = _runtime().gr.Image(label="End", type="pil", height=180)
                    with _runtime().gr.Tab("Audio"):
                        v_audio = _runtime().gr.Audio(label="Audio", type="filepath")
                    with _runtime().gr.Tab("IC-LoRA"):
                        with _runtime().gr.Row():
                            v_ic_lora = _runtime().gr.Dropdown(
                                _runtime().LTX_IC_LORA_CHOICES,
                                value=_runtime().LTX_IC_LORA_OFF,
                                label="Adapter",
                                scale=2,
                            )
                            v_ic_strength = _runtime().gr.Slider(
                                0.0, 2.0, 1.0, step=0.05,
                                label="Adapter Strength",
                                scale=1,
                            )
                            v_ic_attention = _runtime().gr.Slider(
                                0.0, 1.0, 1.0, step=0.05,
                                label="Attention",
                                scale=1,
                            )
                        with _runtime().gr.Row():
                            v_ic_ref_image = _runtime().gr.Image(
                                label="Reference Image / Sheet",
                                type="pil",
                                height=210,
                            )
                            v_ic_ref_video = _runtime().gr.Video(
                                label="Reference / Control Video",
                                height=210,
                            )
                        v_ic_ref_text = _runtime().gr.Textbox(
                            label="Reference Sheet Description",
                            lines=2,
                            placeholder="Characters, props, and location shown in the sheet",
                        )

                with _runtime().gr.Accordion("Generation Settings", open=True):
                    with _runtime().gr.Row():
                        v_quick_ratio = _runtime().gr.Dropdown(
                            _runtime().VIDEO_QUICK_RATIO_CHOICES,
                            value="1:1 Square",
                            label="Large Ratio",
                        )
                        v_quick_ratio_small = _runtime().gr.Dropdown(
                            _runtime().VIDEO_QUICK_RATIO_CHOICES,
                            value=None,
                            label="Small Ratio",
                        )
                    with _runtime().gr.Row():
                        v_w = _runtime().gr.Slider(256, 1280, 1024, step=32, label="Width")
                        v_h = _runtime().gr.Slider(256, 1280, 1024, step=32, label="Height")
                    with _runtime().gr.Row():
                        v_frames = _runtime().gr.Slider(9, _runtime().LTX_VIDEO_MAX_FRAMES, 121, step=8, label="Frames")
                        v_fps = _runtime().gr.Slider(8, 60, 24, step=1, label="FPS")
                        v_quick_dur = _runtime().gr.Slider(3, 30, 5, step=1, label="Duration")
                    v_skip_cleanup = _runtime().gr.Checkbox(True, label="Skip Memory Cleanup")
            with _runtime().gr.Column(scale=4):
                with _runtime().gr.Tabs(elem_id="video-output-tabs"):
                    with _runtime().gr.Tab("Result"):
                        v_out = _runtime().gr.Video(label="Generated Video", height=520, interactive=False)
                        with _runtime().gr.Row():
                            v_to_upscale = _runtime().gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                        v_st = _runtime().gr.Markdown("", elem_id="video-status")
                    with _runtime().gr.Tab("Gallery"):
                        v_gallery = _runtime().gr.Gallery(
                            value=_runtime().get_video_gallery_images(),
                            label="Generated Videos",
                            show_label=True,
                            elem_id="video-gallery",
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height=520,
                            allow_preview=False,
                        )
                        with _runtime().gr.Row():
                            v_gallery_refresh = _runtime().gr.Button("Refresh Videos", size="sm")
                            v_gallery_remove_all = _runtime().gr.Button("Remove All", size="sm", variant="stop")
                
        # Wiring internal UI interactions
        v_quick_ratio.change(apply_quick_ratio, inputs=[v_quick_ratio], outputs=[v_w, v_h])
        v_quick_ratio_small.change(apply_quick_ratio_small, inputs=[v_quick_ratio_small], outputs=[v_w, v_h])
        v_quick_dur.change(apply_quick_duration, inputs=[v_quick_dur, v_fps], outputs=[v_frames])
        v_fps.change(apply_quick_duration, inputs=[v_quick_dur, v_fps], outputs=[v_frames])
                
    return VideoTab(**{
        "tab": video_tab,
        "image1": v_img1,
        "image2": v_img2,
        "image3": v_img3,
        "audio": v_audio,
        "ic_lora": v_ic_lora,
        "ic_ref_image": v_ic_ref_image,
        "ic_ref_video": v_ic_ref_video,
        "ic_ref_text": v_ic_ref_text,
        "ic_strength": v_ic_strength,
        "ic_attention": v_ic_attention,
        "prompt": v_prompt,
        "enhance_btn": v_enhance_btn,
        "negative": v_neg_prompt,
        "width": v_w,
        "height": v_h,
        "frames": v_frames,
        "fps": v_fps,
        "skip_cleanup": v_skip_cleanup,
        "button": v_btn,
        "output": v_out,
        "to_upscale": v_to_upscale,
        "status": v_st,
        "gallery": v_gallery,
        "gallery_refresh": v_gallery_refresh,
        "gallery_remove_all": v_gallery_remove_all,
    })

__all__ = (
    'apply_quick_ratio',
    'apply_quick_ratio_small',
    'apply_quick_duration',
    '_build_video_tab',
)
