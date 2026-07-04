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

def apply_quick_ratio(ratio, size="large"):
    presets = VIDEO_QUICK_RATIO_PRESETS.get(size, VIDEO_QUICK_RATIO_PRESETS["large"])
    ratio_key = ratio.split(" ")[0] if ratio else "3:2"
    next_height, next_width = presets.get(ratio_key, presets["3:2"])
    return gr.update(value=next_width), gr.update(value=next_height)

def apply_quick_ratio_small(ratio):
    return apply_quick_ratio(ratio, size="small")

def apply_quick_duration(duration_str, frame_rate_val):
    fps = float(frame_rate_val or 24)
    if isinstance(duration_str, str):
        duration_val = float(duration_str.replace("s", "")) if duration_str else 5.0
    else:
        duration_val = float(duration_str or 5.0)
    raw_frames = round(duration_val * fps) + 1
    clamped_frames = _snap_ltx_audio_video_frames(raw_frames)
    return gr.update(value=clamped_frames)

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


def _build_video_tab() -> dict[str, Any]:
    with gr.Tab("Video", id=TAB_VIDEO) as video_tab:
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                v_prompt = gr.Textbox(
                    label="Prompt",
                    lines=3,
                    placeholder="Describe the video",
                    elem_id="video-prompt",
                )
                with gr.Row():
                    v_enhance_btn = gr.Button(
                        "Enhance Prompt (Gemma 4)", size="sm",
                        elem_classes=["enhance-btn"],
                    )
                    v_btn = gr.Button("Generate Video", variant="primary", elem_id="video-btn")
                v_neg_prompt = gr.Textbox(label="Negative Prompt", lines=1, value="")

                with gr.Tabs(elem_id="video-input-tabs"):
                    with gr.Tab("Keyframes"):
                        with gr.Row():
                            v_img1 = gr.Image(label="Start", type="pil", height=180)
                            v_img2 = gr.Image(label="Middle", type="pil", height=180)
                            v_img3 = gr.Image(label="End", type="pil", height=180)
                    with gr.Tab("Audio"):
                        v_audio = gr.Audio(label="Audio", type="filepath")
                    with gr.Tab("IC-LoRA"):
                        with gr.Row():
                            v_ic_lora = gr.Dropdown(
                                LTX_IC_LORA_CHOICES,
                                value=LTX_IC_LORA_OFF,
                                label="Adapter",
                                scale=2,
                            )
                            v_ic_strength = gr.Slider(
                                0.0, 2.0, 1.0, step=0.05,
                                label="Adapter Strength",
                                scale=1,
                            )
                            v_ic_attention = gr.Slider(
                                0.0, 1.0, 1.0, step=0.05,
                                label="Attention",
                                scale=1,
                            )
                        with gr.Row():
                            v_ic_ref_image = gr.Image(
                                label="Reference Image / Sheet",
                                type="pil",
                                height=210,
                            )
                            v_ic_ref_video = gr.Video(
                                label="Reference / Control Video",
                                height=210,
                            )
                        v_ic_ref_text = gr.Textbox(
                            label="Reference Sheet Description",
                            lines=2,
                            placeholder="Characters, props, and location shown in the sheet",
                        )

                with gr.Accordion("Generation Settings", open=True):
                    with gr.Row():
                        v_quick_ratio = gr.Dropdown(
                            VIDEO_QUICK_RATIO_CHOICES,
                            value="1:1 Square",
                            label="Large Ratio",
                        )
                        v_quick_ratio_small = gr.Dropdown(
                            VIDEO_QUICK_RATIO_CHOICES,
                            value=None,
                            label="Small Ratio",
                        )
                    with gr.Row():
                        v_w = gr.Slider(256, 1280, 1024, step=32, label="Width")
                        v_h = gr.Slider(256, 1280, 1024, step=32, label="Height")
                    with gr.Row():
                        v_frames = gr.Slider(9, LTX_VIDEO_MAX_FRAMES, 121, step=8, label="Frames")
                        v_fps = gr.Slider(8, 60, 24, step=1, label="FPS")
                        v_quick_dur = gr.Slider(3, 30, 5, step=1, label="Duration")
                    v_skip_cleanup = gr.Checkbox(True, label="Skip Memory Cleanup")
            with gr.Column(scale=4):
                with gr.Tabs(elem_id="video-output-tabs"):
                    with gr.Tab("Result"):
                        v_out = gr.Video(label="Generated Video", height=520, interactive=False)
                        with gr.Row():
                            v_to_upscale = gr.Button("Send to Upscale", size="sm", elem_classes=["send-btn"])
                        v_st = gr.Markdown("", elem_id="video-status")
                    with gr.Tab("Gallery"):
                        v_gallery = gr.Gallery(
                            value=get_video_gallery_images(),
                            label="Generated Videos",
                            show_label=True,
                            elem_id="video-gallery",
                            columns=[2],
                            rows=[2],
                            object_fit="contain",
                            height=520,
                            allow_preview=False,
                        )
                        with gr.Row():
                            v_gallery_refresh = gr.Button("Refresh Videos", size="sm")
                            v_gallery_remove_all = gr.Button("Remove All", size="sm", variant="stop")
                
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
_seal_runtime_module(globals())
