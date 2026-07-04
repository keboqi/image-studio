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
class UpscaleTab(ComponentSet):
    image: Any
    resolution: Any
    max_resolution: Any
    dit: Any
    color: Any
    tiling: Any
    tile_size: Any
    blocks: Any
    seed: Any
    button: Any
    output: Any
    raw: Any
    status: Any
    to_ai_remover: Any
    video_input: Any
    video_resolution: Any
    video_max_resolution: Any
    video_dit: Any
    video_color: Any
    video_tiling: Any
    video_tile_size: Any
    video_blocks: Any
    video_batch: Any
    video_chunk: Any
    video_overlap: Any
    video_seed: Any
    video_button: Any
    video_output: Any
    video_status: Any


def _build_upscale_tab(seedvr2_models: list[str], seedvr2_default: str, seedvr2_available: bool) -> dict[str, Any]:
    with gr.Tab("Upscale", id=TAB_UPSCALE):
        if not seedvr2_available:
            gr.Markdown("**SeedVR2 upscaler is currently unavailable.** Check terminal logs for git or network errors.")
        else:
            gr.Markdown(
                "Upload an image or video and upscale it with the **SeedVR2** diffusion-based upscaler.  \n"
                "Models are auto-downloaded on first use."
            )
        gr.Markdown("### Image Upscale")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                up_img = gr.Image(label="Source Image", type="pil", height=320)
                with gr.Row():
                    up_res = gr.Slider(
                        512, 4096, 2160, step=64,
                        label="Target Resolution (shortest edge)",
                    )
                    up_max = gr.Slider(
                        0, 7680, 3584, step=64,
                        label="Max Resolution (0 = no limit)",
                    )
                up_dit = gr.Dropdown(seedvr2_models, value=seedvr2_default, label="DiT Model")
                up_color = gr.Dropdown(
                    ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    value="lab",
                    label="Color Correction",
                )
                with gr.Accordion("Advanced Upscaling Parameters (Advanced)", open=False):
                    up_tiling = gr.Checkbox(False, label="VAE Tiling (for very high res)")
                    up_tile_sz = gr.Slider(256, 2048, 1024, step=64, label="Tile Size")
                    up_blocks = gr.Slider(
                        0, 36, 0, step=1,
                        label="BlockSwap (0 = off, higher = less VRAM)",
                    )
                up_seed = gr.Number(42, label="Seed", precision=0)
                up_btn = gr.Button(
                    "Upscale", variant="primary", elem_id="upscale-btn",
                    interactive=seedvr2_available,
                )
            with gr.Column(scale=5):
                up_out = gr.Image(
                    label="Upscaled Result", type="filepath",
                    height=520, interactive=False, format="webp",
                )
                with gr.Row():
                    up_to_ai_remover = gr.Button("Send to AI Remover", size="sm", elem_classes=["send-btn"])
                up_st = gr.Markdown("", elem_id="upscale-status")
                up_raw = gr.File(label="Raw PNG Download", interactive=False)

        gr.Markdown("### Video Upscale")
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                vu_in = gr.Video(label="Source Video", height=320)
                with gr.Row():
                    vu_res = gr.Slider(
                        512, 4096, 2160, step=64,
                        label="Target Resolution (shortest edge)",
                    )
                    vu_max = gr.Slider(
                        0, 7680, 3584, step=64,
                        label="Max Resolution (0 = no limit)",
                    )
                vu_dit = gr.Dropdown(seedvr2_models, value=seedvr2_default, label="DiT Model")
                vu_color = gr.Dropdown(
                    ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    value="lab",
                    label="Color Correction",
                )
                with gr.Accordion("Advanced Video Upscaling Parameters", open=False):
                    vu_tiling = gr.Checkbox(False, label="VAE Tiling")
                    vu_tile_sz = gr.Slider(256, 2048, 1024, step=64, label="Tile Size")
                    vu_blocks = gr.Slider(0, 36, 0, step=1, label="BlockSwap")
                    vu_batch = gr.Slider(1, 33, 5, step=4, label="Batch Size")
                    vu_chunk = gr.Slider(0, 257, 25, step=1, label="Chunk Size (0 = whole video)")
                    vu_overlap = gr.Slider(0, 32, 0, step=1, label="Temporal Overlap")
                vu_seed = gr.Number(42, label="Seed", precision=0)
                vu_btn = gr.Button(
                    "Upscale Video", variant="primary", elem_id="video-upscale-btn",
                    interactive=seedvr2_available,
                )
            with gr.Column(scale=5):
                vu_out = gr.Video(label="Upscaled Video", height=520, interactive=False)
                vu_st = gr.Markdown("", elem_id="video-upscale-status")

    return UpscaleTab(**{
        "image": up_img,
        "resolution": up_res,
        "max_resolution": up_max,
        "dit": up_dit,
        "color": up_color,
        "tiling": up_tiling,
        "tile_size": up_tile_sz,
        "blocks": up_blocks,
        "seed": up_seed,
        "button": up_btn,
        "output": up_out,
        "raw": up_raw,
        "status": up_st,
        "to_ai_remover": up_to_ai_remover,
        "video_input": vu_in,
        "video_resolution": vu_res,
        "video_max_resolution": vu_max,
        "video_dit": vu_dit,
        "video_color": vu_color,
        "video_tiling": vu_tiling,
        "video_tile_size": vu_tile_sz,
        "video_blocks": vu_blocks,
        "video_batch": vu_batch,
        "video_chunk": vu_chunk,
        "video_overlap": vu_overlap,
        "video_seed": vu_seed,
        "video_button": vu_btn,
        "video_output": vu_out,
        "video_status": vu_st,
    })

__all__ = (
    '_build_upscale_tab',
)
_seal_runtime_module(globals())
