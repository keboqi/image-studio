"""Image Studio theme and stylesheet loading."""

from __future__ import annotations

from pathlib import Path


def load_css() -> str:
    return Path(__file__).with_name("theme.css").read_text(encoding="utf-8")


def build_theme(gradio):
    return gradio.themes.Base(
        primary_hue=gradio.themes.colors.red,
        secondary_hue=gradio.themes.colors.rose,
        neutral_hue=gradio.themes.colors.gray,
        font=[gradio.themes.GoogleFont("Lora"), "serif"],
        font_mono=[gradio.themes.GoogleFont("Courier New"), "monospace"],
    ).set(
        body_background_fill="#050505",
        body_background_fill_dark="#050505",
        block_background_fill="#120909",
        block_background_fill_dark="#120909",
        block_border_color="rgba(139,0,0,0.25)",
        block_border_color_dark="rgba(139,0,0,0.25)",
        input_background_fill="#0a0505",
        input_background_fill_dark="#0a0505",
        button_primary_background_fill="#8B0000",
        button_primary_background_fill_dark="#8B0000",
        button_primary_text_color="#e8dcc4",
        button_primary_text_color_dark="#e8dcc4",
        button_secondary_background_fill="#120909",
        button_secondary_background_fill_dark="#120909",
        button_secondary_text_color="#e8dcc4",
        button_secondary_text_color_dark="#e8dcc4",
        body_text_color="#e8dcc4",
        body_text_color_dark="#e8dcc4",
        block_title_text_color="#e8dcc4",
        block_title_text_color_dark="#e8dcc4",
        block_radius="2px",
        block_title_radius="2px",
        container_radius="2px",
        checkbox_border_radius="2px",
        input_radius="2px",
        table_radius="2px",
        button_large_radius="2px",
        button_small_radius="2px",
        button_medium_radius="2px",
        block_label_radius="2px",
        block_label_right_radius="2px",
        embed_radius="2px",
    )
