from types import SimpleNamespace

from image_studio.pipelines import seedvr2
from image_studio.ui.components import upscale

FAST_DIT = "seedvr2_distill_6L_1.4B_sharp_fp16.safetensors"
DEFAULT_DIT = "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"


def test_dropdown_labels_fast_mode_without_changing_model_values(monkeypatch):
    monkeypatch.setattr(
        upscale,
        "_runtime",
        lambda: SimpleNamespace(SEEDVR2_FAST_DIT=FAST_DIT),
    )
    choices = upscale._seedvr2_dropdown_choices([FAST_DIT, DEFAULT_DIT], DEFAULT_DIT)
    assert choices == [
        ("Fast — SeedVR2 1.4B distilled FP16 (2×–4× recommended)", FAST_DIT),
        ("Default — SeedVR2 7B FP8", DEFAULT_DIT),
    ]


def test_fast_mode_uses_smaller_vram_budget(monkeypatch):
    runtime = SimpleNamespace(
        SEEDVR2_FAST_DIT=FAST_DIT,
        MODEL_SEEDVR2="seedvr2",
        MODEL_SPECS={"seedvr2": SimpleNamespace(vram_mb=12_000)},
    )
    monkeypatch.setattr(seedvr2, "_runtime", lambda: runtime)
    assert seedvr2._seedvr2_vram_budget(FAST_DIT) == 6_000
    assert seedvr2._seedvr2_vram_budget(DEFAULT_DIT) == 12_000
