from image_studio.generators.base import GenerationRequest
from image_studio.generators.dispatch import (
    SEEDVR2_DEFAULT_DIT,
    SEEDVR2_DIT_MODELS,
    SEEDVR2_FAST_DIT,
    run_generation_request,
)


def test_typed_dispatch_uses_stable_model_id():
    values = dict.fromkeys(GenerationRequest.field_names(), 0)
    values.update(
        mode="Qwen Image",
        prompt="a cat",
        neg_prompt="blur",
        width=768,
        height=512,
        cfg=1.5,
        full_pid_enabled=False,
        full_pid_ckpt="auto",
        full_pid_steps=4,
        full_pid_cfg=1.0,
        seed=42,
    )
    captured = {}

    class Executor:
        def execute(self, model_id, operation, parameters, progress, **kwargs):
            captured.update(
                model_id=model_id,
                operation=operation.value,
                parameters=parameters,
                progress=progress,
                kwargs=kwargs,
            )
            return "routed"

    request = GenerationRequest.from_mapping(values)
    assert run_generation_request(Executor(), request, "p") == "routed"
    assert captured["model_id"] == "qwen-image"
    assert captured["operation"] == "image.generate"
    assert captured["parameters"]["prompt"] == "a cat"
    assert captured["progress"] == "p"


def test_seedvr2_fast_model_is_available_without_changing_the_default():
    assert SEEDVR2_FAST_DIT in SEEDVR2_DIT_MODELS
    assert SEEDVR2_DEFAULT_DIT == (
        "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    )
