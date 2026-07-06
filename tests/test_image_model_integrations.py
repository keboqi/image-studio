from image_studio.core.models import Operation
from image_studio.integrations.image_models import (
    ImageModelFunctions,
    build_image_model_registry,
)


def test_builtin_image_models_share_one_typed_registry():
    calls = []

    def record(name):
        def handler(*args, **kwargs):
            calls.append((name, args, kwargs))
            return name

        return handler

    registry = build_image_model_registry(
        ImageModelFunctions(
            qwen_generate=record("qwen_generate"),
            qwen_edit=record("qwen_edit"),
            zimage_generate=record("zimage_generate"),
            zimage_full_generate=record("zimage_full_generate"),
            hidream_generate=record("hidream_generate"),
            hidream_edit=record("hidream_edit"),
            boogu_generate=record("boogu_generate"),
            boogu_edit=record("boogu_edit"),
            krea2_generate=record("krea2_generate"),
            ideogram_generate=record("ideogram_generate"),
            hidream_model_keys={"Dev": "dev", "Best Quality": "full"},
        )
    )

    assert [
        adapter.spec.display_name
        for adapter in registry.for_operation(Operation.IMAGE_GENERATE)
    ] == ["Qwen Image", "Z-Image", "HiDream-O1", "Ideogram 4", "Boogu-Image", "Krea2"]
    assert [
        adapter.spec.display_name for adapter in registry.for_operation(Operation.IMAGE_EDIT)
    ] == ["Qwen Image Edit", "HiDream-O1", "Boogu-Image"]
    assert registry.resolve("Krea2 Turbo").spec.backend_id == "krea2-comfy"

    result = registry.resolve("qwen-image").execute(
        Operation.IMAGE_GENERATE,
        {"prompt": "cat", "width": 768, "height": 512},
        progress="p",
    )
    assert result == "qwen_generate"
    assert calls[0][0] == "qwen_generate"
    assert calls[0][1][:4] == ("cat", "", 768, 512)
    assert calls[0][2]["progress"] == "p"
