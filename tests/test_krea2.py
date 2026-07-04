from image_studio.services.krea2_comfy import Krea2ComfyService


def test_krea2_workflow_optional_latent_node():
    base = Krea2ComfyService._build_workflow("prompt", 1024, 1024, 8, 1.0, 42)
    with_latent = Krea2ComfyService._build_workflow_with_latent(
        "prompt", 1024, 1024, 8, 1.0, 42
    )
    assert "10" not in base
    assert with_latent["10"]["class_type"] == "SaveLatent"
    assert with_latent["7"] == base["7"]
