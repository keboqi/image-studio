from pathlib import Path


def test_modal_deploy_uses_composed_application_contract():
    source = (Path(__file__).resolve().parents[1] / "modal_deploy.py").read_text(encoding="utf-8")

    for obsolete_access in (
        "application.APP_CONTEXT",
        "application.IMAGE_MODEL_EXECUTOR",
        "application.attach_app_routes",
    ):
        assert obsolete_access not in source

    for composed_access in (
        "application.context",
        "application.model_executor",
        "application.build_ui()",
        "application.attach_routes",
    ):
        assert composed_access in source


def test_modal_deploy_uses_seedvr2_fast_mode_fork():
    source = (Path(__file__).resolve().parents[1] / "modal_deploy.py").read_text(encoding="utf-8")
    assert "keboqi/ComfyUI-SeedVR2_VideoUpscaler.git" in source
