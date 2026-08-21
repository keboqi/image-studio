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


def test_modal_deploy_pins_official_sensenova_upstream():
    source = (Path(__file__).resolve().parents[1] / "modal_deploy.py").read_text(encoding="utf-8")
    assert "github.com/OpenSenseNova/SenseNova-U1/archive/" in source
    assert 'SENSENOVA_UPSTREAM_REVISION = "f71dfb098226d01edc0e4c67b3917a2af71a30ef"' in source


def test_modal_deploy_bakes_krea2_runtime_and_only_persists_models():
    source = (Path(__file__).resolve().parents[1] / "modal_deploy.py").read_text(encoding="utf-8")

    assert 'KREA2_COMFY_DIR = "/opt/krea2-comfy"' in source
    assert 'KREA2_COMFY_VENV = "/opt/krea2-comfy-venv"' in source
    assert '"KREA2_UPDATE_COMFY": "0"' in source
    assert "KREA2_COMFY_MODELS_DIR: KREA2_COMFY_PERSISTENT_MODELS_DIR" in source
    assert 'KREA2_COMFY_DIR: "/persistent_cache/krea2_comfy"' not in source


def test_krea2_launcher_reconciles_requirements_and_dumps_hangs():
    source = (Path(__file__).resolve().parents[1] / "deploy_krea2_comfy.sh").read_text(
        encoding="utf-8"
    )

    assert "requirements_sha256=" in source
    assert "venv_has_runtime_packages" in source
    assert 'rm -f "${INSTALL_STAMP}"' in source
    assert "--debug-hang" in source
    assert 'kill -INT "${pid}"' in source
