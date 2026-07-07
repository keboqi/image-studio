from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUICKSTART = (ROOT / "scripts" / "quickstart.sh").read_text(encoding="utf-8")
CONSTRAINTS = (ROOT / "scripts" / "constraints.txt").read_text(encoding="utf-8")


def test_quickstart_retains_full_service_setup():
    required_fragments = (
        "@earendil-works/pi-coding-agent",
        "wiltodelta/remove-ai-watermarks",
        "uv_install sageattention --no-build-isolation",
        "nunchaku import ok",
        "packaging ninja psutil",
        "Boogu/Boogu-Image-0.1-Turbo",
        "bash deploy_krea2_comfy.sh install",
        "hydra-core==1.3.2",
        "seedvr2_upscaler/requirements.txt",
        "keboqi/ltx-web.git",
        "LTX_WEB_MODEL_MODE=\"${QUICKSTART_MODE}\" LTX_WEB_NO_LAUNCH=1",
        "PIP_CONSTRAINT=\"$CONSTRAINTS_FILE\" bash run.sh",
        "Compatibility check ok:",
    )
    missing = [fragment for fragment in required_fragments if fragment not in QUICKSTART]
    assert not missing, f"quickstart setup blocks were dropped: {missing}"
    assert "uv_install flash-attn --no-build-isolation" not in QUICKSTART


def test_binary_compatibility_repair_runs_after_third_party_setups():
    seedvr2 = QUICKSTART.index("uv_install -r seedvr2_upscaler/requirements.txt")
    ltx = QUICKSTART.index('PIP_CONSTRAINT="$CONSTRAINTS_FILE" bash run.sh')
    repair = QUICKSTART.index('uv_install --upgrade --force-reinstall "numpy<2.0"')
    launch = QUICKSTART.index('python image_studio_webui.py "${LAUNCH_ARGS[@]}"')

    assert seedvr2 < ltx < repair < launch
    assert "numpy<2.0" in CONSTRAINTS
    assert "opencv-python<4.12" in CONSTRAINTS
    assert "opencv-python-headless<4.12" in CONSTRAINTS


def test_quickstart_defaults_to_env_only_and_gates_model_prefetch():
    assert 'IMAGE_STUDIO_QUICKSTART_MODE="${IMAGE_STUDIO_QUICKSTART_MODE:-env-only}"' in QUICKSTART
    assert 'prefetch_models_enabled()' in QUICKSTART
    assert 'if prefetch_models_enabled; then\n  prefetch_optional_models' in QUICKSTART
    assert 'KREA2_MODEL_MODE="${QUICKSTART_MODE}" bash deploy_krea2_comfy.sh install' in QUICKSTART
    assert 'LTX_WEB_MODEL_MODE="${QUICKSTART_MODE}" LTX_WEB_NO_LAUNCH=1' in QUICKSTART
    assert "sed -i" not in QUICKSTART
