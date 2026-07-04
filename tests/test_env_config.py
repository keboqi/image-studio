from image_studio.config import AppConfig
from image_studio.infra.env import env_bool, env_int


def test_typed_env_helpers_handle_invalid_values():
    assert env_int({"COUNT": "bad"}, "COUNT", 7, minimum=2) == 7
    assert env_int({"COUNT": "1"}, "COUNT", 7, minimum=2) == 2
    assert env_bool({"FLAG": "yes"}, "FLAG") is True
    assert env_bool({"FLAG": "off"}, "FLAG", True) is False


def test_app_config_uses_one_explicit_snapshot(tmp_path):
    config = AppConfig.from_env(
        {
            "DIFFUSIONGEMMA_VLLM_PORT": "9001",
            "KREA2_COMFY_PORT": "9002",
            "IMAGE_STUDIO_NO_BOOTSTRAP": "true",
        },
        base_dir=tmp_path,
    )
    assert config.vllm.port == 9001
    assert config.vllm.api_base == "http://127.0.0.1:9001/v1"
    assert config.krea2.server_base == "http://127.0.0.1:9002"
    assert config.no_bootstrap is True


def test_boogu_edit_turbo_model_can_be_configured(tmp_path):
    config = AppConfig.from_env(
        {"BOOGU_IMAGE_EDIT_TURBO_MODEL": "/models/boogu-edit-turbo"},
        base_dir=tmp_path,
    )
    assert config.boogu.edit_turbo_model == "/models/boogu-edit-turbo"
