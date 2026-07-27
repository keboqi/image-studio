import importlib
import re
import sys
import types
from pathlib import Path


class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return Dummy()

    def __getattr__(self, _name):
        return Dummy()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(())

    def set(self, **kwargs):
        return self


class DummyModule(types.ModuleType):
    def __getattr__(self, _name):
        return Dummy()


class GradioError(Exception):
    pass


def _module(name):
    module = DummyModule(name)
    module.__path__ = []
    return module


def _install_runtime_stubs(monkeypatch):
    gradio = _module("gradio")
    gradio.Error = GradioError
    gradio.Progress = lambda *args, **kwargs: Dummy()
    gradio.themes = Dummy()

    torch = _module("torch")
    torch.cuda = Dummy()
    torch.cuda.is_available = lambda: False
    torch.nn = _module("torch.nn")
    torch.nn.Module = type("Module", (), {"__init__": lambda self, *args, **kwargs: None})
    torch.Tensor = type("Tensor", (), {})
    torch.dtype = type("dtype", (), {})
    torch.device = lambda value=None: value or "cpu"
    torch.Generator = Dummy

    diffusers = _module("diffusers")
    for name in ("FlowMatchEulerDiscreteScheduler", "QwenImagePipeline", "QwenImageEditPlusPipeline"):
        setattr(diffusers, name, type(name, (), {}))
    pipelines = _module("diffusers.pipelines")
    z_image = _module("diffusers.pipelines.z_image")
    pipeline_z_image = _module("diffusers.pipelines.z_image.pipeline_z_image")
    pipeline_z_image.ZImagePipeline = type("ZImagePipeline", (), {})
    pil = _module("PIL")
    pil_image = _module("PIL.Image")
    pil_image.Image = type("Image", (), {})
    pil.Image = pil_image

    stubs = {
        "gradio": gradio,
        "torch": torch,
        "torch.nn": torch.nn,
        "numpy": _module("numpy"),
        "cv2": _module("cv2"),
        "PIL": pil,
        "PIL.Image": pil_image,
        "diffusers": diffusers,
        "diffusers.pipelines": pipelines,
        "diffusers.pipelines.z_image": z_image,
        "diffusers.pipelines.z_image.pipeline_z_image": pipeline_z_image,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_application_import_is_lightweight_and_runtime_remains_compatible(monkeypatch):
    sys.modules.pop("image_studio.app", None)
    sys.modules.pop("image_studio.composition", None)
    sys.modules.pop("image_studio.runtime", None)
    app = importlib.import_module("image_studio.app")
    assert "image_studio.runtime" not in sys.modules

    _install_runtime_stubs(monkeypatch)
    monkeypatch.setenv("IMAGE_STUDIO_NO_BOOTSTRAP", "1")
    application = app.create_application()
    runtime = application.runtime
    assert callable(app.build_ui)
    assert callable(runtime.run_generate)
    assert callable(app.attach_app_routes)
    assert runtime.GenerationRequest.field_names()[0] == "mode"
    assert app.build_ui() is not None
    assert runtime.PID_CKPT_2KTO4K_V1PT5 == "2kto4k_v1pt5"
    assert runtime.PID_CKPT_2KTO4K not in runtime.PID_ZIMAGE_CKPT_CHOICES
    for backbone in (
        runtime.PID_BACKBONE_ZIMAGE,
        runtime.PID_BACKBONE_QWEN,
        runtime.PID_BACKBONE_IDEOGRAM4,
    ):
        spec = runtime.PID_CHECKPOINTS[backbone][runtime.PID_CKPT_2KTO4K_V1PT5]
        assert "/PiD_v1pt5_res2kto4k_" in spec.relative_checkpoint_path
        assert runtime._resolve_pid_ckpt_type(backbone, "auto", 1024, 1024) == "2kto4k_v1pt5"
        assert runtime._resolve_pid_ckpt_type(backbone, "2kto4k", 1024, 1024) == "2kto4k_v1pt5"

    pid_model = types.SimpleNamespace(
        config=types.SimpleNamespace(input_caption_key="caption")
    )
    pid_batch = runtime._pid_data_batch(pid_model, "a cat", Dummy(), 0.0)
    assert set(pid_batch) == {"caption", "LQ_latent", "degrade_sigma"}

    package_dir = Path(runtime.__file__).parent
    referenced_names = {
        name
        for path in package_dir.rglob("*.py")
        if path.name not in {"runtime.py", "runtime_access.py"}
        for name in re.findall(r"_runtime\(\)\.([A-Za-z_][A-Za-z0-9_]*)", path.read_text(encoding="utf-8"))
    }
    missing_names = sorted(name for name in referenced_names if not hasattr(runtime, name))
    assert missing_names == []

    values = dict.fromkeys(runtime.GenerationRequest.field_names(), 0)
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

    def execute(model_id, operation, parameters, progress, **kwargs):
        captured.update(
            model_id=model_id,
            operation=operation.value,
            parameters=parameters,
            progress=progress,
            kwargs=kwargs,
        )
        return "routed"

    monkeypatch.setattr(runtime.IMAGE_MODEL_EXECUTOR, "execute", execute)
    request = runtime.GenerationRequest.from_mapping(values)
    assert runtime._run_generation_request(request, progress="p") == "routed"
    assert captured["model_id"] == "qwen-image"
    assert captured["operation"] == "image.generate"
    assert captured["parameters"]["prompt"] == "a cat"
    assert captured["parameters"]["pid_steps"] == 4
    assert captured["progress"] == "p"
