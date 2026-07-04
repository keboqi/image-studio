import importlib
import sys
import types


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

    stubs = {
        "gradio": gradio,
        "torch": torch,
        "torch.nn": torch.nn,
        "cv2": _module("cv2"),
        "diffusers": diffusers,
        "diffusers.pipelines": pipelines,
        "diffusers.pipelines.z_image": z_image,
        "diffusers.pipelines.z_image.pipeline_z_image": pipeline_z_image,
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_application_imports_and_exports_extracted_runtime(monkeypatch):
    _install_runtime_stubs(monkeypatch)
    monkeypatch.setenv("IMAGE_STUDIO_NO_BOOTSTRAP", "1")
    sys.modules.pop("image_studio.app", None)
    app = importlib.import_module("image_studio.app")
    assert callable(app.build_ui)
    assert callable(app.run_generate)
    assert callable(app.attach_app_routes)
    assert app.GenerationRequest.field_names()[0] == "mode"
    assert len(app._RUNTIME_MODULES) >= 30
    assert app.build_ui() is not None
