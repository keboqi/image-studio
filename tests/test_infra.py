import sys

import pytest

from image_studio.errors import BackendUnavailableError
from image_studio.infra.lazy_modules import LazyModuleGroup
from image_studio.infra.managed_service import ManagedScriptConfig, ManagedScriptService
from image_studio.infra.model_manager import ManagedModelSpec, ModelManager


def test_lazy_module_group_is_cached():
    calls = []
    group = LazyModuleGroup("demo", lambda: True, lambda: calls.append(1) or {"value": object()})
    assert group.get() is group.get()
    assert calls == [1]


def test_lazy_module_group_reports_unavailable():
    group = LazyModuleGroup("demo", lambda: False, lambda: {})
    with pytest.raises(BackendUnavailableError):
        group.get()


def test_model_manager_eviction_and_exclusive_groups():
    free = [0]
    manager = ModelManager(lambda: (free[0], 10_000 * 1024 * 1024))
    unloaded = []
    manager.register("old", object(), 100, lambda: (unloaded.append("old"), free.__setitem__(0, 500 * 1024 * 1024)))
    manager.ensure_vram(400)
    assert unloaded == ["old"]

    specs = {
        "a": ManagedModelSpec("a", "A", 1, "family"),
        "b": ManagedModelSpec("b", "B", 1, "family"),
    }
    manager.register("a", object(), 1, lambda: unloaded.append("a"))
    loaded = object()
    assert manager.get_or_load(specs["b"], lambda: loaded, lambda _value: lambda: None, specs=specs) is loaded
    assert "a" in unloaded
    assert manager.keys() == ["b"]


def test_managed_script_service_runs_fake_backend(tmp_path):
    script = tmp_path / "backend.py"
    script.write_text("import sys\nprint(sys.argv[1])\n", encoding="utf-8")
    service = ManagedScriptService(
        ManagedScriptConfig(
            label="fake", manager_key="fake", vram_mb=1,
            script=str(script), shell=sys.executable, shell_env_name="PYTHON",
            ready_timeout=1, start_timeout=5, request_timeout=5,
            working_dir=str(tmp_path),
        )
    )
    result = service.run_script("start", timeout=5)
    assert result.returncode == 0
    assert result.stdout.strip() == "start"
    assert service.wait_until_ready(lambda: True, timeout=0.1)
