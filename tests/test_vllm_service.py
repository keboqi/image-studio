import threading
from types import SimpleNamespace

import pytest

from image_studio.errors import BackendUnavailableError
from image_studio.services.vllm import DiffusionGemmaVllmService


def _service() -> DiffusionGemmaVllmService:
    service = DiffusionGemmaVllmService.__new__(DiffusionGemmaVllmService)
    service.lock = threading.RLock()
    service.config = SimpleNamespace(ready_timeout=0)
    return service


def test_wake_fails_if_launcher_returns_before_backend_is_awake(monkeypatch):
    service = _service()
    monkeypatch.setattr(service, "is_ready", lambda: False)
    monkeypatch.setattr(service, "is_healthy", lambda: True)
    monkeypatch.setattr(service, "is_control_reachable", lambda: True)
    monkeypatch.setattr(
        service,
        "_run_script",
        lambda action, timeout: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    with pytest.raises(BackendUnavailableError, match="still sleeping or unhealthy"):
        service.wake()


def test_wake_existing_only_succeeds_after_ready(monkeypatch):
    service = _service()
    ready = False

    monkeypatch.setattr(service, "is_healthy", lambda: False)
    monkeypatch.setattr(service, "is_control_reachable", lambda: True)
    monkeypatch.setattr(service, "is_ready", lambda: ready)
    monkeypatch.setattr(service, "wake", lambda: None)

    assert service._wake_existing() is False

    ready = True
    assert service._wake_existing() is True
