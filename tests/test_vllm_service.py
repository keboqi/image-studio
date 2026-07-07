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


def test_wake_does_not_wait_for_sleep_state_to_flip(monkeypatch):
    service = _service()
    calls = []

    monkeypatch.setattr(service, "is_ready", lambda: False)
    monkeypatch.setattr(service, "is_healthy", lambda: True)
    monkeypatch.setattr(service, "is_control_reachable", lambda: True)
    monkeypatch.setattr(
        service,
        "_run_script",
        lambda action, timeout: calls.append((action, timeout))
        or SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    assert service.wake() is True

    assert calls == [("wake", 0)]


def test_wake_fails_if_launcher_returns_error(monkeypatch):
    service = _service()

    monkeypatch.setattr(service, "is_ready", lambda: False)
    monkeypatch.setattr(service, "is_healthy", lambda: True)
    monkeypatch.setattr(service, "is_control_reachable", lambda: True)
    monkeypatch.setattr(
        service,
        "_run_script",
        lambda action, timeout: SimpleNamespace(returncode=1, stdout="bad", stderr="boom"),
    )

    with pytest.raises(BackendUnavailableError, match="Failed to wake"):
        service.wake()


def test_wake_existing_succeeds_after_wake_command(monkeypatch):
    service = _service()
    calls = []

    monkeypatch.setattr(service, "is_healthy", lambda: False)
    monkeypatch.setattr(service, "is_control_reachable", lambda: True)
    monkeypatch.setattr(service, "is_ready", lambda: False)
    monkeypatch.setattr(
        service,
        "_run_script",
        lambda action, timeout: calls.append((action, timeout))
        or SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )

    assert service._wake_existing() is True
    assert calls == [("wake", 0)]


def test_wake_existing_falls_back_to_start_when_control_unreachable(monkeypatch):
    service = _service()

    monkeypatch.setattr(service, "is_ready", lambda: False)
    monkeypatch.setattr(service, "is_healthy", lambda: False)
    monkeypatch.setattr(service, "is_control_reachable", lambda: False)

    assert service._wake_existing() is False
