import pytest

from image_studio.core.backends import BackendController, BackendState, CallableBackend
from image_studio.errors import BackendBusyError, BackendUnavailableError


def test_backend_start_is_idempotent_and_active_lease_blocks_stop():
    events = []
    controller = BackendController(
        CallableBackend(
            id="demo",
            label="Demo",
            start_fn=lambda: events.append("start"),
            stop_fn=lambda: events.append("stop"),
        )
    )

    with controller.lease():
        assert controller.state is BackendState.READY
        assert controller.active_leases == 1
        with pytest.raises(BackendBusyError):
            controller.stop()

    with controller.lease():
        pass
    assert events == ["start"]

    controller.stop()
    assert events == ["start", "stop"]
    assert controller.state is BackendState.STOPPED


def test_backend_failed_start_transitions_to_unhealthy():
    def fail():
        raise RuntimeError("boom")

    controller = BackendController(
        CallableBackend(id="broken", label="Broken", start_fn=fail)
    )
    with pytest.raises(BackendUnavailableError, match="Could not start Broken"):
        with controller.lease():
            pass
    assert controller.state is BackendState.UNHEALTHY


def test_backend_restarts_when_external_health_is_lost():
    healthy = [False]
    starts = []

    def start():
        starts.append("start")
        healthy[0] = True

    controller = BackendController(
        CallableBackend(
            id="managed",
            label="Managed",
            start_fn=start,
            health_fn=lambda: healthy[0],
        )
    )
    with controller.lease():
        pass
    healthy[0] = False
    with controller.lease():
        pass
    assert starts == ["start", "start"]
