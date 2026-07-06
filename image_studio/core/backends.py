"""Backend lifecycle state machine and concurrency-safe leases."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from ..errors import BackendBusyError, BackendUnavailableError


class BackendState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"


@dataclass(frozen=True)
class BackendHealth:
    ready: bool
    detail: str = ""


@dataclass(frozen=True)
class CallableBackend:
    id: str
    label: str
    start_fn: Callable[[], object] = lambda: None
    stop_fn: Callable[[], object] = lambda: None
    health_fn: Callable[[], bool] = lambda: True
    max_concurrency: int = 1

    def start(self) -> None:
        self.start_fn()

    def stop(self) -> None:
        self.stop_fn()

    def health(self) -> BackendHealth:
        try:
            ready = bool(self.health_fn())
            return BackendHealth(ready, "ready" if ready else "health check failed")
        except Exception as exc:
            return BackendHealth(False, str(exc))


class BackendController:
    """Serialize startup and enforce backend request capacity."""

    def __init__(self, backend: CallableBackend) -> None:
        if backend.max_concurrency < 1:
            raise ValueError("Backend max_concurrency must be at least one")
        self.backend = backend
        self._state = BackendState.STOPPED
        self._active = 0
        self._condition = threading.Condition(threading.RLock())

    @property
    def state(self) -> BackendState:
        with self._condition:
            return self._state

    @property
    def active_leases(self) -> int:
        with self._condition:
            return self._active

    def _remaining(self, deadline: float | None) -> float | None:
        if deadline is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise BackendUnavailableError(f"Timed out waiting for {self.backend.label}.")
        return remaining

    @contextlib.contextmanager
    def lease(self, timeout: float | None = None) -> Iterator[None]:
        deadline = None if timeout is None else time.monotonic() + timeout
        starter = False
        while True:
            with self._condition:
                if self._state in (BackendState.STARTING, BackendState.STOPPING):
                    self._condition.wait(self._remaining(deadline))
                    continue
                if self._state is BackendState.READY:
                    if self._active == 0 and not self.backend.health().ready:
                        self._state = BackendState.UNHEALTHY
                        continue
                    if self._active < self.backend.max_concurrency:
                        self._active += 1
                        break
                    self._condition.wait(self._remaining(deadline))
                    continue
                self._state = BackendState.STARTING
                starter = True
            if starter:
                try:
                    self.backend.start()
                    health = self.backend.health()
                    if not health.ready:
                        raise BackendUnavailableError(
                            f"{self.backend.label} did not become ready: {health.detail}"
                        )
                except Exception as exc:
                    with self._condition:
                        self._state = BackendState.UNHEALTHY
                        self._condition.notify_all()
                    if isinstance(exc, BackendUnavailableError):
                        raise
                    raise BackendUnavailableError(
                        f"Could not start {self.backend.label}: {exc}"
                    ) from exc
                with self._condition:
                    self._state = BackendState.READY
                    self._condition.notify_all()
                starter = False
        try:
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def stop(self) -> None:
        with self._condition:
            if self._active:
                raise BackendBusyError(
                    f"Cannot stop {self.backend.label}; {self._active} request(s) are active."
                )
            if self._state is BackendState.STOPPED:
                return
            self._state = BackendState.STOPPING
        try:
            self.backend.stop()
        finally:
            with self._condition:
                self._state = BackendState.STOPPED
                self._condition.notify_all()

    def describe(self, include_health: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.backend.id,
            "label": self.backend.label,
            "state": self.state.value,
            "active_leases": self.active_leases,
            "max_concurrency": self.backend.max_concurrency,
        }
        if include_health:
            health = self.backend.health()
            payload["health"] = {"ready": health.ready, "detail": health.detail}
        return payload


class BackendRegistry:
    def __init__(self) -> None:
        self._controllers: dict[str, BackendController] = {}

    def register(self, backend: CallableBackend) -> None:
        if backend.id in self._controllers:
            raise ValueError(f"Backend already registered: {backend.id}")
        self._controllers[backend.id] = BackendController(backend)

    def get(self, backend_id: str) -> BackendController:
        try:
            return self._controllers[backend_id]
        except KeyError:
            raise BackendUnavailableError(f"Unknown backend: {backend_id}") from None

    def lease(self, backend_id: str, timeout: float | None = None):
        return self.get(backend_id).lease(timeout)

    def describe(self, include_health: bool = False) -> list[dict[str, object]]:
        return [item.describe(include_health=include_health) for item in self._controllers.values()]
