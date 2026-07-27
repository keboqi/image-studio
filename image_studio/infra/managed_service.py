"""Shared process lifecycle for script-managed backends."""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..errors import BackendUnavailableError


class ManagedService:
    """Minimal lifecycle implemented by externally managed backends."""

    def start(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class ModelLifecycle(Protocol):
    def is_loaded(self, name: str) -> bool: ...

    def touch(self, name: str) -> None: ...

    def register(
        self,
        name: str,
        pipeline: Any,
        vram_mb: float,
        unload_fn: Callable[[], None],
    ) -> None: ...

    def ensure_vram(self, need_mb: float, exclude: str | None = None) -> None: ...


@dataclass(frozen=True)
class ManagedScriptConfig:
    label: str
    manager_key: str
    vram_mb: int
    script: str
    shell: str
    shell_env_name: str
    ready_timeout: int
    start_timeout: int
    request_timeout: int
    working_dir: str
    environment: Mapping[str, str] = field(default_factory=dict)


class ManagedScriptService(ManagedService):
    """UI-independent lifecycle for a backend controlled by a shell script."""

    def __init__(
        self,
        config: ManagedScriptConfig,
        *,
        model_manager: ModelLifecycle | None = None,
        execution_lock: Any = None,
        bootstrap_allowed: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.mgr_key = config.manager_key
        self.vram_mb = config.vram_mb
        self.script = config.script
        self.lock = threading.RLock()
        self.model_manager = model_manager
        self.execution_lock = execution_lock
        self.bootstrap_allowed = bootstrap_allowed
        self.log = logger or logging.getLogger(__name__)

    def script_env(self) -> dict[str, str]:
        return {**os.environ, **self.config.environment}

    def run_script(self, action: str, timeout: int) -> subprocess.CompletedProcess[str]:
        """Run one launcher action through the same path used by subclasses."""
        return self._run_script(action, timeout)

    # Compatibility aliases retained for existing service subclasses.
    def _script_env(self) -> dict[str, str]:
        return self.script_env()

    def _run_script(self, action: str, timeout: int) -> subprocess.CompletedProcess[str]:
        if not os.path.isfile(self.script):
            raise BackendUnavailableError(
                f"{self.config.label} launcher not found: {self.script}"
            )
        command = [self.config.shell, self.script, action]
        self.log.info(
            "Running %s backend command: %s",
            self.config.label,
            " ".join(command),
        )
        started = time.perf_counter()
        try:
            result = subprocess.run(
                command,
                cwd=self.config.working_dir,
                env=self._script_env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise BackendUnavailableError(
                f"Could not run {self.config.shell!r}. Install it or set "
                f"{self.config.shell_env_name}."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailableError(
                f"Timed out running {self.config.label} launcher ({action})."
            ) from exc
        self.log.info(
            "%s backend command '%s' exited with code %s after %.1fs.",
            self.config.label,
            action,
            result.returncode,
            time.perf_counter() - started,
        )
        return result

    @staticmethod
    def tail(text: str, limit: int = 8000) -> str:
        text = (text or "").strip()
        return text[-limit:] if len(text) > limit else text

    _tail = tail

    def _register(self) -> None:
        manager = self.model_manager
        if manager is None:
            return
        if manager.is_loaded(self.mgr_key):
            manager.touch(self.mgr_key)
            return
        manager.register(self.mgr_key, self, self.vram_mb, unload_fn=self.stop)

    def _raise_action_failure(
        self,
        action: str,
        result: subprocess.CompletedProcess[str],
    ) -> None:
        raise BackendUnavailableError(
            f"Failed to {action} {self.config.label} backend.\n"
            f"STDOUT:\n{self.tail(result.stdout)}\n\n"
            f"STDERR:\n{self.tail(result.stderr)}"
        )

    def _ensure_running(
        self,
        is_ready: Callable[[], bool],
        ready_location: str,
        prepare_existing: Callable[[], bool] | None = None,
    ) -> None:
        if is_ready():
            self._register()
            return
        with self.lock:
            if is_ready():
                self._register()
                return

            lock = self.execution_lock or nullcontext()
            with lock:
                if not is_ready() and self.model_manager is not None:
                    self.model_manager.ensure_vram(self.vram_mb, exclude=self.mgr_key)

            if prepare_existing is not None and prepare_existing():
                self._register()
                return

            result = self._run_script("start", self.config.start_timeout)
            if result.returncode != 0:
                self._raise_action_failure("start", result)

            if not self.wait_until_ready(
                is_ready,
                timeout=self.config.ready_timeout,
                poll_interval=2,
            ):
                raise BackendUnavailableError(
                    f"{self.config.label} backend started but did not become ready before timeout. "
                    f"Check logs with: {self.config.shell} {self.script} logs"
                )
            self.log.info("%s backend is ready at %s.", self.config.label, ready_location)
            self._register()

    def _stop_script(
        self,
        action: str = "stop",
        fallback_action: str | None = None,
    ) -> None:
        if not self.bootstrap_allowed or not os.path.isfile(self.script):
            return
        result = self._run_script(action, 120)
        if result.returncode == 0:
            return
        self.log.warning(
            "%s %s failed: stdout=%s stderr=%s",
            self.config.label,
            action,
            self.tail(result.stdout),
            self.tail(result.stderr),
        )
        if fallback_action is None:
            return
        fallback = self._run_script(fallback_action, 120)
        if fallback.returncode != 0:
            self.log.warning(
                "%s %s fallback failed: stdout=%s stderr=%s",
                self.config.label,
                fallback_action,
                self.tail(fallback.stdout),
                self.tail(fallback.stderr),
            )

    def wait_until_ready(
        self,
        is_ready: Callable[[], bool],
        *,
        timeout: float,
        poll_interval: float = 1.0,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if is_ready():
                return True
            time.sleep(poll_interval)
        return is_ready()
