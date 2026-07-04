"""Shared process lifecycle for script-managed backends."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from ..errors import BackendUnavailableError


class ManagedService:
    """Minimal lifecycle implemented by externally managed backends."""

    def start(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


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
    """UI-independent base for a backend controlled by a shell script."""

    def __init__(self, config: ManagedScriptConfig) -> None:
        self.config = config
        self.mgr_key = config.manager_key
        self.vram_mb = config.vram_mb
        self.script = config.script
        self.lock = threading.RLock()

    def script_env(self) -> dict[str, str]:
        return {**os.environ, **self.config.environment}

    def run_script(self, action: str, timeout: int) -> subprocess.CompletedProcess[str]:
        if not os.path.isfile(self.script):
            raise BackendUnavailableError(f"{self.config.label} launcher not found: {self.script}")
        try:
            return subprocess.run(
                [self.config.shell, self.script, action],
                cwd=self.config.working_dir,
                env=self.script_env(),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise BackendUnavailableError(
                f"Could not run {self.config.shell!r}. Install it or set {self.config.shell_env_name}."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BackendUnavailableError(
                f"Timed out running {self.config.label} launcher ({action})."
            ) from exc

    @staticmethod
    def tail(text: str, limit: int = 8000) -> str:
        text = (text or "").strip()
        return text[-limit:] if len(text) > limit else text

    def wait_until_ready(
        self,
        is_ready: Callable[[], bool],
        *,
        timeout: float,
        poll_interval: float = 1.0,
    ) -> bool:
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if is_ready():
                return True
            time.sleep(poll_interval)
        return is_ready()
