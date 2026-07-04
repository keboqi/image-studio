"""Extracted runtime implementation."""

from __future__ import annotations
from image_studio.errors import BackendUnavailableError
import os

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

@dataclass(frozen=True)
class ManagedScriptConfig:
    """Configuration shared by backends controlled through shell scripts."""

    label: str
    manager_key: str
    vram_mb: int
    script: str
    shell: str
    shell_env_name: str
    ready_timeout: int
    start_timeout: int
    request_timeout: int
    working_dir: str = field(default_factory=os.getcwd)

class ManagedScriptService:
    """Common lifecycle for GPU backends managed by start/stop scripts."""

    def __init__(self, config: ManagedScriptConfig):
        self.config = config
        self.mgr_key = config.manager_key
        self.vram_mb = config.vram_mb
        self.script = config.script
        self.lock = threading.RLock()

    def _script_env(self) -> dict[str, str]:
        return os.environ.copy()

    def _run_script(self, action: str, timeout: int) -> subprocess.CompletedProcess:
        if not os.path.isfile(self.script):
            raise BackendUnavailableError(f"{self.config.label} launcher not found: {self.script}")

        cmd = [self.config.shell, self.script, action]
        log.info("Running %s backend command: %s", self.config.label, " ".join(cmd))
        started_at = time.perf_counter()
        try:
            result = subprocess.run(
                cmd,
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

        log.info(
            "%s backend command '%s' exited with code %s after %.1fs.",
            self.config.label,
            action,
            result.returncode,
            time.perf_counter() - started_at,
        )
        return result

    @staticmethod
    def _tail(text: str, limit: int = 8000) -> str:
        text = (text or "").strip()
        return text[-limit:] if len(text) > limit else text

    def _register(self) -> None:
        if model_mgr.is_loaded(self.mgr_key):
            model_mgr.touch(self.mgr_key)
            return
        model_mgr.register(self.mgr_key, self, self.vram_mb, unload_fn=self.stop)

    def _raise_action_failure(
        self,
        action: str,
        result: subprocess.CompletedProcess,
    ) -> None:
        raise BackendUnavailableError(
            f"Failed to {action} {self.config.label} backend.\n"
            f"STDOUT:\n{self._tail(result.stdout)}\n\n"
            f"STDERR:\n{self._tail(result.stderr)}"
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

            with _inprocess_gpu_lock:
                if not is_ready():
                    model_mgr.ensure_vram(self.vram_mb, exclude=self.mgr_key)

            if prepare_existing is not None and prepare_existing():
                self._register()
                return

            result = self._run_script("start", self.config.start_timeout)
            if result.returncode != 0:
                self._raise_action_failure("start", result)

            deadline = time.time() + self.config.ready_timeout
            while time.time() < deadline:
                if is_ready():
                    log.info(
                        "%s backend is ready at %s.",
                        self.config.label,
                        ready_location,
                    )
                    self._register()
                    return
                time.sleep(2)

            raise BackendUnavailableError(
                f"{self.config.label} backend started but did not become ready before timeout. "
                f"Check logs with: {self.config.shell} {self.script} logs"
            )

    def _stop_script(self, action: str = "stop", fallback_action: str | None = None) -> None:
        if _NO_BOOTSTRAP or not os.path.isfile(self.script):
            return

        result = self._run_script(action, 120)
        if result.returncode == 0:
            return

        log.warning(
            "%s %s failed: stdout=%s stderr=%s",
            self.config.label,
            action,
            self._tail(result.stdout),
            self._tail(result.stderr),
        )
        if fallback_action is None:
            return

        fallback = self._run_script(fallback_action, 120)
        if fallback.returncode != 0:
            log.warning(
                "%s %s fallback failed: stdout=%s stderr=%s",
                self.config.label,
                fallback_action,
                self._tail(fallback.stdout),
                self._tail(fallback.stderr),
            )

__all__ = (
    'ManagedScriptConfig',
    'ManagedScriptService',
)
_seal_runtime_module(globals())
