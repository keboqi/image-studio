"""Optional Git checkout bootstrap used by model integrations."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepoSpec:
    name: str
    repo_url: str
    target_dir: str
    sentinel_files: tuple[str, ...]
    requirements: tuple[str, ...] = ()


class GitBootstrap:
    def __init__(self, allowed: Callable[[str], bool] | None = None):
        self._allowed = allowed or (lambda _name: True)

    def ensure(self, spec: RepoSpec) -> bool:
        if self._has_sentinels(spec):
            return True
        if os.path.isdir(spec.target_dir):
            log.warning("%s directory exists but required files are missing: %s", spec.name, spec.target_dir)
            return False
        if not self._allowed(spec.name):
            return False
        log.info("%s not found; cloning from GitHub...", spec.name)
        try:
            subprocess.check_call(["git", "clone", "--depth", "1", spec.repo_url, spec.target_dir])
            self._install_requirements(spec)
            return self._has_sentinels(spec)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            log.warning("%s unavailable: %s", spec.name, exc)
            return False

    @staticmethod
    def _has_sentinels(spec: RepoSpec) -> bool:
        return all(os.path.isfile(os.path.join(spec.target_dir, path)) for path in spec.sentinel_files)

    @staticmethod
    def _install_requirements(spec: RepoSpec) -> None:
        for requirement in spec.requirements:
            req_path = os.path.join(spec.target_dir, requirement)
            if requirement.endswith(".txt") and os.path.isfile(req_path):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", req_path])
            elif not requirement.endswith(".txt"):
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", requirement])
