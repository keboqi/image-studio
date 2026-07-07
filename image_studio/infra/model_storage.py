"""Disk storage catalog for downloaded model files."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import StorageError, UserInputError


NONE_CHOICE = "(none)"


@dataclass(frozen=True)
class ModelStorageTarget:
    key: str
    display_name: str
    paths: tuple[Path | str, ...] = ()
    hf_repos: tuple[str, ...] = ()
    active_model_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelStorageEntry:
    key: str
    display_name: str
    size_bytes: int
    paths: tuple[Path, ...]
    active_model_keys: tuple[str, ...] = ()

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "size_bytes": self.size_bytes,
            "paths": [str(path) for path in self.paths],
            "active_model_keys": list(self.active_model_keys),
        }


@dataclass(frozen=True)
class ModelStorageRemoval:
    key: str
    display_name: str
    removed_bytes: int
    removed_paths: tuple[Path, ...]

    def describe(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "removed_bytes": self.removed_bytes,
            "removed_paths": [str(path) for path in self.removed_paths],
        }


def format_storage_size(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def default_hf_cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(name)
        if value:
            roots.append(Path(value))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home) / "hub")

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        roots.append(Path(xdg_cache_home) / "huggingface" / "hub")

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return tuple(_dedupe_paths(roots))


def hf_repo_cache_dir(cache_root: Path, repo_id: str) -> Path:
    safe_repo = repo_id.strip().replace("/", "--")
    return Path(cache_root) / f"models--{safe_repo}"


class ModelStorageCatalog:
    def __init__(
        self,
        targets: list[ModelStorageTarget] | tuple[ModelStorageTarget, ...],
        *,
        cache_roots: tuple[Path | str, ...] | None = None,
    ) -> None:
        self._targets = {target.key: target for target in targets}
        self._configured_cache_roots = (
            tuple(Path(root) for root in cache_roots)
            if cache_roots is not None
            else None
        )

    def target(self, key: str) -> ModelStorageTarget:
        try:
            return self._targets[key]
        except KeyError:
            raise UserInputError(f"Unknown model storage target: {key!r}") from None

    def active_model_keys(self, key: str) -> tuple[str, ...]:
        return self.target(key).active_model_keys

    def status(self) -> list[dict[str, Any]]:
        entries = []
        for target in self._targets.values():
            paths = self._existing_paths(target)
            if not paths:
                continue
            entry = ModelStorageEntry(
                key=target.key,
                display_name=target.display_name,
                size_bytes=sum(_path_size(path) for path in paths),
                paths=tuple(paths),
                active_model_keys=target.active_model_keys,
            )
            entries.append(entry.describe())
        return sorted(entries, key=lambda item: item["display_name"].casefold())

    def remove(self, key: str) -> dict[str, Any]:
        if not key or key == NONE_CHOICE:
            return ModelStorageRemoval(NONE_CHOICE, NONE_CHOICE, 0, ()).describe()

        target = self.target(key)
        paths = _prune_nested_paths(self._existing_paths(target))
        removed_bytes = 0
        removed_paths = []
        for path in paths:
            _assert_safe_delete_path(path)
            removed_bytes += _path_size(path)
            _remove_path(path)
            removed_paths.append(path)
        return ModelStorageRemoval(
            key=target.key,
            display_name=target.display_name,
            removed_bytes=removed_bytes,
            removed_paths=tuple(removed_paths),
        ).describe()

    def remove_all(self) -> dict[str, Any]:
        paths = []
        for target in self._targets.values():
            paths.extend(self._existing_paths(target))
        paths = _prune_nested_paths(paths)

        removed_bytes = 0
        removed_paths = []
        for path in paths:
            _assert_safe_delete_path(path)
            removed_bytes += _path_size(path)
            _remove_path(path)
            removed_paths.append(path)
        return ModelStorageRemoval(
            key="all",
            display_name="All downloaded model files",
            removed_bytes=removed_bytes,
            removed_paths=tuple(removed_paths),
        ).describe()

    def _cache_roots(self) -> tuple[Path, ...]:
        if self._configured_cache_roots is not None:
            return self._configured_cache_roots
        return default_hf_cache_roots()

    def _candidate_paths(self, target: ModelStorageTarget) -> list[Path]:
        paths = [Path(path) for path in target.paths if str(path).strip()]
        for repo_id in target.hf_repos:
            repo_id = repo_id.strip()
            if not repo_id:
                continue
            for cache_root in self._cache_roots():
                paths.append(hf_repo_cache_dir(cache_root, repo_id))
        return _dedupe_paths(paths)

    def _existing_paths(self, target: ModelStorageTarget) -> list[Path]:
        return [path for path in self._candidate_paths(target) if path.exists() or path.is_symlink()]


def _dedupe_paths(paths: list[Path] | tuple[Path, ...]) -> list[Path]:
    seen = set()
    unique = []
    for path in paths:
        resolved = _normalise_path(path)
        key = os.path.normcase(str(resolved))
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _normalise_path(path: Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _prune_nested_paths(paths: list[Path]) -> list[Path]:
    pruned: list[Path] = []
    for path in sorted(_dedupe_paths(paths), key=lambda item: len(item.parts)):
        if any(_is_relative_to(path, parent) for parent in pruned):
            continue
        pruned.append(path)
    return pruned


def _path_size(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        try:
            return path.lstat().st_size
        except OSError:
            return 0

    total = 0
    for root, dirs, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in files:
            try:
                total += (root_path / name).lstat().st_size
            except OSError:
                pass
        for name in dirs:
            child = root_path / name
            if child.is_symlink():
                try:
                    total += child.lstat().st_size
                except OSError:
                    pass
    return total


def _assert_safe_delete_path(path: Path) -> None:
    resolved = _normalise_path(path)
    if not resolved.exists() and not resolved.is_symlink():
        return

    anchor = Path(resolved.anchor).resolve(strict=False) if resolved.anchor else None
    home = Path.home().resolve(strict=False)
    if anchor is not None and resolved == anchor:
        raise StorageError(f"Refusing to remove filesystem root: {resolved}")
    if resolved == home:
        raise StorageError(f"Refusing to remove home directory: {resolved}")
    if len(resolved.parts) <= 2:
        raise StorageError(f"Refusing to remove broad filesystem path: {resolved}")


def _remove_path(path: Path) -> None:
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    except OSError as exc:
        raise StorageError(f"Failed to remove downloaded model files at {path}: {exc}") from exc
