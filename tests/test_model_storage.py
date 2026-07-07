from pathlib import Path

from image_studio.infra.model_storage import (
    ModelStorageCatalog,
    ModelStorageTarget,
    format_storage_size,
    hf_repo_cache_dir,
)


def test_model_storage_catalog_detects_and_removes_local_and_hf_downloads(tmp_path):
    local_dir = tmp_path / "models" / "demo"
    local_dir.mkdir(parents=True)
    (local_dir / "weights.bin").write_bytes(b"x" * 5)

    cache_root = tmp_path / "hf" / "hub"
    repo_dir = hf_repo_cache_dir(cache_root, "owner/repo")
    (repo_dir / "blobs").mkdir(parents=True)
    (repo_dir / "blobs" / "abc").write_bytes(b"y" * 7)

    catalog = ModelStorageCatalog(
        [
            ModelStorageTarget(
                key="demo",
                display_name="Demo Model",
                paths=(local_dir,),
                hf_repos=("owner/repo",),
                active_model_keys=("demo",),
            )
        ],
        cache_roots=(cache_root,),
    )

    status = catalog.status()
    assert len(status) == 1
    assert status[0]["display_name"] == "Demo Model"
    assert status[0]["size_bytes"] == 12
    assert status[0]["active_model_keys"] == ["demo"]

    removed = catalog.remove("demo")
    assert removed["removed_bytes"] == 12
    assert not local_dir.exists()
    assert not repo_dir.exists()
    assert catalog.status() == []


def test_model_storage_remove_all_dedupes_shared_paths(tmp_path):
    cache_root = tmp_path / "hf" / "hub"
    repo_dir = hf_repo_cache_dir(cache_root, "owner/shared")
    repo_dir.mkdir(parents=True)
    (repo_dir / "model.safetensors").write_bytes(b"z" * 9)

    catalog = ModelStorageCatalog(
        [
            ModelStorageTarget("a", "A", hf_repos=("owner/shared",)),
            ModelStorageTarget("b", "B", hf_repos=("owner/shared",)),
        ],
        cache_roots=(cache_root,),
    )

    removed = catalog.remove_all()
    assert removed["removed_bytes"] == 9
    assert removed["removed_paths"] == [str(Path(repo_dir).resolve(strict=False))]
    assert not repo_dir.exists()


def test_format_storage_size():
    assert format_storage_size(0) == "0 B"
    assert format_storage_size(1536) == "1.5 KiB"
    assert format_storage_size(2 * 1024 * 1024) == "2.0 MiB"

