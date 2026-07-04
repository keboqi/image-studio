from pathlib import Path

from image_studio.storage.metadata import JsonFileCache, PromptMetadataStore


def test_json_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    cache = JsonFileCache(str(path), schema_version=2)
    cache.set("prompt", {"answer": 42})
    assert JsonFileCache(str(path), schema_version=2).get("prompt") == {"answer": 42}
    assert JsonFileCache(str(path), schema_version=3).get("prompt") is None


def test_prompt_metadata_rejects_paths_outside_store(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()

    def contained(path):
        candidate = Path(path).resolve()
        return str(candidate) if root.resolve() in candidate.parents else None

    store = PromptMetadataStore(contained, suffix=".prompt.json")
    raw = root / "image.png"
    raw.touch()
    store.write(str(raw), {"prompt": "hello"})
    assert store.read(str(raw))["prompt"] == "hello"
    assert store.sidecar_path(str(tmp_path / "outside.png")) is None
