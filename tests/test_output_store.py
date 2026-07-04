from pathlib import Path

from PIL import Image

from image_studio.storage.output_store import OutputStore


def test_output_store_containment_and_preview_roundtrip(tmp_path):
    store = OutputStore(str(tmp_path))
    raw = tmp_path / "sample.png"
    Image.new("RGB", (8, 8), "red").save(raw)

    preview = store.ensure_preview(str(raw))
    assert Path(preview).is_file()
    assert store.raw_path(preview) == str(raw)
    assert store.contained_path(str(raw)) == str(raw.resolve())
    assert store.contained_path(str(tmp_path.parent / "escape.png")) is None


def test_path_from_gradio_shapes(tmp_path):
    store = OutputStore(str(tmp_path))
    assert store.path_from_value({"image": {"path": "x.png"}}) == "x.png"
    assert store.path_from_value({"name": "y.png"}) == "y.png"
