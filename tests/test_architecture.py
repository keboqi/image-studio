from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "image_studio"


def test_runtime_access_is_confined_to_legacy_adapters():
    importers = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*.py")
        if "runtime_access import" in path.read_text(encoding="utf-8")
    }
    assert all(
        path == "runtime.py"
        or path.startswith("pipelines/")
        or path in {
            "generators/boogu.py",
            "generators/chat.py",
            "generators/hidream.py",
            "generators/ideogram.py",
            "generators/krea2.py",
                "generators/qwen.py",
                "generators/sensenova.py",
                "generators/zimage.py",
            "ui/gallery_actions.py",
            "ui/models.py",
        }
        or path.startswith("ui/components/")
        for path in importers
    )
    assert "generators/dispatch.py" not in importers
    assert "ui/layout.py" not in importers
    assert "ui/wiring.py" not in importers


def test_duplicate_runtime_implementations_are_removed():
    assert not (PACKAGE / "runtime_binding.py").exists()
    assert not (PACKAGE / "services" / "managed_runtime.py").exists()
