"""Compatibility launcher for existing Image Studio commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in arguments:
        tests = Path(__file__).resolve().parent / "tests"
        return subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(tests)],
            cwd=Path(__file__).resolve().parent,
            check=False,
        ).returncode

    from image_studio.app import main as app_main

    return app_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
