"""Image Studio command-line application entry point."""

from __future__ import annotations

from . import runtime as _runtime

build_ui = _runtime.build_ui
parse_args = _runtime.parse_args
run_selftest = _runtime.run_selftest
attach_app_routes = _runtime.attach_app_routes
main = _runtime.main


def __getattr__(name: str):
    """Preserve imports of established runtime helpers during migration."""
    return getattr(_runtime, name)


if __name__ == "__main__":
    raise SystemExit(main())
