"""Image Studio command-line application entry point.

Importing this module does not import GPU frameworks or construct services.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .composition import create_application


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Gradio share link",
    )
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--auth", help="user:password for basic auth")
    parser.add_argument(
        "--vllm-proxy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Expose the managed DiffusionGemma vLLM backend at /v1/*",
    )
    parser.add_argument(
        "--vllm-proxy-api-key",
        default=None,
        help="Optional bearer/API key required by the /v1/* vLLM proxy",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the pytest compatibility suite without launching",
    )
    return parser.parse_args(argv)


def run_selftest() -> int:
    environment = os.environ.copy()
    environment["IMAGE_STUDIO_NO_BOOTSTRAP"] = "1"
    tests_dir = Path(__file__).resolve().parents[1] / "tests"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(tests_dir)],
        cwd=tests_dir.parent,
        env=environment,
        check=False,
    )
    return int(result.returncode)


def build_ui(context: Any = None):
    application = create_application()
    return application.runtime.build_ui(context or application.context)


def attach_app_routes(app: Any, **kwargs: Any):
    application = create_application()
    kwargs.setdefault("model_catalog_provider", application.model_executor.catalog)
    return application.runtime.attach_app_routes(app, **kwargs)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()

    application = create_application()
    config = application.context.config
    vllm_proxy = config.vllm.proxy_enabled if args.vllm_proxy is None else args.vllm_proxy
    api_key = (
        config.vllm.proxy_api_key
        if args.vllm_proxy_api_key is None
        else args.vllm_proxy_api_key
    )
    auth = tuple(args.auth.split(":", 1)) if args.auth else None

    gradio_app = application.build_ui()
    gradio_app.queue(max_size=4, default_concurrency_limit=1)
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": args.port,
        "share": args.share,
        "auth": auth,
        "css": application.runtime.CSS,
        "theme": application.runtime.THEME,
    }
    fastapi_app = application.attach_routes(
        gradio_app,
        vllm_proxy=vllm_proxy,
        api_key=api_key,
    )
    if vllm_proxy and fastapi_app is not None:
        launch_kwargs["_app"] = fastapi_app
    if vllm_proxy and args.share and not api_key:
        application.runtime.log.warning(
            "vLLM proxy is enabled with Gradio share and no proxy API key. "
            "Use --vllm-proxy-api-key or IMAGE_STUDIO_VLLM_PROXY_API_KEY for public links."
        )
    gradio_app.launch(**launch_kwargs, prevent_thread_lock=True)
    application.attach_routes(
        gradio_app,
        vllm_proxy=vllm_proxy,
        api_key=api_key,
    )
    block_thread = getattr(gradio_app, "block_thread", None)
    if callable(block_thread):
        block_thread()
    else:
        while True:
            time.sleep(3600)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
