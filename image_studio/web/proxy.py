"""Extracted runtime implementation."""

from __future__ import annotations

# --- extracted runtime implementation ---
import sys as _runtime_sys
from dataclasses import dataclass, field
from image_studio.runtime_binding import bind_module as _bind_runtime_module, seal_module as _seal_runtime_module

_runtime_source = _runtime_sys.modules.get('image_studio.runtime') or _runtime_sys.modules.get('image_studio.app') or _runtime_sys.modules.get('__main__')
if _runtime_source is not None:
    _bind_runtime_module(globals(), vars(_runtime_source))

def _vllm_proxy_error(message: str, error_type: str = "proxy_error") -> dict[str, Any]:
    return {
        "error": {
            "message": str(message),
            "type": error_type,
            "param": None,
            "code": None,
        }
    }

def _vllm_proxy_authorized(headers: Any, api_key: str) -> bool:
    if not api_key:
        return True
    auth = str(headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer ") and auth[7:].strip() == api_key:
        return True
    return str(headers.get("x-api-key") or "").strip() == api_key

def _vllm_proxy_target_url(path: str, query: str = "") -> str:
    quoted_path = urllib.parse.quote(path.strip("/"), safe="/:@._~-")
    url = f"{_diffusiongemma_vllm_service.api_base}/{quoted_path}"
    if query:
        url = f"{url}?{query}"
    return url

def _vllm_proxy_request_headers(headers: Any) -> dict[str, str]:
    proxied: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _VLLM_PROXY_HOP_BY_HOP_HEADERS or lower == "accept-encoding":
            continue
        proxied[key] = value
    return proxied

def _vllm_proxy_response_headers(headers: Any) -> dict[str, str]:
    proxied: dict[str, str] = {}
    for key, value in headers.items():
        lower = key.lower()
        if lower in _VLLM_PROXY_HOP_BY_HOP_HEADERS:
            continue
        proxied[key] = value
    return proxied

def _vllm_proxy_request_wants_stream(body: bytes, content_type: str) -> bool:
    if content_type.lower().startswith("text/event-stream"):
        return True
    if not body:
        return False
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        return False
    return bool(isinstance(payload, dict) and payload.get("stream") is True)

def _vllm_proxy_open(method: str, url: str, headers: dict[str, str], body: bytes):
    data = body if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    response = urllib.request.urlopen(req, timeout=DIFFUSIONGEMMA_VLLM_REQUEST_TIMEOUT)
    return response, int(getattr(response, "status", 200)), response.headers

def _vllm_proxy_iter_response(response):
    try:
        while True:
            chunk = response.read(65536)
            if not chunk:
                break
            yield chunk
    finally:
        response.close()

def _promote_vllm_proxy_routes(fastapi_app: Any) -> None:
    """Keep proxy routes ahead of broad Gradio fallback routes."""
    promote_routes_before_fallback(
        fastapi_app,
        {_VLLM_PROXY_ROUTE_NAME, _VLLM_PROXY_HEALTH_ROUTE_NAME},
    )

def attach_vllm_proxy_routes(blocks: Any, api_key: str = "") -> bool:
    """Expose the managed DiffusionGemma backend as /v1/* on the Gradio server."""
    try:
        import asyncio
        from fastapi import Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ImportError as exc:
        log.warning("Could not enable vLLM proxy routes because FastAPI is unavailable: %s", exc)
        return False

    fastapi_app = getattr(blocks, "app", None)
    if fastapi_app is None:
        log.warning("Could not enable vLLM proxy routes because this Gradio Blocks has no FastAPI app.")
        return False

    route_names = {getattr(route, "name", "") for route in getattr(fastapi_app, "routes", [])}
    if _VLLM_PROXY_ROUTE_NAME in route_names:
        _promote_vllm_proxy_routes(fastapi_app)
        return True

    async def vllm_proxy(path: str, request: Request):
        if not _vllm_proxy_authorized(request.headers, api_key):
            return JSONResponse(
                _vllm_proxy_error("Invalid or missing vLLM proxy API key.", "authentication_error"),
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )

        body = await request.body()
        target_url = _vllm_proxy_target_url(path, request.url.query)
        request_headers = _vllm_proxy_request_headers(request.headers)

        try:
            await asyncio.to_thread(_diffusiongemma_vllm_service.ensure_running)
            response, status_code, response_headers = await asyncio.to_thread(
                _vllm_proxy_open,
                request.method,
                target_url,
                request_headers,
                body,
            )
        except urllib.error.HTTPError as exc:
            detail = await asyncio.to_thread(exc.read)
            return Response(
                content=detail,
                status_code=exc.code,
                media_type=exc.headers.get("content-type", "application/json"),
                headers=_vllm_proxy_response_headers(exc.headers),
            )
        except Exception as exc:
            log.exception("vLLM proxy request failed for %s %s", request.method, target_url)
            return JSONResponse(_vllm_proxy_error(exc, "backend_error"), status_code=503)

        content_type = response_headers.get("content-type", "application/json")
        headers = _vllm_proxy_response_headers(response_headers)
        if _vllm_proxy_request_wants_stream(body, content_type):
            return StreamingResponse(
                _vllm_proxy_iter_response(response),
                status_code=status_code,
                media_type=content_type,
                headers=headers,
            )

        data = await asyncio.to_thread(response.read)
        response.close()
        return Response(
            content=data,
            status_code=status_code,
            media_type=content_type,
            headers=headers,
        )

    async def vllm_proxy_health(request: Request):
        if not _vllm_proxy_authorized(request.headers, api_key):
            return JSONResponse(
                _vllm_proxy_error("Invalid or missing vLLM proxy API key.", "authentication_error"),
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        healthy = await asyncio.to_thread(_diffusiongemma_vllm_service.is_healthy)
        sleeping = await asyncio.to_thread(_diffusiongemma_vllm_service.is_sleeping)
        return JSONResponse({
            "ok": healthy and not sleeping,
            "healthy": healthy,
            "sleeping": sleeping,
            "backend": _diffusiongemma_vllm_service.api_base,
            "model": _diffusiongemma_vllm_service.model,
        })

    fastapi_app.add_api_route(
        "/v1/{path:path}",
        vllm_proxy,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        name=_VLLM_PROXY_ROUTE_NAME,
        include_in_schema=False,
    )
    fastapi_app.add_api_route(
        "/vllm/health",
        vllm_proxy_health,
        methods=["GET"],
        name=_VLLM_PROXY_HEALTH_ROUTE_NAME,
        include_in_schema=False,
    )
    _promote_vllm_proxy_routes(fastapi_app)
    log.info(
        "vLLM proxy enabled at /v1/* -> %s (model=%s).",
        _diffusiongemma_vllm_service.api_base,
        _diffusiongemma_vllm_service.model,
    )
    return True

__all__ = (
    '_vllm_proxy_error',
    '_vllm_proxy_authorized',
    '_vllm_proxy_target_url',
    '_vllm_proxy_request_headers',
    '_vllm_proxy_response_headers',
    '_vllm_proxy_request_wants_stream',
    '_vllm_proxy_open',
    '_vllm_proxy_iter_response',
    '_promote_vllm_proxy_routes',
    'attach_vllm_proxy_routes',
)
_seal_runtime_module(globals())
