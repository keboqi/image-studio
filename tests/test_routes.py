import ast
from pathlib import Path
from types import SimpleNamespace

from image_studio.web.routes import PUBLIC_API_ENDPOINTS, promote_routes_before_fallback


def test_public_api_contract_is_stable():
    assert PUBLIC_API_ENDPOINTS == (
        ("generate", 35, 4),
        ("edit", 18, 4),
        ("upscale", 9, 4),
        ("ai_remover", 3, 4),
        ("generate_video", 17, 3),
        ("upscale_video", 12, 3),
    )


def test_request_dataclass_arities_match_public_contract():
    request_path = Path(__file__).parents[1] / "image_studio" / "generators" / "base.py"
    tree = ast.parse(request_path.read_text(encoding="utf-8"))
    field_counts = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in {"GenerationRequest", "EditRequest"}:
            field_counts[node.name] = sum(
                isinstance(item, ast.AnnAssign) for item in node.body
            )
    arities = {name: inputs for name, inputs, _outputs in PUBLIC_API_ENDPOINTS}
    assert arities["generate"] == field_counts["GenerationRequest"]
    assert arities["edit"] == field_counts["EditRequest"]


def test_wiring_exposes_all_six_public_api_names():
    wiring_path = Path(__file__).parents[1] / "image_studio" / "ui" / "wiring.py"
    tree = ast.parse(wiring_path.read_text(encoding="utf-8"))
    names = {
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "api_name" and isinstance(keyword.value, ast.Constant)
    }
    assert names == {name for name, _inputs, _outputs in PUBLIC_API_ENDPOINTS}


def test_named_routes_are_promoted_before_fallback():
    def route(name, path):
        return SimpleNamespace(name=name, path=path)

    routes = [route("root", "/"), route("fallback", "/{path:path}"), route("proxy", "/v1/{path:path}")]
    app = SimpleNamespace(router=SimpleNamespace(routes=routes))
    promote_routes_before_fallback(app, {"proxy"})
    assert [item.name for item in routes] == ["root", "proxy", "fallback"]
