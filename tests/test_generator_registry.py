from dataclasses import dataclass

from image_studio.generators.registry import RequestHandlerRegistry


@dataclass
class Request:
    value: int


def test_request_handler_registry_dispatches_and_falls_back():
    registry = RequestHandlerRegistry(lambda request, progress=None: ("default", request.value, progress))
    registry.register("special", lambda request, progress=None: ("special", request.value, progress))
    assert registry.dispatch("special", Request(2), progress="p") == ("special", 2, "p")
    assert registry.dispatch("missing", Request(3)) == ("default", 3, None)
    assert registry.modes() == ("special",)
