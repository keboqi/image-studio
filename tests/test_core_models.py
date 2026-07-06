from dataclasses import dataclass, field

import pytest

from image_studio.core.backends import BackendRegistry, CallableBackend
from image_studio.core.executor import ModelExecutor
from image_studio.core.models import (
    DataclassModelAdapter,
    ModelRegistry,
    ModelSpec,
    Operation,
    OperationBinding,
)
from image_studio.errors import ModelNotFoundError, UserInputError


@dataclass(frozen=True)
class DemoParameters:
    prompt: str
    steps: int = field(default=4, metadata={"minimum": 1, "maximum": 20})
    token: str = field(default="private", metadata={"secret": True})


def _demo_adapter(calls):
    return DataclassModelAdapter(
        ModelSpec(
            id="demo-model",
            display_name="Demo Model",
            backend_id="demo-backend",
            operations=(Operation.IMAGE_GENERATE,),
            aliases=("legacy demo",),
        ),
        {
            Operation.IMAGE_GENERATE: OperationBinding(
                DemoParameters,
                lambda parameters, progress: calls.append((parameters, progress)) or "result",
            )
        },
    )


def test_registry_resolves_ids_and_aliases_and_describes_typed_schema():
    calls = []
    registry = ModelRegistry()
    registry.register(_demo_adapter(calls))

    assert registry.resolve("DEMO-MODEL") is registry.resolve("legacy demo")
    description = registry.describe()[0]
    parameters = {
        item["name"]: item
        for item in description["operations"][Operation.IMAGE_GENERATE.value]
    }
    assert parameters["prompt"]["required"] is True
    assert parameters["steps"]["minimum"] == 1
    assert "default" not in parameters["token"]
    assert parameters["token"]["secret"] is True

    with pytest.raises(ModelNotFoundError):
        registry.resolve("missing")


def test_dataclass_adapter_rejects_unknown_parameters_in_strict_mode():
    adapter = _demo_adapter([])
    with pytest.raises(UserInputError, match="Unknown parameters"):
        adapter.execute(
            Operation.IMAGE_GENERATE,
            {"prompt": "hello", "surprise": True},
        )

    with pytest.raises(UserInputError, match="minimum is 1"):
        adapter.execute(
            Operation.IMAGE_GENERATE,
            {"prompt": "hello", "steps": 0},
        )

    with pytest.raises(UserInputError, match="expected int"):
        adapter.execute(
            Operation.IMAGE_GENERATE,
            {"prompt": "hello", "steps": "four"},
        )


def test_executor_validates_parameters_and_uses_backend_lease():
    calls = []
    starts = []
    models = ModelRegistry()
    models.register(_demo_adapter(calls))
    backends = BackendRegistry()
    backends.register(
        CallableBackend(
            id="demo-backend",
            label="Demo Backend",
            start_fn=lambda: starts.append("start"),
        )
    )
    executor = ModelExecutor(models, backends)

    assert executor.execute(
        "legacy demo",
        Operation.IMAGE_GENERATE,
        {"prompt": "hello", "steps": 8},
        progress="progress",
    ) == "result"
    assert starts == ["start"]
    assert calls[0][0] == DemoParameters(prompt="hello", steps=8)
    assert calls[0][1] == "progress"
    assert executor.catalog()["schema_version"] == 1
