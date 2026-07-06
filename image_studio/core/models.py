"""Model metadata, typed parameter binding, and explicit registration."""

from __future__ import annotations

import dataclasses
import types
from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass
from enum import StrEnum
from typing import Any, Protocol, Union, get_args, get_origin, get_type_hints

from ..errors import ModelNotFoundError, UserInputError


class Operation(StrEnum):
    IMAGE_GENERATE = "image.generate"
    IMAGE_EDIT = "image.edit"
    IMAGE_UPSCALE = "image.upscale"
    VIDEO_GENERATE = "video.generate"
    VIDEO_UPSCALE = "video.upscale"
    CHAT = "chat"


def _type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        non_none = [item for item in get_args(annotation) if item is not type(None)]
        return _type_name(non_none[0]) if len(non_none) == 1 else "any"
    if annotation in (str, int, float, bool):
        return annotation.__name__
    return "any"


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    type: str
    required: bool
    default: Any = None
    label: str = ""
    description: str = ""
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[Any, ...] = ()
    secret: bool = False

    def describe(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "required": self.required,
            "label": self.label or self.name.replace("_", " ").title(),
        }
        if not self.required and not self.secret:
            payload["default"] = self.default
        for name in ("description", "minimum", "maximum"):
            value = getattr(self, name)
            if value not in (None, ""):
                payload[name] = value
        if self.choices:
            payload["choices"] = list(self.choices)
        if self.secret:
            payload["secret"] = True
        return payload


def parameters_for_dataclass(parameter_type: type[Any]) -> tuple[ParameterSpec, ...]:
    if not dataclasses.is_dataclass(parameter_type):
        raise TypeError(f"Parameter type must be a dataclass: {parameter_type!r}")
    hints = get_type_hints(parameter_type)
    specs = []
    for item in dataclasses.fields(parameter_type):
        required = item.default is MISSING and item.default_factory is MISSING
        default = None
        if item.default is not MISSING:
            default = item.default
        elif item.default_factory is not MISSING:
            default = item.default_factory()
        metadata = item.metadata
        specs.append(
            ParameterSpec(
                name=item.name,
                type=_type_name(hints.get(item.name, Any)),
                required=required,
                default=default,
                label=str(metadata.get("label", "")),
                description=str(metadata.get("description", "")),
                minimum=metadata.get("minimum"),
                maximum=metadata.get("maximum"),
                choices=tuple(metadata.get("choices", ())),
                secret=bool(metadata.get("secret", False)),
            )
        )
    return tuple(specs)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display_name: str
    backend_id: str
    operations: tuple[Operation, ...]
    description: str = ""
    aliases: tuple[str, ...] = ()
    order: int = 100

    def describe(self, schemas: Mapping[Operation, tuple[ParameterSpec, ...]]) -> dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "backend_id": self.backend_id,
            "description": self.description,
            "order": self.order,
            "operations": {
                operation.value: [parameter.describe() for parameter in schemas[operation]]
                for operation in self.operations
            },
        }


@dataclass(frozen=True)
class OperationBinding:
    parameter_type: type[Any]
    handler: Callable[[Any, Any], Any]


class ModelAdapter(Protocol):
    spec: ModelSpec

    def parameter_schema(self, operation: Operation) -> tuple[ParameterSpec, ...]: ...

    def execute(
        self,
        operation: Operation,
        parameters: Mapping[str, Any],
        progress: Any = None,
        *,
        strict: bool = True,
    ) -> Any: ...


class DataclassModelAdapter:
    """Bind external mappings to per-model dataclasses before invocation."""

    def __init__(self, spec: ModelSpec, bindings: Mapping[Operation, OperationBinding]) -> None:
        self.spec = spec
        self._bindings = dict(bindings)
        missing = set(spec.operations) - set(bindings)
        if missing:
            raise ValueError(f"Missing bindings for {spec.id}: {sorted(item.value for item in missing)}")
        self._schemas = {
            operation: parameters_for_dataclass(binding.parameter_type)
            for operation, binding in self._bindings.items()
        }

    def parameter_schema(self, operation: Operation) -> tuple[ParameterSpec, ...]:
        try:
            return self._schemas[operation]
        except KeyError:
            raise UserInputError(f"{self.spec.display_name} does not support {operation.value}.") from None

    def execute(
        self,
        operation: Operation,
        parameters: Mapping[str, Any],
        progress: Any = None,
        *,
        strict: bool = True,
    ) -> Any:
        try:
            binding = self._bindings[operation]
        except KeyError:
            raise UserInputError(f"{self.spec.display_name} does not support {operation.value}.") from None

        fields = {item.name for item in dataclasses.fields(binding.parameter_type)}
        unknown = set(parameters) - fields
        if strict and unknown:
            raise UserInputError(
                f"Unknown parameters for {self.spec.display_name}: {', '.join(sorted(unknown))}"
            )
        values = {name: parameters[name] for name in fields if name in parameters}
        schema_by_name = {item.name: item for item in self._schemas[operation]}
        for name, value in values.items():
            schema = schema_by_name[name]
            if value is None or schema.type == "any":
                continue
            valid_type = {
                "str": isinstance(value, str),
                "int": isinstance(value, int) and not isinstance(value, bool),
                "float": isinstance(value, int | float) and not isinstance(value, bool),
                "bool": isinstance(value, bool),
            }.get(schema.type, True)
            if not valid_type:
                raise UserInputError(
                    f"Invalid parameter {name!r} for {self.spec.display_name}: "
                    f"expected {schema.type}, got {type(value).__name__}."
                )
            if schema.choices and value not in schema.choices:
                raise UserInputError(
                    f"Invalid parameter {name!r} for {self.spec.display_name}: "
                    f"choose one of {', '.join(map(str, schema.choices))}."
                )
            if schema.minimum is not None and value < schema.minimum:
                raise UserInputError(
                    f"Invalid parameter {name!r} for {self.spec.display_name}: "
                    f"minimum is {schema.minimum}."
                )
            if schema.maximum is not None and value > schema.maximum:
                raise UserInputError(
                    f"Invalid parameter {name!r} for {self.spec.display_name}: "
                    f"maximum is {schema.maximum}."
                )
        try:
            typed_parameters = binding.parameter_type(**values)
        except TypeError as exc:
            raise UserInputError(f"Invalid parameters for {self.spec.display_name}: {exc}") from exc
        return binding.handler(typed_parameters, progress)


class ModelRegistry:
    """Case-insensitive registry with stable IDs and legacy aliases."""

    def __init__(self) -> None:
        self._adapters: dict[str, ModelAdapter] = {}
        self._lookups: dict[str, str] = {}

    @staticmethod
    def _lookup_key(value: str) -> str:
        return value.strip().casefold()

    def register(self, adapter: ModelAdapter) -> None:
        model_id = adapter.spec.id
        if model_id in self._adapters:
            raise ValueError(f"Model already registered: {model_id}")
        names = (model_id, adapter.spec.display_name, *adapter.spec.aliases)
        for name in names:
            key = self._lookup_key(name)
            if key in self._lookups:
                raise ValueError(f"Model name or alias already registered: {name}")
        self._adapters[model_id] = adapter
        for name in names:
            self._lookups[self._lookup_key(name)] = model_id

    def resolve(self, identifier: str) -> ModelAdapter:
        model_id = self._lookups.get(self._lookup_key(identifier or ""))
        if model_id is None:
            raise ModelNotFoundError(f"Unknown model: {identifier!r}")
        return self._adapters[model_id]

    def for_operation(self, operation: Operation) -> tuple[ModelAdapter, ...]:
        return tuple(
            sorted(
                (
                    adapter
                    for adapter in self._adapters.values()
                    if operation in adapter.spec.operations
                ),
                key=lambda adapter: (adapter.spec.order, adapter.spec.display_name),
            )
        )

    def describe(self) -> list[dict[str, Any]]:
        return [
            adapter.spec.describe(
                {operation: adapter.parameter_schema(operation) for operation in adapter.spec.operations}
            )
            for adapter in self._adapters.values()
        ]
