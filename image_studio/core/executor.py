"""Application service for validated model execution."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

from .backends import BackendRegistry
from .models import ModelRegistry, Operation

log = logging.getLogger(__name__)


class ModelExecutor:
    def __init__(
        self,
        models: ModelRegistry,
        backends: BackendRegistry,
        *,
        lease_timeout: float | None = None,
    ) -> None:
        self.models = models
        self.backends = backends
        self.lease_timeout = lease_timeout

    def execute(
        self,
        model_id: str,
        operation: Operation,
        parameters: Mapping[str, Any],
        progress: Any = None,
        *,
        strict: bool = True,
        request_id: str | None = None,
    ) -> Any:
        request_id = request_id or uuid.uuid4().hex
        adapter = self.models.resolve(model_id)
        started = time.perf_counter()
        log.info(
            "Model request started | request_id=%s model=%s operation=%s backend=%s",
            request_id,
            adapter.spec.id,
            operation.value,
            adapter.spec.backend_id,
        )
        try:
            with self.backends.lease(adapter.spec.backend_id, timeout=self.lease_timeout):
                return adapter.execute(operation, parameters, progress, strict=strict)
        finally:
            log.info(
                "Model request finished | request_id=%s model=%s operation=%s elapsed=%.3f",
                request_id,
                adapter.spec.id,
                operation.value,
                time.perf_counter() - started,
            )

    def catalog(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "models": self.models.describe(),
            "backends": self.backends.describe(),
        }
