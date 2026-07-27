# Image Studio architecture

Image Studio separates a **model** from the **backend** that executes it. A model owns its
capabilities, parameter schema, validation, and payload adaptation. A backend owns process
lifecycle, health, and concurrency. `ModelExecutor` joins the two for UI and API callers.

```text
Gradio / API
    -> ModelExecutor
       -> ModelRegistry -> DataclassModelAdapter
       -> BackendRegistry -> concurrency-safe backend lease
```

## Core contracts

- `ModelSpec` gives a model a stable ID, display name, backend ID, operations, aliases, and UI order.
- `OperationBinding` pairs one operation with a model-specific parameter dataclass and handler.
- `ModelRegistry` resolves stable IDs and legacy labels without importing heavyweight model code.
- `BackendController` serializes startup, limits concurrent requests, and prevents stop while leased.
- `ModelExecutor` validates parameters, acquires a backend lease, invokes the adapter, and logs timing.

The existing flat Gradio requests remain compatibility facades. Their dispatch functions translate
legacy fields into the per-model dataclasses before calling `IMAGE_MODEL_EXECUTOR`.

## Adding a model

1. Add a frozen parameter dataclass in `integrations/` for each supported operation.
2. Add a `DataclassModelAdapter` with a stable lowercase ID and an existing backend ID.
3. Inject the implementation callable through the integration's function bundle.
4. Add a contract test that resolves the ID and legacy aliases, validates its schema, and records the
   adapted invocation.

Do not import CUDA, Diffusers, or model repositories from the integration module. The injected
callable remains responsible for lazy loading.

## Adding a backend

1. Implement idempotent start, stop, and health callables.
2. Register a `CallableBackend` at application composition time.
3. Set an explicit concurrency capacity. GPU backends should default to one unless proven safe.
4. Test startup failure, repeated leases, capacity, and stop-during-request behavior.

Adapters may share a backend. This allows several models to use the same local GPU or managed
service without duplicating lifecycle code.

## Discovery

`GET /api/models` returns schema version 1 of the model catalog. Secret defaults are never included.
Clients should use stable model IDs; display names and compatibility aliases are presentation details.

## Composition and compatibility

`image_studio.app` is a lightweight entry point. It defers heavyweight GPU imports and process
construction to `composition.create_application()`. The resulting `AppContext` owns the model
manager, output store, managed services, locks, typed image executor, storage catalog, and UI
actions.

The historical flat Gradio handlers remain at the outer compatibility boundary, but they no longer
copy the runtime namespace into feature modules. New integrations must use constructor or
`AppContext` injection and must not import `runtime_access`.
