# Refactor implementation

The refactor was delivered as seven phases while preserving the existing Gradio and HTTP contracts.

## 1. Baseline and contract locks

- The six public Gradio endpoints retain their names and input/output arities.
- Stable model IDs are tested independently from display labels.
- Import smoke coverage verifies that `image_studio.app` does not load the GPU runtime.

## 2. Composition and quality gates

- `composition.create_application()` is the single lazy application factory.
- `AppContext` carries application-owned services into UI construction and wiring.
- CPU tests, Ruff, and mypy run in GitHub Actions on Python 3.12.

## 3. Service ownership and concurrency

- Script-managed backends share `infra.managed_service.ManagedScriptService`.
- vLLM, Krea2, LTX video, and AI-removal services receive configuration and collaborators explicitly.
- `ModelExecutor` and backend leases own image-generation concurrency; Gradio callbacks no longer add a
  second image-generation lock.

## 4. Typed dispatch

- `GenerationRequest` and `EditRequest` preserve the flat Gradio contracts.
- Dispatch translates those requests into model-specific parameter dataclasses through stable model IDs.
- The old label-keyed handler registry was removed.

## 5. Typed UI boundaries

- Each tab returns a typed component dataclass instead of a string-keyed mapping.
- `UiActions` is the explicit callable boundary between composition and Gradio wiring.
- Layout construction requires an `AppContext`.

## 6. Runtime extraction

- Runtime namespace copying into feature modules and `runtime_binding.py` were removed.
- Storage, managed services, web routes, layout, and event wiring no longer depend on runtime globals.
- The remaining read-only compatibility accessor is limited to historical GPU/UI adapter functions;
  a smoke assertion guarantees every referenced compatibility symbol is present.

## 7. Verification and maintenance

Run the same commands used by CI:

```bash
python -m pytest
python -m ruff check image_studio tests
python -m mypy image_studio
python -m compileall -q image_studio tests
```

New integrations must use typed registries, constructor injection, or `AppContext`. They must not add
new `runtime_access` imports.
