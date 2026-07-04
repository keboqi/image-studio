import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch is not available in this test runtime", allow_module_level=True)

from image_studio.pipelines.ideogram.lora import Ideogram4LoRAGroup, Ideogram4LoRALinear


def test_runtime_lora_linear_applies_tiny_adapter():
    base = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.weight.zero_()
    group = Ideogram4LoRAGroup(
        module_name="demo",
        lora_a=torch.tensor([[1.0, 0.0]]),
        lora_b=torch.tensor([[2.0], [3.0]]),
    )
    wrapped = Ideogram4LoRALinear(base, group, strength=1.0)
    result = wrapped(torch.tensor([[1.0, 9.0]]))
    assert torch.allclose(result, torch.tensor([[2.0, 3.0]]))
