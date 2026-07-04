import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch is not available in this test runtime", allow_module_level=True)

from image_studio.errors import UserInputError
from image_studio.pipelines.pid import patchify_flux2_raw_latents, validate_dims


def test_flux2_patchify_shape_and_order():
    raw = torch.arange(1 * 32 * 4 * 6).reshape(1, 32, 4, 6)
    packed = patchify_flux2_raw_latents(raw)
    assert packed.shape == (1, 128, 2, 3)
    expected = raw.reshape(1, 32, 2, 2, 3, 2).permute(0, 1, 3, 5, 2, 4).reshape(1, 128, 2, 3)
    assert torch.equal(packed, expected)


def test_flux2_patchify_validates_layout():
    with pytest.raises(UserInputError):
        patchify_flux2_raw_latents(torch.zeros(1, 31, 4, 4))
    with pytest.raises(UserInputError):
        patchify_flux2_raw_latents(torch.zeros(1, 32, 3, 4))
    assert validate_dims(512, 256) == (2048, 1024)
