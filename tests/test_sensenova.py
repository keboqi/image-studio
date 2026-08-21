import pytest

from image_studio.errors import UserInputError
from image_studio.generators.sensenova import _sensenova_aspect_size
from image_studio.validation import validate_sensenova_dims


def test_sensenova_dimensions_require_upstream_32_pixel_grid():
    assert validate_sensenova_dims(1024, 768) == (1024, 768)
    with pytest.raises(UserInputError, match="multiples of 32"):
        validate_sensenova_dims(1008, 768)


def test_sensenova_single_image_aspect_preservation_snaps_to_32():
    width, height = _sensenova_aspect_size(1600, 900, 1024 * 1024)
    assert width % 32 == 0
    assert height % 32 == 0
    assert width > height
    assert (width / height) == pytest.approx(16 / 9, rel=0.04)
