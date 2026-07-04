import pytest

from image_studio.errors import UserInputError
from image_studio.validation import (
    is_ltx_audio_video_frame_count,
    snap_ltx_audio_video_frames,
    validate_boogu_dims,
    validate_dims,
    validate_ideogram_dims,
)


def test_dimension_validation():
    assert validate_dims(1024, 768) == (1024, 768)
    assert validate_ideogram_dims(1536, 256) == (1536, 256)
    assert validate_boogu_dims(2048, 1024) == (2048, 1024)
    with pytest.raises(UserInputError):
        validate_dims(1023, 768)
    with pytest.raises(UserInputError):
        validate_ideogram_dims(1792, 256)
    with pytest.raises(UserInputError):
        validate_boogu_dims(4096, 1024)


def test_ltx_audio_frame_snapping():
    assert snap_ltx_audio_video_frames(121) == 121
    assert snap_ltx_audio_video_frames(120) == 121
    assert snap_ltx_audio_video_frames(1) == 9
    assert is_ltx_audio_video_frame_count(121)
    assert not is_ltx_audio_video_frame_count(120)
