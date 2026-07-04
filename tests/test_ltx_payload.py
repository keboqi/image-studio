import pytest

from image_studio.errors import UserInputError
from image_studio.services.ltx_video import build_payload


def test_ltx_payload_audio_and_ic_lora_validation():
    payload = build_payload(
        prompt="hello", negative_prompt="", keyframes=[], width=1024, height=1024,
        frames=121, fps=24, skip_memory_cleanup=True, pipeline_type="a2vid_two_stage",
        audio_base64="data", audio_filename="voice.wav",
    )
    assert payload["audio_max_duration"] == pytest.approx(121 / 24)
    assert payload["num_inference_steps"] == 8

    with pytest.raises(UserInputError):
        build_payload(
            prompt="hello", negative_prompt="", keyframes=[], width=1000, height=1024,
            frames=121, fps=24, skip_memory_cleanup=True, pipeline_type="a2vid_two_stage",
            audio_base64="data",
        )
    with pytest.raises(UserInputError):
        build_payload(
            prompt="hello", negative_prompt="", keyframes=[], width=1024, height=1024,
            frames=121, fps=24, skip_memory_cleanup=True, pipeline_type="ic_lora",
            ic_lora_key="demo", reference_image_base64="image", video_conditioning=[{"video": "x"}],
        )
