import json

import pytest

from image_studio.parsing import extract_json_object, fix_unescaped_json_newlines, parse_enhance_json
from image_studio.pipelines.ideogram.prompting import (
    clean_malformed_json_caption,
    normalize_caption_object,
    parse_caption,
)


def test_extract_json_object_honors_nesting_and_quoted_braces():
    assert extract_json_object('prefix {"a":{"text":"}"},"b":1} suffix') == '{"a":{"text":"}"},"b":1}'
    assert extract_json_object("no object") is None


def test_parse_enhance_json_repairs_literal_newlines():
    parsed = parse_enhance_json('{"prompt":"line one\nline two","reasoning":"ok"}')
    assert parsed["prompt"] == "line one\nline two"
    assert json.loads(fix_unescaped_json_newlines('{"x":"a\nb"}')) == {"x": "a\nb"}
    with pytest.raises(ValueError):
        parse_enhance_json('{"reasoning":"missing prompt"}')


def test_ideogram_caption_cleanup_and_normalization():
    malformed = '{"high_level_description":"cup","style_description":{"lighting":"soft","bad":1},"compositional_deconstruction":{"elements":[{"desc":"cup"}]},""}'
    parsed = json.loads(parse_caption(malformed, "1:1"))
    assert parsed["aspect_ratio"] == "1:1"
    assert parsed["style_description"] == {"lighting": "soft"}
    assert parsed["compositional_deconstruction"]["elements"][0]["type"] == "obj"
    assert clean_malformed_json_caption('{"a":1,""}') == '{"a":1}'
    assert normalize_caption_object({}, "4:3") == {"aspect_ratio": "4:3"}
