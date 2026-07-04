"""Pure JSON extraction and prompt-enhancer parsing helpers."""

from __future__ import annotations

import json


def strip_markdown_fences(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_object(text: str) -> str | None:
    text = strip_markdown_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return text[start:]


def fix_unescaped_json_newlines(text: str) -> str:
    output: list[str] = []
    in_string = False
    escape_next = False
    for char in text:
        if escape_next:
            output.append(char)
            escape_next = False
        elif char == "\\" and in_string:
            output.append(char)
            escape_next = True
        elif char == '"':
            in_string = not in_string
            output.append(char)
        elif in_string and char == "\n":
            output.append("\\n")
        elif not (in_string and char == "\r"):
            output.append(char)
    return "".join(output)


def parse_enhance_json(text: str) -> dict:
    stripped = strip_markdown_fences(text)
    block = extract_json_object(stripped) or stripped
    for candidate in (block, fix_unescaped_json_newlines(block)):
        try:
            data = json.loads(candidate)
            if not isinstance(data, dict):
                raise ValueError("Prompt enhancer JSON was not an object.")
            prompt = data.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Prompt enhancer JSON is missing a non-empty prompt.")
            return data
        except (json.JSONDecodeError, ValueError):
            continue
    raise ValueError(f"Failed to parse prompt enhancer JSON: {stripped[:500]}")
