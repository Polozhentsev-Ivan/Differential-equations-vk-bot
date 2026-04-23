"""Утилиты для работы с Markdown файлами."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def read_markdown(path: str | Path) -> str:
    """Читает .md файл и возвращает его содержимое."""
    return Path(path).read_text(encoding="utf-8")


def write_markdown(path: str | Path, content: str) -> None:
    """Записывает содержимое в .md файл."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def extract_json(text: str) -> dict[str, Any]:
    """Извлекает JSON из текста, который может содержать markdown code blocks.

    Обрабатывает форматы:
        - Чистый JSON: ``{"key": "value"}``
        - Markdown блок: ````json\\n{...}\\n````
        - JSON внутри текста
    """
    text = text.strip()

    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(0))

    raise ValueError(f"No valid JSON found in text: {text[:200]}")
