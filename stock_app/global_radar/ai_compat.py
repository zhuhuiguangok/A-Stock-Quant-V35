"""复用研报模块的 AI 服务（DeepSeek 主 / Moonshot 备）。"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _service():
    from ..research.ai_service import AIService

    return AIService()


def ai_available() -> bool:
    return _service().available


def ask_json(prompt: str):
    return _service().ask_json(prompt)
