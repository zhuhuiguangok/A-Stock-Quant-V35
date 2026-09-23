"""AI 摘要服务（同步版）。

移植自原项目 ai_service.py：DeepSeek 主 → Moonshot 备，OpenAI 兼容接口。
Key 读取顺序：环境变量 DEEPSEEK_API_KEY → 原项目 backend/.env（迁移过渡期兼容）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import httpx


def _literal_eval_safe(text: str):
    import ast

    return ast.literal_eval(text)


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEEPSEEK_MODEL = "deepseek-chat"
MOONSHOT_MODEL = "moonshot-v1-8k"

_LEGACY_ENV_PATH = Path(r"D:\codex-workspace\AI每日研报调研平台\backend\.env")


def _load_legacy_env(key: str) -> str:
    """原 FastAPI 项目的 .env 里读取 key，避免迁移后还要手动复制配置。"""
    try:
        if _LEGACY_ENV_PATH.exists():
            for line in _LEGACY_ENV_PATH.read_text("utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{key}="):
                    value = line.split("=", 1)[1].strip()
                    if value and "your_" not in value.lower():
                        return value
    except Exception:
        pass
    return ""


class AIService:
    def __init__(self):
        self.deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "") or _load_legacy_env("DEEPSEEK_API_KEY")
        self.moonshot_key = os.environ.get("MOONSHOT_API_KEY", "") or _load_legacy_env("MOONSHOT_API_KEY")
        self.deepseek_model = DEEPSEEK_MODEL
        self.moonshot_model = MOONSHOT_MODEL

    @property
    def available(self) -> bool:
        return bool(self.deepseek_key or self.moonshot_key)

    def summarize_report(self, title: str, content: str) -> Optional[Dict[str, Any]]:
        return self.ask_json(self._build_prompt(title, content))

    def ask_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        result = self._call_deepseek(prompt)
        if result:
            return result
        return self._call_moonshot(prompt)

    def _build_prompt(self, title: str, content: str) -> str:
        return f"""你是一名资深 A 股投资分析师。请阅读以下券商研报，输出结构化 JSON：

{{
  "summary": "300 字左右的投资逻辑摘要",
  "investment_points": ["要点1", "要点2", "要点3"],
  "tags": ["行业标签1", "概念标签2"],
  "sentiment": "bullish|neutral|bearish",
  "rating": "买入|增持|中性|减持",
  "risks": ["风险1", "风险2"],
  "target_price": "12.5",
  "target_stocks": ["600519"]
}}

字段说明：
- summary: 提炼研报核心投资逻辑，突出数据与结论
- investment_points: 3-5 条最值得关注的要点，每条一句话
- tags: 行业与概念标签
- sentiment: bullish=看多, neutral=中性, bearish=看空
- rating: 研报给出的投资评级，若无法判断填"中性"
- risks: 2-4 条研报提及或隐含的主要风险
- target_price: 研报给出的目标价数字（仅数字，如 12.5），未提及填 null
- target_stocks: 涉及的具体股票代码数组，无则空数组

研报标题：{title}

研报内容：
{content[:12000]}"""

    def _post_chat(self, base_url: str, key: str, model: str, prompt: str, json_mode: bool) -> Optional[Dict]:
        try:
            with httpx.Client(timeout=60.0, trust_env=False) as client:
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                if not json_mode:
                    start, end = content.find("{"), content.rfind("}") + 1
                    if start < 0 or end <= start:
                        return None
                    content = content[start:end]
                return json.loads(content)
        except Exception:
            return None

    def _call_deepseek(self, prompt: str) -> Optional[Dict]:
        if not self.deepseek_key:
            return None
        return self._post_chat(DEEPSEEK_BASE_URL, self.deepseek_key, self.deepseek_model, prompt, json_mode=True)

    def _call_moonshot(self, prompt: str) -> Optional[Dict]:
        if not self.moonshot_key:
            return None
        return self._post_chat(MOONSHOT_BASE_URL, self.moonshot_key, self.moonshot_model, prompt, json_mode=False)

    @staticmethod
    def _as_list(value, sep_pattern=r"[；;\n]"):
        """把 AI 返回的字段规范化为数组（模型偶尔返回字符串甚至 Python repr 风格）。"""
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return []
        for loader in (json.loads, _literal_eval_safe):
            try:
                parsed = loader(text)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if str(v).strip()]
            except Exception:
                continue
        import re

        parts = [p.strip(" '\"") for p in re.split(sep_pattern, text)]
        return [p for p in parts if p]

    def parse_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """AI JSON → ResearchSummary 字段（JSONField 直接存 list，不再 dumps）。"""
        return {
            "summary": str(result.get("summary") or ""),
            "tags": self._as_list(result.get("tags")),
            "investment_points": self._as_list(result.get("investment_points")),
            "sentiment": result.get("sentiment") or "neutral",
            "target_stocks": self._as_list(result.get("target_stocks")),
            "rating": result.get("rating") or "中性",
            "risks": self._as_list(result.get("risks")),
            "target_price": str(result.get("target_price")) if result.get("target_price") else "",
            "ai_model": self.deepseek_model,
        }
