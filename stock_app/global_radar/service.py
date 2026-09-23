"""全球雷达服务层：数据编排、AI 晨报、结果落盘。"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

from ..bull_bear_service import BULL_BEAR_DATA_ROOT

GLOBAL_RADAR_ROOT = Path(BULL_BEAR_DATA_ROOT).parent / "global_radar"
DASHBOARD_PATH = GLOBAL_RADAR_ROOT / "dashboard_latest.json"
BRIEF_PATH = GLOBAL_RADAR_ROOT / "brief_latest.json"


def _cross_module_context() -> tuple:
    """跨模块：资金流关注度榜 + 研报共振（供 A 股受益方向三角验证）。"""
    attention = []
    try:
        ff_path = Path(BULL_BEAR_DATA_ROOT) / "results" / "fund_flow_latest.json"
        if ff_path.exists():
            attention = (json.loads(ff_path.read_text("utf-8")).get("attention_ranking")) or []
    except Exception:
        attention = []
    resonance = {}
    try:
        from ..fund_flow_service import _research_resonance

        resonance = _research_resonance(attention) or {}
    except Exception:
        resonance = {}
    return attention, resonance


def build_dashboard() -> Dict:
    """采集 → 分析 → 完整面板 payload（不落盘）。"""
    from . import collectors
    from . import analysis

    failed = []

    assets = collectors.fetch_asset_closes()
    if not assets:
        failed.append("全球行情")
    us10y = collectors.fetch_us10y()
    if us10y is None:
        failed.append("美债收益率")
    btc = collectors.fetch_btc()
    macro = collectors.fetch_macro_china()
    if not macro:
        failed.append("中国宏观")
    margin = collectors.fetch_margin()
    if margin is None:
        failed.append("两融")
    qvix = collectors.fetch_qvix()
    if qvix is None:
        failed.append("QVIX")
    breadth = collectors.fetch_breadth()

    attention, resonance = _cross_module_context()
    payload = {
        "date": date.today().isoformat(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "asset_heat": analysis.build_asset_heat(assets, us10y, btc),
        "risk_appetite": analysis.build_risk_appetite(assets, us10y),
        "fear_greed": analysis.build_fear_greed(qvix, margin, breadth, assets),
        "clock_cn": analysis.build_clock_china(macro),
        "clock_us": analysis.build_clock_us(assets),
        "transmission": analysis.build_transmission(assets, us10y),
        "a_share_directions": analysis.build_a_share_directions(assets, us10y, attention, resonance),
        "meta": {
            "assets_ok": sorted(assets.keys()),
            "sources_failed": failed,
            "us10y_ok": us10y is not None,
            "btc_ok": btc is not None,
            "macro_ok": bool(macro),
            "margin_ok": margin is not None,
            "qvix_ok": qvix is not None,
            "breadth_ok": breadth is not None,
            "cross_flow_ok": bool(attention),
            "cross_research_ok": bool(resonance),
        },
    }
    return analysis.decorate(payload)


def save_dashboard(payload: Dict) -> None:
    GLOBAL_RADAR_ROOT.mkdir(parents=True, exist_ok=True)
    DASHBOARD_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")


def read_dashboard() -> Optional[Dict]:
    if not DASHBOARD_PATH.exists():
        return None
    try:
        return json.loads(DASHBOARD_PATH.read_text("utf-8"))
    except Exception:
        return None


def read_brief() -> Optional[Dict]:
    if not BRIEF_PATH.exists():
        return None
    try:
        brief = json.loads(BRIEF_PATH.read_text("utf-8"))
        return brief if brief.get("date") == date.today().isoformat() else None
    except Exception:
        return None


def _brief_facts(payload: Dict) -> str:
    """把面板数据压成 AI 可读的事实清单。"""
    lines = []
    ra = payload.get("risk_appetite") or {}
    if ra.get("score") is not None:
        lines.append(f"全球风险偏好分 {ra['score']}/100（{ra['label']}）")
    fg = payload.get("fear_greed") or {}
    if fg.get("score") is not None:
        lines.append(f"A股恐惧贪婪分 {fg['score']}/100（{fg['label']}）")
        if fg.get("qvix"):
            lines.append(f"50ETF期权QVIX {fg['qvix']}")
        if fg.get("margin_yi"):
            m5 = fg.get("margin_chg5_pct")
            lines.append(f"两融余额 {fg['margin_yi']:.0f}亿" + (f"（5日 {m5:+.1f}%）" if m5 is not None else ""))
        b = fg.get("breadth") or {}
        if b.get("up") is not None:
            lines.append(f"涨跌家数 {b['up']}:{b['down']}，涨停约 {b['limit_up']} 家，成交 {b['amount_yi']:.0f} 亿")
    for row in (payload.get("asset_heat") or {}).get("rows", []):
        c1, c5, c20 = row.get("chg_1d"), row.get("chg_5d"), row.get("chg_20d")

        def fmt(v):
            return "--" if v is None else f"{v:+.1f}%"

        lines.append(f"{row['name']} 最新{row['last']}（1日 {fmt(c1)}，5日 {fmt(c5)}，20日 {fmt(c20)}）")
    for key in ("clock_cn", "clock_us"):
        clock = payload.get(key) or {}
        if clock.get("ok"):
            lines.append(f"{clock['region']}宏观时钟：{clock['quadrant_name']}，配置建议 {clock['asset']}")
    for chain in payload.get("transmission") or []:
        if chain.get("state_dir"):
            lines.append(f"传导链·{chain['name']}：{chain['state']}——{chain['impact']}")
    dirs = (payload.get("a_share_directions") or {}).get("directions") or []
    if dirs:
        top = "、".join(f"{d['direction']}({d['level']}确认)" for d in dirs[:4])
        lines.append(f"规则推导的A股受益方向：{top}")
    meta = payload.get("meta") or {}
    if meta.get("sources_failed"):
        lines.append(f"（数据缺失项：{'、'.join(meta['sources_failed'])}，解读时忽略）")
    return "\n".join(lines)


def generate_ai_brief(payload: Optional[Dict] = None) -> Dict:
    """AI 全球晨报：全球发生了什么 → 对 A 股意味着什么 → 今日关注方向。"""
    payload = payload or read_dashboard() or build_dashboard()
    from .ai_compat import ai_available, ask_json

    if not ai_available():
        return {"status": "skipped", "message": "未配置 AI Key（DEEPSEEK_API_KEY）"}
    facts = _brief_facts(payload)
    try:
        from ..models import DailyBrief as ResearchDailyBrief

        rb = ResearchDailyBrief.objects.order_by("-brief_date").first()
        if rb and rb.overview:
            facts += f"\n\n（今日券商研报晨报观点供参考：{rb.overview[:200]}）"
    except Exception:
        pass

    prompt = f"""你是一名全球宏观策略分析师，每天为不懂金融的 A 股投资者写晨报（受众是普通人，要讲大白话）。以下是今日全球市场数据与信号事实：

{facts}

请输出 JSON：
{{
  "one_liner": "25字以内超大白话总结，用天气或生活比喻（如'全球天气晴，A股偏暖，资源股最吃香'）",
  "weather": "从 ☀️ ⛅ ☁️ 🌧️ ⛈️ 🌪️ 中选一个代表今天全球市场的天气",
  "headline": "50字以内，今天全球市场一句话定性",
  "points": ["3-5条要点，每条一句话，像跟朋友聊天一样讲数据背后的含义"],
  "impact": "80字以内，这些变化对A股老百姓钱包的含义（利好/利空哪些方向）",
  "style": "40字以内，A股风格研判：偏成长还是价值、大盘还是小盘、进攻还是防御，一句理由",
  "risk_radar": "60字以内，最值得盯的1-2个风险点，用大白话说清楚会怎样",
  "focus": "60字以内，今日建议关注的方向与节奏",
  "directions": [
    {{"name": "方向名（如：AI算力/有色/银行）", "logic": "25字以内，大白话逻辑（'钱正在流入+券商都看好'这种）"}}
  ],
  "sentiment": "bullish|neutral|bearish 对A股短期（1-2周）的倾向"
}}

要求：全程说人话，可以把行情比喻成天气/温度/心情；directions 给3-4个A股方向，优先有共振确认的；数据缺失项不要提及；输出合法 JSON。"""

    result = ask_json(prompt)
    if not result:
        return {"status": "error", "message": "AI 生成失败"}
    brief = {
        "date": date.today().isoformat(),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "one_liner": result.get("one_liner", ""),
        "weather": result.get("weather", "⛅"),
        "headline": result.get("headline", ""),
        "points": result.get("points") or [],
        "impact": result.get("impact", ""),
        "style": result.get("style", ""),
        "risk_radar": result.get("risk_radar", ""),
        "focus": result.get("focus", ""),
        "directions": [d for d in (result.get("directions") or []) if isinstance(d, dict) and d.get("name")][:5],
        "sentiment": result.get("sentiment", "neutral"),
    }
    GLOBAL_RADAR_ROOT.mkdir(parents=True, exist_ok=True)
    BRIEF_PATH.write_text(json.dumps(brief, ensure_ascii=False, indent=2), "utf-8")
    return {"status": "success", "brief": brief}
