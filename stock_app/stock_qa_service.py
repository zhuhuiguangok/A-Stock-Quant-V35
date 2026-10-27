# -*- coding: utf-8 -*-
"""个股「问一问」小 Agent：聚合全模块数据为一份个股档案，DeepSeek 用大白话答疑。

Agent 结构 = 自动工具编排（行情/技术面/选股信号/板块/研报情绪/持仓/周期 7 路数据
全部自动拉取）+ 单轮富上下文问答 + 会话历史追问。所有取数失败都降级为"暂无数据"，
不阻断回答。
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JEV_DIR = PROJECT_ROOT / "data_cache" / "jev"
SECTOR_FORECAST_PATH = JEV_DIR / "sector_forecast_latest.json"
FUND_FLOW_PATH = PROJECT_ROOT / "data_cache" / "bull_bear" / "results" / "fund_flow_latest.json"
CYCLE_PATH = PROJECT_ROOT / "data_cache" / "bull_bear" / "results" / "cycle_signal_latest.json"
LAST_SELECTION_PATH = PROJECT_ROOT / "data_cache" / "last_selection.json"

_DEEPSEEK_BASE = "https://api.deepseek.com"
_DEEPSEEK_MODEL = "deepseek-chat"

_SESSIONS: Dict[str, List[dict]] = {}
_SESSION_LOCK = threading.Lock()
_MAX_TURNS = 8


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


# ────────────────────────── 取数工具 ──────────────────────────

def _tencent_quote(code: str) -> dict:
    """腾讯实时行情：名称/现价/涨跌幅（GBK，秒回）。"""
    base, _, suffix = code.partition(".")
    base = base.strip()
    if not base.isdigit():
        return {}
    if suffix.upper() == "SH" or base.startswith("6"):
        tcode = f"sh{base}"
    elif suffix.upper() == "BJ" or base.startswith(("43", "83", "87", "92")):
        tcode = f"bj{base}"
    else:
        tcode = f"sz{base}"
    try:
        with httpx.Client(timeout=8.0, trust_env=False) as client:
            r = client.get(f"http://qt.gtimg.cn/q={tcode}")
        parts = r.content.decode("gbk", errors="ignore").split("~")
        if len(parts) < 32:
            return {}
        return {
            "名称": parts[1],
            "现价": float(parts[3]) if parts[3] else None,
            "涨跌幅": float(parts[32]) if parts[32] else None,
        }
    except Exception:
        return {}


def _industry_of(code: str) -> str:
    """行业：先查本地池（real_data parquet 的 industry 列）。"""
    import glob

    import pandas as pd

    base = code.split(".")[0]
    try:
        for fp in glob.glob(str(PROJECT_ROOT / "data_cache" / "real_data_*.parquet")):
            try:
                df = pd.read_parquet(fp, columns=["ts_code", "industry"])
            except Exception:
                continue
            hit = df[df["ts_code"].astype(str).str.split(".").str[0] == base]
            if not hit.empty:
                v = hit["industry"].iloc[-1]
                if v and str(v) != "nan":
                    return str(v)
    except Exception:
        pass
    return ""


def _technicals(code: str) -> dict:
    """技术面：MA5/20/60 位置、20日涨跌、距60日高低点（复用成绩单的K线加载器）。"""
    from .selection_scorecard import _close_map

    closes_map = _close_map(code)
    days = sorted(closes_map)
    if len(days) < 10:
        return {}
    closes = [closes_map[d] for d in days]

    def ma(n: int) -> Optional[float]:
        return round(sum(closes[-n:]) / n, 2) if len(closes) >= n else None

    px = closes[-1]
    hi60, lo60 = max(closes[-60:]), min(closes[-60:])
    chg20 = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else None
    return {
        "现价": px,
        "MA5": ma(5), "MA20": ma(20), "MA60": ma(60),
        "20日涨跌%": round(chg20, 2) if chg20 is not None else None,
        "距60日高点%": round((px / hi60 - 1) * 100, 2),
        "距60日低点%": round((px / lo60 - 1) * 100, 2),
        "数据日": days[-1],
    }


def _selection_signal(code: str) -> dict:
    last = _read_json(LAST_SELECTION_PATH).get("result") or {}
    base = code.split(".")[0]
    for bucket, key in (("趋势档", "trend_stocks"), ("抄底档", "bottom_stocks")):
        for s in last.get(key) or []:
            if str(s.get("ts_code") or "").split(".")[0] == base:
                return {"在榜": bucket, "买点": s.get("buy_score"), "数据日": last.get("data_date")}
    return {"在榜": None, "数据日": last.get("data_date")}


def _sector_view(industry: str) -> dict:
    from .industry_map import to_sw_industry

    out = {"行业": industry or "未知"}
    sf = _read_json(SECTOR_FORECAST_PATH)
    sw = to_sw_industry(industry)
    for p in sf.get("gbm_picks") or []:
        if p.get("sector") in (sw, industry):
            out["板块预测p_up"] = p.get("p_up")
            break
    ff = _read_json(FUND_FLOW_PATH)
    for list_key in ("top_inflow", "top_outflow", "rows"):
        for r in ff.get(list_key) or []:
            if isinstance(r, dict) and (r.get("sector_name") or r.get("name")) in (industry, sw):
                out["行业主力净流入"] = r.get("main_force_net")
                break
    return out


def _research_view(code: str) -> dict:
    """近60天券商研报覆盖与情绪。"""
    try:
        from .models.research import ResearchReport, ResearchSummary

        base = code.split(".")[0]
        since = date.today() - timedelta(days=60)
        reports = (ResearchReport.objects
                   .filter(stock_code__contains=base, report_date__gte=since)
                   .order_by("-report_date")[:8])
        items, moods = [], {"bullish": 0, "neutral": 0, "bearish": 0}
        for rep in reports:
            mood = "neutral"
            title_line = rep.title[:40]
            try:
                summ = rep.summary
                if summ is not None:
                    mood = summ.sentiment or "neutral"
                    if summ.summary:
                        title_line = f"{rep.title[:26]}…：{summ.summary[:60]}"
            except Exception:
                pass
            moods[mood] = moods.get(mood, 0) + 1
            items.append(f"[{rep.report_date}]{title_line}")
        total = sum(moods.values())
        if not total:
            return {"覆盖": "近60天无券商研报"}
        lean = max(moods, key=moods.get)
        return {
            "近60天研报": total, "看多": moods["bullish"], "中性": moods["neutral"], "看空": moods["bearish"],
            "情绪倾向": {"bullish": "偏多", "neutral": "中性", "bearish": "偏空"}[lean],
            "要点": items[:4],
        }
    except Exception as exc:
        logger.warning(f"研报视图失败 {code}: {exc}")
        return {"覆盖": "暂无"}


def _holding_view(code: str) -> dict:
    try:
        from .models import PortfolioHolding
        from .portfolio_service import get_default_account, normalize_code

        base = code.split(".")[0]
        h = (PortfolioHolding.objects
             .filter(account=get_default_account(), status="holding")
             .filter(code__contains=base).first())
        if not h:
            return {"持仓": "未持有"}
        return {"持仓": "持有中", "股数": h.shares, "成本": float(h.avg_cost)}
    except Exception:
        return {"持仓": "未知"}


def collect_context(code: str) -> dict:
    """聚合 7 路数据为一份个股档案。"""
    from .bull_bear_service import get_dashboard_payload  # noqa: F401 确保目录就绪

    quote = _tencent_quote(code)
    industry = _industry_of(code)
    ctx = {
        "行情": quote or {"名称": code, "现价": None},
        "技术面": _technicals(code),
        "选股信号": _selection_signal(code),
        "板块": _sector_view(industry),
        "研报情绪": _research_view(code),
        "持仓状态": _holding_view(code),
        "大盘周期": (lambda c: {"日期": c.get("date"), "温度": c.get("cycle_score"),
                              "阶段": c.get("phase"), "建议仓位": c.get("current_position")})(
            _read_json(CYCLE_PATH)),
    }
    return ctx


def _fmt_context(ctx: dict) -> str:
    lines = ["【个股档案（系统自动聚合，可能有缺失）】"]
    for k, v in ctx.items():
        if v:
            lines.append(f"◆ {k}: {json.dumps(v, ensure_ascii=False)}")
    return "\n".join(lines)


_SYSTEM = (
    "你是A股平台的个股分析小助手。用户给你一份由系统自动聚合的个股档案，请用大白话回答。规则：\n"
    "1. 先亮结论（能买/谨慎/别碰/看情况），再给 2-4 条依据，每条注明来自档案哪部分（如'技术面''研报情绪'）。\n"
    "2. 必须说主要风险，哪怕结论偏多。\n"
    "3. 档案里没有的数据就明说'档案里没有'，不要编造。\n"
    "4. 不承诺收益，不给精确买卖点位；仓位问题结合大盘周期建议。\n"
    "5. 全程中文，口语化，控制在 300 字以内。"
)


def _deepseek_chat(messages: List[dict]) -> Optional[str]:
    try:
        from .research.ai_service import _load_legacy_env

        key = __import__("os").environ.get("DEEPSEEK_API_KEY", "") or _load_legacy_env("DEEPSEEK_API_KEY")
        if not key:
            return None
        with httpx.Client(timeout=90.0, trust_env=False) as client:
            r = client.post(f"{_DEEPSEEK_BASE}/chat/completions",
                            headers={"Authorization": f"Bearer {key}"},
                            json={"model": _DEEPSEEK_MODEL, "messages": messages,
                                  "temperature": 0.5, "max_tokens": 800})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning(f"问一问 LLM 调用失败: {exc}")
        return None


def ask(code: str, question: str = "", session_id: str = "") -> dict:
    code = str(code or "").strip()
    if not code:
        return {"ok": False, "answer": "请提供股票代码"}
    question = (question or "").strip() or "现在能买吗？主要风险是什么？"
    sid = session_id or uuid.uuid4().hex[:12]

    with _SESSION_LOCK:
        history = list(_SESSIONS.get(sid, []))

    ctx = collect_context(code)
    messages = [{"role": "system", "content": _SYSTEM}]
    if history:
        messages.append({"role": "system",
                         "content": "此前对话（同一只股票的追问，供参考）：\n" + "\n".join(
                             f"问: {t['q']}\n答: {t['a'][:300]}" for t in history[-4:])})
    messages.append({"role": "user",
                     "content": f"股票代码：{code}\n用户问题：{question}\n\n{_fmt_context(ctx)}"})
    answer = _deepseek_chat(messages)
    if not answer:
        answer = ("⚠️ AI 服务暂时不可用（DeepSeek 未配置或网络失败）。"
                  "以下是系统档案摘要：\n" + _fmt_context(ctx)[:500])

    with _SESSION_LOCK:
        sess = _SESSIONS.setdefault(sid, [])
        sess.append({"q": question, "a": answer, "code": code})
        if len(sess) > _MAX_TURNS:
            del sess[:len(sess) - _MAX_TURNS]

    return {"ok": True, "session_id": sid, "question": question,
            "answer": answer, "名称": (ctx.get("行情") or {}).get("名称") or code}
