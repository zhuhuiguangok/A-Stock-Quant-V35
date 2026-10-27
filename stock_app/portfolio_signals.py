# -*- coding: utf-8 -*-
"""持仓信号联动：把持仓与选股结果/周期建议仓位/板块资金流对齐，给大白话行动建议。

纯读取聚合，不改动持仓数据。数据源全部走已有模块，失败降级为缺省提示。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAST_SELECTION_PATH = PROJECT_ROOT / "data_cache" / "last_selection.json"
FUND_FLOW_PATH = PROJECT_ROOT / "data_cache" / "bull_bear" / "results" / "fund_flow_latest.json"
SECTOR_FORECAST_PATH = PROJECT_ROOT / "data_cache" / "jev" / "sector_forecast_latest.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def _cycle_signal() -> dict:
    try:
        from .bull_bear_service import get_dashboard_payload  # noqa: F401  (确保缓存目录就绪)
    except Exception:
        pass
    p = PROJECT_ROOT / "data_cache" / "bull_bear" / "results" / "cycle_signal_latest.json"
    return _read_json(p)


def build_signals() -> dict:
    from .portfolio_service import get_default_account, _d, normalize_code
    from .models import PortfolioHolding

    account = get_default_account()
    holdings = list(PortfolioHolding.objects.filter(account=account, status="holding"))
    result: Dict = {"holdings": [], "advice": [], "cycle": {}, "generated_at": None}

    # ── 周期建议仓位 ──
    cyc = _cycle_signal()
    score = float(cyc.get("cycle_score") or 50)
    position = float(cyc.get("current_position") or 0.5)
    phase = str(cyc.get("phase") or "中性")
    total_capital = float(_d(account.total_capital))
    result["cycle"] = {
        "date": cyc.get("date"), "温度": score, "阶段": phase, "建议仓位": position,
        "总资金": total_capital,
        "建议持仓市值": round(total_capital * position, 0),
    }

    # ── 选股结果映射 ──
    last = _read_json(LAST_SELECTION_PATH)
    sel = last.get("result") or {}
    pick_map: Dict[str, dict] = {}
    for bucket, key in (("trend", "trend_stocks"), ("bottom", "bottom_stocks")):
        for s in sel.get(key) or []:
            code = str(s.get("ts_code") or "").split(".")[0]
            if code:
                pick_map.setdefault(code, {"bucket": bucket, "buy_score": s.get("buy_score"),
                                           "industry": s.get("industry"), "name": s.get("name"),
                                           "data_date": sel.get("data_date")})
    # 板块预测 p_up
    sf = _read_json(SECTOR_FORECAST_PATH)
    sector_pup = {p.get("sector"): p.get("p_up") for p in (sf.get("gbm_picks") or [])}
    # 板块资金流
    ff = _read_json(FUND_FLOW_PATH)
    ff_rows = ff.get("top_inflow") or ff.get("rows") or []
    industry_flow = {}
    for r in ff_rows if isinstance(ff_rows, list) else []:
        name = r.get("sector_name") or r.get("name")
        if name:
            industry_flow[str(name)] = r.get("main_force_net")

    total_mv = 0.0
    rows: List[dict] = []
    from .portfolio_service import serialize_holding  # 复用现有序列化
    try:
        from .research.market_utils import get_latest_price
    except Exception:
        get_latest_price = None

    for h in holdings:
        code = normalize_code(h.code)
        info = pick_map.get(code)
        cur = {}
        if get_latest_price is not None:
            try:
                px = get_latest_price(code)
                if px:
                    cur = {"close": px}
            except Exception:
                cur = {}
        ser = serialize_holding(h, current_data={code: cur} if cur else {})
        mv = float(ser.get("market_value") or 0)
        total_mv += mv
        ind = h.industry or (info or {}).get("industry") or ""
        display_name = h.name or (info or {}).get("name") or h.code
        rows.append({
            "code": h.code,
            "name": display_name,
            "industry": ind,
            "shares": ser.get("shares"),
            "avg_cost": ser.get("avg_cost"),
            "现价": ser.get("current_price"),
            "市值": mv,
            "盈亏率": ser.get("pnl_pct"),
            "选股信号": ({"档位": info["bucket"], "买点": info.get("buy_score"),
                        "数据日": info.get("data_date")} if info else None),
            "板块预测p_up": sector_pup.get(ind),
            "行业主力净流入": industry_flow.get(ind),
        })

    result["holdings"] = rows
    result["generated_at"] = __import__("datetime").datetime.now().isoformat(timespec="minutes")

    # ── 大白话建议 ──
    advice: List[str] = []
    held_mv_pct = (total_mv / total_capital * 100) if total_capital else 0
    advice.append(f"当前持仓市值 {total_mv:,.0f} 元，占总资金 {held_mv_pct:.0f}%；"
                  f"周期建议仓位 {position:.0%}（≈{total_capital*position:,.0f} 元）→ "
                  + ("仓位偏重，考虑减到建议线附近" if total_mv > total_capital * position * 1.2
                     else "仓位偏轻，可按信号分批补" if total_mv < total_capital * position * 0.8
                     else "仓位与建议基本匹配"))
    in_trend = [r for r in rows if (r.get("选股信号") or {}).get("档位") == "trend"]
    in_bottom = [r for r in rows if (r.get("选股信号") or {}).get("档位") == "bottom"]
    if in_trend:
        advice.append("✅ 持仓在最新趋势档："
                      + "、".join(f"{r['name']}(买点{r['选股信号']['买点']})" for r in in_trend[:3])
                      + " —— 强势延续中，可继续持有")
    if in_bottom:
        advice.append("🟡 持仓在最新抄底档："
                      + "、".join(f"{r['name']}(买点{r['选股信号']['买点']})" for r in in_bottom[:3])
                      + " —— 系统认为超跌回补中，别急着割")
    risky = [r for r in rows if r.get("盈亏率") is not None and float(r["盈亏率"]) <= -8]
    if risky:
        advice.append("⚠️ 深度浮亏（≤-8%）："
                      + "、".join(f"{r['name']}({float(r['盈亏率']):.0f}%)" for r in risky[:3])
                      + " —— 结合止损巡检（持仓巡检页）确认是否破位")
    result["advice"] = advice
    return result
