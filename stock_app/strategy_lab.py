from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


BEIJING_CHAOJIA_STRATEGY_ID = "beijing_chaoshao_first_board"
YANGJIA_XINFA_STRATEGY_ID = "yangjia_emotion_leader"
ZHAO_LAOGE_STRATEGY_ID = "zhao_erge_second_board_leader"
QIAO_BANGZHU_STRATEGY_ID = "qiao_bangzhu_strong_low_suction"
FANG_XINXIA_STRATEGY_ID = "fang_xinxia_big_money_trend"


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str = BEIJING_CHAOJIA_STRATEGY_ID
    strategy_name: str = "北京炒家首板策略"
    mv_min_yi: float = 20.0
    mv_max_yi: float = 100.0
    price_min: float = 3.0
    price_max: float = 25.0
    top_n: int = 30
    recent_limit_days: int = 10
    recent_limit_max_count: int = 3
    strong_limit_10d_max: int = 3
    strong_limit_30d_max: int = 5
    exclude_bj: bool = True
    per_stock_position_pct: float = 10.0
    max_total_position_pct: float = 60.0
    stop_loss_pct: float = -5.0
    holding_days_min: int = 1
    holding_days_max: int = 3


STRATEGIES = {
    BEIJING_CHAOJIA_STRATEGY_ID: StrategyConfig(),
    YANGJIA_XINFA_STRATEGY_ID: StrategyConfig(
        strategy_id=YANGJIA_XINFA_STRATEGY_ID,
        strategy_name="炒股养家心法·情绪龙头策略",
        mv_min_yi=20.0,
        mv_max_yi=300.0,
        price_min=3.0,
        price_max=80.0,
        top_n=30,
        recent_limit_days=20,
        recent_limit_max_count=6,
        strong_limit_10d_max=5,
        strong_limit_30d_max=10,
        per_stock_position_pct=12.0,
        max_total_position_pct=70.0,
        stop_loss_pct=-6.0,
        holding_days_min=1,
        holding_days_max=5,
    ),
    ZHAO_LAOGE_STRATEGY_ID: StrategyConfig(
        strategy_id=ZHAO_LAOGE_STRATEGY_ID,
        strategy_name="赵老哥·新题材二板龙头策略",
        mv_min_yi=20.0,
        mv_max_yi=250.0,
        price_min=3.0,
        price_max=60.0,
        top_n=30,
        recent_limit_days=10,
        recent_limit_max_count=4,
        strong_limit_10d_max=5,
        strong_limit_30d_max=9,
        per_stock_position_pct=10.0,
        max_total_position_pct=70.0,
        stop_loss_pct=-6.0,
        holding_days_min=1,
        holding_days_max=4,
    ),
    QIAO_BANGZHU_STRATEGY_ID: StrategyConfig(
        strategy_id=QIAO_BANGZHU_STRATEGY_ID,
        strategy_name="乔帮主·强势股低吸策略",
        mv_min_yi=20.0,
        mv_max_yi=500.0,
        price_min=3.0,
        price_max=100.0,
        top_n=30,
        recent_limit_days=20,
        recent_limit_max_count=5,
        strong_limit_10d_max=5,
        strong_limit_30d_max=10,
        per_stock_position_pct=8.0,
        max_total_position_pct=50.0,
        stop_loss_pct=-5.0,
        holding_days_min=2,
        holding_days_max=8,
    ),
    FANG_XINXIA_STRATEGY_ID: StrategyConfig(
        strategy_id=FANG_XINXIA_STRATEGY_ID,
        strategy_name="方新侠·大成交趋势龙头策略",
        mv_min_yi=80.0,
        mv_max_yi=2000.0,
        price_min=5.0,
        price_max=180.0,
        top_n=30,
        recent_limit_days=20,
        recent_limit_max_count=6,
        strong_limit_10d_max=6,
        strong_limit_30d_max=12,
        per_stock_position_pct=10.0,
        max_total_position_pct=60.0,
        stop_loss_pct=-7.0,
        holding_days_min=3,
        holding_days_max=15,
    ),
}


def _num(s: pd.Series, default=0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _clip_score(value, low=0.0, high=100.0) -> float:
    return float(np.clip(value, low, high))


def _score_in_range(value, low, high, soft_low=None, soft_high=None) -> float:
    """100 in [low, high], linearly decays to 0 outside a soft range."""
    value = float(value or 0)
    soft_low = low * 0.5 if soft_low is None else soft_low
    soft_high = high * 1.5 if soft_high is None else soft_high
    if low <= value <= high:
        return 100.0
    if value < low:
        if value <= soft_low:
            return 0.0
        return _clip_score((value - soft_low) / (low - soft_low) * 100)
    if value >= soft_high:
        return 0.0
    return _clip_score((soft_high - value) / (soft_high - high) * 100)


def _prepare_history(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if data.empty:
        return data
    if "trade_date" not in data.columns:
        raise ValueError("策略选股需要 trade_date 字段")
    if "ts_code" not in data.columns:
        if "code" in data.columns:
            data["ts_code"] = data["code"]
        else:
            raise ValueError("策略选股需要 ts_code/code 字段")

    data["trade_date"] = data["trade_date"].astype(str)
    data = data.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    numeric_cols = [
        "open", "high", "low", "close", "pre_close", "vol", "amount",
        "turnover_rate", "total_mv", "circ_mv", "float_mv", "pe", "pb",
        "fd_amount", "limit_amount", "open_times",
    ]
    for col in numeric_cols:
        if col in data.columns:
            data[col] = _num(data[col])

    if "pct_chg" not in data.columns or data["pct_chg"].isna().all():
        pre_close = data.groupby("ts_code")["close"].shift(1)
        data["pct_chg"] = (data["close"] / pre_close - 1) * 100
    data["pct_chg"] = _num(data["pct_chg"])

    g = data.groupby("ts_code", group_keys=False)
    data["ma5"] = g["close"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    data["ma10"] = g["close"].transform(lambda s: s.rolling(10, min_periods=5).mean())
    data["ma20"] = g["close"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    data["high20"] = g["high"].transform(lambda s: s.rolling(20, min_periods=10).max()) if "high" in data.columns else data["close"]
    data["avg_amount20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean()) if "amount" in data.columns else 0
    data["prev_pct_chg"] = g["pct_chg"].shift(1).fillna(0)
    data["is_limit_up"] = data["pct_chg"] >= 9.5
    data["prev_limit_up"] = g["is_limit_up"].shift(1).fillna(False).astype(bool)
    data["limit_up_5d"] = g["is_limit_up"].transform(lambda s: s.rolling(5, min_periods=1).sum())
    data["limit_up_10d"] = g["is_limit_up"].transform(lambda s: s.rolling(10, min_periods=1).sum())
    data["limit_up_20d"] = g["is_limit_up"].transform(lambda s: s.rolling(20, min_periods=1).sum())
    data["limit_up_30d"] = g["is_limit_up"].transform(lambda s: s.rolling(30, min_periods=1).sum())
    data["return_5d"] = g["close"].pct_change(5).fillna(0) * 100
    data["return_20d"] = g["close"].pct_change(20).fillna(0) * 100
    data["volume_ratio"] = np.where(data["avg_amount20"] > 0, data["amount"] / data["avg_amount20"], 1.0)

    mv_candidates = []
    for candidate in ("circ_mv", "float_mv", "total_mv"):
        if candidate in data.columns:
            value = pd.to_numeric(data[candidate], errors="coerce")
            mv_candidates.append(value.where(value > 0))
    if mv_candidates:
        mv_source = pd.concat(mv_candidates, axis=1).bfill(axis=1).iloc[:, 0].fillna(0)
        data["market_value_yi"] = mv_source / 10000
    else:
        data["market_value_yi"] = 0.0

    zeros = pd.Series(0, index=data.index)
    data["one_word_limit"] = (
        data.get("open", zeros).astype(float).round(3).eq(data["close"].astype(float).round(3)) &
        data.get("high", zeros).astype(float).round(3).eq(data["close"].astype(float).round(3)) &
        data.get("low", zeros).astype(float).round(3).eq(data["close"].astype(float).round(3))
    )
    return data


def _latest_cross_section(data: pd.DataFrame, trade_date: str | None = None) -> pd.DataFrame:
    if data.empty:
        return data.copy()
    target_date = str(trade_date or data["trade_date"].max())
    latest = data.loc[data["trade_date"].astype(str).eq(target_date)].copy()
    if latest.empty:
        return latest
    idx = latest.groupby("ts_code")["trade_date"].idxmax()
    return latest.loc[idx].copy().reset_index(drop=True)


def _stock_name(row: pd.Series) -> str:
    return str(row.get("name") or row.get("stock_name") or row.get("ts_code") or "")


def _clean_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


def _to_bool(value, default=True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "否"}
    return bool(value)


def run_beijing_chaoshao_strategy(
    df: pd.DataFrame,
    *,
    mv_min_yi: float = 20.0,
    mv_max_yi: float = 100.0,
    price_min: float = 3.0,
    price_max: float = 25.0,
    top_n: int = 30,
    recent_limit_days: int = 10,
    recent_limit_max_count: int = 3,
    strong_limit_10d_max: int = 3,
    strong_limit_30d_max: int = 5,
    exclude_bj: bool = True,
) -> Dict:
    """Run an explainable first-limit-up strategy inspired by 北京炒家 style.

    初筛条件只负责入池：首板、低价、小市值、非一字连续板、非妖股。
    评分只使用已经存在或能真实补全的字段：封板质量、技术形态、股性活跃。
    """
    cfg = StrategyConfig(
        mv_min_yi=float(mv_min_yi),
        mv_max_yi=float(mv_max_yi),
        price_min=float(price_min),
        price_max=float(price_max),
        top_n=int(top_n),
        recent_limit_days=int(recent_limit_days),
        recent_limit_max_count=int(recent_limit_max_count),
        strong_limit_10d_max=int(strong_limit_10d_max),
        strong_limit_30d_max=int(strong_limit_30d_max),
        exclude_bj=bool(exclude_bj),
    )
    warnings: List[str] = []
    if df is None or df.empty:
        return {
            "strategy": _strategy_meta(cfg),
            "picks": [],
            "trade_date": None,
            "stats": {"input_rows": 0},
            "warnings": ["没有可用行情数据"],
        }

    data = _prepare_history(df)
    trade_date = str(data["trade_date"].max())
    latest = _latest_cross_section(data, trade_date=trade_date)
    input_count = int(latest["ts_code"].nunique())

    if "name" in latest.columns:
        st_mask = latest["name"].astype(str).str.contains("ST|退", regex=True, na=False)
        latest = latest.loc[~st_mask].copy()
    if cfg.exclude_bj:
        latest = latest.loc[~latest["ts_code"].astype(str).str.endswith(".BJ")].copy()

    recent_col = f"limit_up_{cfg.recent_limit_days}d"
    if recent_col not in latest.columns:
        warnings.append(f"暂不支持 {cfg.recent_limit_days} 日涨停次数，已回退为 10 日")
        recent_col = "limit_up_10d"

    latest = latest.loc[
        (latest["is_limit_up"]) &
        (~latest["prev_limit_up"]) &
        (latest[recent_col] <= cfg.recent_limit_max_count) &
        (latest["limit_up_10d"] <= cfg.strong_limit_10d_max) &
        (latest["limit_up_30d"] <= cfg.strong_limit_30d_max) &
        (~latest["one_word_limit"].fillna(False)) &
        (latest["close"] >= cfg.price_min) &
        (latest["close"] <= cfg.price_max) &
        (latest["market_value_yi"] >= cfg.mv_min_yi) &
        (latest["market_value_yi"] <= cfg.mv_max_yi)
    ].copy()

    if latest.empty:
        return {
            "strategy": _strategy_meta(cfg),
            "picks": [],
            "trade_date": trade_date,
            "stats": {
                "input_stocks": input_count,
                "candidate_count": 0,
                "selected_count": 0,
                "mv_min_yi": cfg.mv_min_yi,
                "mv_max_yi": cfg.mv_max_yi,
                "price_min": cfg.price_min,
                "price_max": cfg.price_max,
                "recent_limit_days": cfg.recent_limit_days,
                "recent_limit_max_count": cfg.recent_limit_max_count,
                "exclude_bj": cfg.exclude_bj,
            },
            "warnings": warnings + ["当日未发现满足“首板”条件的候选股"],
        }

    if "limit_times" not in latest.columns and "fd_amount" not in latest.columns and "limit_amount" not in latest.columns:
        warnings.append("缺少封单金额/炸板次数字段，封板坚决维度不参与评分；请检查 limit_list_d 数据补全")

    rows = []
    for _, row in latest.iterrows():
        price = float(row.get("close") or row.get("current_price") or 0)
        mv_yi = float(row.get("market_value_yi") or 0)
        turnover = float(row.get("turnover_rate") or row.get("turnover_ratio") or 0)
        vol_ratio = float(row.get("volume_ratio") or 1)
        amount_yi = float(row.get("amount") or 0) / 100000
        fd_amount_yi = float(row.get("fd_amount") or row.get("limit_amount") or 0) / 100000000
        open_times = float(row.get("open_times") or 0)
        if pd.isna(fd_amount_yi):
            fd_amount_yi = 0.0
        if pd.isna(open_times):
            open_times = 0.0
        first_time = _clean_text(row.get("first_time"))
        last_time = _clean_text(row.get("last_time"))
        limit10 = float(row.get("limit_up_10d") or 0)
        close = float(row.get("close") or 0)
        ma5 = float(row.get("ma5") or 0)
        ma10 = float(row.get("ma10") or 0)
        ma20 = float(row.get("ma20") or 0)
        high20 = float(row.get("high20") or close)

        score_parts = []
        dimensions = {}
        if fd_amount_yi > 0 or open_times > 0 or first_time or last_time:
            seal_score = _clip_score(50 + min(fd_amount_yi, 20.0) * 2.0 + max(0, 8 - open_times) * 3.0)
            score_parts.append(("封板坚决", seal_score, 0.35))
            dimensions["封板坚决"] = round(seal_score, 1)

        tech_score = 0.0
        if ma5 > 0 and close >= ma5:
            tech_score += 25
        if ma10 > 0 and ma5 >= ma10:
            tech_score += 20
        if ma20 > 0 and ma10 >= ma20:
            tech_score += 15
        if high20 > 0 and close >= high20 * 0.98:
            tech_score += 25
        if float(row.get("return_20d") or 0) < 80:
            tech_score += 15
        tech_score = _clip_score(tech_score)
        score_parts.append(("技术形态", tech_score, 0.35))
        dimensions["技术形态"] = round(tech_score, 1)

        if turnover > 0 or amount_yi > 0:
            activity_score = _clip_score(
                _score_in_range(turnover, 3.0, 18.0, soft_low=0.5, soft_high=35.0) * 0.65
                + min(vol_ratio, 3.0) / 3.0 * 35
            )
            score_parts.append(("股性活跃", activity_score, 0.30))
            dimensions["股性活跃"] = round(activity_score, 1)

        if not score_parts:
            total_score = 0.0
            warnings.append("候选股缺少可评分字段，本次评分为0")
        else:
            weight_sum = sum(w for _, _, w in score_parts)
            total_score = sum(score * weight for _, score, weight in score_parts) / weight_sum

        confidence = _clip_score(55 + total_score * 0.40)
        position_pct = min(cfg.per_stock_position_pct, cfg.max_total_position_pct)
        signals = [
            "首板涨停" if row.get("is_limit_up") else "",
            f"市值{mv_yi:.0f}亿" if mv_yi > 0 else "",
            f"换手{turnover:.1f}%" if turnover > 0 else "",
            "均线多头/突破" if tech_score >= 60 else "技术形态待确认",
            "近期有涨停但非连续妖股" if 1 <= limit10 <= 3 else "",
        ]
        signals = [s for s in signals if s]
        code = str(row.get("ts_code") or row.get("code") or "").split(".")[0]
        rows.append({
            "code": code,
            "ts_code": str(row.get("ts_code") or ""),
            "name": _stock_name(row),
            "industry": str(row.get("industry") or ""),
            "current_price": round(price, 3),
            "pct_chg": round(float(row.get("pct_chg") or 0), 3),
            "buy_score": round(total_score, 3),
            "strategy_score": round(total_score, 3),
            "confidence": round(confidence, 1),
            "ml_score": round(total_score, 3),
            "dual_score": round(total_score, 3),
            "trend_score": round(tech_score, 3),
            "bottom_score": 0.0,
            "rule_score": round(total_score, 3),
            "position_pct": round(position_pct, 2),
            "position_advice": f"单票{position_pct:.0f}% / 总仓≤{cfg.max_total_position_pct:.0f}%",
            "position_level": "aggressive",
            "market_value": round(mv_yi, 3),
            "turnover_rate": round(turnover, 3),
            "fd_amount_yi": round(fd_amount_yi, 3),
            "open_times": int(open_times),
            "first_limit_time": first_time,
            "last_limit_time": last_time,
            "pe": round(float(row.get("pe") or 0), 3),
            "pb": round(float(row.get("pb") or 0), 3),
            "rsi": round(float(row.get("rsi") or 0), 3),
            "pmt_return_5d": round(float(row.get("return_5d") or 0), 3),
            "pmt_return_20d": round(float(row.get("return_20d") or 0), 3),
            "buy_signals": signals,
            "strategy_id": cfg.strategy_id,
            "strategy_name": cfg.strategy_name,
            "strategy_dimensions": dimensions,
            "holding_days_plan": f"{cfg.holding_days_min}-{cfg.holding_days_max}天",
            "entry_rule": "次日开盘或分时低吸；不追连续加速板",
            "exit_rule": {
                "涨停继续持有": True,
                "跌破分时均线卖出": True,
                "跌破5日线卖出": True,
                "单票硬止损": f"{cfg.stop_loss_pct:.0f}%",
                "3日不强则退出": True,
            },
            "max_total_position_pct": cfg.max_total_position_pct,
            "stop_loss_pct": cfg.stop_loss_pct,
            "source": cfg.strategy_id,
            "type": "strategy",
        })

    picks = sorted(rows, key=lambda x: x["strategy_score"], reverse=True)[: cfg.top_n]
    return {
        "strategy": _strategy_meta(cfg),
        "picks": picks,
        "trade_date": trade_date,
        "stats": {
            "input_stocks": input_count,
            "candidate_count": len(rows),
            "selected_count": len(picks),
            "mv_min_yi": cfg.mv_min_yi,
            "mv_max_yi": cfg.mv_max_yi,
            "price_min": cfg.price_min,
            "price_max": cfg.price_max,
            "recent_limit_days": cfg.recent_limit_days,
            "recent_limit_max_count": cfg.recent_limit_max_count,
            "exclude_bj": cfg.exclude_bj,
            "filter_note": "首板/低价/流通市值/非妖股约束仅用于初筛，不参与评分",
            "max_total_position_pct": cfg.max_total_position_pct,
            "per_stock_position_pct": cfg.per_stock_position_pct,
        },
        "warnings": warnings,
    }


def _market_emotion_snapshot(latest: pd.DataFrame) -> Dict:
    if latest is None or latest.empty:
        return {"score": 0.0, "level": "冰点", "limit_up_count": 0, "limit_down_count": 0, "strong_count": 0, "weak_count": 0, "median_pct": 0.0}
    pct = _num(latest.get("pct_chg", pd.Series(0, index=latest.index)))
    total = max(len(latest), 1)
    limit_up_count = int((pct >= 9.5).sum())
    limit_down_count = int((pct <= -9.5).sum())
    strong_count = int((pct >= 3.0).sum())
    weak_count = int((pct <= -3.0).sum())
    median_pct = float(pct.median()) if len(pct) else 0.0
    score = (
        min(limit_up_count, 120) / 120 * 32
        + min(strong_count / total, 0.25) / 0.25 * 24
        + _clip_score((median_pct + 2.0) / 4.5 * 22)
        + max(0, 22 - min(limit_down_count, 80) / 80 * 22)
        - min(weak_count / total, 0.20) / 0.20 * 10
    )
    score = _clip_score(score)
    level = "主升/高潮" if score >= 72 else "修复/可参与" if score >= 56 else "分歧/试错" if score >= 40 else "退潮/防守"
    return {
        "score": round(score, 2),
        "level": level,
        "limit_up_count": limit_up_count,
        "limit_down_count": limit_down_count,
        "strong_count": strong_count,
        "weak_count": weak_count,
        "median_pct": round(median_pct, 3),
    }


def run_yangjia_xinfa_strategy(
    df: pd.DataFrame,
    *,
    mv_min_yi: float = 20.0,
    mv_max_yi: float = 300.0,
    price_min: float = 3.0,
    price_max: float = 80.0,
    top_n: int = 30,
    recent_limit_days: int = 20,
    recent_limit_max_count: int = 6,
    strong_limit_10d_max: int = 5,
    strong_limit_30d_max: int = 10,
    exclude_bj: bool = True,
) -> Dict:
    """炒股养家公开心法的量化适配版：情绪周期、主流热点、龙头人气、市场合力、风报比、动态仓位。"""
    base_cfg = STRATEGIES[YANGJIA_XINFA_STRATEGY_ID]
    cfg = StrategyConfig(
        strategy_id=YANGJIA_XINFA_STRATEGY_ID,
        strategy_name=base_cfg.strategy_name,
        mv_min_yi=float(mv_min_yi),
        mv_max_yi=float(mv_max_yi),
        price_min=float(price_min),
        price_max=float(price_max),
        top_n=int(top_n),
        recent_limit_days=int(recent_limit_days),
        recent_limit_max_count=int(recent_limit_max_count),
        strong_limit_10d_max=int(strong_limit_10d_max),
        strong_limit_30d_max=int(strong_limit_30d_max),
        exclude_bj=bool(exclude_bj),
        per_stock_position_pct=base_cfg.per_stock_position_pct,
        max_total_position_pct=base_cfg.max_total_position_pct,
        stop_loss_pct=base_cfg.stop_loss_pct,
        holding_days_min=base_cfg.holding_days_min,
        holding_days_max=base_cfg.holding_days_max,
    )
    warnings: List[str] = []
    if df is None or df.empty:
        return {"strategy": _strategy_meta(cfg), "picks": [], "trade_date": None, "stats": {"input_rows": 0}, "warnings": ["没有可用行情数据"]}

    data = _prepare_history(df)
    trade_date = str(data["trade_date"].max())
    latest = _latest_cross_section(data, trade_date=trade_date)
    input_count = int(latest["ts_code"].nunique())
    emotion = _market_emotion_snapshot(latest)

    if "name" in latest.columns:
        latest = latest.loc[~latest["name"].astype(str).str.contains("ST|退", regex=True, na=False)].copy()
    if cfg.exclude_bj:
        latest = latest.loc[~latest["ts_code"].astype(str).str.endswith(".BJ")].copy()
    if "industry" not in latest.columns or latest["industry"].fillna("").astype(str).str.strip().eq("").all():
        warnings.append("缺少行业/题材字段，主流热点用全市场强度代理；建议补 stock_basic/概念热度数据")
        latest["industry"] = "未知主线"
    latest["industry"] = latest["industry"].fillna("未知主线").astype(str).replace({"": "未知主线"})

    latest = latest.loc[
        (latest["close"] >= cfg.price_min) &
        (latest["close"] <= cfg.price_max) &
        (latest["market_value_yi"] >= cfg.mv_min_yi) &
        (latest["market_value_yi"] <= cfg.mv_max_yi) &
        (~latest["one_word_limit"].fillna(False)) &
        (latest["pct_chg"] > 0) &
        ((latest["is_limit_up"]) | (latest["limit_up_5d"] >= 1) | ((latest["close"] >= latest["high20"] * 0.94) & (latest["return_5d"] >= 3))) &
        (latest["limit_up_10d"] <= cfg.strong_limit_10d_max) &
        (latest["limit_up_30d"] <= cfg.strong_limit_30d_max)
    ].copy()

    if latest.empty:
        return {
            "strategy": _strategy_meta(cfg),
            "picks": [],
            "trade_date": trade_date,
            "stats": {"input_stocks": input_count, "candidate_count": 0, "selected_count": 0, "market_emotion": emotion, "filter_note": "心法初筛：非ST、非一字、正收益、近端强势/涨停记忆、市值价格约束、不过度妖化"},
            "warnings": warnings + ["当前没有满足养家心法情绪/龙头初筛的候选"],
        }

    if "fd_amount" not in latest.columns and "limit_amount" not in latest.columns:
        warnings.append("缺少封单金额字段，市场合力维度用成交额/换手/放量代理")
    if "open_times" not in latest.columns:
        warnings.append("缺少炸板次数字段，分歧质量用换手和涨停记忆代理")

    latest["amount_yi"] = _num(latest.get("amount", pd.Series(0, index=latest.index))) / 100000
    latest["fd_amount_yi"] = _num(latest.get("fd_amount", latest.get("limit_amount", pd.Series(0, index=latest.index)))) / 100000000
    latest["open_times_num"] = _num(latest.get("open_times", pd.Series(0, index=latest.index)))
    industry_stats = latest.groupby("industry").agg(
        industry_count=("ts_code", "count"),
        industry_limit_up=("is_limit_up", "sum"),
        industry_avg_pct=("pct_chg", "mean"),
        industry_amount_yi=("amount_yi", "sum"),
    ).reset_index()
    industry_stats["hot_score"] = (
        industry_stats["industry_limit_up"].clip(0, 8) / 8 * 45
        + industry_stats["industry_count"].clip(0, 20) / 20 * 20
        + industry_stats["industry_avg_pct"].clip(0, 10) / 10 * 20
        + industry_stats["industry_amount_yi"].rank(pct=True).fillna(0) * 15
    )
    latest = latest.merge(industry_stats[["industry", "industry_count", "industry_limit_up", "industry_avg_pct", "hot_score"]], on="industry", how="left")

    rows = []
    for _, row in latest.iterrows():
        price = float(row.get("close") or 0)
        mv_yi = float(row.get("market_value_yi") or 0)
        turnover = float(row.get("turnover_rate") or row.get("turnover_ratio") or 0)
        amount_yi = float(row.get("amount_yi") or 0)
        fd_amount_yi = float(row.get("fd_amount_yi") or 0)
        open_times = float(row.get("open_times_num") or 0)
        limit5 = float(row.get("limit_up_5d") or 0)
        ret5 = float(row.get("return_5d") or 0)
        ret20 = float(row.get("return_20d") or 0)
        pct = float(row.get("pct_chg") or 0)
        close = float(row.get("close") or 0)
        ma5 = float(row.get("ma5") or 0)
        ma10 = float(row.get("ma10") or 0)
        ma20 = float(row.get("ma20") or 0)
        high20 = float(row.get("high20") or close)

        emotion_score = float(emotion["score"])
        mainline_score = _clip_score(float(row.get("hot_score") or 0))
        leader_score = _clip_score((100 if row.get("is_limit_up") else min(max(pct, 0), 10) / 10 * 45) + min(limit5, 3) / 3 * 18 + min(max(ret5, 0), 35) / 35 * 17 + (20 if close >= high20 * 0.97 else 0))
        consensus_score = _clip_score(_score_in_range(turnover, 4, 22, soft_low=1, soft_high=45) * 0.40 + min(amount_yi, 80) / 80 * 24 + min(fd_amount_yi, 20) / 20 * 22 + max(0, 14 - min(open_times, 7) * 2))
        rr_score = 0.0
        if ma5 > 0 and close >= ma5:
            rr_score += 18
        if ma10 > 0 and ma5 >= ma10:
            rr_score += 15
        if ma20 > 0 and ma10 >= ma20:
            rr_score += 12
        rr_score += _score_in_range(mv_yi, cfg.mv_min_yi, cfg.mv_max_yi, soft_low=8, soft_high=max(cfg.mv_max_yi * 1.8, 500)) * 0.25
        rr_score += _score_in_range(price, cfg.price_min, cfg.price_max, soft_low=1, soft_high=max(cfg.price_max * 1.5, 120)) * 0.15
        if ret20 < 120:
            rr_score += 15
        rr_score = _clip_score(rr_score)

        if emotion_score >= 72:
            per_pos, max_pos, pos_level = min(15.0, cfg.per_stock_position_pct + 3), min(80.0, cfg.max_total_position_pct + 10), "aggressive"
        elif emotion_score >= 56:
            per_pos, max_pos, pos_level = cfg.per_stock_position_pct, cfg.max_total_position_pct, "normal"
        elif emotion_score >= 40:
            per_pos, max_pos, pos_level = 6.0, 35.0, "cautious"
        else:
            per_pos, max_pos, pos_level = 3.0, 15.0, "light"

        total_score = emotion_score * 0.22 + mainline_score * 0.23 + leader_score * 0.23 + consensus_score * 0.17 + rr_score * 0.15
        confidence = _clip_score(50 + total_score * 0.42)
        dimensions = {"市场情绪周期": round(emotion_score, 1), "主流热点强度": round(mainline_score, 1), "龙头人气地位": round(leader_score, 1), "市场合力承接": round(consensus_score, 1), "风险收益比": round(rr_score, 1)}
        signals = [f"情绪周期：{emotion['level']}", f"主线：{row.get('industry')} 涨停{int(row.get('industry_limit_up') or 0)}只", "涨停/近端强势人气" if row.get("is_limit_up") or limit5 >= 1 else "", f"换手{turnover:.1f}% 成交{amount_yi:.1f}亿" if turnover > 0 or amount_yi > 0 else "", "均线多头/接近新高" if rr_score >= 65 else "位置需控制仓位"]
        code = str(row.get("ts_code") or row.get("code") or "").split(".")[0]
        rows.append({
            "code": code, "ts_code": str(row.get("ts_code") or ""), "name": _stock_name(row), "industry": str(row.get("industry") or ""),
            "current_price": round(price, 3), "pct_chg": round(pct, 3), "buy_score": round(total_score, 3), "strategy_score": round(total_score, 3),
            "confidence": round(confidence, 1), "ml_score": round(total_score, 3), "dual_score": round(total_score, 3),
            "trend_score": round(leader_score, 3), "bottom_score": round(rr_score, 3), "rule_score": round(total_score, 3),
            "position_pct": round(per_pos, 2), "position_advice": f"单票{per_pos:.0f}% / 总仓≤{max_pos:.0f}%", "position_level": pos_level,
            "market_value": round(mv_yi, 3), "turnover_rate": round(turnover, 3), "fd_amount_yi": round(fd_amount_yi, 3), "open_times": int(open_times),
            "first_limit_time": _clean_text(row.get("first_time")), "last_limit_time": _clean_text(row.get("last_time")),
            "pe": round(float(row.get("pe") or 0), 3), "pb": round(float(row.get("pb") or 0), 3), "rsi": round(float(row.get("rsi") or 0), 3),
            "pmt_return_5d": round(ret5, 3), "pmt_return_20d": round(ret20, 3), "buy_signals": [s for s in signals if s],
            "strategy_id": cfg.strategy_id, "strategy_name": cfg.strategy_name, "strategy_dimensions": dimensions,
            "holding_days_plan": f"{cfg.holding_days_min}-{cfg.holding_days_max}天",
            "entry_rule": "跟随主流热点龙头；分歧转一致/回流确认后低吸或打板，不做杂毛",
            "exit_rule": {"情绪退潮降仓": True, "不及预期次日走弱卖出": True, "跌破5日线/核心承接失效": True, "单票硬止损": f"{cfg.stop_loss_pct:.0f}%", "非主流/非龙头不恋战": True},
            "max_total_position_pct": max_pos, "stop_loss_pct": cfg.stop_loss_pct, "market_emotion": emotion, "source": cfg.strategy_id, "type": "strategy",
        })

    picks = sorted(rows, key=lambda x: x["strategy_score"], reverse=True)[: cfg.top_n]
    return {
        "strategy": _strategy_meta(cfg),
        "picks": picks,
        "trade_date": trade_date,
        "stats": {
            "input_stocks": input_count, "candidate_count": len(rows), "selected_count": len(picks), "market_emotion": emotion,
            "mv_min_yi": cfg.mv_min_yi, "mv_max_yi": cfg.mv_max_yi, "price_min": cfg.price_min, "price_max": cfg.price_max,
            "recent_limit_days": cfg.recent_limit_days, "recent_limit_max_count": cfg.recent_limit_max_count, "exclude_bj": cfg.exclude_bj,
            "filter_note": "养家心法初筛：主流热点/情绪强势/龙头人气/非ST非一字/不过度妖化；评分按情绪周期、主流、龙头、合力、风报比",
            "max_total_position_pct": cfg.max_total_position_pct, "per_stock_position_pct": cfg.per_stock_position_pct,
        },
        "warnings": warnings,
    }


def _prepare_named_master_base(df: pd.DataFrame, cfg: StrategyConfig, warnings: List[str]):
    data = _prepare_history(df)
    trade_date = str(data["trade_date"].max())
    latest = _latest_cross_section(data, trade_date=trade_date)
    input_count = int(latest["ts_code"].nunique())
    emotion = _market_emotion_snapshot(latest)
    if "name" in latest.columns:
        latest = latest.loc[~latest["name"].astype(str).str.contains("ST|退", regex=True, na=False)].copy()
    if cfg.exclude_bj:
        latest = latest.loc[~latest["ts_code"].astype(str).str.endswith(".BJ")].copy()
    if "industry" not in latest.columns or latest["industry"].fillna("").astype(str).str.strip().eq("").all():
        warnings.append("缺少行业/题材字段，主线热度用全市场强度代理；建议补 stock_basic/概念热度数据")
        latest["industry"] = "未知主线"
    latest["industry"] = latest["industry"].fillna("未知主线").astype(str).replace({"": "未知主线"})
    latest["amount_yi"] = _num(latest.get("amount", pd.Series(0, index=latest.index))) / 100000
    latest["fd_amount_yi"] = _num(latest.get("fd_amount", latest.get("limit_amount", pd.Series(0, index=latest.index)))) / 100000000
    latest["open_times_num"] = _num(latest.get("open_times", pd.Series(0, index=latest.index)))
    if "fd_amount" not in latest.columns and "limit_amount" not in latest.columns:
        warnings.append("缺少封单金额字段，资金合力用成交额/换手/放量代理")
    if "open_times" not in latest.columns:
        warnings.append("缺少炸板次数字段，分歧承接用换手和涨停记忆代理")
    return data, latest, trade_date, input_count, emotion


def _attach_mainline_hotness(latest: pd.DataFrame) -> pd.DataFrame:
    if latest.empty:
        return latest
    industry_stats = latest.groupby("industry").agg(
        industry_count=("ts_code", "count"),
        industry_limit_up=("is_limit_up", "sum"),
        industry_strong=("pct_chg", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) >= 3).sum())),
        industry_avg_pct=("pct_chg", "mean"),
        industry_amount_yi=("amount_yi", "sum"),
    ).reset_index()
    industry_stats["hot_score"] = (
        industry_stats["industry_limit_up"].clip(0, 8) / 8 * 38
        + industry_stats["industry_strong"].clip(0, 20) / 20 * 22
        + industry_stats["industry_avg_pct"].clip(0, 10) / 10 * 20
        + industry_stats["industry_amount_yi"].rank(pct=True).fillna(0) * 20
    )
    return latest.merge(
        industry_stats[["industry", "industry_count", "industry_limit_up", "industry_strong", "industry_avg_pct", "industry_amount_yi", "hot_score"]],
        on="industry",
        how="left",
    )


def _dynamic_position(cfg: StrategyConfig, emotion_score: float, strong: bool = False):
    if emotion_score >= 72:
        return min(15.0, cfg.per_stock_position_pct + (4 if strong else 2)), min(80.0, cfg.max_total_position_pct + 10), "aggressive"
    if emotion_score >= 56:
        return cfg.per_stock_position_pct, cfg.max_total_position_pct, "normal"
    if emotion_score >= 40:
        return min(6.0, cfg.per_stock_position_pct), min(35.0, cfg.max_total_position_pct), "cautious"
    return 3.0, 15.0, "light"


def _common_pick_payload(row: pd.Series, cfg: StrategyConfig, total_score: float, confidence: float, dimensions: Dict, signals: List[str],
                         trend_score: float, bottom_score: float, position_pct: float, max_pos: float, pos_level: str,
                         entry_rule: str, exit_rule: Dict, emotion: Dict) -> Dict:
    code = str(row.get("ts_code") or row.get("code") or "").split(".")[0]
    price = float(row.get("close") or row.get("current_price") or 0)
    return {
        "code": code,
        "ts_code": str(row.get("ts_code") or ""),
        "name": _stock_name(row),
        "industry": str(row.get("industry") or ""),
        "current_price": round(price, 3),
        "pct_chg": round(float(row.get("pct_chg") or 0), 3),
        "buy_score": round(total_score, 3),
        "strategy_score": round(total_score, 3),
        "confidence": round(confidence, 1),
        "ml_score": round(total_score, 3),
        "dual_score": round(total_score, 3),
        "trend_score": round(trend_score, 3),
        "bottom_score": round(bottom_score, 3),
        "rule_score": round(total_score, 3),
        "position_pct": round(position_pct, 2),
        "position_advice": f"单票{position_pct:.0f}% / 总仓≤{max_pos:.0f}%",
        "position_level": pos_level,
        "market_value": round(float(row.get("market_value_yi") or 0), 3),
        "turnover_rate": round(float(row.get("turnover_rate") or row.get("turnover_ratio") or 0), 3),
        "fd_amount_yi": round(float(row.get("fd_amount_yi") or 0), 3),
        "open_times": int(float(row.get("open_times_num") or 0)),
        "first_limit_time": _clean_text(row.get("first_time")),
        "last_limit_time": _clean_text(row.get("last_time")),
        "pe": round(float(row.get("pe") or 0), 3),
        "pb": round(float(row.get("pb") or 0), 3),
        "rsi": round(float(row.get("rsi") or 0), 3),
        "pmt_return_5d": round(float(row.get("return_5d") or 0), 3),
        "pmt_return_20d": round(float(row.get("return_20d") or 0), 3),
        "buy_signals": [s for s in signals if s],
        "strategy_id": cfg.strategy_id,
        "strategy_name": cfg.strategy_name,
        "strategy_dimensions": dimensions,
        "holding_days_plan": f"{cfg.holding_days_min}-{cfg.holding_days_max}天",
        "entry_rule": entry_rule,
        "exit_rule": exit_rule,
        "max_total_position_pct": max_pos,
        "stop_loss_pct": cfg.stop_loss_pct,
        "market_emotion": emotion,
        "source": cfg.strategy_id,
        "type": "strategy",
    }


def _empty_strategy_result(cfg: StrategyConfig, trade_date, input_count, emotion, warnings, note: str):
    return {
        "strategy": _strategy_meta(cfg),
        "picks": [],
        "trade_date": trade_date,
        "stats": {
            "input_stocks": input_count,
            "candidate_count": 0,
            "selected_count": 0,
            "market_emotion": emotion,
            "filter_note": note,
        },
        "warnings": warnings + ["当前没有满足该名家策略初筛的候选"],
    }


def _stats_payload(cfg: StrategyConfig, input_count: int, rows: List[Dict], picks: List[Dict], emotion: Dict, note: str) -> Dict:
    return {
        "input_stocks": input_count,
        "candidate_count": len(rows),
        "selected_count": len(picks),
        "market_emotion": emotion,
        "mv_min_yi": cfg.mv_min_yi,
        "mv_max_yi": cfg.mv_max_yi,
        "price_min": cfg.price_min,
        "price_max": cfg.price_max,
        "recent_limit_days": cfg.recent_limit_days,
        "recent_limit_max_count": cfg.recent_limit_max_count,
        "exclude_bj": cfg.exclude_bj,
        "filter_note": note,
        "max_total_position_pct": cfg.max_total_position_pct,
        "per_stock_position_pct": cfg.per_stock_position_pct,
    }


def run_zhao_laoge_strategy(
    df: pd.DataFrame,
    *,
    mv_min_yi: float = 20.0,
    mv_max_yi: float = 250.0,
    price_min: float = 3.0,
    price_max: float = 60.0,
    top_n: int = 30,
    recent_limit_days: int = 10,
    recent_limit_max_count: int = 4,
    strong_limit_10d_max: int = 5,
    strong_limit_30d_max: int = 9,
    exclude_bj: bool = True,
) -> Dict:
    cfg0 = STRATEGIES[ZHAO_LAOGE_STRATEGY_ID]
    cfg = StrategyConfig(strategy_id=ZHAO_LAOGE_STRATEGY_ID, strategy_name=cfg0.strategy_name, mv_min_yi=float(mv_min_yi), mv_max_yi=float(mv_max_yi), price_min=float(price_min), price_max=float(price_max), top_n=int(top_n), recent_limit_days=int(recent_limit_days), recent_limit_max_count=int(recent_limit_max_count), strong_limit_10d_max=int(strong_limit_10d_max), strong_limit_30d_max=int(strong_limit_30d_max), exclude_bj=bool(exclude_bj), per_stock_position_pct=cfg0.per_stock_position_pct, max_total_position_pct=cfg0.max_total_position_pct, stop_loss_pct=cfg0.stop_loss_pct, holding_days_min=cfg0.holding_days_min, holding_days_max=cfg0.holding_days_max)
    warnings: List[str] = []
    if df is None or df.empty:
        return {"strategy": _strategy_meta(cfg), "picks": [], "trade_date": None, "stats": {"input_rows": 0}, "warnings": ["没有可用行情数据"]}
    _, latest, trade_date, input_count, emotion = _prepare_named_master_base(df, cfg, warnings)
    latest = _attach_mainline_hotness(latest)
    latest = latest.loc[
        (latest["close"].between(cfg.price_min, cfg.price_max)) &
        (latest["market_value_yi"].between(cfg.mv_min_yi, cfg.mv_max_yi)) &
        (~latest["one_word_limit"].fillna(False)) &
        (latest["is_limit_up"]) &
        (latest["limit_up_5d"].between(2, cfg.recent_limit_max_count)) &
        (latest["limit_up_10d"] <= cfg.strong_limit_10d_max) &
        (latest["limit_up_30d"] <= cfg.strong_limit_30d_max)
    ].copy()
    note = "赵老哥初筛：二板/近端二板确认、主线题材共振、换手承接、非一字、不过度妖化"
    if latest.empty:
        return _empty_strategy_result(cfg, trade_date, input_count, emotion, warnings, note)
    rows = []
    for _, row in latest.iterrows():
        turnover = float(row.get("turnover_rate") or row.get("turnover_ratio") or 0)
        amount_yi = float(row.get("amount_yi") or 0)
        fd_amount_yi = float(row.get("fd_amount_yi") or 0)
        open_times = float(row.get("open_times_num") or 0)
        hot = _clip_score(float(row.get("hot_score") or 0))
        board_score = _clip_score(55 + min(float(row.get("limit_up_5d") or 0), 3) / 3 * 30 + max(0, 8 - open_times) * 2)
        money_score = _clip_score(_score_in_range(turnover, 5, 25, soft_low=1, soft_high=45) * 0.45 + min(amount_yi, 60) / 60 * 35 + min(fd_amount_yi, 15) / 15 * 20)
        risk_score = _clip_score(_score_in_range(float(row.get("market_value_yi") or 0), cfg.mv_min_yi, cfg.mv_max_yi, soft_low=8, soft_high=500) * 0.45 + _score_in_range(float(row.get("close") or 0), cfg.price_min, cfg.price_max, soft_low=1, soft_high=100) * 0.25 + (30 if float(row.get("return_20d") or 0) < 110 else 0))
        total = emotion["score"] * 0.18 + hot * 0.27 + board_score * 0.25 + money_score * 0.18 + risk_score * 0.12
        pos, max_pos, level = _dynamic_position(cfg, float(emotion["score"]), strong=hot >= 70 and board_score >= 80)
        dims = {"情绪周期": round(float(emotion["score"]), 1), "新题材主线": round(hot, 1), "二板龙头确认": round(board_score, 1), "资金承接": round(money_score, 1), "风险收益比": round(risk_score, 1)}
        signals = [f"情绪：{emotion['level']}", f"主线{row.get('industry')} 涨停{int(row.get('industry_limit_up') or 0)}只", "二板/近端二板确认", f"换手{turnover:.1f}% 成交{amount_yi:.1f}亿"]
        rows.append(_common_pick_payload(row, cfg, total, _clip_score(50 + total * 0.43), dims, signals, board_score, risk_score, pos, max_pos, level, "新题材二板确认；主线共振时打板/分歧回封参与，弱市降低仓位", {"不及预期卖出": True, "炸板回封失败离场": True, "跌破5日线退出": True, "单票硬止损": f"{cfg.stop_loss_pct:.0f}%"}, emotion))
    picks = sorted(rows, key=lambda x: x["strategy_score"], reverse=True)[: cfg.top_n]
    return {"strategy": _strategy_meta(cfg), "picks": picks, "trade_date": trade_date, "stats": _stats_payload(cfg, input_count, rows, picks, emotion, note), "warnings": warnings}


def run_qiao_bangzhu_strategy(
    df: pd.DataFrame,
    *,
    mv_min_yi: float = 20.0,
    mv_max_yi: float = 500.0,
    price_min: float = 3.0,
    price_max: float = 100.0,
    top_n: int = 30,
    recent_limit_days: int = 20,
    recent_limit_max_count: int = 5,
    strong_limit_10d_max: int = 5,
    strong_limit_30d_max: int = 10,
    exclude_bj: bool = True,
) -> Dict:
    cfg0 = STRATEGIES[QIAO_BANGZHU_STRATEGY_ID]
    cfg = StrategyConfig(strategy_id=QIAO_BANGZHU_STRATEGY_ID, strategy_name=cfg0.strategy_name, mv_min_yi=float(mv_min_yi), mv_max_yi=float(mv_max_yi), price_min=float(price_min), price_max=float(price_max), top_n=int(top_n), recent_limit_days=int(recent_limit_days), recent_limit_max_count=int(recent_limit_max_count), strong_limit_10d_max=int(strong_limit_10d_max), strong_limit_30d_max=int(strong_limit_30d_max), exclude_bj=bool(exclude_bj), per_stock_position_pct=cfg0.per_stock_position_pct, max_total_position_pct=cfg0.max_total_position_pct, stop_loss_pct=cfg0.stop_loss_pct, holding_days_min=cfg0.holding_days_min, holding_days_max=cfg0.holding_days_max)
    warnings: List[str] = []
    if df is None or df.empty:
        return {"strategy": _strategy_meta(cfg), "picks": [], "trade_date": None, "stats": {"input_rows": 0}, "warnings": ["没有可用行情数据"]}
    _, latest, trade_date, input_count, emotion = _prepare_named_master_base(df, cfg, warnings)
    latest = _attach_mainline_hotness(latest)
    pullback = ((latest["close"] <= latest["ma5"] * 1.035) & (latest["close"] >= latest["ma20"] * 0.97)) | ((latest["pct_chg"].between(-4.5, 2.5)) & (latest["return_20d"] > 8))
    latest = latest.loc[
        (latest["close"].between(cfg.price_min, cfg.price_max)) &
        (latest["market_value_yi"].between(cfg.mv_min_yi, cfg.mv_max_yi)) &
        (latest["return_20d"] >= 8) &
        (latest["limit_up_20d"] >= 1) &
        (latest["limit_up_30d"] <= cfg.strong_limit_30d_max) &
        (pullback) &
        (latest["ma5"] >= latest["ma20"] * 0.98)
    ].copy()
    note = "乔帮主初筛：强势股回踩、主线仍在、缩量/温和换手承接、趋势未坏，不追高"
    if latest.empty:
        return _empty_strategy_result(cfg, trade_date, input_count, emotion, warnings, note)
    rows = []
    for _, row in latest.iterrows():
        turnover = float(row.get("turnover_rate") or row.get("turnover_ratio") or 0)
        amount_yi = float(row.get("amount_yi") or 0)
        hot = _clip_score(float(row.get("hot_score") or 0))
        strength = _clip_score(min(float(row.get("return_20d") or 0), 80) / 80 * 45 + min(float(row.get("return_5d") or 0) + 8, 30) / 30 * 25 + min(float(row.get("limit_up_20d") or 0), 4) / 4 * 30)
        pullback_score = _clip_score(100 - abs(float(row.get("close") or 0) / max(float(row.get("ma10") or row.get("ma5") or 1), 1) - 1) * 800)
        support = _clip_score(_score_in_range(turnover, 2.5, 14, soft_low=0.8, soft_high=28) * 0.55 + min(amount_yi, 40) / 40 * 25 + (20 if float(row.get("close") or 0) >= float(row.get("ma20") or 0) else 0))
        total = emotion["score"] * 0.12 + hot * 0.20 + strength * 0.25 + pullback_score * 0.25 + support * 0.18
        pos, max_pos, level = _dynamic_position(cfg, float(emotion["score"]), strong=False)
        dims = {"情绪环境": round(float(emotion["score"]), 1), "主线热度": round(hot, 1), "前期强度": round(strength, 1), "回踩质量": round(pullback_score, 1), "承接质量": round(support, 1)}
        signals = [f"强势回踩：20日{float(row.get('return_20d') or 0):.1f}%", f"主线{row.get('industry')} 强势{int(row.get('industry_strong') or 0)}只", f"换手{turnover:.1f}% 成交{amount_yi:.1f}亿", "靠近均线低吸区"]
        rows.append(_common_pick_payload(row, cfg, total, _clip_score(50 + total * 0.42), dims, signals, strength, pullback_score, pos, max_pos, level, "强势股回踩5/10/20日线附近低吸，放量反包确认加仓，不追高", {"跌破20日线退出": True, "回踩无承接卖出": True, "主线退潮减仓": True, "单票硬止损": f"{cfg.stop_loss_pct:.0f}%"}, emotion))
    picks = sorted(rows, key=lambda x: x["strategy_score"], reverse=True)[: cfg.top_n]
    return {"strategy": _strategy_meta(cfg), "picks": picks, "trade_date": trade_date, "stats": _stats_payload(cfg, input_count, rows, picks, emotion, note), "warnings": warnings}


def run_fang_xinxia_strategy(
    df: pd.DataFrame,
    *,
    mv_min_yi: float = 80.0,
    mv_max_yi: float = 2000.0,
    price_min: float = 5.0,
    price_max: float = 180.0,
    top_n: int = 30,
    recent_limit_days: int = 20,
    recent_limit_max_count: int = 6,
    strong_limit_10d_max: int = 6,
    strong_limit_30d_max: int = 12,
    exclude_bj: bool = True,
) -> Dict:
    cfg0 = STRATEGIES[FANG_XINXIA_STRATEGY_ID]
    cfg = StrategyConfig(strategy_id=FANG_XINXIA_STRATEGY_ID, strategy_name=cfg0.strategy_name, mv_min_yi=float(mv_min_yi), mv_max_yi=float(mv_max_yi), price_min=float(price_min), price_max=float(price_max), top_n=int(top_n), recent_limit_days=int(recent_limit_days), recent_limit_max_count=int(recent_limit_max_count), strong_limit_10d_max=int(strong_limit_10d_max), strong_limit_30d_max=int(strong_limit_30d_max), exclude_bj=bool(exclude_bj), per_stock_position_pct=cfg0.per_stock_position_pct, max_total_position_pct=cfg0.max_total_position_pct, stop_loss_pct=cfg0.stop_loss_pct, holding_days_min=cfg0.holding_days_min, holding_days_max=cfg0.holding_days_max)
    warnings: List[str] = []
    if df is None or df.empty:
        return {"strategy": _strategy_meta(cfg), "picks": [], "trade_date": None, "stats": {"input_rows": 0}, "warnings": ["没有可用行情数据"]}
    _, latest, trade_date, input_count, emotion = _prepare_named_master_base(df, cfg, warnings)
    latest = _attach_mainline_hotness(latest)
    latest = latest.loc[
        (latest["close"].between(cfg.price_min, cfg.price_max)) &
        (latest["market_value_yi"].between(cfg.mv_min_yi, cfg.mv_max_yi)) &
        (latest["amount_yi"] >= 8) &
        (latest["close"] >= latest["ma20"]) &
        (latest["ma5"] >= latest["ma20"] * 0.98) &
        ((latest["return_20d"] >= 10) | (latest["close"] >= latest["high20"] * 0.95)) &
        (latest["pct_chg"] > -3)
    ].copy()
    note = "方新侠初筛：大成交、大市值/中大盘、趋势龙头、主线板块、资金容量足"
    if latest.empty:
        return _empty_strategy_result(cfg, trade_date, input_count, emotion, warnings, note)
    rows = []
    for _, row in latest.iterrows():
        turnover = float(row.get("turnover_rate") or row.get("turnover_ratio") or 0)
        amount_yi = float(row.get("amount_yi") or 0)
        mv_yi = float(row.get("market_value_yi") or 0)
        hot = _clip_score(float(row.get("hot_score") or 0))
        capacity = _clip_score(min(amount_yi, 150) / 150 * 55 + _score_in_range(mv_yi, cfg.mv_min_yi, cfg.mv_max_yi, soft_low=40, soft_high=3000) * 0.45)
        trend = _clip_score(min(max(float(row.get("return_20d") or 0), 0), 80) / 80 * 35 + (30 if float(row.get("close") or 0) >= float(row.get("high20") or 0) * 0.96 else 0) + (20 if float(row.get("ma5") or 0) >= float(row.get("ma10") or 0) >= float(row.get("ma20") or 0) else 0) + min(max(float(row.get("pct_chg") or 0), 0), 10) / 10 * 15)
        consensus = _clip_score(_score_in_range(turnover, 2, 18, soft_low=0.5, soft_high=35) * 0.45 + min(amount_yi, 120) / 120 * 35 + min(float(row.get("fd_amount_yi") or 0), 20) / 20 * 20)
        total = emotion["score"] * 0.12 + hot * 0.22 + capacity * 0.24 + trend * 0.27 + consensus * 0.15
        pos, max_pos, level = _dynamic_position(cfg, float(emotion["score"]), strong=capacity >= 70 and trend >= 70)
        dims = {"情绪环境": round(float(emotion["score"]), 1), "主线容量": round(hot, 1), "大资金容量": round(capacity, 1), "趋势龙头": round(trend, 1), "资金共振": round(consensus, 1)}
        signals = [f"成交{amount_yi:.1f}亿 市值{mv_yi:.0f}亿", f"主线{row.get('industry')} 成交{float(row.get('industry_amount_yi') or 0):.1f}亿", "趋势多头/接近新高", f"换手{turnover:.1f}%"]
        rows.append(_common_pick_payload(row, cfg, total, _clip_score(50 + total * 0.42), dims, signals, trend, capacity, pos, max_pos, level, "大成交趋势龙头分歧低吸/突破确认，适合容量资金，不做小票杂毛", {"趋势破位退出": True, "板块退潮减仓": True, "放量长阴止损": True, "单票硬止损": f"{cfg.stop_loss_pct:.0f}%"}, emotion))
    picks = sorted(rows, key=lambda x: x["strategy_score"], reverse=True)[: cfg.top_n]
    return {"strategy": _strategy_meta(cfg), "picks": picks, "trade_date": trade_date, "stats": _stats_payload(cfg, input_count, rows, picks, emotion, note), "warnings": warnings}


def _strategy_meta(cfg: StrategyConfig) -> Dict:
    if cfg.strategy_id == ZHAO_LAOGE_STRATEGY_ID:
        return {
            "id": cfg.strategy_id,
            "name": cfg.strategy_name,
            "description": "按赵老哥公开风格量化：新题材、二板确认、主线共振、资金承接，强调大题材龙头而非杂毛。",
            "position_rules": {"holding_period": f"{cfg.holding_days_min}-{cfg.holding_days_max}天", "per_stock_position_pct": cfg.per_stock_position_pct, "max_total_position_pct": cfg.max_total_position_pct},
            "entry_rules": ["新题材二板", "主线共振", "分歧回封", "资金承接"],
            "exit_rules": ["不及预期卖出", "炸板回封失败离场", "跌破5日线退出", "情绪退潮降仓"],
            "risk_notes": ["二板接力波动较高", "题材热度缺失时使用行业热度代理"],
        }
    if cfg.strategy_id == QIAO_BANGZHU_STRATEGY_ID:
        return {
            "id": cfg.strategy_id,
            "name": cfg.strategy_name,
            "description": "按乔帮主公开风格量化：强势股低吸，关注回踩均线、缩量承接、主线未退潮，不追高。",
            "position_rules": {"holding_period": f"{cfg.holding_days_min}-{cfg.holding_days_max}天", "per_stock_position_pct": cfg.per_stock_position_pct, "max_total_position_pct": cfg.max_total_position_pct},
            "entry_rules": ["强势股回踩", "5/10/20日线附近低吸", "缩量承接", "主线仍在"],
            "exit_rules": ["跌破20日线退出", "回踩无承接卖出", "主线退潮减仓", "硬止损"],
            "risk_notes": ["低吸需等待确认", "日线代理无法完全替代分时承接"],
        }
    if cfg.strategy_id == FANG_XINXIA_STRATEGY_ID:
        return {
            "id": cfg.strategy_id,
            "name": cfg.strategy_name,
            "description": "按方新侠公开风格量化：大成交、大容量、趋势龙头，关注板块主线和资金共振。",
            "position_rules": {"holding_period": f"{cfg.holding_days_min}-{cfg.holding_days_max}天", "per_stock_position_pct": cfg.per_stock_position_pct, "max_total_position_pct": cfg.max_total_position_pct},
            "entry_rules": ["大成交趋势龙头", "板块主线", "分歧低吸/突破确认", "容量资金共振"],
            "exit_rules": ["趋势破位退出", "板块退潮减仓", "放量长阴止损", "硬止损"],
            "risk_notes": ["更偏中大盘容量票", "成交额和行业热度为资金共振代理"],
        }
    if cfg.strategy_id == YANGJIA_XINFA_STRATEGY_ID:
        return {
            "id": cfg.strategy_id,
            "name": cfg.strategy_name,
            "description": "按炒股养家公开心法做量化适配：情绪周期优先，围绕主流热点和龙头人气，重视市场合力与风险收益比，仓位随情绪强弱切换。",
            "position_rules": {
                "holding_period": f"{cfg.holding_days_min}-{cfg.holding_days_max}天",
                "per_stock_position_pct": cfg.per_stock_position_pct,
                "max_total_position_pct": cfg.max_total_position_pct,
            },
            "entry_rules": ["主流热点龙头", "分歧转一致", "弱转强/回流确认", "不做杂毛"],
            "exit_rules": ["情绪退潮降仓", "不及预期卖出", "跌破5日线/承接失效", "单票硬止损", "非主流不恋战"],
            "risk_notes": ["公开心法量化适配，不构成投资建议", "题材/席位/盘口缺失时使用行业热度、涨停池、成交换手代理"],
        }
    return {
        "id": cfg.strategy_id,
        "name": cfg.strategy_name,
        "description": "首板、3~25元、20~100亿流通市值为初筛条件；评分仅使用已补全/已存在的封板质量、技术形态、股性活跃字段",
        "position_rules": {
            "holding_period": f"{cfg.holding_days_min}-{cfg.holding_days_max}天",
            "per_stock_position_pct": cfg.per_stock_position_pct,
            "max_total_position_pct": cfg.max_total_position_pct,
        },
        "entry_rules": ["次日开盘", "分时低吸"],
        "exit_rules": ["涨停继续持有", "跌破分时均线卖出", "跌破5日线卖出", "单票-5%硬止损", "3日不强则退出"],
        "risk_notes": ["涨停策略波动较高", "初筛条件不参与评分", "缺失且无法补全的数据不会被硬塞进评分"],
    }


def run_strategy(strategy_id: str, df: pd.DataFrame, params: Dict | None = None) -> Dict:
    params = params or {}
    if strategy_id in {ZHAO_LAOGE_STRATEGY_ID, QIAO_BANGZHU_STRATEGY_ID, FANG_XINXIA_STRATEGY_ID}:
        cfg = STRATEGIES[strategy_id]
        fn = {
            ZHAO_LAOGE_STRATEGY_ID: run_zhao_laoge_strategy,
            QIAO_BANGZHU_STRATEGY_ID: run_qiao_bangzhu_strategy,
            FANG_XINXIA_STRATEGY_ID: run_fang_xinxia_strategy,
        }[strategy_id]
        return fn(
            df,
            mv_min_yi=float(params.get("mv_min_yi", params.get("mv_min", cfg.mv_min_yi))),
            mv_max_yi=float(params.get("mv_max_yi", params.get("mv_max", cfg.mv_max_yi))),
            price_min=float(params.get("price_min", cfg.price_min)),
            price_max=float(params.get("price_max", cfg.price_max)),
            top_n=int(params.get("top_n", cfg.top_n)),
            recent_limit_days=int(params.get("recent_limit_days", cfg.recent_limit_days)),
            recent_limit_max_count=int(params.get("recent_limit_max_count", cfg.recent_limit_max_count)),
            strong_limit_10d_max=int(params.get("strong_limit_10d_max", cfg.strong_limit_10d_max)),
            strong_limit_30d_max=int(params.get("strong_limit_30d_max", cfg.strong_limit_30d_max)),
            exclude_bj=_to_bool(params.get("exclude_bj"), True),
        )
    if strategy_id == YANGJIA_XINFA_STRATEGY_ID:
        cfg = STRATEGIES[YANGJIA_XINFA_STRATEGY_ID]
        return run_yangjia_xinfa_strategy(
            df,
            mv_min_yi=float(params.get("mv_min_yi", params.get("mv_min", cfg.mv_min_yi))),
            mv_max_yi=float(params.get("mv_max_yi", params.get("mv_max", cfg.mv_max_yi))),
            price_min=float(params.get("price_min", cfg.price_min)),
            price_max=float(params.get("price_max", cfg.price_max)),
            top_n=int(params.get("top_n", cfg.top_n)),
            recent_limit_days=int(params.get("recent_limit_days", cfg.recent_limit_days)),
            recent_limit_max_count=int(params.get("recent_limit_max_count", cfg.recent_limit_max_count)),
            strong_limit_10d_max=int(params.get("strong_limit_10d_max", cfg.strong_limit_10d_max)),
            strong_limit_30d_max=int(params.get("strong_limit_30d_max", cfg.strong_limit_30d_max)),
            exclude_bj=_to_bool(params.get("exclude_bj"), True),
        )
    if strategy_id != BEIJING_CHAOJIA_STRATEGY_ID:
        raise ValueError(f"未知策略: {strategy_id}")
    return run_beijing_chaoshao_strategy(
        df,
        mv_min_yi=float(params.get("mv_min_yi", params.get("mv_min", 20))),
        mv_max_yi=float(params.get("mv_max_yi", params.get("mv_max", 100))),
        price_min=float(params.get("price_min", 3)),
        price_max=float(params.get("price_max", 25)),
        top_n=int(params.get("top_n", 30)),
        recent_limit_days=int(params.get("recent_limit_days", 10)),
        recent_limit_max_count=int(params.get("recent_limit_max_count", 3)),
        strong_limit_10d_max=int(params.get("strong_limit_10d_max", 3)),
        strong_limit_30d_max=int(params.get("strong_limit_30d_max", 5)),
        exclude_bj=_to_bool(params.get("exclude_bj"), True),
    )
