# -*- coding: utf-8 -*-
"""选股成绩单：把每次 AI 选股落库为 SelectionRun，回填 T+1/T+5/T+20 收益并聚合。

数据源优先级：本地 real_data 池 parquet（快、稳）→ 腾讯日K（全市场兜底，含北交所/科创板）。
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAST_SELECTION_PATH = PROJECT_ROOT / "data_cache" / "last_selection.json"

_HORIZONS = (1, 5, 20)


# ────────────────────────── 记录 ──────────────────────────

def record_selection_run(result: dict, pool: str = "") -> Optional[int]:
    """把一次成功的选股结果落库（在 _run_analysis_task 持久化处调用）。"""
    try:
        from .models import SelectionRun

        data_date = str(result.get("data_date") or "").replace("-", "")[:8]
        if not data_date:
            return None
        picks = []
        for bucket, key in (("trend", "trend_stocks"), ("bottom", "bottom_stocks")):
            for s in result.get(key) or []:
                picks.append({
                    "code": str(s.get("ts_code") or s.get("code") or ""),
                    "name": str(s.get("name") or ""),
                    "bucket": bucket,
                    "buy_score": float(s.get("buy_score") or 0),
                    "industry": str(s.get("industry") or ""),
                })
        # 同一数据日+池只保留一条，重跑覆盖
        run, created = SelectionRun.objects.update_or_create(
            data_date=data_date, pool=pool or "csi1000",
            defaults={
                "regime": str((result.get("market_regime") or {}).get("regime") or "neutral"),
                "total_count": len(picks),
                "trend_count": sum(1 for p in picks if p["bucket"] == "trend"),
                "bottom_count": sum(1 for p in picks if p["bucket"] == "bottom"),
                "picks": picks,
                "stats": result.get("stats") or {},
            },
        )
        return run.id
    except Exception as exc:
        logger.warning(f"选股成绩单落库失败: {exc}")
        return None


def backfill_from_last_selection() -> int:
    """冷启动：把持久化的最近一次选股导入成绩单（幂等）。

    同一 data_date 已存在任意池的 run 时跳过——防止与正式钩子记录的
    run 形成"同日双胞胎"双重计入统计。
    """
    try:
        if not LAST_SELECTION_PATH.exists():
            return 0
        payload = json.loads(LAST_SELECTION_PATH.read_text("utf-8"))
        result = payload.get("result") or {}
        data_date = str(result.get("data_date") or "").replace("-", "")[:8]
        if not data_date:
            return 0
        from .models import SelectionRun

        if SelectionRun.objects.filter(data_date=data_date).exists():
            return 0
        rid = record_selection_run(result, pool="backfill")
        return 1 if rid else 0
    except Exception as exc:
        logger.warning(f"成绩单回填失败: {exc}")
        return 0


# ────────────────────────── 收益回填 ──────────────────────────

_POOL_CLOSES: Optional[Dict[str, Dict[str, float]]] = None


def _pool_closes_all() -> Dict[str, Dict[str, float]]:
    """一次性把 real_data 池的收盘价载入内存 {6位代码: {日期: close}}（避免逐股扫文件）。"""
    global _POOL_CLOSES
    if _POOL_CLOSES is not None:
        return _POOL_CLOSES
    import glob

    import pandas as pd

    out: Dict[str, Dict[str, float]] = {}
    for fp in glob.glob(str(PROJECT_ROOT / "data_cache" / "real_data_*.parquet")):
        try:
            df = pd.read_parquet(fp, columns=["ts_code", "trade_date", "close"])
        except Exception:
            continue
        for tc, d, c in zip(df["ts_code"].astype(str), df["trade_date"].astype(str), df["close"]):
            try:
                out.setdefault(tc.split(".")[0], {})[str(d)] = float(c)
            except (TypeError, ValueError):
                continue
    _POOL_CLOSES = out
    return out


def _close_map(code: str) -> Dict[str, float]:
    """本地池优先，不足 10 根 K 线时腾讯日K兜底（含北交所/科创板）。"""
    base = code.split(".")[0]
    closes = dict(_pool_closes_all().get(base) or {})
    if len(closes) >= 10:
        return closes
    closes = {**closes, **_load_close_from_tencent(code, days=90)}
    return closes


def _load_close_from_tencent(code: str, days: int = 60) -> Dict[str, float]:
    """腾讯日K兜底（全市场含北交所），返回 {YYYYMMDD: close}。"""
    import httpx

    tcode = _to_tencent_code(code)
    if not tcode:
        return {}
    out: Dict[str, float] = {}
    try:
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            r = client.get("https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
                           params={"param": f"{tcode},day,,,{days},qfq"})
            node = (r.json().get("data") or {}).get(tcode) or {}
        rows = node.get("qfqday") or node.get("day") or []
        for row in rows:
            try:
                out[str(row[0]).replace("-", "")] = float(row[2])
            except (TypeError, ValueError, IndexError):
                continue
    except Exception as exc:
        logger.warning(f"腾讯日K读取失败 {code}: {exc}")
    return out


def _to_tencent_code(code: str) -> str:
    base, _, suffix = code.partition(".")
    base = base.strip()
    if not base.isdigit():
        return ""
    if suffix.upper() == "BJ" or base.startswith(("43", "83", "87", "92")):
        return f"bj{base}"
    if suffix.upper() == "SH" or base.startswith("6"):
        return f"sh{base}"
    return f"sz{base}"


def _compute_returns(closes: Dict[str, float], signal_day: str) -> Dict[str, Optional[float]]:
    days = sorted(closes)
    if signal_day not in days:
        before = [d for d in days if d < signal_day]
        if not before:
            return {}
        signal_day = before[-1]  # 信号日非交易日（周末跑）→ 用之前最近交易日
    i = days.index(signal_day)
    base = closes[signal_day]
    out: Dict[str, Optional[float]] = {}
    for n in _HORIZONS:
        j = i + n
        if j < len(days) and base:
            out[f"t{n}"] = round((closes[days[j]] / base - 1) * 100, 2)
        else:
            out[f"t{n}"] = None  # 窗口未走完，留待下次回填
    return out


def refresh_returns(max_runs: int = 30) -> int:
    """为近期 runs 回填收益（增量：已有完整收益的跳过）。返回更新条数。"""
    from django.utils import timezone

    from .models import SelectionRun

    updated = 0
    runs = SelectionRun.objects.order_by("-run_at")[:max_runs]
    for run in runs:
        picks = run.picks or []
        if not picks:
            continue
        ret_map = dict(run.returns or {})
        pending = [p for p in picks
                   if p.get("code") and (p["code"] not in ret_map
                                         or any(ret_map[p["code"]].get(f"t{n}") is None for n in _HORIZONS
                                                if p["code"] in ret_map))]
        if not pending:
            continue
        data_date = run.data_date
        for p in pending:
            code = p["code"]
            closes = _close_map(code)
            if len(closes) < 2:
                continue
            rets = _compute_returns(closes, data_date)
            if rets:
                ret_map[code] = rets
                updated += 1
        run.returns = ret_map
        run.returns_updated_at = timezone.now()
        run.save(update_fields=["returns", "returns_updated_at"])
    return updated


# ────────────────────────── 聚合 ──────────────────────────

def aggregate_scorecard() -> dict:
    """全量聚合：总体 + 分档（趋势/抄底）的样本数/均收益/胜率，附近期 runs。"""
    from .models import SelectionRun

    def _bucket_stats(runs, bucket: Optional[str] = None) -> dict:
        per_h = {f"t{n}": [] for n in _HORIZONS}
        win = {f"t{n}": [0, 0] for n in _HORIZONS}
        for run in runs:
            for p in run.picks or []:
                if bucket and p.get("bucket") != bucket:
                    continue
                rets = (run.returns or {}).get(p.get("code")) or {}
                for n in _HORIZONS:
                    v = rets.get(f"t{n}")
                    if v is not None:
                        per_h[f"t{n}"].append(v)
                        win[f"t{n}"][1] += 1
                        if v > 0:
                            win[f"t{n}"][0] += 1
        out = {}
        for n in _HORIZONS:
            k = f"t{n}"
            vals = per_h[k]
            out[k] = {
                "样本": len(vals),
                "均收益": round(sum(vals) / len(vals), 2) if vals else None,
                "胜率": round(win[k][0] / win[k][1] * 100, 1) if win[k][1] else None,
            }
        return out

    runs = list(SelectionRun.objects.order_by("-run_at")[:60])
    recent = []
    for run in runs[:20]:
        done = sum(1 for p in (run.picks or [])
                   if ((run.returns or {}).get(p.get("code")) or {}).get("t20") is not None)
        recent.append({
            "id": run.id,
            "data_date": run.data_date,
            "run_at": run.run_at.isoformat(timespec="minutes") if run.run_at else "",
            "pool": run.pool,
            "regime": run.regime,
            "total": run.total_count,
            "trend": run.trend_count,
            "bottom": run.bottom_count,
            "回填进度": f"{done}/{run.total_count}",
        })
    return {
        "生成时间": datetime.now().isoformat(timespec="minutes"),
        "覆盖runs": len(runs),
        "总体": _bucket_stats(runs),
        "分档": {
            "趋势档": _bucket_stats(runs, bucket="trend"),
            "抄底档": _bucket_stats(runs, bucket="bottom"),
        },
        "近期runs": recent,
    }
