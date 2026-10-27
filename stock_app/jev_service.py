# -*- coding: utf-8 -*-
"""市场数据缓存与特征层（原 Jev 易术模块的数据部分）。

2026-10-28 起 Jev(TypeSafe) LLM 预测层已下线（区域封锁 451 + 验证无超额）。
本模块保留为全平台共享的数据基础设施：
  - 指数/辅助指数/两融/宽度/隔夜缺口/涨停池/FRED 序列的缓存加载（带过期自愈）
  - 技术信号、v2~v10 市场特征（GBM 与 DeepSeek 次要视角的输入）
  - 干支/五行/八卦确定性推导与交易日历（DeepSeek 上下文继续使用）

消费方：forecast_service（GBM 主信号 + DeepSeek）、sector_forecast_service。
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JEV_DIR = PROJECT_ROOT / "data_cache" / "jev"

# ── 供应商配置：TypeSafe「系统一」(jev-latest) 原生格式 ────────────────
# 端点 POST {base}/v1/systemone，Bearer 认证，非 OpenAI 兼容：
#   body = {model, state: {...}, questions: {key: {type:"choice",
#           question, criteria: {选项: 描述}}}}
#   resp = {"model": "jev-1.13.0", "answers": {key: {choice, confidence,
#           probabilities: {选项: 概率}}}, "usage": {...}}
def _load_legacy_env(key: str) -> str:
    """迁移过渡：从旧项目 .env 读 key。"""
    for cand in (PROJECT_ROOT / ".env", PROJECT_ROOT / "backend" / ".env"):
        try:
            for line in cand.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip().startswith(key):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            continue
    return ""


def _sanitize_state(obj):
    """NaN/Inf → None，保证 state 可被严格 JSON 序列化（服务端拒绝 NaN 字面量）。"""
    if isinstance(obj, dict):
        return {str(k): _sanitize_state(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_state(v) for v in obj]
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        try:
            f = float(obj)
        except (TypeError, ValueError, OverflowError):
            return None
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    return obj


# 连续失败熔断：TypeSafe 区域封锁(451)等场景下，服务端要 20+ 秒才慢速拒绝，
# 每次预测白等多轮。连续失败后 10 分钟内直接跳过 LLM 调用。
# ═══════════════════════════════════════════════════════════════════
# 干支 / 五行 / 八卦（确定性推导，纯公历日期输入）
# ═══════════════════════════════════════════════════════════════════
TIANGAN = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
DIZHI = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
GAN_WUXING = ["木", "木", "火", "火", "土", "土", "金", "金", "水", "水"]
ZHI_WUXING = ["水", "土", "木", "木", "土", "火", "火", "土", "金", "金", "土", "水"]
WUXING_SHENG = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}
BAGUA = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]
BAGUA_ATTR = {"乾": "天/刚健", "兑": "泽/喜悦", "离": "火/明动", "震": "雷/奋动",
              "巽": "风/进入", "坎": "水/险陷", "艮": "山/止息", "坤": "地/柔顺"}

# 锚点：1949-10-01 为甲子日（史载开国大典日干支，广泛引用的公历对照锚）
_ANCHOR_DATE = _date(1949, 10, 1)
_ANCHOR_INDEX = 0  # 甲子 = 干0支0


def day_ganzhi(d: _date) -> Tuple[str, str, int]:
    """日干支：以 1949-10-01=甲子 为锚，按天数差模 60。返回 (干, 支, 序号0-59)。"""
    diff = (d - _ANCHOR_DATE).days
    idx = diff % 60
    return TIANGAN[idx % 10], DIZHI[idx % 12], idx


def year_ganzhi(d: _date) -> Tuple[str, str]:
    """年干支（以立春 2 月 4 日为界，简化处理；节气精确分钟不必要于此场景）。"""
    y = d.year if (d.month, d.day) >= (2, 4) else d.year - 1
    idx = (y - 1984) % 60  # 1984 为甲子年
    return TIANGAN[idx % 10], DIZHI[idx % 12]


def wuxing_summary(d: _date) -> Dict[str, str]:
    g, z, _ = day_ganzhi(d)
    return {"日干": g, "日干五行": GAN_WUXING[TIANGAN.index(g)],
            "日支": z, "日支五行": ZHI_WUXING[DIZHI.index(z)],
            "年柱": "".join(year_ganzhi(d))}


def bagua_for_day(d: _date) -> Dict[str, str]:
    """日卦象：先天八卦数派生（确定性；从日干支序推数起卦，非时间起卦）。"""
    _, _, idx = day_ganzhi(d)
    upper = BAGUA[idx % 8]
    lower = BAGUA[(idx // 8 + idx % 10) % 8]  # 干支双因子派生上下卦
    return {"上卦": upper, "下卦": lower,
            "卦名": f"{BAGUA_ATTR[lower].split('/')[0]}上{BAGUA_ATTR[upper].split('/')[0]}下",
            "意象": f"上{upper}({BAGUA_ATTR[upper]}) 下{lower}({BAGUA_ATTR[lower]})"}


# ── 24 节气（寿星公式，21世纪精度 ±1 天）────────────────────────────
_SOLAR_C = [  # 每月两个节气的 C 值（21世纪）
    (5.4055, 20.12), (3.87, 18.73), (5.63, 20.646), (4.81, 20.1),
    (5.52, 21.04), (5.678, 21.37), (7.108, 22.83), (7.5, 23.13),
    (7.646, 23.042), (8.318, 23.438), (7.438, 22.36), (7.18, 21.94),
]
_SOLAR_NAMES = ["小寒", "大寒", "立春", "雨水", "惊蛰", "春分", "清明", "谷雨",
                "立夏", "小满", "芒种", "夏至", "小暑", "大暑", "立秋", "处暑",
                "白露", "秋分", "寒露", "霜降", "立冬", "小雪", "大雪", "冬至"]


def _solar_term_dates(year: int) -> Dict[str, str]:
    """某年 24 节气交节日（YYYY-MM-DD）。寿星公式：L 用两位年份的闰数。"""
    out = {}
    y = year % 100
    for mi, (c1, c2) in enumerate(_SOLAR_C, 1):
        for ci, c in enumerate((c1, c2)):
            name = _SOLAR_NAMES[mi * 2 - 2 + ci]
            L = (y - 1) // 4 if mi <= 2 else y // 4  # 1-2月节气用 (Y-1)/4
            day = int(y * 0.2422 + c) - L
            out[name] = f"{year:04d}-{mi:02d}-{day:02d}"
    return out


def solar_term_info(d: _date) -> Dict:
    """当日所处节气 + 节气内进度 + 距下一节气。"""
    all_terms = []
    for y in (d.year - 1, d.year, d.year + 1):
        for name, ds in _solar_term_dates(y).items():
            all_terms.append((ds, name, y))
    ordered = sorted(all_terms)  # 按日期排序，避免同名键跨年覆盖
    prev_name = prev_date = next_name = next_date = None
    for ds, name, _y in ordered:
        dd = _date.fromisoformat(ds)
        if dd <= d:
            prev_name, prev_date = name, dd
        elif next_name is None:
            next_name, next_date = name, dd
            break
    into = (d - prev_date).days if prev_date else 0
    gap = (next_date - d).days if next_date else 15
    cycle = 15.2
    return {"节气": prev_name or "—", "节气第N天": into + 1,
            "距下一节气": gap,
            "节气进度0-1": round(min(1.0, into / cycle), 2),
            "相邻节气": f"{prev_name}→{next_name}"}


def yiji_context(d: _date) -> Dict:
    """给定日期的完整易术上下文（供提示词）。"""
    return {
        "公历": d.isoformat(),
        "星期": "一二三四五六日"[d.weekday()],
        **wuxing_summary(d),
        **bagua_for_day(d),
    }


# ═══════════════════════════════════════════════════════════════════
# 市场数据（只允许 ≤ T 的行进入提示词——防穿越的唯一入口）
# ═══════════════════════════════════════════════════════════════════
INDEX_CACHE = JEV_DIR / "index_daily_000001sh.json"
INDEX_AMT_CACHE = JEV_DIR / "index_amount_000001sh.json"
INDEX300_CACHE = JEV_DIR / "index_daily_000300sh.json"
CHINEXT_CACHE = JEV_DIR / "index_daily_399006sz.json"
BREADTH_CACHE = JEV_DIR / "breadth_by_date.json"
LIMIT_CACHE = JEV_DIR / "limit_stats_by_date.json"
MARGIN_CACHE = JEV_DIR / "margin_sse.json"
VIX_CACHE = JEV_DIR / "vix_close.json"
USDCNY_CACHE = JEV_DIR / "usdcny_close.json"
GAP_CACHE = JEV_DIR / "gap_by_date.json"
ZT_CACHE = JEV_DIR / "zt_pool_history.json"
FEATV = 10  # 特征版本：v10=v8+易术数字化(节气/干支五行/阴阳) 进 GBM


def _parse_json(text: Optional[str]) -> Optional[Dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _cache_stale(mapping: Dict, max_lag_days: int = 3) -> bool:
    """缓存最新日期键落后今天超过 max_lag_days 个自然日 → 需要刷新。"""
    try:
        latest = max(str(k) for k in mapping)
        d = _date.fromisoformat(f"{latest[:4]}-{latest[4:6]}-{latest[6:]}")
        return (_date.today() - d).days > max_lag_days
    except Exception:
        return True


def _pool_max_date() -> str:
    """real_data 合并池的最新交易日（本地 parquet，稳健数据源）。"""
    import glob as _g

    mx = "0"
    for fp in _g.glob(str(PROJECT_ROOT / "data_cache" / "real_data_*.parquet")):
        try:
            import pandas as pd

            mx = max(mx, str(pd.read_parquet(fp, columns=["trade_date"])["trade_date"].astype(str).max()))
        except Exception:
            continue
    return mx


def _pool_stale(mapping: Dict) -> bool:
    """缓存最新日期键落后 real_data 池最新交易日 → 需要刷新（池是唯一真相）。"""
    try:
        latest = max(str(k) for k in mapping)
        return latest < _pool_max_date()
    except Exception:
        return True


def _load_index_closes(force: bool = False) -> List[Tuple[str, float]]:
    """上证指数日收盘。缓存过期（最新日落后>3天）自动重拉；拉取失败用旧缓存。"""
    cached = None
    if INDEX_CACHE.exists():
        try:
            cached = [(a, float(b)) for a, b in json.loads(INDEX_CACHE.read_text("utf-8"))]
        except Exception:
            cached = None
    stale = True
    if cached:
        try:
            latest = _date.fromisoformat(f"{cached[-1][0][:4]}-{cached[-1][0][4:6]}-{cached[-1][0][6:]}")
            stale = (_date.today() - latest).days > 3
        except Exception:
            stale = True
    if force or stale or not cached:
        try:
            rows = _fetch_index_from_tushare()
            if rows:
                JEV_DIR.mkdir(parents=True, exist_ok=True)
                INDEX_CACHE.write_text(json.dumps(rows), encoding="utf-8")
                return rows
        except Exception as exc:
            logger.warning(f"Jev: 指数缓存刷新失败，使用本地缓存: {exc}")
    return cached or []


def _next_trading_day(after: _date) -> _date:
    """下一交易日（Tushare 交易日历缓存；失败则仅跳过周末）。"""
    cal_path = JEV_DIR / "trade_cal.json"
    open_days = None
    today = _date.today()
    if cal_path.exists():
        try:
            cal = json.loads(cal_path.read_text("utf-8"))
            if cal.get("fetched_for") == today.isoformat():
                open_days = set(cal.get("open") or [])
        except Exception:
            open_days = None
    if open_days is None:
        try:
            from .tushare_client import create_tushare_pro

            pro = create_tushare_pro(timeout=30)
            df = pro.trade_cal(exchange="SSE",
                               start_date=after.strftime("%Y%m%d"),
                               end_date=(after + timedelta(days=40)).strftime("%Y%m%d"))
            open_days = sorted(df[df["is_open"] == 1]["cal_date"].astype(str))
            JEV_DIR.mkdir(parents=True, exist_ok=True)
            cal_path.write_text(json.dumps({"fetched_for": today.isoformat(), "open": open_days}),
                                encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Jev: 交易日历获取失败，退化为跳过周末: {exc}")
    d = after + timedelta(days=1)
    for _ in range(30):
        if open_days is None:
            if d.weekday() < 5:
                return d
        elif d.strftime("%Y%m%d") in open_days:
            return d
        d += timedelta(days=1)
    return after + timedelta(days=1)


def _fetch_index_from_tushare(days: int = 4300) -> List[Tuple[str, float]]:
    """上证指数日收盘，默认拉 2015 年起（约 2700 交易日，跨 4 轮牛熊）。"""
    from .tushare_client import create_tushare_pro

    pro = create_tushare_pro(timeout=30)
    end = _date.today()
    start = end - timedelta(days=days)
    df = pro.index_daily(ts_code="000001.SH", start_date=start.strftime("%Y%m%d"),
                         end_date=end.strftime("%Y%m%d"))
    rows = sorted(zip(df["trade_date"].astype(str), df["close"].astype(float)))
    try:
        amt = {d: float(a) for d, a in
               sorted(zip(df["trade_date"].astype(str), df["amount"].astype(float)))}
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        INDEX_AMT_CACHE.write_text(json.dumps(amt), encoding="utf-8")
    except Exception:
        pass
    return [(d, float(c)) for d, c in rows]


def _load_fred_cached(series_id: str, cache_name: str, force: bool = False) -> Dict[str, float]:
    """FRED 日序列全量缓存（date→value），过期自动重拉（FRED T+1 披露，留 4 天余量）。"""
    path = JEV_DIR / cache_name
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text("utf-8"))
            if cached and not _cache_stale(cached, max_lag_days=4):
                return cached
        except Exception:
            pass
    try:
        with httpx.Client(timeout=20.0, headers=_HEADERS, trust_env=False) as client:
            r = client.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                           params={"id": series_id})
            rows = [line.split(",") for line in r.text.strip().splitlines()[1:] if "," in line]
        mapping = {d.replace("-", ""): float(v) for d, v in rows if v not in ("", ".")}
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping), encoding="utf-8")
        return mapping
    except Exception as exc:
        logger.warning(f"Jev: FRED {series_id} 拉取失败: {exc}")
        return {}


def _technical_signals(rows: List[Tuple[str, float]], as_of: _date) -> Dict:
    """公式化技术信号（全部由 ≤as_of 的收盘序列计算）。

    RSI14 / MACD(12,26,9) 状态 / 均线排列 / 量能比(5日/20日成交额) / 动量质量。
    """
    closes = [c for _, c in rows]
    if len(closes) < 30:
        return {}

    def _ema(series, n):
        k = 2 / (n + 1)
        out = [series[0]]
        for v in series[1:]:
            out.append(out[-1] + k * (v - out[-1]))
        return out

    # RSI14（Wilder 简化）
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0))
        losses.append(max(-ch, 0))
    window = gains[-14:]
    avg_g = sum(window) / 14 or 1e-9
    avg_l = sum(losses[-14:]) / 14 or 1e-9
    rsi = round(100 - 100 / (1 + avg_g / avg_l), 1)

    # MACD
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = [a - b for a, b in zip(ema12, ema26)]
    dea = _ema(dif, 9)
    hist_now = (dif[-1] - dea[-1]) * 2
    hist_prev = (dif[-2] - dea[-2]) * 2
    macd_state = ("金叉" if hist_now > 0 and hist_prev <= 0
                  else "死叉" if hist_now <= 0 and hist_prev > 0
                  else "多头" if hist_now > 0 else "空头")

    def _ma(n):
        return sum(closes[-n:]) / n

    ma5, ma20, ma60 = _ma(5), _ma(20), _ma(60)
    if ma5 > ma20 > ma60:
        ma_align = "多头排列"
    elif ma5 < ma20 < ma60:
        ma_align = "空头排列"
    else:
        ma_align = "纠缠"

    # 量能比：5日成交额 / 20日成交额
    vol_ratio = None
    try:
        if INDEX_AMT_CACHE.exists():
            amt = json.loads(INDEX_AMT_CACHE.read_text("utf-8"))
            vals = [v for d, v in sorted(amt.items()) if d <= as_of.strftime("%Y%m%d")]
            if len(vals) >= 20 and sum(vals[-20:]) > 0:
                vol_ratio = round((sum(vals[-5:]) / 5) / (sum(vals[-20:]) / 20), 3)
    except Exception:
        vol_ratio = None

    return {"RSI14": rsi, "MACD": macd_state, "MACD柱": round(hist_now, 2),
            "均线排列": ma_align, "量能比_5v20": vol_ratio,
            "动能质量": ("加速上行" if closes[-1] > closes[-6] and ret20_quality(closes) > 0
                     else "反弹乏力" if closes[-1] <= closes[-6] else "温和")}


def ret20_quality(closes):
    return closes[-1] / closes[-21] - 1 if len(closes) > 21 else 0


def _slice_as_of(rows: List[Tuple[str, float]], as_of: _date, lookback: int = 40) -> List[Tuple[str, float]]:
    """防穿越唯一入口：只返回 date <= as_of 的最后 lookback 行。"""
    kept = [(d, c) for d, c in rows if str(d) <= as_of.strftime("%Y%m%d")]
    return kept[-lookback:]


def _market_features(rows: List[Tuple[str, float]]) -> Dict:
    closes = [c for _, c in rows]
    if len(closes) < 6:
        return {}
    def ret(n):
        return round((closes[-1] / closes[-1 - n] - 1) * 100, 2) if len(closes) > n else None
    rets = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))][-20:]
    vol = round(float(np_std(rets)) * 100, 2) if len(rets) >= 5 else None
    return {"最新收盘": closes[-1], "涨跌_1日%": ret(1), "涨跌_5日%": ret(5),
            "涨跌_20日%": ret(20), "20日年化波动%": vol,
            "区间": f"{rows[0][0]}~{rows[-1][0]}"}


# ── v2 多维特征（FEATV=2）────────────────────────────────────────────
def _load_aux_index_close(ts_code: str, cache_name: str, force: bool = False) -> Dict[str, float]:
    """辅助指数（沪深300/创业板指）date→close 字典，带缓存+过期自动重拉。"""
    path = JEV_DIR / cache_name
    if path.exists() and not force:
        try:
            cached = json.loads(path.read_text("utf-8"))
            if cached and not _cache_stale(cached):
                return cached
        except Exception:
            pass
    try:
        from .tushare_client import create_tushare_pro

        pro = create_tushare_pro(timeout=30)
        end = _date.today()
        df = pro.index_daily(ts_code=ts_code,
                             start_date=(end - timedelta(days=1150)).strftime("%Y%m%d"),
                             end_date=end.strftime("%Y%m%d"))
        mapping = {d: float(c) for d, c in sorted(zip(df["trade_date"].astype(str),
                                                      df["close"].astype(float)))}
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mapping), encoding="utf-8")
        return mapping
    except Exception as exc:
        logger.warning(f"Jev: 辅助指数 {ts_code} 拉取失败: {exc}")
        return {}


def _load_breadth(as_of: _date, force: bool = False) -> Dict[str, float]:
    """全市场宽度：每交易日上涨家数占比（来自 real_data 合并池，池更新后自动重算）。"""
    if BREADTH_CACHE.exists() and not force:
        try:
            cached = json.loads(BREADTH_CACHE.read_text("utf-8"))
            if cached and not _pool_stale(cached):
                return cached
        except Exception:
            pass
    try:
        import glob as _glob

        import pandas as pd

        frames = []
        for fp in _glob.glob(str(PROJECT_ROOT / "data_cache" / "real_data_*.parquet")):
            try:
                frames.append(pd.read_parquet(fp, columns=["trade_date", "close", "ts_code"]))
            except Exception:
                continue
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
        df["date"] = df["trade_date"].astype(str)
        df = df.sort_values(["ts_code", "date"]) if "ts_code" in df.columns else df.sort_values("date")
        if "ts_code" in df.columns:
            df["prev_close"] = df.groupby("ts_code")["close"].shift(1)
            up = df[df["prev_close"].notna()].assign(up=lambda x: x["close"] > x["prev_close"])
            breadth = up.groupby("date")["up"].mean().round(4)
        else:
            breadth = pd.Series(dtype=float)
        mapping = {d: float(v) for d, v in breadth.items()}
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        BREADTH_CACHE.write_text(json.dumps(mapping), encoding="utf-8")
        return mapping
    except Exception as exc:
        logger.warning(f"Jev: 宽度计算失败: {exc}")
        return {}


def _load_margin(force: bool = False) -> Dict[str, float]:
    """两融余额（沪，亿元），过期自动重拉（T+1 披露，留 4 天余量）。"""
    if MARGIN_CACHE.exists() and not force:
        try:
            cached = json.loads(MARGIN_CACHE.read_text("utf-8"))
            if cached and not _cache_stale(cached, max_lag_days=4):
                return cached
        except Exception:
            pass
    try:
        from .tushare_client import create_tushare_pro

        pro = create_tushare_pro(timeout=30)
        end = _date.today()
        df = pro.margin(exchange_id="SSE",
                        start_date=(end - timedelta(days=800)).strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"))
        mapping = {d: round(float(v) / 1e8, 1) for d, v in
                   sorted(zip(df["trade_date"].astype(str), df["rzye"].astype(float)))}
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        MARGIN_CACHE.write_text(json.dumps(mapping), encoding="utf-8")
        return mapping
    except Exception as exc:
        logger.warning(f"Jev: 两融数据拉取失败: {exc}")
        return {}


def _load_limit_stats(as_of: _date, force: bool = False) -> Dict[str, Dict]:
    """涨跌停宽度：每交易日 {涨停家数, 跌停家数}（主板±9.7%近似，真实涨跌停价差异已知偏差）。

    由 real_data 合并池计算，池更新后自动重算。
    """
    if LIMIT_CACHE.exists() and not force:
        try:
            cached = json.loads(LIMIT_CACHE.read_text("utf-8"))
            if cached and not _pool_stale(cached):
                return cached
        except Exception:
            pass
    try:
        import glob as _glob

        import pandas as pd

        frames = []
        for fp in _glob.glob(str(PROJECT_ROOT / "data_cache" / "real_data_*.parquet")):
            try:
                frames.append(pd.read_parquet(fp, columns=["trade_date", "close", "ts_code"]))
            except Exception:
                continue
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
        df["date"] = df["trade_date"].astype(str)
        df = df.sort_values(["ts_code", "date"])
        df["prev_close"] = df.groupby("ts_code")["close"].shift(1)
        df = df[df["prev_close"].notna() & (df["prev_close"] > 0)]
        df["chg"] = df["close"] / df["prev_close"] - 1
        g = df.groupby("date")["chg"]
        stats = pd.DataFrame({
            "limit_up": g.apply(lambda s: int((s >= 0.097).sum())),
            "limit_down": g.apply(lambda s: int((s <= -0.097).sum())),
            "up_ratio": g.apply(lambda s: float((s > 0).mean())),
        })
        mapping = {d: {"limit_up": int(r.limit_up), "limit_down": int(r.limit_down),
                       "up_ratio": round(float(r.up_ratio), 4)}
                   for d, r in stats.iterrows()}
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        LIMIT_CACHE.write_text(json.dumps(mapping), encoding="utf-8")
        return mapping
    except Exception as exc:
        logger.warning(f"Jev: 涨跌停统计失败: {exc}")
        return {}


def _macro_known(as_of: _date) -> Dict:
    """宏观已知值（发布滞后≥1个月的最近值，无穿越）。"""
    try:
        from .tushare_client import create_tushare_pro

        pro = create_tushare_pro(timeout=30)
        out = {}
        pub_cut = as_of.strftime("%Y%m")  # 当月数据未发布，取上月及以前
        pmi = pro.cn_pmi()
        if pmi is not None and not pmi.empty:
            pmi.columns = [str(c).lower() for c in pmi.columns]
            sub = pmi[(pmi["month"] < pub_cut)].dropna(subset=["pmi010000"])
            if not sub.empty:
                row = sub.iloc[0]
                out["最近PMI"] = float(row["pmi010000"])
                out["PMI月份"] = str(row["month"])
        cpi = pro.cn_cpi()
        if cpi is not None and not cpi.empty:
            cpi.columns = [str(c).lower() for c in cpi.columns]
            sub = cpi[(cpi["month"] < pub_cut)].dropna(subset=["nt_yoy"])
            if not sub.empty:
                row = sub.iloc[0]
                out["最近CPI同比%"] = float(row["nt_yoy"])
        return out
    except Exception as exc:
        logger.warning(f"Jev: 宏观已知值获取失败: {exc}")
        return {}


def _load_gap_series(force: bool = False) -> Dict[str, float]:
    """隔夜缺口：全市场(开盘/昨收-1)中位数，按日（real_data 池更新后自动重算）。"""
    if GAP_CACHE.exists() and not force:
        try:
            cached = json.loads(GAP_CACHE.read_text("utf-8"))
            if cached and not _pool_stale(cached):
                return cached
        except Exception:
            pass
    try:
        import glob as _glob

        import pandas as pd

        frames = []
        for fp in _glob.glob(str(PROJECT_ROOT / "data_cache" / "real_data_*.parquet")):
            try:
                frames.append(pd.read_parquet(fp, columns=["trade_date", "close", "open", "ts_code"]))
            except Exception:
                continue
        df = pd.concat(frames, ignore_index=True)
        df["date"] = df["trade_date"].astype(str)
        df = df.sort_values(["ts_code", "date"])
        df["prev_close"] = df.groupby("ts_code")["close"].shift(1)
        df = df[df["prev_close"].notna() & (df["prev_close"] > 0)]
        df["gap"] = df["open"] / df["prev_close"] - 1
        mapping = {d: round(float(v), 5) for d, v in df.groupby("date")["gap"].median().items()}
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        GAP_CACHE.write_text(json.dumps(mapping), encoding="utf-8")
        return mapping
    except Exception as exc:
        logger.warning(f"Jev: 缺口序列计算失败: {exc}")
        return {}


def _load_zt_history(force: bool = False, back_days: int = 0) -> Dict[str, Dict]:
    """真实涨停池情绪：由本地 limit_pool_*.parquet 文件计算（Tushare 已入池，
    不再走 akshare 逐日回溯）。cover 区间=这些文件涵盖的交易日。"""
    if ZT_CACHE.exists() and not force:
        try:
            saved = json.loads(ZT_CACHE.read_text("utf-8"))
            if saved.get("backfilled"):
                data = saved.get("data") or {}
                # 最新 limit_pool 文件比缓存新 → 重算（涨停池文件是唯一真相）
                import glob as _g

                pool_days = [f.split("limit_pool_")[-1].split(".")[0]
                             for f in _g.glob(str(PROJECT_ROOT / "data_cache" / "limit_pool_*.parquet"))]
                if not pool_days or max(pool_days) <= max(data or {"0": None}):
                    return data
        except Exception:
            pass
    try:
        import glob as _glob

        import pandas as pd

        out = {}
        for fp in sorted(_glob.glob(str(PROJECT_ROOT / "data_cache" / "limit_pool_*.parquet"))):
            try:
                df = pd.read_parquet(fp)
                day = str(df["trade_date"].iloc[0]) if "trade_date" in df.columns else ""
                day = day.replace("-", "")[:8]
                if not day:
                    continue
                nzt = len(df)
                max_lb = 0
                if "limit_times" in df.columns:
                    max_lb = int(pd.to_numeric(df["limit_times"], errors="coerce").max() or 0)
                zha = int((df.get("open_times", 0).fillna(0) > 0).sum()) if "open_times" in df.columns else 0
                out[day] = {"zt": nzt, "zb": zha,
                            "zha_rate": round(zha / max(1, nzt), 3), "max_lb": max_lb}
            except Exception:
                continue
        JEV_DIR.mkdir(parents=True, exist_ok=True)
        ZT_CACHE.write_text(json.dumps({"back_days": 0, "backfilled": True,
                                        "source": "local_limit_pool_parquet", "data": out}),
                            encoding="utf-8")
        return out
    except Exception as exc:
        logger.warning(f"Jev: 本地涨停池计算失败: {exc}")
        return {}


def _yiji_numeric(d: _date) -> Dict[str, float]:
    """易术维度数字化（供 GBM 等模型食用）：
    节气进度/距下一节气、日干支生克强度、阴阳标签、年柱五行。"""
    from collections import defaultdict

    ctx = yiji_context(d)
    gan_wx = {"木": 0, "火": 1, "土": 2, "金": 3, "水": 4}
    gan_yinyang = {"甲": 1, "乙": 0, "丙": 1, "丁": 0, "戊": 1, "己": 0,
                   "庚": 1, "辛": 0, "壬": 1, "癸": 0}
    out = {
        "节气进度0-1": ctx.get("节气进度0-1") or 0,
        "距下一节气": ctx.get("距下一节气") or 15,
        "日干五行码": gan_wx.get(ctx.get("日干五行"), 2),
        "日支五行码": gan_wx.get(ctx.get("日支五行"), 2),
        "干支阴阳": gan_yinyang.get(ctx.get("日干"), 0),
        "日柱五行相生": 0.0,
    }
    wg, wz = ctx.get("日干五行"), ctx.get("日支五行")
    if wg and wz:
        if WUXING_SHENG.get(wg) == wz:
            out["日柱五行相生"] = 1.0
        elif WUXING_SHENG.get(wz) == wg:
            out["日柱五行相生"] = -1.0
    return out


def _market_features_v10(as_of: _date, force: bool = False) -> Dict:
    """v10 = v8 全部 + 易术维度数字化（节气/干支五行/阴阳）。"""
    feats = _market_features_v8(as_of, force=force)
    if not feats:
        return {}
    feats.update(_yiji_numeric(as_of))
    feats["特征版本"] = f"v{FEATV}"
    return feats


def _market_features_v8(as_of: _date, force: bool = False) -> Dict:
    """v8 = v5 + 隔夜缺口 + 真实涨停池情绪（涨停/炸板率/最高连板）。"""
    feats = _market_features_v5(as_of, force=force)
    if not feats:
        return {}
    key = as_of.strftime("%Y%m%d")

    gap = _load_gap_series(force=force)
    g_vals = sorted((d, v) for d, v in gap.items() if d <= key)
    if len(g_vals) >= 6:
        feats["隔夜缺口%"] = round(g_vals[-1][1] * 100, 2)
        feats["缺口5日均值%"] = round(sum(v for _, v in g_vals[-5:]) / 5 * 100, 2)

    zt = _load_zt_history(force=force, back_days=0)
    z_keys = [d for d in sorted(zt) if d <= key]
    if z_keys:
        recent = [zt[d] for d in z_keys[-5:]]
        last = recent[-1]
        feats["真实涨停家数"] = last.get("zt")
        feats["炸板率%"] = round(last.get("zha_rate", 0) * 100, 1)
        feats["最高连板"] = max(r.get("max_lb", 0) for r in recent)
    else:
        # 真实池缺失时回退近似值（LIMIT_CACHE 的 ±9.7% 统计）
        lim = json.loads(LIMIT_CACHE.read_text("utf-8")) if LIMIT_CACHE.exists() else {}
        limd = lim.get(key) or {}
        if limd:
            feats["真实涨停家数"] = int(limd.get("limit_up", 0))
            feats["炸板率%"] = None
            feats["最高连板"] = None
    feats["特征版本"] = f"v{FEATV}"
    return feats


def _market_features_v5(as_of: _date, force: bool = False) -> Dict:
    """v5 全维度特征：v4 全部 + 外盘(VIX/SPX隔夜/USDCNY) + 长周期 + 周内效应。

    全部维度仅使用 ≤ as_of 的数据（外盘用其最后收盘日 ≤ as_of，天然无穿越）。
    """
    feats = _market_features_v4(as_of, force=force)
    if not feats:
        return {}
    rows = _slice_as_of(_load_index_closes(force=force), as_of, lookback=1400)
    closes = [c for _, c in rows]
    dmap = dict(rows)
    dates = [d for d, _ in rows]
    key = as_of.strftime("%Y%m%d")

    def ret(n):
        return round((closes[-1] / closes[-1 - n] - 1) * 100, 2) if len(closes) > n else None

    if len(closes) > 120:
        feats["120日趋势%"] = round((closes[-1] / closes[-121] - 1) * 100, 2)
    if len(closes) > 250:
        feats["250日趋势%"] = round((closes[-1] / closes[-251] - 1) * 100, 2)
        win250 = closes[-250:]
        feats["250日位置%"] = round((closes[-1] - min(win250)) / max(1e-9, max(win250) - min(win250)) * 100, 1)

    # 量能：成交额 5/20 比（若缓存有）
    try:
        if INDEX_AMT_CACHE.exists():
            amt_all = json.loads(INDEX_AMT_CACHE.read_text("utf-8"))
            vals = [v for d, v in sorted(amt_all.items()) if d <= key]
            if len(vals) >= 20 and sum(vals[-20:]) > 0:
                feats["成交额比_5v20"] = round((sum(vals[-5:]) / 5) / (sum(vals[-20:]) / 20), 3)
    except Exception:
        pass

    # 外盘：VIX 水平与 5 日变化、标普隔夜(上一收盘日)、USDCNY 20 日
    vix = _load_fred_cached("VIXCLS", VIX_CACHE.name, force=force)
    v_vals = sorted((d, v) for d, v in vix.items() if d <= key)
    if len(v_vals) >= 6:
        feats["VIX"] = v_vals[-1][1]
        feats["VIX_5日变化%"] = round((v_vals[-1][1] / v_vals[-6][1] - 1) * 100, 2)
    spx = _load_fred_cached("SP500", "spx_close.json", force=force)
    s_vals = sorted((d, v) for d, v in spx.items() if d <= key)
    if len(s_vals) >= 2:
        feats["SPX隔夜%"] = round((s_vals[-1][1] / s_vals[-2][1] - 1) * 100, 2)
    fx = _load_fred_cached("DEXCHUS", USDCNY_CACHE.name, force=force)
    f_vals = sorted((d, v) for d, v in fx.items() if d <= key)
    if len(f_vals) >= 21:
        feats["USDCNY_20日%"] = round((f_vals[-1][1] / f_vals[-21][1] - 1) * 100, 3)

    # 周内效应（已知的日历事实）
    feats["星期"] = as_of.weekday() + 1

    # 辅助指数也拉全历史（供元模型 GBM 用长序列）
    _load_aux_index_close("000300.SH", INDEX300_CACHE.name, force=force)
    _load_aux_index_close("399006.SZ", CHINEXT_CACHE.name, force=force)
    feats["特征版本"] = f"v{FEATV}"
    return feats


def _market_features_v4(as_of: _date, force: bool = False) -> Dict:
    """v4 = v3 + 24节气 + 涨跌停宽度 + 情绪复合分 + 宏观已知值 + 长趋势。"""
    feats = _market_features_v3(as_of, force=force)
    if not feats:
        return {}
    feats.update(solar_term_info(as_of))

    limits = _load_limit_stats(as_of, force=force)
    key = as_of.strftime("%Y%m%d")
    lu = ld = None
    if key in limits:
        lu, ld = limits[key]["limit_up"], limits[key]["limit_down"]
        feats["涨停家数"] = lu
        feats["跌停家数"] = ld
        feats["涨跌停比"] = round(lu / max(1, ld), 2)

    # 情绪复合分（-100~100）：宽度+涨跌停比+量能+两融，公式透明
    up_ratio = limits.get(key, {}).get("up_ratio")
    vol_r = feats.get("量能比_5v20")
    margin_chg = feats.get("两融20日变化%")
    sent = 0.0
    if up_ratio is not None:
        sent += 60.0 * max(-1.0, min(1.0, (up_ratio - 0.5) / 0.15))
    if lu is not None and ld:
        sent += 20.0 * max(-1.0, min(1.0, (lu - ld) / max(20, lu + ld)))
    if vol_r:
        sent += 10.0 * max(-1.0, min(1.0, (vol_r - 1.0) / 0.25))
    if margin_chg is not None:
        sent += 10.0 * max(-1.0, min(1.0, margin_chg / 3.0))
    feats["情绪复合分"] = round(max(-100.0, min(100.0, sent)), 0)

    # 长趋势与突破
    rows = _slice_as_of(_load_index_closes(force=force), as_of, lookback=270)
    closes = [c for _, c in rows]
    if len(closes) > 120:
        feats["120日趋势%"] = round((closes[-1] / closes[-121] - 1) * 100, 2)
    if len(closes) >= 20:
        feats["突破20日高"] = bool(closes[-1] >= max(closes[-21:-1]))

    m = _macro_known(as_of)
    if m:
        feats.update(m)
    feats["特征版本"] = f"v{FEATV}"
    return feats


def _market_features_v3(as_of: _date, force: bool = False) -> Dict:
    """v3 = v2 多维 + 公式化技术信号 + 历史基准涨概率（全部 ≤ as_of）。"""
    rows = _slice_as_of(_load_index_closes(force=force), as_of, lookback=130)
    feats = _market_features_v2(as_of, force=force)
    if not feats:
        return {}
    feats.update(_technical_signals(rows, as_of))
    closes = [c for _, c in rows]
    ups = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i - 1])
    feats["历史基准涨概率%"] = round(ups / (len(closes) - 1) * 100, 1) if len(closes) > 1 else None
    feats["特征版本"] = f"v{FEATV}"
    return feats


def _market_features_v2(as_of: _date, force: bool = False) -> Dict:
    """v2 多维特征：主指数 + 结构指数 + 宽度 + 两融（全部 ≤ as_of，无穿越）。"""
    rows = _slice_as_of(_load_index_closes(force=force), as_of, lookback=80)
    base = _market_features(rows)
    if not base:
        return {}
    closes = [c for _, c in rows]
    # 60日位置：现价在近60日高低区间的分位
    win = closes[-60:]
    pos60 = round((closes[-1] - min(win)) / max(1e-9, (max(win) - min(win))) * 100, 1) if len(win) >= 20 else None
    base["60日位置%"] = pos60

    def idx_ret(mapping: Dict[str, float], n: int) -> Optional[float]:
        vals = sorted((d, v) for d, v in mapping.items() if d <= as_of.strftime("%Y%m%d"))
        if len(vals) <= n:
            return None
        return round((vals[-1][1] / vals[-1 - n][1] - 1) * 100, 2)

    hs300 = _load_aux_index_close("000300.SH", INDEX300_CACHE.name, force=force)
    cyb = _load_aux_index_close("399006.SZ", CHINEXT_CACHE.name, force=force)
    base["沪深300_5日%"] = idx_ret(hs300, 5)
    base["沪深300_20日%"] = idx_ret(hs300, 20)
    base["创业板_5日%"] = idx_ret(cyb, 5)
    base["创业板_20日%"] = idx_ret(cyb, 20)

    breadth = _load_breadth(as_of, force=force)
    b_vals = sorted((d, v) for d, v in breadth.items() if d <= as_of.strftime("%Y%m%d"))
    if len(b_vals) >= 6:
        latest_ratio = b_vals[-1][1]
        b5 = [v for d, v in b_vals[-5:]]
        base["上涨家数占比%"] = round(latest_ratio * 100, 1)
        base["宽度5日均值%"] = round(sum(b5) / len(b5) * 100, 1)
    margin = _load_margin(force=force)
    m_vals = sorted((d, v) for d, v in margin.items() if d <= as_of.strftime("%Y%m%d"))
    if len(m_vals) >= 21:
        base["两融余额_亿"] = m_vals[-1][1]
        base["两融20日变化%"] = round((m_vals[-1][1] / m_vals[-21][1] - 1) * 100, 2)
    base["特征版本"] = f"v{FEATV}"
    return base


def np_std(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# ═══════════════════════════════════════════════════════════════════
# 任务 A：大盘明日涨跌概率
# ═══════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════
# 持久化与回填
# ═══════════════════════════════════════════════════════════════════
def _clamp01(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


