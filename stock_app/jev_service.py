# -*- coding: utf-8 -*-
"""Jev 易术 AI 预判模块。

两个预判任务（LLM 供应商可配置，OpenAI 兼容格式）：
  A. 大盘明日上涨概率（市场数据 + 天干地支/五行/八卦 → p_up）
  B. 概念/行业板块买入价值与持续性概率

供应商：
  默认使用平台已验证的 DeepSeek（https://api.deepseek.com, deepseek-chat）。
  拿到 Jev 地址后设置环境变量即可切换，无需改代码：
      JEV_API_BASE=https://<jev域名>      JEV_API_KEY=apikey_xxx
      JEV_MODEL=<模型名>

样本不穿越保证（防未来函数）：
  - 预测 T+1 时，提示词只允许包含交易日 ≤ T 的数据（_slice_as_of 强制截断）；
  - 干支/五行/卦象由公历日期确定性推导（T+1 的日期是已知的日历事实，非未来数据）；
  - 验证采用 walk-forward：逐日滚动，每天只用截至当日的信息生成预测，
    再与 T+1 真实涨跌比对；内/外样本按时间前 2/3 与后 1/3 切分仅用于报告漂移。

持久化：data_cache/jev/ 下的 predictions.jsonl（追加式）与 report_*.json。
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
PREDICTIONS_PATH = JEV_DIR / "predictions.jsonl"

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


def get_provider() -> Dict[str, str]:
    base = os.environ.get("JEV_API_BASE", "") or _load_legacy_env("JEV_API_BASE") \
        or "https://api.typesafe.ai"
    key = os.environ.get("JEV_API_KEY", "") or _load_legacy_env("JEV_API_KEY")
    model = os.environ.get("JEV_MODEL", "") or _load_legacy_env("JEV_MODEL") or "jev-latest"
    kind = "systemone" if "typesafe" in base.lower() else "openai"
    return {"base": base.rstrip("/"), "key": key, "model": model, "kind": kind}


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


def _systemone_ask(state: Dict, questions: Dict) -> Optional[Dict]:
    """TypeSafe systemone 原生调用 → answers 字典。失败返回 None。"""
    p = get_provider()
    if not p["key"]:
        logger.warning("Jev预判: 未配置 JEV_API_KEY")
        return None
    try:
        with httpx.Client(timeout=90.0, trust_env=False) as client:
            r = client.post(f"{p['base']}/v1/systemone",
                            headers={"Authorization": f"Bearer {p['key']}"},
                            json={"model": p["model"],
                                  "state": _sanitize_state(state),
                                  "questions": questions})
            r.raise_for_status()
            data = r.json()
            return data.get("answers") or {}
    except Exception as exc:
        logger.warning(f"Jev预判: systemone调用失败（{p['base']}/{p['model']}）: {exc}")
        return None


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


def _http_chat(prompt: str, json_mode: bool = True) -> Optional[str]:
    p = get_provider()
    if not p["key"]:
        logger.warning("Jev预判: 未配置 API key（JEV_API_KEY/DEEPSEEK_API_KEY）")
        return None
    body = {
        "model": p["model"],
        "messages": [
            {"role": "system", "content": "你是严谨的量化分析师，兼通易术象数。只输出合法 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 900,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    try:
        with httpx.Client(timeout=90.0, trust_env=False) as client:
            r = client.post(f"{p['base']}/chat/completions",
                            headers={"Authorization": f"Bearer {p['key']}"},
                            json=body)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning(f"Jev预判: LLM调用失败（{p['base']}/{p['model']}）: {exc}")
        return None


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


def predict_index_next() -> Optional[Dict]:
    """预测"下一个交易日"（自动跳过周末节假日，指数缓存强制刷新）。"""
    rows = _load_index_closes(force=True)
    if not rows:
        return None
    latest = max(d for d, _ in rows)
    y, m, dd = int(latest[:4]), int(latest[4:6]), int(latest[6:8])
    target = _next_trading_day(_date(y, m, dd))
    return predict_index(target)


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
    """FRED 日序列全量缓存（date→value）。"""
    path = JEV_DIR / cache_name
    if path.exists() and not force:
        try:
            return json.loads(path.read_text("utf-8"))
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
    """辅助指数（沪深300/创业板指）date→close 字典，带缓存。"""
    path = JEV_DIR / cache_name
    if path.exists() and not force:
        try:
            return json.loads(path.read_text("utf-8"))
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
    """全市场宽度：每交易日上涨家数占比（来自 real_data 合并池，一次性计算后缓存）。"""
    if BREADTH_CACHE.exists() and not force:
        try:
            return json.loads(BREADTH_CACHE.read_text("utf-8"))
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
    """两融余额（沪，亿元）date→balance。"""
    if MARGIN_CACHE.exists() and not force:
        try:
            return json.loads(MARGIN_CACHE.read_text("utf-8"))
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

    由 real_data 合并池一次性计算后缓存。
    """
    if LIMIT_CACHE.exists() and not force:
        try:
            return json.loads(LIMIT_CACHE.read_text("utf-8"))
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
    """隔夜缺口：全市场(开盘/昨收-1)中位数，按日。一次性从 real_data 池计算。"""
    if GAP_CACHE.exists() and not force:
        try:
            return json.loads(GAP_CACHE.read_text("utf-8"))
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
                return saved.get("data") or {}
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
def predict_index(target_date: _date) -> Optional[Dict]:
    """预测 target_date 当日上证涨跌。使用 ≤ target_date-1 的 v5 特征（无穿越）。"""
    as_of = target_date - timedelta(days=1)
    feats = _market_features_v5(as_of)
    if not feats:
        logger.warning("Jev预判: 指数数据不足")
        return None
    yiji = yiji_context(target_date)
    p = get_provider()
    rec = {
        "kind": "index", "target_date": target_date.isoformat(),
        "as_of": as_of.isoformat(), "provider": p["model"],
        "featv": FEATV,
        "features": feats, "yiji": yiji,
    }

    if p["kind"] == "systemone":
        state = {"market": feats, "yiji": yiji, "target_date": target_date.isoformat()}
        answers = _systemone_ask(state, {
            "up_tomorrow": {
                "type": "choice",
                "question": ("Quant + Yijing metaphysics view: will the Shanghai index "
                             f"CLOSE UP on {target_date.isoformat()}? Weigh momentum, "
                             "volatility and the day-pillar five-element relations and hexagram imagery."),
                "criteria": {
                    "UP": "index closes higher than previous close",
                    "DOWN": "index closes lower than previous close",
                },
            },
            "conviction": {
                "type": "choice",
                "question": "How strong is your conviction in the direction above?",
                "criteria": {
                    "high": "clear signal, would act on it",
                    "medium": "signal present but noisy",
                    "low": "coin-flip territory",
                },
            },
        })
        if not answers or "up_tomorrow" not in answers:
            return None
        a = answers["up_tomorrow"]
        probs = a.get("probabilities") or {}
        rec["p_up"] = _clamp01(probs.get("UP", 0.0))
        rec["direction"] = "up" if a.get("choice") == "UP" else "down"
        conv = (answers.get("conviction") or {}).get("choice")
        rec["conviction"] = {"high": 0.85, "medium": 0.55, "low": 0.3}.get(conv, 0.5)
        rec["reason"] = f"[系统一结构化决策] choice={a.get('choice')} confidence={a.get('confidence')}"
        _append_prediction(rec)
        return rec

    # DeepSeek 等 OpenAI 兼容文本模型（v3 提示词：具名信号+基准锚定）
    base_up = feats.get("历史基准涨概率%")
    prompt = (
        "任务：预测上证指数在目标交易日的上涨概率 p_up。\n"
        f"目标交易日：{target_date.isoformat()}\n\n"
        f"【计算公式信号（截至 {as_of.isoformat()} 收盘，绝无未来数据）】\n"
        f"{json.dumps(feats, ensure_ascii=False)}\n\n"
        "【目标日的干支五行卦象（辅助维度，权重应低于量化信号）】\n"
        f"{json.dumps(yiji, ensure_ascii=False)}\n\n"
        f"【重要锚定】历史上该指数约 {base_up}% 的交易日上涨。"
        "没有强信号时 p_up 应贴近该基准，只有在多项信号同向时才显著偏离。\n"
        "输出 JSON：\n"
        '{"p_up": 0~1, "edge": "偏离基准的原因(无偏离写 none)", '
        '"reason": "60字内：RSI/MACD/均线/量能各一句+易术一句"}'
    )
    out = _parse_json(_http_chat(prompt))
    if not out or "p_up" not in out:
        return None
    rec.update({
        "p_up": _clamp01(out.get("p_up")), "direction": out.get("direction"),
        "conviction": _clamp01(out.get("conviction")), "reason": str(out.get("reason", ""))[:200],
    })
    _append_prediction(rec)
    return rec


# ═══════════════════════════════════════════════════════════════════
# 任务 B：板块买入价值 + 持续性概率
# ═══════════════════════════════════════════════════════════════════
def predict_sectors(target_date: _date, top_n: int = 8) -> Optional[Dict]:
    """对行业板块给出买入价值与持续性。使用 ≤ target_date-1 的板块资金流（无穿越）。"""
    sectors = _sector_snapshot(target_date)
    if not sectors:
        logger.warning("Jev预判: 板块资金流缓存不足")
        return None
    yiji = yiji_context(target_date)
    p = get_provider()
    sec_map = {s["sector"]: s for s in sectors}

    if p["kind"] == "systemone":
        # 批量类型化问题：每板块"值得买吗"+"动能会持续吗"，一次调用全出
        questions = {}
        for s in sectors[:12]:
            name = s["sector"]
            questions[f"buy::{name}"] = {
                "type": "choice",
                "question": f"Considering the 5-day main-force net inflow ({s['net_5d_yi']}亿) "
                            f"and Yijing day-pillar context, is {name} worth buying next session?",
                "criteria": {"buy": "worth buying", "pass": "not attractive"},
            }
            questions[f"persist::{name}"] = {
                "type": "choice",
                "question": f"Will the net inflow into {name} CONTINUE over the next 3 sessions?",
                "criteria": {"sustain": "inflow continues", "fade": "momentum fades"},
            }
        answers = _systemone_ask(
            {"sector_flows": sectors[:12], "yiji": yiji, "target_date": target_date.isoformat()},
            questions)
        if not answers:
            return None
        picks, avoid, pass_scores = [], [], {}
        for key, a in answers.items():
            kind, _, name = key.partition("::")
            probs = a.get("probabilities") or {}
            if kind == "buy":
                score = _clamp01(probs.get("buy", 0.0))
                if name not in sec_map:
                    continue
                if a.get("choice") == "buy" and score >= 0.5:
                    picks.append({"sector": name, "persistence": 0.0,
                                  "reason": f"买入概率{score:.0%}", "_buy": score})
                else:
                    pass_scores[name] = score
                    avoid.append(name)
            elif kind == "persist":
                for pk in picks:
                    if pk["sector"] == name:
                        pk["persistence"] = _clamp01(probs.get("sustain", 0.0))
        if not picks and pass_scores:
            # 保守市况：无板块达 50% 买入阈值 → 给 buy 概率参考榜（明确标注未达标）
            for name, sc in sorted(pass_scores.items(), key=lambda kv: -kv[1])[:5]:
                picks.append({"sector": name, "persistence": 0.0,
                              "reason": f"买入概率{sc:.0%}（未达50%阈值，仅供观察）", "_buy": sc})
        picks = sorted(picks, key=lambda x: -x.get("_buy", 0))[:top_n]
        rec = {
            "kind": "sectors", "target_date": target_date.isoformat(),
            "as_of": sectors[0].get("as_of", ""), "provider": p["model"],
            "picks": picks, "avoid": avoid[:4], "yiji": yiji,
        }
        _append_prediction(rec)
        return rec

    # DeepSeek 等 OpenAI 兼容文本模型
    prompt = (
        "任务：从下列行业板块中选出最值得买入的板块并评估行情持续性。\n"
        f"目标交易日：{target_date.isoformat()}\n\n"
        "【板块资金流（近5日主力净额亿元，截至昨日，绝无未来数据）】\n"
        f"{json.dumps(sectors[:20], ensure_ascii=False)}\n\n"
        "【目标日的干支五行卦象】\n"
        f"{json.dumps(yiji, ensure_ascii=False)}\n\n"
        "结合资金动能与五行生克（如资金属水而日柱属土则受克承压），输出 JSON：\n"
        '{"picks": [{"sector": "板块名", "persistence": 0~1 持续性概率, "reason": "40字内"}, '
        f'"...最多{top_n}个"], "avoid": ["最不建议的2个板块"]}}'
    )
    out = _parse_json(_http_chat(prompt))
    if not out or not out.get("picks"):
        return None
    known = {s["sector"] for s in sectors}
    picks = [pk for pk in out["picks"] if isinstance(pk, dict) and pk.get("sector") in known][:top_n]
    for pk in picks:
        pk["persistence"] = _clamp01(pk.get("persistence"))
    rec = {
        "kind": "sectors", "target_date": target_date.isoformat(),
        "as_of": sectors[0].get("as_of", ""), "provider": p["model"],
        "picks": picks, "avoid": [str(a) for a in (out.get("avoid") or [])][:4],
        "yiji": yiji,
    }
    _append_prediction(rec)
    return rec


def _sector_snapshot(as_of: _date) -> List[Dict]:
    """从资金流缓存取 ≤ as_of 的最新一日板块净流入排名。"""
    import pandas as pd

    cache_dir = PROJECT_ROOT / "data_cache" / "bull_bear" / "cache"
    cands = list(cache_dir.glob("fund_flow_sector*.parquet")) + list(cache_dir.glob("*sector*.parquet"))
    frames = []
    for p in cands:
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True)
    df = df[pd.to_datetime(df["date"].astype(str), errors="coerce").notna()]
    df["date"] = df["date"].astype(str)
    df = df[df["date"] <= as_of.strftime("%Y%m%d")]
    if df.empty:
        return []
    latest = df["date"].max()
    day = df[df["date"] == latest]
    if "main_force_net" not in day.columns:
        return []
    g = day.groupby("sector_name")["main_force_net"].sum().sort_values(ascending=False)
    out = [{"sector": k, "net_5d_yi": round(float(v) / 1e8, 2), "as_of": latest}
           for k, v in g.head(20).items()]
    return out


# ═══════════════════════════════════════════════════════════════════
# 持久化与回填
# ═══════════════════════════════════════════════════════════════════
def _append_prediction(rec: Dict) -> None:
    """追加预测记录；同 kind+目标日+模型+特征版本 已存在时跳过。"""
    JEV_DIR.mkdir(parents=True, exist_ok=True)
    for old in _load_predictions():
        if (old.get("kind") == rec.get("kind")
                and old.get("target_date") == rec.get("target_date")
                and old.get("provider") == rec.get("provider")
                and old.get("featv") == rec.get("featv")):
            return
    with open(PREDICTIONS_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_predictions() -> List[Dict]:
    if not PREDICTIONS_PATH.exists():
        return []
    out = []
    for line in PREDICTIONS_PATH.read_text("utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _clamp01(v) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


# ═══════════════════════════════════════════════════════════════════
# Walk-forward 验证（样本不穿越 + 内外样本报告）
# ═══════════════════════════════════════════════════════════════════
def validate(days: int = 30) -> Dict:
    """逐日 walk-forward：每天用 ≤T 数据预测 T+1，与真实涨跌比对。

    优化：已预测过的日期直接复用历史记录（不重复扣 LLM 调用），增量补算新日期。
    内样本 = 评估窗前 2/3；外样本 = 后 1/3（检验提示词同分布假设是否漂移）。
    """
    rows = _load_index_closes()
    if len(rows) < days + 6:
        days = max(5, len(rows) - 6)
    date_list = [d for d, _ in rows[-(days + 1):]]
    reused = {(p.get("target_date"), p.get("provider")): p
              for p in _load_predictions()
              if p.get("kind") == "index" and p.get("featv") == FEATV
              and p.get("provider") == get_provider()["model"]}
    model = get_provider()["model"]
    results, fresh_calls = [], 0
    for i in range(len(date_list) - 1):
        t = _date.fromisoformat(f"{date_list[i][:4]}-{date_list[i][4:6]}-{date_list[i][6:]}")
        t1 = _date.fromisoformat(f"{date_list[i+1][:4]}-{date_list[i+1][4:6]}-{date_list[i+1][6:]}")
        rec = reused.get((t1.isoformat(), model))
        if rec and "p_up" in rec:
            pass  # 复用历史预测，不再扣调用
        else:
            rec = predict_index(t1)
            fresh_calls += 1
        if not rec:
            continue
        c_t = dict(rows)[date_list[i]]
        c_t1 = dict(rows)[date_list[i + 1]]
        actual_up = c_t1 > c_t
        results.append({
            "target_date": t1.isoformat(), "p_up": rec["p_up"],
            "actual_up": actual_up, "correct": (rec["p_up"] > 0.5) == actual_up,
        })
    if not results:
        return {"status": "failed", "message": "无可用预测样本"}

    split = int(len(results) * 2 / 3)
    in_s, out_s = results[:split], results[split:]

    def _stats(seg):
        n = len(seg)
        acc = sum(r["correct"] for r in seg) / n if n else 0.0
        brier = sum((r["p_up"] - (1 if r["actual_up"] else 0)) ** 2 for r in seg) / n if n else 0.0
        base = sum(r["actual_up"] for r in seg) / n if n else 0.0
        buckets = {}
        for r in seg:
            k = min(4, int(r["p_up"] * 5))
            buckets.setdefault(k, []).append(r["actual_up"])
        calib = {f"{k*20}-{k*20+20}%": round(sum(v)/len(v), 3) for k, v in sorted(buckets.items()) if v}
        return {"n": n, "accuracy": round(acc, 3), "brier": round(brier, 3),
                "base_rate_up": round(base, 3), "edge": round(acc - base, 3),
                "calibration": calib}

    report = {
        "status": "success", "days": days, "provider": model,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fresh_llm_calls": fresh_calls,
        "overall": _stats(results),
        "in_sample": _stats(in_s), "out_sample": _stats(out_s) if out_s else None,
        "sector_validation": validate_sectors(days),
        "detail": results[-10:],
    }
    JEV_DIR.mkdir(parents=True, exist_ok=True)
    (JEV_DIR / f"report_index_{datetime.now().strftime('%Y%m%d_%H%M')}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def validate_sectors(days: int = 20) -> Optional[Dict]:
    """板块推荐持续性验证：pick 的 persistence 是否兑现为后续3日真净流入。

    实际持续 = 该板块之后 3 个交易日的主力净额合计 > 0（用资金流缓存，无未来函数：
    只比对 as_of 之后的已实现数据）。分桶报告高/中持续性组的兑现率。
    """
    import pandas as pd

    picks_all = [p for p in _load_predictions()
                 if p.get("kind") == "sectors" and p.get("as_of")]
    if not picks_all:
        return None
    cache_dir = PROJECT_ROOT / "data_cache" / "bull_bear" / "cache"
    frames = []
    for p in cache_dir.glob("*sector*.parquet"):
        try:
            frames.append(pd.read_parquet(p))
        except Exception:
            continue
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["date"] = df["date"].astype(str)
    df = df[pd.to_datetime(df["date"], errors="coerce").notna()]
    dates = sorted(df["date"].unique())
    date_pos = {d: i for i, d in enumerate(dates)}

    rows, hit_hi = [], []
    for rec in picks_all[-days:]:
        as_of = str(rec.get("as_of", ""))[:8].replace("-", "")
        if as_of not in date_pos:
            continue
        future = dates[date_pos[as_of] + 1: date_pos[as_of] + 4]  # 之后3个交易日
        if not future:
            continue
        for pick in rec.get("picks") or []:
            sector = pick.get("sector")
            sub = df[(df["sector_name"] == sector) & (df["date"].isin(future))]
            if sub.empty or "main_force_net" not in sub.columns:
                continue
            actual_continue = float(sub["main_force_net"].sum()) > 0
            persistence = float(pick.get("persistence") or 0)
            hit = (persistence >= 0.5) == actual_continue if persistence >= 0.4 else None
            rows.append({"as_of": as_of, "sector": sector, "persistence": persistence,
                         "actual_continue": actual_continue})
            if persistence >= 0.6:
                hit_hi.append(1 if actual_continue else 0)

    if not rows:
        return None
    n = len(rows)
    hi_n = len(hit_hi)
    hi_rate = sum(hit_hi) / hi_n if hi_n else None
    return {
        "n": n, "high_persist_n": hi_n,
        "high_persist_continue_rate": round(hi_rate, 3) if hi_rate is not None else None,
        "mean_persistence": round(sum(r["persistence"] for r in rows) / n, 3),
        "note": "高持续性(≥0.6)组的后续3日真净流入兑现率",
    }
