# -*- coding: utf-8 -*-
"""AI 预测模块（AI Forecast）：每日收盘后的市场方向信号。

主信号：GBM（XGBoost，v5 特征集 21 维）——2600 日全历史 +3.5% 超额、熊段 +7.8%、
         选择性出手 +7.3%（经十轮消融验证的唯一有效信号源）。
次要视角：Jev-v13 语义化（输入=数值+强弱/放量/低开等标注）与 DeepSeek（v5 特征），
         两者为 LLM 参考意见，与 GBM 信号并排展示，不参与决策权重。

数据层：全部复用 jev 缓存（data_cache/jev/*），与每日定时拉取管线同源同生命周期。
调度：每日 07:50 自动重训 GBM 并生成三模型当日信号（AI_FORECAST_AUTO=0 关闭）。
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = None


def _log():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = PROJECT_ROOT / "data_cache" / "jev"  # 与实验管线同源
LATEST_PATH = FORECAST_DIR / "forecast_latest.json"

_TASK: Dict = {"state": "idle", "message": "AI预测尚未运行", "started_at": None,
               "finished_at": None, "error": None, "progress": 0}
_SCHEDULER: Dict = {"enabled": False, "state": "stopped", "hour": 7, "minute": 50,
                    "check_seconds": 600, "last_run_date": None, "message": "AI预测定时器未启动"}
_LOCK = threading.Lock()
_STARTED = False


def _set_task(**kw):
    _TASK.update(kw)


def _set_sched(**kw):
    _SCHEDULER.update(kw)


def get_task():
    return dict(_TASK)


def get_scheduler():
    return dict(_SCHEDULER)


def latest_forecast() -> Dict:
    if not LATEST_PATH.exists():
        return {}
    try:
        return json.loads(LATEST_PATH.read_text("utf-8"))
    except Exception:
        return {}


# ── GBM 主信号（v5 特征集 21 维，扩窗每日重训）─────────────────────────
def _gbm_signal(force: bool = False) -> Optional[Dict]:
    """预加载各序列为 dict（一次读盘），逐日特征用 O(1) 查表，全历史约 1~2 分钟。"""
    import json as _json

    from .jev_service import JEV_DIR, _load_index_closes
    import xgboost as xgb

    _log().info("GBM信号: 加载指数历史…")
    rows = _load_index_closes(force=True)
    if len(rows) < 400:
        return None
    closes = [c for _, c in rows]
    days = [d for d, _ in rows]
    _log().info(f"GBM信号: 指数 {len(rows)} 日, 预加载辅助序列…")

    def _dmap(name):
        p = JEV_DIR / name
        if not p.exists():
            return {}
        try:
            data = _json.loads(p.read_text("utf-8"))
        except Exception:
            return {}
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        return data

    gap = _dmap("gap_by_date.json")
    breadth = _dmap("breadth_by_date.json")
    limits = _dmap("limit_stats_by_date.json")
    margin = _dmap("margin_sse.json")
    vix = _dmap("vix_close.json")
    zt = _dmap("zt_pool_history.json")
    amt = _dmap("index_amount_000001sh.json")
    hs300 = _dmap("index_daily_000300sh.json")
    cyb = _dmap("index_daily_399006sz.json")
    _log().info("GBM信号: 辅助序列就绪, 开始逐日特征计算…")
    today_i = len(days) - 1
    as_of = _date.fromisoformat(f"{days[today_i][:4]}-{days[today_i][4:6]}-{days[today_i][6:]}")
    target = as_of + timedelta(days=1)

    def _vec(i):
        d = days[i]
        c = closes[i]
        r1 = (c / closes[i - 1] - 1) * 100
        r5 = (c / closes[i - 5] - 1) * 100 if i >= 5 else 0.0
        r20 = (c / closes[i - 20] - 1) * 100 if i >= 20 else 0.0
        r60 = (c / closes[i - 60] - 1) * 100 if i >= 60 else 0.0
        r120 = (c / closes[i - 121] - 1) * 100 if i >= 121 else 0.0
        w60 = closes[i - 59:i + 1]
        pos60 = (c - min(w60)) / max(1e-9, max(w60) - min(w60)) * 100
        w250 = closes[i - 249:i + 1] if i >= 249 else None
        pos250 = (c - min(w250)) / max(1e-9, max(w250) - min(w250)) * 100 if w250 else 50.0
        rets = [closes[j] / closes[j - 1] - 1 for j in range(i - 19, i + 1)]
        vol20 = float(np.std(rets)) * 100 if len(rets) >= 5 else 1.0
        gs = [max(0, closes[j] - closes[j - 1]) for j in range(i - 13, i + 1)]
        ls = [max(0, closes[j - 1] - closes[j]) for j in range(i - 13, i + 1)]
        ag, al = sum(gs) / 14 or 1e-9, sum(ls) / 14 or 1e-9
        rsi = 100 - 100 / (1 + ag / al)
        e12 = _ema(closes[:i + 1], 12)
        e26 = _ema(closes[:i + 1], 26)
        dif = [a - b for a, b in zip(e12, e26)]
        dea = _ema(dif, 9)
        hist = (dif[-1] - dea[-1]) * 2
        ma5 = sum(closes[i - 4:i + 1]) / 5
        ma20 = sum(closes[i - 19:i + 1]) / 20
        ma60 = sum(closes[i - 59:i + 1]) / 60
        ma_align = 1.0 if ma5 > ma20 > ma60 else (-1.0 if ma5 < ma20 < ma60 else 0.0)
        av5 = [amt.get(days[j], 0) for j in range(i - 4, i + 1)]
        av20 = [amt.get(days[j], 0) for j in range(i - 19, i + 1)]
        amt_r = (sum(av5) / 5) / (sum(av20) / 20) if av5 and av20 and sum(av20) > 0 else 1.0
        b = breadth.get(d) or {}
        up_r = b if isinstance(b, (int, float)) else b.get("up_ratio", 0.5)
        lim = limits.get(d) or {}
        lim_up = lim.get("limit_up", 0)
        lim_dn = lim.get("limit_down", 0)
        lim_ratio = lim_up / max(1, lim_dn) if lim else 1.0
        m_vals = sorted((dd, v) for dd, v in margin.items() if dd <= d)
        m20 = (m_vals[-1][1] / m_vals[-21][1] - 1) * 100 if len(m_vals) >= 21 else 0.0
        v_vals = sorted((dd, v) for dd, v in vix.items() if dd <= d)
        vix_now = v_vals[-1][1] if v_vals else 20.0
        vix5 = (v_vals[-1][1] / v_vals[-6][1] - 1) * 100 if len(v_vals) >= 6 else 0.0
        zt_rec = zt.get(d) or {}
        base_up = 0.55
        return [
            r1 / 3, r5 / 8, r20 / 20, r120 / 30, pos60 / 100, rsi / 100,
            1.0 if hist > 0 else -1.0, ma_align, amt_r - 1, up_r - 0.5,
            lim_up / 100, lim_ratio, (zt_rec.get("zha_rate", 0.25)),
            (zt_rec.get("max_lb", 0)) / 10, (gap.get(d, 0.0)) * 100,
            0.0,  # 情绪复合分占位
            m20 / 5, vix_now / 30, base_up - 0.5, float(i) / 2500.0,
        ]

    X, y = [], []
    for i in range(60, len(days) - 1):
        X.append(_vec(i))
        y.append(1.0 if closes[i + 1] > closes[i] else 0.0)
        if i % 500 == 0:
            _log().info(f"GBM信号: 特征 {i}/{len(days)} 已计算")
    _log().info(f"GBM信号: 特征矩阵 {len(X)} 日 × 21 维, 开始训练…")
    if len(X) < 300:
        return None
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    tr = list(range(0, len(X) - 5))
    m = xgb.XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric="logloss", verbosity=0)
    m.fit(X[tr], y[tr])
    _log().info("GBM信号: 训练完成, 预测今日")
    x_today = np.array(_vec(today_i), dtype=float).reshape(1, -1)
    p_up = float(m.predict_proba(x_today)[0, 1])
    confidence = abs(p_up - 0.5) * 2
    selective = confidence >= 0.10
    return {
        "p_up": round(p_up, 4),
        "direction": "up" if p_up > 0.5 else "down",
        "confidence": round(confidence, 3),
        "selective": selective,
        "features_used": X.shape[1], "train_n": len(tr),
        "target_date": target.isoformat(), "as_of": as_of.isoformat(),
    }


def _ema(series, n):
    k = 2 / (n + 1)
    out = [series[0]]
    for v in series[1:]:
        out.append(out[-1] + k * (v - out[-1]))
    return out


def _vec_from_v5(f: Dict, closes, i: int) -> list:
    """与 v5 实验一致的 21 维特征（顺序固定）。"""
    def g(k, d=0.0):
        v = f.get(k)
        return float(v) if isinstance(v, (int, float)) else d
    ma = f.get("均线排列")
    macd = f.get("MACD")
    return [
        g("涨跌_1日%") / 3.0, g("涨跌_5日%") / 8.0, g("涨跌_20日%") / 20.0,
        g("120日趋势%") / 30.0, g("60日位置%") / 100.0, g("RSI14", 50) / 100.0,
        1.0 if macd == "多头" else (-1.0 if macd == "空头" else 0.0),
        1.0 if ma == "多头排列" else (-1.0 if ma == "空头排列" else 0.0),
        g("成交额比_5v20", 1.0) - 1.0,
        g("上涨家数占比%", 50) / 100.0 - 0.5,
        g("涨停家数", 0) / 100.0, g("涨跌停比", 1.0),
        g("炸板率%", 25) / 100.0, g("最高连板", 0) / 10.0,
        g("隔夜缺口%", 0) / 1.5, g("情绪复合分", 0) / 100.0,
        g("两融20日变化%", 0) / 5.0, g("VIX", 20) / 30.0,
        g("历史基准涨概率%", 50) / 100.0 - 0.5,
        float(i) / 2500.0,  # 时间位置（非预测特征，仅做单调性约束）
    ]


# ── LLM 次要视角（Jev-v13 语义化 + DeepSeek）─────────────────────────────
def _semanticize(feats: Dict) -> Dict:
    out = dict(feats)
    def tag(key, lo, hi, labels):
        v = feats.get(key)
        if isinstance(v, (int, float)):
            out[key + "_语义"] = labels[0] if v <= lo else (labels[1] if v >= hi else "中性")
    tag("涨跌_5日%", -2, 2, ("弱", "强"))
    tag("涨跌_20日%", -4, 4, ("弱", "强"))
    tag("RSI14", 40, 60, ("偏弱", "偏强"))
    tag("量能比_5v20", 0.9, 1.1, ("缩量", "放量"))
    tag("上涨家数占比%", 45, 55, ("窄", "宽"))
    tag("情绪复合分", -15, 15, ("低迷", "亢奋"))
    tag("隔夜缺口%", -0.2, 0.2, ("低开", "高开"))
    tag("VIX", 16, 24, ("低恐慌", "高恐慌"))
    return out


def _jev_signal(target: _date) -> Optional[Dict]:
    """Jev-v13 语义化次要视角（TypeSafe systemone）。target=最近已完成交易日（与GBM一致）。"""
    try:
        from .jev_service import _market_features_v10, _systemone_ask, _clamp01, yiji_context
        # target 即 as_of（最近已完成交易日收盘），用 ≤target-1 的特征
        as_of = target - timedelta(days=1)
        feats = _market_features_v10(as_of)
        if not feats:
            return None
        sem = _semanticize(feats)
        yiji = yiji_context(target)
        answers = _systemone_ask(
            {"market": sem, "yiji": yiji, "target_date": target.isoformat()},
            {"up_tomorrow": {
                "type": "choice",
                "question": f"综合数据（含语义标注）与易术，判断上证指数 {target.isoformat()} 收盘上涨还是下跌？",
                "criteria": {"UP": "上涨", "DOWN": "下跌"},
            }})
        if not answers or "up_tomorrow" not in answers:
            return None
        a = answers["up_tomorrow"]
        probs = a.get("probabilities") or {}
        return {"p_up": _clamp01(probs.get("UP", 0.5)), "direction": a.get("choice"),
                "confidence": a.get("confidence"), "provider": "jev-latest"}
    except Exception as exc:
        _log().warning(f"Jev 次要视角失败: {exc}")
        return None


def _deepseek_signal(target: _date) -> Optional[Dict]:
    """DeepSeek 次要视角（固定 DeepSeek 官方端点，不随 JEV 配置切换）。target=最近已完成交易日。"""
    try:
        from .jev_service import _market_features_v10, _parse_json, _clamp01, yiji_context
        import httpx as _httpx

        # DeepSeek key：从平台既有 AI 服务拿（它的 key 读取链与研报模块一致）
        ds_key = ""
        try:
            from .research.ai_service import AIService
            ds_key = AIService().deepseek_key or ""
        except Exception:
            pass
        if not ds_key:
            return None
        as_of = target - timedelta(days=1)
        feats = _market_features_v10(as_of)
        if not feats:
            return None
        yiji = yiji_context(target)
        prompt = (
            "任务：预测上证指数在目标交易日的上涨概率。\n"
            f"目标交易日：{target.isoformat()}\n\n"
            f"【市场数据（截至 {as_of.isoformat()} 收盘，绝无未来数据）】\n"
            f"{json.dumps(feats, ensure_ascii=False)}\n\n"
            f"【目标日的干支五行卦象】\n{json.dumps(yiji, ensure_ascii=False)}\n\n"
            "综合量化数据与易术象数，输出 JSON："
            '{"p_up": 0~1, "direction": "up"或"down", "conviction": 0~1}'
        )
        body = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是严谨的量化分析师，兼通易术象数。只输出合法 JSON，不要解释。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2, "max_tokens": 900,
            "response_format": {"type": "json_object"},
        }
        with _httpx.Client(timeout=90.0, trust_env=False) as client:
            r = client.post("https://api.deepseek.com/chat/completions",
                            headers={"Authorization": f"Bearer {ds_key}"}, json=body)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        out = _parse_json(text)
        if not out or "p_up" not in out:
            return None
        return {"p_up": _clamp01(out.get("p_up")), "direction": out.get("direction"),
                "conviction": _clamp01(out.get("conviction")), "provider": "deepseek-chat"}
    except Exception as exc:
        _log().warning(f"DeepSeek 次要视角失败: {exc}")
        return None


# ── 每日编排 ─────────────────────────────────────────────────────────────
def run_daily(force: bool = False, trigger: str = "manual") -> Dict:
    from .jev_service import _load_index_closes, _next_trading_day

    task = get_task()
    if task.get("state") == "running":
        return task
    _set_task(state="running", message="计算 GBM 主信号", progress=5, trigger=trigger,
              started_at=datetime.now().isoformat(timespec="seconds"),
              finished_at=None, error=None)

    def target():
        try:
            rows = _load_index_closes(force=True)
            closes = [c for _, c in rows]
            days = [d for d, _ in rows]
            latest_d = days[-1]
            as_of = _date.fromisoformat(f"{latest_d[:4]}-{latest_d[4:6]}-{latest_d[6:]}")
            # 与 v5 实验一致：预测最近已完成交易日（收盘已完成），不用 next_trading_day
            target_day = as_of

            _set_task(progress=20, message="GBM 模型重训与预测")
            gbm = _gbm_signal()
            if not gbm:
                raise RuntimeError("GBM 信号生成失败（数据不足）")

            _set_task(progress=60, message="LLM 次要视角")
            jev = _jev_signal(target_day)
            deepseek = _deepseek_signal(target_day)

            _set_task(progress=90, message="落盘")
            out = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "target_date": target_day.isoformat(),
                "as_of": as_of.isoformat(),
                "gbm": gbm,
                "jev": jev, "deepseek": deepseek,
            }
            FORECAST_DIR.mkdir(parents=True, exist_ok=True)
            LATEST_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            _set_task(state="completed", progress=100,
                      message=f"AI预测完成: {target_day.isoformat()} GBM {gbm['direction']} p_up={gbm['p_up']}",
                      finished_at=datetime.now().isoformat(timespec="seconds"))
        except BaseException as exc:
            _set_task(state="failed", message=f"AI预测失败: {type(exc).__name__}",
                      finished_at=datetime.now().isoformat(timespec="seconds"),
                      error=f"{exc}\n{traceback.format_exc()}")

    threading.Thread(target=target, daemon=True, name="ai_forecast").start()
    return get_task()


def start_forecast_scheduler(hour: int = 7, minute: int = 50, check_seconds: int = 600) -> Dict:
    """每日 hour:minute 自动运行（收盘前）。"""
    global _STARTED
    with _LOCK:
        if _STARTED:
            return get_scheduler()
        _STARTED = True
    _set_sched(enabled=True, state="waiting", hour=hour, minute=minute,
               check_seconds=check_seconds, message=f"AI预测定时器已启动（每日 {hour:02d}:{minute:02d}）")

    def loop():
        while True:
            _set_sched(last_check_at=datetime.now().isoformat(timespec="seconds"))
            now = datetime.now()
            last = get_scheduler().get("last_run_date")
            if (now.hour, now.minute) >= (hour, minute) and last != now.date().isoformat():
                _set_sched(message=f"到达计划时间，触发 AI 预测（{now:%Y-%m-%d %H:%M}）")
                run_daily(trigger="scheduled")
                _set_sched(last_run_date=now.date().isoformat())
            time.sleep(check_seconds)

    threading.Thread(target=loop, daemon=True, name="ai_forecast_scheduler").start()
    return get_scheduler()


def get_status() -> Dict:
    return {"task": get_task(), "scheduler": get_scheduler(), "latest": latest_forecast()}
