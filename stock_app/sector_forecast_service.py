# -*- coding: utf-8 -*-
"""板块预测服务：每日 08:00 自动生成板块 Top10（GBM 主视角 + Jev 语义化次要视角）。

GBM 主视角：板块 6 维特征（涨跌/动量/量能/宽度/大盘方向/资金流）每日全历史重训。
Jev 语义化次要视角：板块行级语义标注（强流入/放量/宽/大盘向好）+ 类型化问题。
数据层：复用 jev 缓存（sector_index_daily.parquet 与指数/市场同源），与每日定时管线共享生命周期。
"""
from __future__ import annotations

import json
import threading
import traceback
from datetime import date as _date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = None


def _log():
    global logger
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    return logger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_DIR = PROJECT_ROOT / "data_cache" / "jev"
SECTOR_PATH = FORECAST_DIR / "sector_forecast_latest.json"
SECTOR_INDEX_PARQUET = FORECAST_DIR / "sector_index_daily.parquet"

_TASK: Dict = {"state": "idle", "message": "板块预测尚未运行", "started_at": None,
               "finished_at": None, "error": None, "progress": 0}
_SCHEDULER: Dict = {"enabled": False, "state": "stopped", "hour": 8, "minute": 0,
                    "check_seconds": 600, "last_run_date": None, "message": "板块预测定时器未启动"}
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


def latest_sector_forecast() -> Dict:
    if not SECTOR_PATH.exists():
        return {}
    try:
        return json.loads(SECTOR_PATH.read_text("utf-8"))
    except Exception:
        return {}


def _load_sector_df():
    import pandas as pd

    df = pd.read_parquet(SECTOR_INDEX_PARQUET)
    df["date"] = df["trade_date"].astype(str)
    df["chg"] = pd.to_numeric(df["pct_change"], errors="coerce")
    df["amt"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.sort_values(["ts_code", "date"])
    df["y"] = df.groupby("ts_code")["chg"].shift(-1).gt(0).astype(float)
    df["chg_5d"] = df.groupby("ts_code")["chg"].rolling(5).mean().reset_index(level=0, drop=True)
    df["amt_5v20"] = (df.groupby("ts_code")["amt"].rolling(5).mean().reset_index(level=0, drop=True)
                      / df.groupby("ts_code")["amt"].rolling(20).mean().reset_index(level=0, drop=True))
    df = df.dropna(subset=["chg", "y", "chg_5d", "amt_5v20"])
    return df


def _enrich(df):
    """补大盘方向/宽度/资金流（行业名映射 + 连续资金流维度）。"""
    from .jev_service import _load_index_closes
    idx_rows = _load_index_closes(force=True)
    closes = [c for _, c in idx_rows]
    days_idx = [d for d, _ in idx_rows]
    idx_map = {days_idx[i]: (closes[i + 1] > closes[i]) for i in range(len(closes) - 1)}
    df["idx_up"] = df["date"].map(idx_map).fillna(False).astype(float)
    ff_path = PROJECT_ROOT / "data_cache" / "bull_bear" / "cache" / "fund_flow_sector.parquet"
    if ff_path.exists():
        import pandas as pd
        ff = pd.read_parquet(ff_path)
        ff["date"] = ff["date"].astype(str)
        ff = ff.rename(columns={"sector_name": "name"})
        df = df.merge(ff[["date", "name", "main_force_net"]], on=["date", "name"], how="left")
        df["main_force_net"] = df["main_force_net"].fillna(0.0)
    # 连续资金流天数（v18 验证对 LLM 有增益的维度）
    df = df.sort_values(["name", "date"])
    df["flow_streak"] = df.groupby("name")["main_force_net"].transform(_flow_streak)
    # 板块宽度：申万一级名 → real_data Tushare 行业名映射；无对应时用全市场宽度替代
    import pandas as pd
    INDUSTRY_MAP = {
        "农林牧渔": "农业综合", "基础化工": "化工原料", "钢铁": "普钢", "有色金属": "小金属",
        "电子": "元器件", "家用电器": "家用电器", "食品饮料": "食品", "纺织服饰": "服饰",
        "轻工制造": "造纸", "医药生物": "化学制药", "公用事业": "电力", "交通运输": "水运",
        "房地产": "区域地产", "商贸零售": "商贸代理", "社会服务": "旅游服务", "综合": "农业综合",
        "建筑材料": "其他建材", "建筑装饰": "建筑工程", "电力设备": "电气设备", "国防军工": "航空",
        "计算机": "软件服务", "传媒": "互联网", "通信": "通信设备", "银行": "银行",
        "非银金融": "证券", "汽车": "汽车配件", "机械设备": "专用机械", "煤炭": "煤炭开采",
        "石油石化": "石油开采", "环保": "环境保护", "美容护理": "日用化工",
    }
    frames = []
    for fp in (PROJECT_ROOT / "data_cache").glob("real_data_*.parquet"):
        try:
            frames.append(pd.read_parquet(fp, columns=["trade_date", "industry", "close", "ts_code"]))
        except Exception:
            continue
    real = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not real.empty:
        real["date"] = real["trade_date"].astype(str)
        real = real.sort_values(["ts_code", "date"])
        real["prev"] = real.groupby("ts_code")["close"].shift(1)
        real = real[real["prev"].notna() & (real["prev"] > 0)]
        real["up"] = real["close"] > real["prev"]
        # 按日全市场宽度（兜底）
        mkt_width = real.groupby("date")["up"].mean().to_dict()
        df["_mkt_breadth"] = df["date"].map(mkt_width).fillna(0.5)
        # 按行业宽度（映射后的 real_data industry）
        real["_mapped"] = real["industry"].map({v: k for k, v in INDUSTRY_MAP.items()})
        width = (real[real["_mapped"].notna()]
                 .groupby(["date", "_mapped"])["up"].mean().reset_index()
                 .rename(columns={"_mapped": "name", "up": "sector_breadth"}))
        df = df.merge(width, on=["date", "name"], how="left")
        df["sector_breadth"] = df["sector_breadth"].fillna(df["_mkt_breadth"])
    else:
        df["sector_breadth"] = 0.5
    df = df.drop(columns=["_mkt_breadth"], errors="ignore")
    return df


def _semantic_sector(grp, idx_chg: float):
    import pandas as pd
    out = grp.copy()
    out["资金语义"] = pd.cut(out["main_force_net"].fillna(0),
                           bins=[-1e20, -1e8, 1e8, 1e20],
                           labels=["净流出", "弱流入", "强流入"])
    out["连续语义"] = out["flow_streak"].fillna(0).apply(
        lambda x: f"连续{x:+.0f}天流入" if x > 0 else (f"连续{abs(x):+.0f}天流出" if x < 0 else "无连续")
    )
    out["涨跌语义"] = pd.cut(out["chg"].fillna(0),
                           bins=[-100, -1, 1, 100],
                           labels=["弱", "中", "强"])
    out["量能语义"] = pd.cut(out["amt_5v20"].fillna(1),
                           bins=[0, 0.9, 1.1, 99],
                           labels=["缩量", "中", "放量"])
    out["宽度语义"] = pd.cut(out["sector_breadth"].fillna(0.5),
                           bins=[0, 0.42, 0.58, 1],
                           labels=["窄", "中", "宽"])
    out["大盘语义"] = "向好" if idx_chg > 0 else "向差"
    return out


FEATS = ["chg", "chg_5d", "amt_5v20", "idx_up", "sector_breadth", "main_force_net", "flow_streak"]


def _flow_streak(series):
    """连续正/负资金流天数（v18 验证对 LLM 有增益的维度）。"""
    s = (series > 0).astype(int)
    blocks = (s.diff().fillna(0) != 0).cumsum()
    run = s.groupby(blocks).cumsum()
    return run * np.sign(series)


def run_sector_forecast(top_n: int = 10, trigger: str = "manual") -> Dict:
    task = get_task()
    if task.get("state") == "running":
        return task
    _set_task(state="running", message="构建板块特征", progress=10, trigger=trigger,
              started_at=datetime.now().isoformat(timespec="seconds"),
              finished_at=None, error=None)

    def target():
        try:
            _set_task(progress=25, message="加载板块历史与大盘数据")
            df = _load_sector_df()
            df = _enrich(df)
            days = sorted(df["date"].unique())
            latest = days[-1]
            train_dates = days[:-1]
            df_tr = df[df["date"].isin(train_dates)]
            grp_today = df[df["date"] == latest].copy()
            if grp_today.empty:
                raise RuntimeError("无最新板块数据")

            import xgboost as xgb
            _set_task(progress=50, message="GBM 板块模型重训与预测")
            m = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08,
                                  subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", verbosity=0)
            m.fit(df_tr[FEATS].fillna(0).values, df_tr["y"].values)
            grp_today["p_up"] = m.predict_proba(grp_today[FEATS].fillna(0).values)[:, 1]
            gbm_top = grp_today.nlargest(top_n, "p_up")
            gbm_picks = [
                {"sector": r["name"], "p_up": round(float(r["p_up"]), 4),
                 "chg": round(float(r["chg"]), 2),
                 "mf_yi": round(float(r["main_force_net"]) / 1e8, 2)}
                for _, r in gbm_top.iterrows()
            ]

            _set_task(progress=75, message="Jev 语义化板块次要视角")
            from .jev_service import (_load_index_closes, _systemone_ask, _clamp01,
                                      yiji_context)
            from .jev_service import _load_index_closes as _lic
            idx_rows = _lic(force=True)
            closes = [c for _, c in idx_rows]
            days_idx = [d for _, in idx_rows] if False else [d for d, _ in idx_rows]
            idx_chg = 0.0
            if latest in days_idx:
                i = days_idx.index(latest)
                if i > 0:
                    idx_chg = (closes[i] / closes[i - 1] - 1) * 100
            sem = _semantic_sector(grp_today.nlargest(12, "main_force_net"), idx_chg)
            yiji = yiji_context(_date.fromisoformat(latest[:4] + "-" + latest[4:6] + "-" + latest[6:]))
            sectors = sem["name"].tolist()
            ctx = sem[["name", "chg", "chg_5d", "amt_5v20", "sector_breadth", "flow_streak",
                       "资金语义", "连续语义", "涨跌语义", "量能语义", "宽度语义", "大盘语义"]].to_dict("records")
            jev_picks = []
            try:
                questions = {}
                for s in sectors:
                    questions[f"up::{s}"] = {"type": "choice",
                                              "question": f"Will sector {s} CLOSE UP tomorrow? (semantic hints)",
                                              "criteria": {"up": "up", "down": "down"}}
                answers = _systemone_ask(
                    {"sectors_semantic": ctx, "yiji": yiji, "target_date": latest}, questions)
                if answers:
                    scored = []
                    for s in sectors:
                        a = answers.get(f"up::{s}")
                        if a:
                            probs = a.get("probabilities") or {}
                            p = _clamp01(probs.get("up", 0.0))
                            if a.get("choice") == "up" or p >= 0.5:
                                scored.append({"sector": s, "p_up": round(p, 4)})
                    jev_picks = sorted(scored, key=lambda x: -x["p_up"])[:top_n]
            except Exception as exc:
                _log().warning(f"Jev 板块次要视角失败: {exc}")

            _set_task(progress=95, message="落盘")
            out = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "target_date": latest,
                "gbm_picks": gbm_picks,
                "jev_picks": jev_picks,
            }
            FORECAST_DIR.mkdir(parents=True, exist_ok=True)
            SECTOR_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
            _set_task(state="completed", progress=100,
                      message=f"板块预测完成: {latest} GBM Top{top_n} / Jev {len(jev_picks)} 个",
                      finished_at=datetime.now().isoformat(timespec="seconds"))
        except BaseException as exc:
            _set_task(state="failed", message=f"板块预测失败: {type(exc).__name__}",
                      finished_at=datetime.now().isoformat(timespec="seconds"),
                      error=f"{exc}\n{traceback.format_exc()}")

    threading.Thread(target=target, daemon=True, name="sector_forecast").start()
    return get_task()


def start_sector_forecast_scheduler(hour: int = 8, minute: int = 0, check_seconds: int = 600) -> Dict:
    """每日 hour:minute 自动运行板块预测。"""
    global _STARTED
    with _LOCK:
        if _STARTED:
            return get_scheduler()
        _STARTED = True
    _set_sched(enabled=True, state="waiting", hour=hour, minute=minute,
               check_seconds=check_seconds, message=f"板块预测定时器已启动（每日 {hour:02d}:{minute:02d}）")

    def loop():
        while True:
            _set_sched(last_check_at=datetime.now().isoformat(timespec="seconds"))
            now = datetime.now()
            last = get_scheduler().get("last_run_date")
            if (now.hour, now.minute) >= (hour, minute) and last != now.date().isoformat():
                _set_sched(message=f"到达计划时间，触发板块预测（{now:%Y-%m-%d %H:%M}）")
                run_sector_forecast(trigger="scheduled")
                _set_sched(last_run_date=now.date().isoformat())
            import time as _time
            _time.sleep(check_seconds)

    threading.Thread(target=loop, daemon=True, name="sector_forecast_scheduler").start()
    return get_scheduler()


def get_status() -> Dict:
    return {"task": get_task(), "scheduler": get_scheduler(), "latest": latest_sector_forecast()}
