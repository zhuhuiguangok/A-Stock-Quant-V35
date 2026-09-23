"""Scheduled weekly model retraining on top of locally cached data.

默认每周六 09:00 用 data_cache 下最新/最大的 real_data parquet 重训全部模型
（XGBoost / AI 引擎 / 双树模型）。训练直接在服务进程内的引擎单例上进行，
完成后无需重启即可生效，产物同时落盘 models/ 供冷启动加载。

环境变量：
    MODEL_RETRAIN_AUTO            默认 1，置 0/false 关闭
    MODEL_RETRAIN_WEEKDAY         默认 sat（支持 mon..sun 或 0-6）
    MODEL_RETRAIN_HOUR            默认 9（到达该小时后的首次检查即触发）
    MODEL_RETRAIN_CHECK_SECONDS   默认 600（检查间隔）
"""
from __future__ import annotations

import glob
import os
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data_cache"
MODELS_DIR = PROJECT_ROOT / "models"
RETRAIN_LOG = PROJECT_ROOT / "logs" / "retrain.log"


def _tlog(message: str) -> None:
    """直写日志文件（绕开 logging 配置黑洞，训练线程的可观测性保险）。"""
    try:
        RETRAIN_LOG.parent.mkdir(exist_ok=True)
        with open(RETRAIN_LOG, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now().isoformat(timespec='seconds')} {message}\n")
    except OSError:
        pass

_WEEKDAY_NAMES = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}

_LOCK = threading.Lock()
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER_STARTED = False
_TASK: Dict = {
    "state": "idle",
    "message": "模型重训尚未运行",
    "trigger": None,
    "data_file": None,
    "rows": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}
_SCHEDULER: Dict = {
    "enabled": False,
    "state": "stopped",
    "weekday": 5,
    "hour": 9,
    "check_seconds": 600,
    "last_check_at": None,
    "last_run_date": None,
    "next_check_at": None,
    "message": "模型定时重训未启动",
}


def parse_weekday(value, default: int = 5) -> int:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _WEEKDAY_NAMES:
        return _WEEKDAY_NAMES[text]
    try:
        num = int(text)
        return num if 0 <= num <= 6 else default
    except ValueError:
        return default


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _set_task(**kwargs) -> None:
    with _LOCK:
        _TASK.update(kwargs)


def _set_scheduler(**kwargs) -> None:
    with _SCHEDULER_LOCK:
        _SCHEDULER.update(kwargs)


def get_task() -> Dict:
    with _LOCK:
        return dict(_TASK)


def get_scheduler_status() -> Dict:
    with _SCHEDULER_LOCK:
        return dict(_SCHEDULER)


def get_retrain_status() -> Dict:
    return {"task": get_task(), "scheduler": get_scheduler_status()}


def _pick_best_local_data() -> Path:
    """自动选择行数最多、最新的 real_data parquet（与 retrain_with_local_data.py 同策略）。"""
    cands = glob.glob(str(CACHE_DIR / "real_data_*.parquet"))
    if not cands:
        raise FileNotFoundError("data_cache 下没有 real_data_*.parquet，无法训练")
    best, best_sz = None, -1
    for p in cands:
        try:
            sz = os.path.getsize(p)
            if "real_data_all_" in p:
                sz *= 2  # 优先全市场数据
            if sz > best_sz:
                best, best_sz = p, sz
        except OSError:
            pass
    if best is None:
        raise FileNotFoundError("data_cache 下的 real_data parquet 均不可读")
    return Path(best)


def _load_combined_training_data():
    """合并 data_cache 下全部 real_data parquet 作为训练集，让模型吃到最长历史周期。

    各时期缓存列可能不一致（股票池/版本差异），这里取公共列交集再拼接；
    按 (股票, 日期) 去重后排序。返回 (DataFrame, 用到的文件名列表)。
    可用 MODEL_RETRAIN_ALL_DATA=0 关闭，回退为单文件策略 _pick_best_local_data。
    """
    import pandas as pd

    if os.environ.get("MODEL_RETRAIN_ALL_DATA", "1").strip().lower() in ("0", "false", "no"):
        best = _pick_best_local_data()
        return pd.read_parquet(best), [best.name]

    paths = sorted(glob.glob(str(CACHE_DIR / "real_data_*.parquet")))
    if not paths:
        raise FileNotFoundError("data_cache 下没有 real_data_*.parquet，无法训练")
    frames, used = [], []
    for p in paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
        used.append(Path(p).name)
    if not frames:
        raise FileNotFoundError("data_cache 下的 real_data parquet 均不可读")

    if len(frames) == 1:
        return frames[0], used

    common = set.intersection(*(set(f.columns) for f in frames))
    keep = [c for c in frames[0].columns if c in common]
    merged = pd.concat([f[keep] for f in frames], ignore_index=True)
    key_cols = [c for c in ("ts_code", "trade_date", "code", "date") if c in merged.columns]
    before = len(merged)
    if key_cols:
        merged = merged.drop_duplicates(subset=key_cols, keep="last").sort_values(key_cols)
    merged = merged.reset_index(drop=True)
    _set_task(message=f"已合并 {len(used)} 份历史缓存（{before:,} 行去重后 {len(merged):,} 行）")
    return merged, used


def _data_span(df) -> str:
    for col in ("trade_date", "date"):
        if col in df.columns:
            s = df[col].astype(str)
            return f"{s.min()} ~ {s.max()}"
    return "跨度未知"


def _model_artifacts() -> Dict:
    out = {}
    for name in ("trend.pkl", "bottom.pkl", "ai_engine_mlp.pth", "ai_engine_smart_xgnn_xgb.pkl"):
        path = MODELS_DIR / name
        if path.exists():
            out[name] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    return out


def _stratified_sample(df, max_rows: int):
    """按年份分层等比抽样，保留牛熊周期各阶段占比，防止大内存 OOM。

    max_rows<=0 表示不限制。抽样保持行序（时间序），随机种子固定可复现。
    返回 (抽样后的 df, 抽样说明文本)。
    """
    import pandas as pd

    if max_rows <= 0 or len(df) <= max_rows:
        return df, f"全量 {len(df):,} 行"

    date_col = "trade_date" if "trade_date" in df.columns else ("date" if "date" in df.columns else None)
    if date_col is not None:
        year_key = df[date_col].astype(str).str[:4]
        frac = max_rows / len(df)
        parts = []
        for _, part in df.groupby(year_key, sort=False):
            take = part.sample(n=max(1, min(len(part), round(len(part) * frac))), random_state=42)
            parts.append(take.sort_index())
        sampled = pd.concat(parts, ignore_index=True)
    else:
        sampled = df.sample(n=max_rows, random_state=42).sort_index().reset_index(drop=True)
    note = f"{len(df):,} 行按年分层抽样至 {len(sampled):,} 行（保牛熊各期占比）"
    return sampled, note


def _prune_stocks_for_memory(df, max_raw_rows: int):
    """因子计算前的内存闸门：超出上限时按"整只股票"随机剔除。

    必须在 calculate_rule_factors 之前调用——技术因子依赖个股连续时序
    （MA/回看窗口），不能按行抽；丢整只股票既控内存又不破坏因子正确性。
    返回 (df, 说明文本)。max_raw_rows<=0 表示不限制。
    """
    import pandas as pd

    if max_raw_rows <= 0 or len(df) <= max_raw_rows or "ts_code" not in df.columns:
        return df, "未触发"
    codes = df["ts_code"].drop_duplicates()
    keep_frac = max(0.1, max_raw_rows / len(df))
    keep = set(codes.sample(frac=keep_frac, random_state=42))
    pruned = df[df["ts_code"].isin(keep)].reset_index(drop=True)
    note = f"股票级闸门: {len(codes)} 只保留 {len(keep)} 只（{len(df):,}→{len(pruned):,} 行）"
    return pruned, note


def _run_retrain() -> Dict:
    import gc

    import pandas as pd

    from .factor_engine import calculate_rule_factors
    from .views import v19_enhanced_engine

    df, used_files = _load_combined_training_data()
    span = _data_span(df)
    _tlog(f"开始重训: {len(used_files)} 份缓存, {len(df):,} 行, 跨度 {span}")
    _set_task(
        message=f"读取训练数据 {len(used_files)} 份缓存，历史跨度 {span}",
        data_file=" + ".join(used_files[:3]) + (" 等" if len(used_files) > 3 else ""),
    )
    _tlog("计算技术因子…")
    _set_task(message=f"计算技术因子（{len(df):,} 行，跨度 {span}）", rows=len(df))
    max_raw = int(os.environ.get("MODEL_RETRAIN_MAX_RAW_ROWS", "2000000") or 0)
    df, prune_note = _prune_stocks_for_memory(df, max_raw)
    if prune_note != "未触发":
        _tlog(prune_note)
    df = calculate_rule_factors(df)
    _tlog(f"因子完成: {df.shape[1]} 列")

    max_rows = int(os.environ.get("MODEL_RETRAIN_MAX_ROWS", "800000") or 0)
    df, sample_note = _stratified_sample(df, max_rows)
    df = df.reset_index(drop=True)
    gc.collect()
    _tlog(f"采样后提交训练: {sample_note}, {df.shape[1]} 列")
    _set_task(message=f"训练全部模型（XGBoost / AI引擎 / 双树模型，{sample_note}）", rows=len(df))
    trained = v19_enhanced_engine.train_all_models(df)
    _tlog(f"train_all_models 返回 trained={trained}")
    return {
        "trained": bool(trained),
        "data_file": " + ".join(used_files),
        "span": span,
        "rows": len(df),
        "sample_note": sample_note,
        "artifacts": _model_artifacts(),
    }


def start_retrain(trigger: str = "manual") -> Dict:
    """后台线程执行重训；已在运行时直接返回当前任务状态。"""
    task = get_task()
    if task.get("state") == "running":
        return task

    _set_task(
        state="running",
        trigger=trigger,
        message="准备重训模型",
        started_at=_iso_now(),
        finished_at=None,
        error=None,
    )

    def target() -> None:
        _tlog(f"训练线程启动 (trigger={trigger})")
        try:
            result = _run_retrain()
            _set_task(
                state="completed",
                message=(
                    f"重训完成: {result.get('rows', 0):,} 行（{result.get('sample_note', '')}）/ "
                    f"{result.get('span', '跨度未知')} / trained={result['trained']}"
                ),
                finished_at=_iso_now(),
                error=None,
            )
            _tlog(f"重训完成: trained={result['trained']}")
        except BaseException as exc:  # 训练线程的兜底：SystemExit/segfault包装等 Exception 接不住的也要落状态
            _tlog(f"重训异常: {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
            _set_task(
                state="failed",
                message=f"模型重训失败: {type(exc).__name__}: {exc}",
                finished_at=_iso_now(),
                error=f"{exc}\n{traceback.format_exc()}",
            )
        finally:
            # 状态保险：任何逃逸路径都不允许把任务永远留在 running。
            # 故意不依赖 _set_task（它自身出错时保险必须仍然生效），直接改字典。
            if get_task().get("state") == "running":
                _tlog("finally 保险触发: 线程结束但状态仍为 running，强制置 failed")
                try:
                    _TASK.update({
                        "state": "failed",
                        "message": "模型重训失败: 训练线程异常终止",
                        "finished_at": _iso_now(),
                        "error": "training thread terminated while state=running",
                    })
                except Exception:
                    pass

    threading.Thread(target=target, daemon=True, name="model_retrain").start()
    return get_task()


def _should_run_now(now: datetime, weekday: int, hour: int, last_run_date: Optional[str]) -> bool:
    """到达设定星期且过了设定小时、且今天还没跑过时触发。"""
    return (
        now.weekday() == weekday
        and now.hour >= hour
        and last_run_date != now.date().isoformat()
    )


def start_model_retrain_scheduler(weekday: int = 5, hour: int = 9, check_seconds: int = 600) -> Dict:
    """每周定时重训调度器，每个 Django 进程只启动一次。"""
    global _SCHEDULER_STARTED
    weekday = min(6, max(0, int(weekday)))
    hour = min(23, max(0, int(hour)))
    check_seconds = max(60, int(check_seconds or 600))

    with _SCHEDULER_LOCK:
        if _SCHEDULER_STARTED:
            _SCHEDULER.update({"enabled": True, "message": "模型定时重训调度器已在运行"})
            return dict(_SCHEDULER)
        _SCHEDULER_STARTED = True
        _SCHEDULER.update({
            "enabled": True,
            "state": "waiting",
            "weekday": weekday,
            "hour": hour,
            "check_seconds": check_seconds,
            "message": f"模型定时重训已启动，每周{_WEEKDAY_NAMES_REV[weekday]} {hour:02d}:00 后触发",
        })

    def loop() -> None:
        while True:
            next_at = (datetime.now() + timedelta(seconds=check_seconds)).isoformat(timespec="seconds")
            _set_scheduler(state="waiting", last_check_at=_iso_now(), next_check_at=next_at)
            time.sleep(check_seconds)
            now = datetime.now()
            status = get_scheduler_status()
            if not _should_run_now(now, weekday, hour, status.get("last_run_date")):
                continue
            if get_task().get("state") == "running":
                continue
            _set_scheduler(last_run_date=now.date().isoformat(), message=f"到达计划时间，触发重训（{now:%Y-%m-%d %H:%M}）")
            start_retrain(trigger="scheduled")

    threading.Thread(target=loop, daemon=True, name="model_retrain_scheduler").start()
    return get_scheduler_status()


_WEEKDAY_NAMES_REV = {v: k for k, v in _WEEKDAY_NAMES.items()}
