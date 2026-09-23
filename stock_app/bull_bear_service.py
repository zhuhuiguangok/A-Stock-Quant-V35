"""Bull/Bear cycle dashboard integration.

First version: reuse the local ``stock-bull-bear-tushare`` analysis package as
the calculation engine, but keep all generated data under this Django project.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import traceback
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BULL_BEAR_SOURCE_ROOT = Path(os.environ.get(
    "BULL_BEAR_SOURCE_ROOT",
    r"D:\codex-workspace\stock-bull-bear-tushare",
))
_NESTED_BULL_BEAR_SOURCE_ROOT = BULL_BEAR_SOURCE_ROOT / "stock-bull-bear-tushare"
if not (BULL_BEAR_SOURCE_ROOT / "app").exists() and (_NESTED_BULL_BEAR_SOURCE_ROOT / "app").exists():
    BULL_BEAR_SOURCE_ROOT = _NESTED_BULL_BEAR_SOURCE_ROOT
BULL_BEAR_DATA_ROOT = PROJECT_ROOT / "data_cache" / "bull_bear"
BULL_BEAR_SOURCE_DATA_ROOT = Path(os.environ.get(
    "BULL_BEAR_SOURCE_DATA_ROOT",
    str(BULL_BEAR_SOURCE_ROOT / "data"),
))

_LOCK = threading.Lock()
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER_STARTED = False
# 每个 "今天:期望交易日" 组合的失效重试次数，避免节假日每小时空跑 Tushare
_STALE_ATTEMPTS: Dict[str, int] = {}
_MAX_STALE_ATTEMPTS = int(os.environ.get("BULL_BEAR_AUTO_UPDATE_MAX_STALE_ATTEMPTS", "3"))
_TASK: Dict = {
    "state": "idle",
    "message": "尚未启动",
    "stage": "idle",
    "progress": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}
_SCHEDULER: Dict = {
    "enabled": False,
    "state": "stopped",
    "interval_seconds": 3600,
    "initial_delay_seconds": 60,
    "last_check_at": None,
    "last_started_at": None,
    "last_skip_reason": None,
    "next_check_at": None,
    "message": "自动联网补数未启动",
}


def _ensure_source_importable() -> None:
    if not BULL_BEAR_SOURCE_ROOT.exists():
        raise FileNotFoundError(f"未找到牛熊周期工程: {BULL_BEAR_SOURCE_ROOT}")
    root = str(BULL_BEAR_SOURCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _set_task(**kwargs) -> None:
    with _LOCK:
        _TASK.update(kwargs)


def get_task() -> Dict:
    with _LOCK:
        return dict(_TASK)


def _set_scheduler(**kwargs) -> None:
    with _SCHEDULER_LOCK:
        _SCHEDULER.update(kwargs)


def get_scheduler_status() -> Dict:
    with _SCHEDULER_LOCK:
        return dict(_SCHEDULER)


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _iso_after(seconds: int) -> str:
    return (datetime.now() + timedelta(seconds=max(0, int(seconds)))).isoformat(timespec="seconds")


def _latest_completed_trade_date(now: datetime | None = None) -> str:
    """返回"数据应当已齐全"的最近一个交易日（仅按周末推断，节假日靠重试上限兜底）。

    15:45 之后当天 Tushare 日线/资金流基本完整；盘中与早盘期望上一个交易日。
    """
    now = now or datetime.now()
    day = now.date()
    if now.weekday() < 5 and (now.hour, now.minute) >= (15, 45):
        candidate = day
    else:
        candidate = day - timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate.strftime("%Y%m%d")


def _cached_data_fresh(expected: str) -> tuple:
    """检查本地缓存（股票日线 + 资金流）是否已覆盖期望交易日。"""
    stale = []
    for name in ("stocks", "fund_flow_sector"):
        meta = _read_json(BULL_BEAR_DATA_ROOT / "cache" / f"{name}.meta.json")
        end = str(meta.get("end") or "")
        if not end or end < expected:
            stale.append(f"{name}={end or '无'}")
    if stale:
        return False, f"期望>={expected}，落后: {', '.join(stale)}"
    return True, f"stocks/fund_flow 均已覆盖最新交易日 {expected}"


def _copy_file_if_needed(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if dst.exists() and dst.stat().st_size == src.stat().st_size and int(dst.stat().st_mtime) >= int(src.stat().st_mtime):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def bootstrap_from_source_data(force: bool = False, include_cache: bool = False) -> Dict:
    """Import existing stock-bull-bear-tushare data into this Django project.

    This keeps the module usable immediately while the heavier incremental update
    remains available from the UI.
    """
    if not BULL_BEAR_SOURCE_DATA_ROOT.exists():
        return {"copied": 0, "message": f"未找到源数据目录: {BULL_BEAR_SOURCE_DATA_ROOT}"}

    signal_dst = BULL_BEAR_DATA_ROOT / "results" / "cycle_signal_latest.json"
    cache_dst = BULL_BEAR_DATA_ROOT / "cache" / "stocks.parquet"
    if signal_dst.exists() and (cache_dst.exists() or not include_cache) and not force:
        return {"copied": 0, "message": "主工程已有牛熊周期数据"}

    copied = 0
    subdirs = ["results", "processed", "pic"] + (["cache"] if include_cache else [])
    for subdir in subdirs:
        src_dir = BULL_BEAR_SOURCE_DATA_ROOT / subdir
        if not src_dir.exists():
            continue
        patterns = ["*.json", "*.parquet", "*.png", "*.csv"] if subdir == "cache" else ["*"]
        for pattern in patterns:
            for src in src_dir.glob(pattern):
                if _copy_file_if_needed(src, BULL_BEAR_DATA_ROOT / subdir / src.name):
                    copied += 1

    return {"copied": copied, "message": f"已导入 {copied} 个牛熊周期数据文件"}


def _run_update(token: str) -> Dict:
    _ensure_source_importable()

    # The custom Tushare endpoint is plain HTTP; avoid system proxy interference.
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["TUSHARE_TOKEN"] = token

    from app.config import Settings
    from app.data_sources.akshare_client import AkShareClient
    from app.data_sources.cache import ParquetCache
    from app.data_sources.market_builder import MarketDataBuilder
    from app.data_sources.tushare_client import create_pro_client
    from app.pipeline import AnalysisPipeline

    _set_task(stage="bootstrap", progress=3, message="检查/导入已有牛熊周期缓存")
    bootstrap_from_source_data(force=False, include_cache=True)

    settings = Settings(data_root=BULL_BEAR_DATA_ROOT)
    settings.ensure_dirs()
    _set_task(stage="client", progress=6, message="初始化 Tushare / AkShare 数据源")
    pro = create_pro_client(token, settings.tushare_http_url, timeout=settings.tushare_timeout)
    builder = MarketDataBuilder(
        pro,
        ParquetCache(settings.cache_dir),
        settings.start_date,
        AkShareClient(),
        workers=settings.tushare_workers,
    )

    def progress(i: int, total: int, date: str) -> None:
        if total:
            pct = 8 + int(min(72, (i / total) * 72))
            _set_task(stage="fetch_stocks", progress=pct, message=f"拉取交易日 {date} ({i}/{total})")
        else:
            _set_task(stage="fetch_stocks", progress=76, message="股票日线缓存已是最新")

    _set_task(stage="fetch", progress=8, message="拉取/增量补齐股票与指数数据")
    frames, sources = builder.build_all(progress=progress)
    _set_task(stage="calculate", progress=84, message="计算牛熊周期指标与图表")
    result = AnalysisPipeline(data_root=BULL_BEAR_DATA_ROOT).run(frames, sources)
    _set_task(stage="finalize", progress=98, message="整理牛熊周期报告")
    return result


def _run_refresh_from_cache() -> Dict:
    """Recalculate indicators/charts from local cache without network calls."""
    _ensure_source_importable()
    _set_task(stage="bootstrap", progress=5, message="检查/导入本地牛熊周期缓存")
    bootstrap_from_source_data(force=False, include_cache=True)

    from app.pipeline import AnalysisPipeline

    import pandas as pd

    cache_dir = BULL_BEAR_DATA_ROOT / "cache"
    required = ["stocks", "sh_index", "hs300_pe", "sz50_pe", "zz500_pe", "zz1000_pe", "bond"]
    frames = {}
    sources = {}
    for idx, name in enumerate(required, 1):
        path = cache_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"缺少牛熊周期缓存: {path}")
        _set_task(stage="load_cache", progress=5 + int(idx / len(required) * 35), message=f"读取缓存 {name}")
        frames[name] = pd.read_parquet(path)
        meta = _read_json(cache_dir / f"{name}.meta.json")
        sources[name] = meta.get("source", "本地缓存")

        sources[name] = meta.get("source", "本地缓存")
    result = AnalysisPipeline(data_root=BULL_BEAR_DATA_ROOT).run(frames, sources)
    _set_task(stage="finalize", progress=98, message="刷新牛熊周期图表和结果")
    return result


def _start_background(kind: str, runner, started_message: str) -> Dict:
    task = get_task()
    if task.get("state") == "running":
        return task

    now = datetime.now().isoformat(timespec="seconds")
    _set_task(state="running", stage="starting", progress=1, message=started_message, started_at=now, finished_at=None, error=None)

    def target() -> None:
        try:
            result = runner()
            _set_task(
                state="completed",
                stage="completed",
                progress=100,
                message=f"{kind}完成: {result.get('cycle_score')} / {result.get('phase')}",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=None,
            )
        except Exception as exc:  # pragma: no cover - details are surfaced through API
            _set_task(
                state="failed",
                stage="failed",
                message=f"{kind}失败",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=f"{exc}\n{traceback.format_exc()}",
            )

    threading.Thread(target=target, daemon=True, name=f"bull_bear_{kind}").start()
    return get_task()


def start_refresh_from_cache() -> Dict:
    return _start_background("本地刷新", _run_refresh_from_cache, "准备基于本地缓存刷新牛熊周期")


def start_update(token: str) -> Dict:
    if not token:
        raise ValueError("Tushare Token 不能为空")
    return _start_background("联网补数", lambda: _run_update(token), "准备联网增量更新牛熊周期数据")


def start_unified_update(token: str) -> Dict:
    """Start the unified network backfill for bull/bear and fund-flow data."""
    bull_bear_task = start_update(token)
    fund_flow_task = None
    fund_flow_error = None
    try:
        from .fund_flow_service import start_update as start_fund_flow_update

        fund_flow_task = start_fund_flow_update()
    except Exception as exc:  # pragma: no cover - returned to API/scheduler status
        fund_flow_error = str(exc)
    return {
        "bull_bear_task": bull_bear_task,
        "fund_flow_task": fund_flow_task,
        "fund_flow_error": fund_flow_error,
    }


def start_auto_update_scheduler(token_provider, interval_seconds: int = 3600, initial_delay_seconds: int = 60) -> Dict:
    """Start hourly automatic network update scheduler once per Django process."""
    global _SCHEDULER_STARTED
    interval_seconds = max(300, int(interval_seconds or 3600))
    initial_delay_seconds = max(0, int(initial_delay_seconds or 0))

    with _SCHEDULER_LOCK:
        if _SCHEDULER_STARTED:
            _SCHEDULER.update({
                "enabled": True,
                "interval_seconds": interval_seconds,
                "initial_delay_seconds": initial_delay_seconds,
                "message": "自动联网补数调度器已在运行",
            })
            return dict(_SCHEDULER)
        _SCHEDULER_STARTED = True
        _SCHEDULER.update({
            "enabled": True,
            "state": "waiting",
            "interval_seconds": interval_seconds,
            "initial_delay_seconds": initial_delay_seconds,
            "next_check_at": _iso_after(initial_delay_seconds),
            "message": f"自动联网补数已启动，每 {interval_seconds // 60} 分钟检查一次",
        })

    def loop() -> None:
        delay = initial_delay_seconds
        while True:
            _set_scheduler(state="waiting", next_check_at=_iso_after(delay))
            time.sleep(delay)
            now = _iso_now()
            _set_scheduler(state="checking", last_check_at=now, next_check_at=None, message="检查是否需要启动联网补数")

            task = get_task()
            if task.get("state") == "running":
                _set_scheduler(
                    state="waiting",
                    last_skip_reason=f"已有任务运行中: {task.get('stage')}",
                    next_check_at=_iso_after(interval_seconds),
                    message="已有牛熊周期任务运行中，本轮自动补数跳过",
                )
                delay = interval_seconds
                continue

            # 【新鲜度跳过】缓存已覆盖最新交易日就不联网，避免闭市时段白跑 Tushare
            expected = _latest_completed_trade_date()
            fresh, fresh_reason = _cached_data_fresh(expected)
            if fresh:
                _set_scheduler(
                    state="waiting",
                    last_skip_reason=fresh_reason,
                    next_check_at=_iso_after(interval_seconds),
                    message=f"数据已最新（{expected}），本轮跳过联网补数",
                )
                delay = interval_seconds
                continue

            # 【失效重试上限】节假日缓存永远等不到"新交易日"，每天最多尝试 N 次
            attempt_key = f"{datetime.now():%Y%m%d}:{expected}"
            attempts = _STALE_ATTEMPTS.get(attempt_key, 0)
            if attempts >= _MAX_STALE_ATTEMPTS:
                for old_key in [k for k in _STALE_ATTEMPTS if not k.startswith(f"{datetime.now():%Y%m%d}:")]:
                    _STALE_ATTEMPTS.pop(old_key, None)
                _set_scheduler(
                    state="waiting",
                    last_skip_reason=f"{fresh_reason}；今日已尝试 {attempts} 次",
                    next_check_at=_iso_after(interval_seconds),
                    message=f"今日已尝试 {attempts} 次仍未获取 {expected} 数据（可能节假日），明日再试",
                )
                delay = interval_seconds
                continue
            _STALE_ATTEMPTS[attempt_key] = attempts + 1

            try:
                token = token_provider() if callable(token_provider) else token_provider
                if not token:
                    raise ValueError("Tushare Token 为空")
                started = start_unified_update(token)
                bull_bear_task = started.get("bull_bear_task") or {}
                fund_flow_task = started.get("fund_flow_task") or {}
                fund_flow_msg = fund_flow_task.get("stage") or started.get("fund_flow_error") or "skipped"
                _set_scheduler(
                    state="waiting",
                    last_started_at=_iso_now(),
                    last_skip_reason=None,
                    next_check_at=_iso_after(interval_seconds),
                    message=f"已自动启动统一联网补数: {bull_bear_task.get('stage')} / fund_flow={fund_flow_msg}",
                )
            except Exception as exc:
                _set_scheduler(
                    state="error",
                    last_skip_reason=str(exc),
                    next_check_at=_iso_after(interval_seconds),
                    message=f"自动联网补数启动失败: {exc}",
                )

            delay = interval_seconds

    threading.Thread(target=loop, daemon=True, name="bull_bear_auto_update_scheduler").start()
    return get_scheduler_status()


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def _read_cache_metadata() -> List[Dict]:
    cache_dir = BULL_BEAR_DATA_ROOT / "cache"
    if not cache_dir.exists() and BULL_BEAR_SOURCE_DATA_ROOT.exists():
        cache_dir = BULL_BEAR_SOURCE_DATA_ROOT / "cache"
    items: List[Dict] = []
    if not cache_dir.exists():
        return items
    for path in sorted(cache_dir.glob("*.meta.json")):
        meta = _read_json(path)
        if meta:
            items.append({
                "dataset": meta.get("dataset") or path.stem.replace(".meta", ""),
                "source": meta.get("source", "--"),
                "rows": meta.get("rows", 0),
                "start": meta.get("start"),
                "end": meta.get("end"),
                "updated_at": meta.get("updated_at"),
            })
    return items


def get_dashboard_payload() -> Dict:
    bootstrap = bootstrap_from_source_data(force=False, include_cache=False)
    results_dir = BULL_BEAR_DATA_ROOT / "results"
    processed_dir = BULL_BEAR_DATA_ROOT / "processed"
    pic_dir = BULL_BEAR_DATA_ROOT / "pic"
    signal = _read_json(results_dir / "cycle_signal_latest.json")

    history: List[Dict] = []
    history_path = processed_dir / "cycle_score_history.parquet"
    if history_path.exists():
        import pandas as pd

        df = pd.read_parquet(history_path)
        if not df.empty:
            df = df.tail(260).copy()
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].astype(str)
            keep = [c for c in ["trade_date", "cycle_score", "indicator_count"] if c in df.columns]
            history = df[keep].to_dict("records")

    def chart_priority(path: Path) -> tuple[int, str]:
        name = path.name
        if name == "cycle_position.png":
            return (0, name)
        if "equity_premium" in name:
            return (10, name)
        if "pe_valuation_沪深300" in name:
            return (20, name)
        if "pe_valuation_上证50" in name:
            return (21, name)
        if "pe_valuation_中证500" in name:
            return (22, name)
        if "pe_valuation_中证1000" in name:
            return (23, name)
        if "price_percentile" in name:
            return (30, name)
        if "market_turnover" in name:
            return (31, name)
        if "ma250_bias" in name:
            return (40, name)
        if "market_crowdedness" in name:
            return (41, name)
        if "herding_rate" in name:
            return (42, name)
        if "below_net_asset" in name:
            return (50, name)
        return (99, name)

    charts = []
    if pic_dir.exists():
        for path in sorted(pic_dir.glob("*.png"), key=chart_priority):
            charts.append({
                "name": path.stem,
                "file": path.name,
                "url": f"/api/bull-bear/chart/{path.name}/",
                "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            })

    return {
        "signal": signal,
        "history": history,
        "charts": charts,
        "cache": _read_cache_metadata(),
        "bootstrap": bootstrap,
        "task": get_task(),
        "scheduler": get_scheduler_status(),
        "model_retrain": _model_retrain_status(),
        "data_root": str(BULL_BEAR_DATA_ROOT),
        "source_root": str(BULL_BEAR_SOURCE_ROOT),
    }


def _model_retrain_status() -> Dict:
    try:
        from .model_retrain_service import get_retrain_status

        return get_retrain_status()
    except Exception:
        return {}


def chart_path(filename: str) -> Path:
    safe = Path(filename).name
    path = BULL_BEAR_DATA_ROOT / "pic" / safe
    if not path.exists() or path.suffix.lower() != ".png":
        raise FileNotFoundError(safe)
    return path
