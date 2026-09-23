"""研报模块后台任务与每日调度（复用 bull_bear_service 骨架）。

每日 1 次精简版流水线（默认 08:00，env RESEARCH_PIPELINE_AUTO/HOUR/MINUTE 可调），
当天已成功跑过则跳过（新鲜度跳过）。
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import date, datetime, timedelta
from typing import Callable, Dict, Optional


_LOCK = threading.Lock()
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULER_STARTED = False
_TASK: Dict = {
    "state": "idle",
    "message": "研报流水线尚未运行",
    "stage": "idle",
    "progress": 0,
    "trigger": None,
    "started_at": None,
    "finished_at": None,
    "error": None,
    "result": None,
}
_SCHEDULER: Dict = {
    "enabled": False,
    "state": "stopped",
    "hour": 8,
    "minute": 0,
    "check_seconds": 600,
    "last_check_at": None,
    "last_run_date": None,
    "next_check_at": None,
    "last_skip_reason": None,
    "message": "研报定时流水线未启动",
}


def _iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


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


def _run_pipeline() -> dict:
    from .research.pipeline import run_daily_pipeline

    def progress(msg: str) -> None:
        stage_map = {
            "抓取": ("fetch", 15), "AI": ("summarize", 40), "推荐": ("recommend", 60),
            "晨报": ("brief", 75), "盈利": ("forecast", 82), "事件": ("events", 88),
            "选股": ("picks", 92), "回填": ("returns", 96),
        }
        for key, (stage, pct) in stage_map.items():
            if key in msg:
                _set_task(stage=stage, progress=pct, message=msg)
                break

    return run_daily_pipeline(progress=progress)


def start_update(trigger: str = "manual") -> Dict:
    """后台线程执行完整流水线；已在运行时直接返回当前任务状态。"""
    task = get_task()
    if task.get("state") == "running":
        return task

    _set_task(
        state="running", stage="starting", progress=2, trigger=trigger,
        message="准备运行研报流水线", started_at=_iso_now(),
        finished_at=None, error=None, result=None,
    )

    def target() -> None:
        try:
            result = _run_pipeline()
            summary = result.get("summary") or {}
            crawl = result.get("crawl") or {}
            ok_sources = [k for k, v in crawl.items() if isinstance(v, dict) and v.get("status") == "success"]
            _set_task(
                state="completed", stage="completed", progress=100,
                message=(f"流水线完成: 新增研报 {sum(v.get('count', 0) for v in crawl.values() if isinstance(v, dict))} 篇 / "
                         f"AI摘要 {summary.get('processed', 0)} 篇 / 数据源 {len(ok_sources)}/{len(crawl)} 可用"),
                finished_at=_iso_now(), error=None, result=result,
            )
            if trigger == "scheduled":
                _set_scheduler(last_run_date=date.today().isoformat())
        except Exception as exc:  # pragma: no cover - surfaced through API
            _set_task(
                state="failed", stage="failed", progress=100,
                message="研报流水线失败", finished_at=_iso_now(),
                error=f"{exc}\n{traceback.format_exc()}",
            )

    threading.Thread(target=target, daemon=True, name="research_pipeline").start()
    return get_task()


def _should_run_now(now: datetime, hour: int, minute: int, last_run_date: Optional[str]) -> bool:
    return (
        (now.hour, now.minute) >= (hour, minute)
        and last_run_date != now.date().isoformat()
    )


def start_pipeline_scheduler(hour: int = 8, minute: int = 0, check_seconds: int = 600) -> Dict:
    """每日一次定时流水线，每个 Django 进程只启动一次。"""
    global _SCHEDULER_STARTED
    hour = min(23, max(0, int(hour)))
    minute = min(59, max(0, int(minute)))
    check_seconds = max(60, int(check_seconds or 600))

    with _SCHEDULER_LOCK:
        if _SCHEDULER_STARTED:
            _SCHEDULER.update({"enabled": True, "message": "研报定时流水线调度器已在运行"})
            return dict(_SCHEDULER)
        _SCHEDULER_STARTED = True
        _SCHEDULER.update({
            "enabled": True, "state": "waiting", "hour": hour, "minute": minute,
            "check_seconds": check_seconds,
            "message": f"研报定时流水线已启动，每日 {hour:02d}:{minute:02d} 后触发一次",
        })

    def loop() -> None:
        while True:
            next_at = (datetime.now() + timedelta(seconds=check_seconds)).isoformat(timespec="seconds")
            _set_scheduler(state="waiting", last_check_at=_iso_now(), next_check_at=next_at)
            time.sleep(check_seconds)
            now = datetime.now()
            status = get_scheduler_status()
            if get_task().get("state") == "running":
                continue
            if not _should_run_now(now, hour, minute, status.get("last_run_date")):
                continue
            _set_scheduler(last_skip_reason=None, message=f"到达计划时间，触发研报流水线（{now:%Y-%m-%d %H:%M}）")
            start_update(trigger="scheduled")

    threading.Thread(target=loop, daemon=True, name="research_pipeline_scheduler").start()
    return get_scheduler_status()
