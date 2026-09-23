"""全球雷达后台任务与每日调度（默认 08:20，晚于研报流水线 08:00 以便 AI 晨报引用研报观点）。"""
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
    "message": "全球雷达尚未运行",
    "stage": "idle",
    "progress": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}
_SCHEDULER: Dict = {
    "enabled": False,
    "state": "stopped",
    "hour": 8,
    "minute": 20,
    "check_seconds": 600,
    "last_check_at": None,
    "last_run_date": None,
    "next_check_at": None,
    "message": "全球雷达定时任务未启动",
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


def _run_update() -> Dict:
    from .global_radar import service as gr_service

    _set_task(stage="collect", progress=15, message="拉取全球行情与宏观数据")
    payload = gr_service.build_dashboard()
    _set_task(stage="save", progress=55, message="生成全球雷达面板")
    gr_service.save_dashboard(payload)
    _set_task(stage="brief", progress=75, message="AI 生成全球晨报")
    brief_result = gr_service.generate_ai_brief(payload)
    payload["brief"] = (brief_result.get("brief") if brief_result.get("status") == "success" else gr_service.read_brief())
    gr_service.save_dashboard(payload)
    return {"payload": payload, "brief_status": brief_result.get("status")}


def start_update(trigger: str = "manual") -> Dict:
    """后台线程执行：采集 → 面板 → AI 晨报。已在运行时直接返回状态。"""
    task = get_task()
    if task.get("state") == "running":
        return task

    _set_task(state="running", stage="starting", progress=2, message="准备运行全球雷达", started_at=_iso_now(), finished_at=None, error=None)

    def target() -> None:
        try:
            result = _run_update()
            meta = (result["payload"].get("meta") or {})
            ok_n = len(meta.get("assets_ok") or [])
            failed = meta.get("sources_failed") or []
            _set_task(
                state="completed",
                stage="completed",
                progress=100,
                message=(f"全球雷达完成: {ok_n} 项资产 / 晨报 {result['brief_status']}"
                         + (f" / 缺失: {'、'.join(failed)}" if failed else "")),
                finished_at=_iso_now(),
                error=None,
            )
            if trigger == "scheduled":
                _set_scheduler(last_run_date=date.today().isoformat())
        except Exception as exc:  # pragma: no cover - surfaced through API
            _set_task(state="failed", stage="failed", progress=100, message="全球雷达失败", finished_at=_iso_now(), error=f"{exc}\n{traceback.format_exc()}")

    threading.Thread(target=target, daemon=True, name="global_radar_update").start()
    return get_task()


def _should_run_now(now: datetime, hour: int, minute: int, last_run_date: Optional[str]) -> bool:
    return (now.hour, now.minute) >= (hour, minute) and last_run_date != now.date().isoformat()


def start_scheduler(hour: int = 8, minute: int = 20, check_seconds: int = 600) -> Dict:
    """每日一次定时更新，每个 Django 进程只启动一次。"""
    global _SCHEDULER_STARTED
    hour = min(23, max(0, int(hour)))
    minute = min(59, max(0, int(minute)))
    check_seconds = max(60, int(check_seconds or 600))

    with _SCHEDULER_LOCK:
        if _SCHEDULER_STARTED:
            _SCHEDULER.update({"enabled": True, "message": "全球雷达定时任务已在运行"})
            return dict(_SCHEDULER)
        _SCHEDULER_STARTED = True
        _SCHEDULER.update({
            "enabled": True, "state": "waiting", "hour": hour, "minute": minute,
            "check_seconds": check_seconds,
            "message": f"全球雷达已启动，每日 {hour:02d}:{minute:02d} 自动更新",
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
            _set_scheduler(message=f"到达计划时间，触发全球雷达（{now:%Y-%m-%d %H:%M}）")
            start_update(trigger="scheduled")

    threading.Thread(target=loop, daemon=True, name="global_radar_scheduler").start()
    return get_scheduler_status()
