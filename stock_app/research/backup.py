# -*- coding: utf-8 -*-
"""研报数据库备份保全：sqlite 在线备份 API → data_cache/backup/。

每日研报流水线末尾调用 _maybe_backup_db()：
- 当日已备份则跳过；否则用 sqlite3 backup API 生成一致性快照
  （WAL 模式安全，不锁业务连接），保留最近 keep 份。
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKUP_DIR = PROJECT_ROOT / "data_cache" / "backup"
DB_PATH = PROJECT_ROOT / "db.sqlite3"
KEEP = 7


def backup_research_db(keep: int = KEEP) -> dict:
    """一致性备份 sqlite → data_cache/backup/research_db_<日期>.sqlite3，保留最近 keep 份。"""
    backup_dir = BACKUP_DIR
    backup_dir.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    target = backup_dir / f"research_db_{today}.sqlite3"
    if target.exists():
        return {"status": "skip", "file": str(target), "message": "当日已备份"}
    if not DB_PATH.exists():
        return {"status": "skip", "message": "db.sqlite3 不存在"}

    src = sqlite3.connect(str(DB_PATH))
    dst = sqlite3.connect(str(target))
    try:
        with dst:
            src.backup(dst)
    finally:
        src.close()
        dst.close()

    size_mb = target.stat().st_size / 1e6
    # 清理旧备份，保留最近 keep 份
    backups = sorted(backup_dir.glob("research_db_*.sqlite3"))
    removed = 0
    for old in backups[:-keep]:
        try:
            old.unlink()
            removed += 1
        except OSError:
            continue
    logger.info(f"📚 研报库备份完成: {target.name} ({size_mb:.0f}MB, 清理{removed}份旧备份)")
    return {"status": "success", "file": str(target), "size_mb": round(size_mb, 1), "removed": removed}


def _maybe_backup_db() -> dict:
    try:
        return backup_research_db()
    except Exception as exc:
        logger.warning(f"研报库备份失败(不影响主流程): {exc}")
        return {"status": "error", "message": str(exc)}
