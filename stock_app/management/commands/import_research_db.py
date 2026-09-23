# -*- coding: utf-8 -*-
"""从原 FastAPI 项目的 SQLite 库导入研报历史数据。

用法：
    python manage.py import_research_db                       # 默认读 D:\codex-workspace\AI每日研报调研平台\backend\research.db
    python manage.py import_research_db --path <sqlite文件>
可重复执行（按唯一键跳过已导入数据）。
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from stock_app.models import (
    DailyBrief,
    DailyRecommendation,
    EarningsForecast,
    MarketEvent,
    ResearchReport,
    ResearchSource,
    ResearchSummary,
    StockPick,
)

DEFAULT_DB = Path(r"D:\codex-workspace\AI每日研报调研平台\backend\research.db")


def _parse_date(val):
    return val[:10] if val else None


def _parse_dt(val):
    if not val:
        return None
    try:
        return datetime.fromisoformat(str(val)[:19])
    except ValueError:
        return None


def _loads(val, default):
    if not val:
        return default
    try:
        return json.loads(val)
    except Exception:
        return default


class Command(BaseCommand):
    help = "导入原研报平台 SQLite 历史数据"

    def add_arguments(self, parser):
        parser.add_argument("--path", default=str(DEFAULT_DB))

    @transaction.atomic
    def handle(self, *args, **options):
        db_path = Path(options["path"])
        if not db_path.exists():
            self.stderr.write(f"源库不存在: {db_path}")
            return
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        stats = {}

        # sources
        for row in conn.execute("SELECT * FROM sources"):
            ResearchSource.objects.update_or_create(
                code=row["code"],
                defaults={
                    "name": row["name"], "base_url": row["base_url"],
                    "status": row["status"] or "active", "weight": row["weight"] or 1.0,
                    "last_crawled_at": _parse_dt(row["last_crawled_at"]),
                },
            )
        stats["sources"] = ResearchSource.objects.count()

        # reports（旧 id → 新对象映射）
        old_to_new = {}
        new_count = 0
        for row in conn.execute("SELECT * FROM reports ORDER BY id"):
            source = ResearchSource.objects.filter(code=conn.execute(
                "SELECT code FROM sources WHERE id=?", (row["source_id"],)
            ).fetchone()[0]).first() if row["source_id"] else None
            if source is None:
                continue
            obj, created = ResearchReport.objects.get_or_create(
                source=source, external_id=row["external_id"],
                defaults={
                    "title": (row["title"] or "")[:500],
                    "stock_code": (row["stock_code"] or "")[:16],
                    "stock_name": (row["stock_name"] or "")[:64],
                    "industry": (row["industry"] or "")[:64],
                    "author": (row["author"] or "")[:128],
                    "institution": (row["institution"] or "")[:128],
                    "report_date": _parse_date(row["report_date"]),
                    "content_text": row["content_text"] or "",
                    "pdf_url": (row["pdf_url"] or "")[:500],
                    "source_url": (row["source_url"] or "")[:500],
                    "status": row["status"] or "pending",
                },
            )
            old_to_new[row["id"]] = obj.id
            new_count += int(created)
        stats["reports"] = f"+{new_count}/{ResearchReport.objects.count()}"

        # summaries
        summary_count = 0
        for row in conn.execute("SELECT * FROM report_summaries"):
            new_id = old_to_new.get(row["report_id"])
            if new_id is None:
                continue
            _, created = ResearchSummary.objects.update_or_create(
                report_id=new_id,
                defaults={
                    "summary": row["summary"] or "",
                    "tags": _loads(row["tags"], []),
                    "investment_points": _loads(row["investment_points"], []),
                    "sentiment": row["sentiment"] or "neutral",
                    "target_stocks": _loads(row["target_stocks"], []),
                    "rating": (row["rating"] or "")[:32],
                    "risks": _loads(row["risks"], []),
                    "target_price": (row["target_price"] or "")[:64],
                    "ai_model": (row["ai_model"] or "")[:64],
                },
            )
            summary_count += int(created)
        stats["summaries"] = f"+{summary_count}/{ResearchSummary.objects.count()}"

        # daily briefs
        for row in conn.execute("SELECT * FROM daily_briefs"):
            DailyBrief.objects.update_or_create(
                brief_date=_parse_date(row["brief_date"]),
                defaults={
                    "overview": row["overview"] or "",
                    "advice": row["advice"] or "",
                    "short_term": row["short_term"] or "",
                    "mid_term": row["mid_term"] or "",
                    "long_term": row["long_term"] or "",
                    "industries": _loads(row["industries"], []),
                    "stocks": _loads(row["stocks"], []),
                    "sentiment_stats": _loads(row["sentiment_stats"], {}),
                    "ai_model": (row["ai_model"] or "")[:64],
                },
            )
        stats["briefs"] = DailyBrief.objects.count()

        # recommendations
        for row in conn.execute("SELECT * FROM daily_recommendations"):
            new_id = old_to_new.get(row["report_id"])
            if new_id is None:
                continue
            DailyRecommendation.objects.get_or_create(
                report_id=new_id, recommend_date=_parse_date(row["recommend_date"]), rank=row["rank"],
                defaults={"score": row["score"] or 0, "reason": row["reason"] or ""},
            )
        stats["recommendations"] = DailyRecommendation.objects.count()

        # stock picks
        for row in conn.execute("SELECT * FROM stock_picks"):
            StockPick.objects.get_or_create(
                pick_date=_parse_date(row["pick_date"]), stock_code=row["stock_code"],
                source=row["source"] or "daily_brief",
                defaults={
                    "stock_name": (row["stock_name"] or "")[:64],
                    "price_at_pick": row["price_at_pick"],
                    "ret_t1": row["ret_t1"], "ret_t5": row["ret_t5"], "ret_t20": row["ret_t20"],
                },
            )
        stats["picks"] = StockPick.objects.count()

        # market events
        for row in conn.execute("SELECT * FROM market_events"):
            MarketEvent.objects.get_or_create(
                event_date=_parse_date(row["event_date"]), stock_code=row["stock_code"] or "",
                title=(row["title"] or "")[:250],
                defaults={
                    "stock_name": (row["stock_name"] or "")[:64],
                    "event_type": (row["event_type"] or "")[:32],
                    "detail": row["detail"] or "",
                },
            )
        stats["events"] = MarketEvent.objects.count()

        # earnings forecasts
        for row in conn.execute("SELECT * FROM earnings_forecasts"):
            new_id = old_to_new.get(row["report_id"])
            if new_id is None:
                continue
            EarningsForecast.objects.create(
                report_id=new_id,
                stock_code=row["stock_code"],
                stock_name=(row["stock_name"] or "")[:64],
                forecast_year=row["forecast_year"],
                profit_value=row["profit_value"],
                profit_text=(row["profit_text"] or "")[:128],
                eps_value=row["eps_value"],
            )
        stats["forecasts"] = EarningsForecast.objects.count()

        conn.close()
        for key, val in stats.items():
            self.stdout.write(f"  {key}: {val}")
        self.stdout.write(self.style.SUCCESS("导入完成"))
