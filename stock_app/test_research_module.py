# -*- coding: utf-8 -*-
"""AI研报模块单元测试：模型约束、评分、流水线（mock）、导入命令、信号读取。"""
import json
import os
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from django.test import TestCase

from stock_app.models import (
    DailyBrief,
    DailyRecommendation,
    EarningsForecast,
    MarketEvent,
    ResearchPreference,
    ResearchReport,
    ResearchSource,
    ResearchSummary,
    StockPick,
)
from stock_app.research import services, signals
from stock_app.research.crawlers import CRAWLER_REGISTRY
from stock_app.research.forecast_extractor import extract_eps_forecasts, extract_profit_forecasts
from stock_app.research.market_utils import extract_target_price, institution_weight, normalize_industry


class MarketUtilsTest(unittest.TestCase):
    def test_institution_weight_tiers(self):
        self.assertEqual(institution_weight("中信证券"), 1.5)
        self.assertEqual(institution_weight("某基金公司"), 1.0)
        self.assertEqual(institution_weight("野村"), 0.8)
        self.assertEqual(institution_weight(None), 0.8)

    def test_normalize_industry(self):
        self.assertEqual(normalize_industry("白酒"), "食品饮料")
        self.assertEqual(normalize_industry("光模块CPO"), "AI算力")
        self.assertEqual(normalize_industry("未知行业"), "未知行业")

    def test_extract_target_price(self):
        self.assertEqual(extract_target_price("给予目标价12.5元"), "12.5")
        self.assertIsNone(extract_target_price("没有价格"))


class ForecastExtractorTest(unittest.TestCase):
    def test_profit_patterns(self):
        text = "预计2026年归母净利润10.5亿元，2027年净利润13.2亿元"
        results = extract_profit_forecasts(text)
        profits = {r["profit"] for r in results}
        self.assertIn(10.5, profits)

    def test_eps_patterns(self):
        results = extract_eps_forecasts("预计2026年EPS为1.25元")
        self.assertTrue(any(abs(r["eps"] - 1.25) < 1e-9 for r in results))

    def test_empty(self):
        self.assertEqual(extract_profit_forecasts(""), [])
        self.assertEqual(extract_eps_forecasts(None), [])


class CrawlerRegistryTest(unittest.TestCase):
    def test_registry_has_six_sources(self):
        self.assertEqual(
            set(CRAWLER_REGISTRY), {"iyanbao", "hibor", "eastmoney", "sina", "fxbaogao", "it199"}
        )


class ModelConstraintTest(TestCase):
    def test_report_unique_external_id(self):
        source = ResearchSource.objects.create(name="t", code="t", base_url="http://t")
        ResearchReport.objects.create(
            source=source, external_id="x1", title="A", source_url="http://a", report_date=date.today()
        )
        from django.db import IntegrityError, transaction

        with self.assertRaises(IntegrityError), transaction.atomic():
            ResearchReport.objects.create(
                source=source, external_id="x1", title="B", source_url="http://b", report_date=date.today()
            )

    def test_preference_singleton(self):
        p1 = ResearchPreference.get_solo()
        p2 = ResearchPreference.get_solo()
        self.assertEqual(p1.pk, p2.pk)


def _make_report(title="贵州茅台深度研究", institution="中信证券", sentiment="bullish",
                 rating="买入", tags=None, target_stocks=None, report_date=None, **kw):
    source, _ = ResearchSource.objects.get_or_create(name="测试源", code="test", base_url="http://t")
    report = ResearchReport.objects.create(
        source=source, external_id=kw.get("external_id", title),
        title=title, source_url="http://t", institution=institution,
        report_date=report_date or date.today(),
        stock_code=kw.get("stock_code", ""), stock_name=kw.get("stock_name", ""),
        content_text=kw.get("content_text", "正文"),
    )
    ResearchSummary.objects.create(
        report=report, summary="摘要", tags=tags or [], investment_points=["要点1"],
        sentiment=sentiment, rating=rating, target_stocks=target_stocks or [],
        risks=["风险1"], ai_model="test",
    )
    report.status = ResearchReport.STATUS_SUMMARIZED
    report.save()
    return report


class RecommendationTest(TestCase):
    def test_score_prefers_watched_stock(self):
        pref = ResearchPreference.get_solo()
        pref.watch_stocks = ["600519"]
        pref.save()
        _make_report("贵州茅台深度研究", stock_code="600519", stock_name="贵州茅台")
        _make_report("无关行业报告", external_id="other")

        result = services.generate_daily_recommendations(limit=10)
        self.assertEqual(result["count"], 2)
        first = DailyRecommendation.objects.filter(rank=1).first()
        self.assertIn("自选股", first.reason)

    def test_freshness_decay(self):
        old = _make_report("旧报告", external_id="old", report_date=date.today() - timedelta(days=5))
        new = _make_report("新报告", external_id="new", report_date=date.today())
        services.generate_daily_recommendations(limit=10)
        ranks = {r.report.external_id: r.rank for r in DailyRecommendation.objects.all()}
        self.assertLess(ranks["new"], ranks["old"])


class ConsensusSignalTest(TestCase):
    def test_consensus_aggregation(self):
        for i in range(3):
            _make_report(
                f"茅台报告{i}", external_id=f"mt{i}", institution="中信证券",
                sentiment="bullish", rating="买入", target_stocks=["600519"],
            )
        with mock.patch.object(signals, "get_stock_change", return_value=-8.0):
            result = signals.get_stock_consensus(days=7, limit=10)
        self.assertTrue(result)
        top = result[0]
        self.assertEqual(top["stock_code"], "600519")
        self.assertEqual(top["report_count"], 3)
        # 3份全买入 + 20日跌8% → 多空分歧
        self.assertTrue(top["divergence"])

    def test_forecast_consensus(self):
        report = _make_report("巨人网络研究", stock_code="002558", stock_name="巨人网络")
        for year, profit in [(2026, 40.0), (2026, 42.0), (2026, 46.0)]:
            EarningsForecast.objects.create(
                report=report, stock_code="002558", stock_name="巨人网络",
                forecast_year=year, profit_value=profit,
            )
        result = signals.get_forecast_consensus(limit=10)
        self.assertTrue(result)
        entry = next(r for r in result if r["stock_code"] == "002558")
        self.assertEqual(entry["forecast_count"], 3)
        self.assertAlmostEqual(entry["avg_profit"], 42.67, places=1)

    def test_tracking_stats(self):
        StockPick.objects.create(
            pick_date=date.today() - timedelta(days=25), stock_code="600519",
            stock_name="贵州茅台", price_at_pick=100.0, ret_t1=1.5, ret_t5=2.0, ret_t20=-3.0,
        )
        stats = signals.get_tracking_stats()
        self.assertEqual(stats["t1"]["count"], 1)
        self.assertEqual(stats["t1"]["win_rate"], 100.0)
        self.assertEqual(stats["t20"]["win_rate"], 0.0)


class EventExtractionTest(TestCase):
    def test_extract_events_from_text(self):
        future = (date.today() + timedelta(days=10)).strftime("%Y年%m月%d日").replace("年0", "年")
        report = _make_report(
            f"某公司公告 股东大会{future}召开", external_id="evt1",
            content_text=f"公司宣布股东大会{future}在公司会议室召开，审议年度报告。",
        )
        saved = signals.extract_events_from_reports(days=7)
        self.assertGreaterEqual(saved, 1)
        self.assertTrue(MarketEvent.objects.filter(event_type="股东大会").exists())


class PipelineTest(TestCase):
    def test_run_daily_pipeline_with_mocks(self):
        fake_raw = [{
            "external_id": "crawl_1", "title": "爬取的研报", "source_url": "http://x",
            "report_date": date.today(), "institution": "中信证券", "content_text": "内容",
        }]
        fake_crawler = mock.MagicMock()
        fake_crawler.base_url = "http://x"
        fake_crawler.fetch_list.return_value = fake_raw
        fake_crawler.fetch_detail.return_value = {"content_text": "详情", "pdf_url": None}
        fake_ai_result = {
            "summary": "s", "tags": ["AI算力"], "investment_points": ["p"],
            "sentiment": "bullish", "rating": "买入", "risks": ["r"],
            "target_price": "12.5", "target_stocks": ["600519"],
        }
        with mock.patch.dict(CRAWLER_REGISTRY, {"test": lambda: fake_crawler}, clear=True), \
             mock.patch.object(services.AIService, "summarize_report", return_value=fake_ai_result), \
             mock.patch.object(services, "generate_daily_brief", return_value={"status": "skipped"}), \
             mock.patch.object(services, "update_pick_returns", return_value=0):
            from stock_app.research.pipeline import run_daily_pipeline

            result = run_daily_pipeline()
        self.assertEqual(result["crawl"]["test"]["status"], "success")
        self.assertEqual(result["crawl"]["test"]["count"], 1)
        self.assertEqual(result["summary"]["processed"], 1)
        self.assertTrue(ResearchReport.objects.filter(external_id="crawl_1", status="summarized").exists())


class ImportCommandTest(TestCase):
    def _build_source_db(self, tmpdir: str) -> str:
        db_path = str(Path(tmpdir) / "research.db")
        conn = sqlite3.connect(db_path)
        conn.executescript("""
        CREATE TABLE sources (id INTEGER PRIMARY KEY, name TEXT, code TEXT UNIQUE, base_url TEXT, status TEXT, weight REAL, last_crawled_at TEXT);
        CREATE TABLE reports (id INTEGER PRIMARY KEY, source_id INTEGER, external_id TEXT, title TEXT, stock_code TEXT, stock_name TEXT, industry TEXT, author TEXT, institution TEXT, report_date TEXT, content_text TEXT, pdf_url TEXT, source_url TEXT, status TEXT);
        CREATE TABLE report_summaries (id INTEGER PRIMARY KEY, report_id INTEGER, summary TEXT, tags TEXT, investment_points TEXT, sentiment TEXT, target_stocks TEXT, rating TEXT, risks TEXT, target_price TEXT, ai_model TEXT);
        CREATE TABLE daily_briefs (id INTEGER PRIMARY KEY, brief_date TEXT UNIQUE, overview TEXT, advice TEXT, short_term TEXT, mid_term TEXT, long_term TEXT, industries TEXT, stocks TEXT, sentiment_stats TEXT, ai_model TEXT);
        CREATE TABLE daily_recommendations (id INTEGER PRIMARY KEY, report_id INTEGER, recommend_date TEXT, score REAL, rank INTEGER, reason TEXT);
        CREATE TABLE stock_picks (id INTEGER PRIMARY KEY, pick_date TEXT, stock_code TEXT, stock_name TEXT, source TEXT, price_at_pick REAL, ret_t1 REAL, ret_t5 REAL, ret_t20 REAL);
        CREATE TABLE market_events (id INTEGER PRIMARY KEY, event_date TEXT, stock_code TEXT, stock_name TEXT, event_type TEXT, title TEXT, detail TEXT);
        CREATE TABLE earnings_forecasts (id INTEGER PRIMARY KEY, report_id INTEGER, stock_code TEXT, stock_name TEXT, forecast_year INTEGER, profit_value REAL, profit_text TEXT, eps_value REAL);
        """)
        conn.execute("INSERT INTO sources VALUES (1,'测试','test','http://t','active',1.0,NULL)")
        conn.execute("INSERT INTO reports VALUES (1,1,'r1','标题一','600519','贵州茅台','食品饮料','analyst','中信证券','2026-08-20','正文','http://p','http://s','summarized')")
        conn.execute("INSERT INTO report_summaries VALUES (1,1,'摘要','[\"白酒\"]','[\"要点\"]','bullish','[\"600519\"]','买入','[\"风险\"]','2000','deepseek-chat')")
        conn.execute("INSERT INTO daily_briefs VALUES (1,'2026-08-20','概览','建议','短','中','长','[\"食品饮料\"]','[{\"name\":\"贵州茅台\",\"code\":\"600519\"}]','{\"bullish\":1}','deepseek-chat')")
        conn.execute("INSERT INTO daily_recommendations VALUES (1,1,'2026-08-20',0.9,1,'原因')")
        conn.execute("INSERT INTO stock_picks VALUES (1,'2026-08-20','600519','贵州茅台','daily_brief',1500.0,1.0,2.0,3.0)")
        conn.execute("INSERT INTO market_events VALUES (1,'2026-08-25','600519','贵州茅台','股东大会','股东大会','细节')")
        conn.execute("INSERT INTO earnings_forecasts VALUES (1,1,'600519','贵州茅台',2026,800.0,'净利800亿',60.0)")
        conn.commit()
        conn.close()
        return db_path

    def test_import_idempotent(self):
        from django.core.management import call_command

        with tempfile.TemporaryDirectory() as tmp:
            db_path = self._build_source_db(tmp)
            call_command("import_research_db", path=db_path)
            self.assertEqual(ResearchReport.objects.count(), 1)
            report = ResearchReport.objects.get()
            self.assertEqual(report.summary.tags, ["白酒"])
            self.assertEqual(report.summary.target_stocks, ["600519"])
            brief = DailyBrief.objects.get()
            self.assertEqual(brief.stocks[0]["code"], "600519")
            # 再导一次不重复
            call_command("import_research_db", path=db_path)
            self.assertEqual(ResearchReport.objects.count(), 1)
            self.assertEqual(DailyRecommendation.objects.count(), 1)
            self.assertEqual(MarketEvent.objects.count(), 1)
