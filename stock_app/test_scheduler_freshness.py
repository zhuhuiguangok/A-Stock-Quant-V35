# -*- coding: utf-8 -*-
"""调度器新鲜度跳过、资金流增量回补、模型定时重训的单元测试。"""
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import pandas as pd

from stock_app import bull_bear_service as bbs
from stock_app import fund_flow_service as ffs
from stock_app import model_retrain_service as mrs


class LatestCompletedTradeDateTest(unittest.TestCase):
    def test_saturday_falls_back_to_friday(self):
        self.assertEqual(
            bbs._latest_completed_trade_date(datetime(2026, 8, 15, 20, 0)), "20260814"
        )

    def test_sunday_falls_back_to_friday(self):
        self.assertEqual(
            bbs._latest_completed_trade_date(datetime(2026, 8, 16, 9, 0)), "20260814"
        )

    def test_monday_morning_expects_previous_friday(self):
        self.assertEqual(
            bbs._latest_completed_trade_date(datetime(2026, 8, 17, 10, 0)), "20260814"
        )

    def test_monday_after_close_expects_today(self):
        self.assertEqual(
            bbs._latest_completed_trade_date(datetime(2026, 8, 17, 16, 0)), "20260817"
        )

    def test_wednesday_before_1545_expects_tuesday(self):
        self.assertEqual(
            bbs._latest_completed_trade_date(datetime(2026, 8, 19, 15, 30)), "20260818"
        )


class CachedDataFreshTest(unittest.TestCase):
    def _write_meta(self, root: Path, name: str, end: str):
        cache = root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / f"{name}.meta.json").write_text(json.dumps({"end": end}), "utf-8")

    def test_fresh_when_both_cover_expected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_meta(root, "stocks", "20260819")
            self._write_meta(root, "fund_flow_sector", "20260819")
            with mock.patch.object(bbs, "BULL_BEAR_DATA_ROOT", root):
                fresh, reason = bbs._cached_data_fresh("20260819")
            self.assertTrue(fresh, reason)

    def test_stale_when_fund_flow_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_meta(root, "stocks", "20260819")
            self._write_meta(root, "fund_flow_sector", "20260817")
            with mock.patch.object(bbs, "BULL_BEAR_DATA_ROOT", root):
                fresh, reason = bbs._cached_data_fresh("20260819")
            self.assertFalse(fresh)
            self.assertIn("fund_flow_sector", reason)

    def test_stale_when_meta_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(bbs, "BULL_BEAR_DATA_ROOT", Path(tmp)):
                fresh, _ = bbs._cached_data_fresh("20260819")
            self.assertFalse(fresh)


class _FakePro:
    """记录 moneyflow 调用次数的最小 Tushare pro 替身。"""

    def __init__(self):
        self.moneyflow_calls = []

    def moneyflow(self, trade_date, fields=None):
        self.moneyflow_calls.append(trade_date)
        return pd.DataFrame({
            "ts_code": ["000001.SZ"],
            "trade_date": [trade_date],
            "buy_lg_amount": [100.0],
            "sell_lg_amount": [50.0],
            "buy_elg_amount": [10.0],
            "sell_elg_amount": [5.0],
            "net_mf_amount": [55.0],
        })

    def stock_basic(self, exchange="", list_status="L", fields=None):
        return pd.DataFrame({"ts_code": ["000001.SZ"], "industry": ["银行"]})

    def daily(self, trade_date, fields=None):
        return pd.DataFrame({"ts_code": ["000001.SZ"], "pct_chg": [1.0]})


class IncrementalBackfillTest(unittest.TestCase):
    def test_cached_dates_are_skipped(self):
        candidates = ffs._candidate_trade_dates(max_days=20)
        existing = set(candidates[1:])  # 仅缺口 = 最近一个交易日
        client = ffs.UnifiedFundFlowClient(pro_factory=None, backfill_days=10)
        client.existing_dates = existing
        pro = _FakePro()
        frame = client._tushare_backfill(pro)
        self.assertEqual(pro.moneyflow_calls, [candidates[0]])
        self.assertIsNone(client.noop_reason)
        self.assertFalse(frame.empty)

    def test_all_cached_returns_noop(self):
        candidates = ffs._candidate_trade_dates(max_days=20)
        client = ffs.UnifiedFundFlowClient(pro_factory=None, backfill_days=10)
        client.existing_dates = set(candidates)
        pro = _FakePro()
        frame = client._tushare_backfill(pro)
        self.assertEqual(pro.moneyflow_calls, [])
        self.assertTrue(frame.empty)
        self.assertIsNotNone(client.noop_reason)


class RetrainSchedulerGateTest(unittest.TestCase):
    def test_triggers_on_scheduled_weekday_after_hour(self):
        now = datetime(2026, 8, 15, 9, 30)  # 周六 09:30
        self.assertTrue(mrs._should_run_now(now, weekday=5, hour=9, last_run_date=None))

    def test_skips_before_hour(self):
        now = datetime(2026, 8, 15, 8, 30)
        self.assertFalse(mrs._should_run_now(now, weekday=5, hour=9, last_run_date=None))

    def test_skips_wrong_weekday(self):
        now = datetime(2026, 8, 16, 10, 0)  # 周日
        self.assertFalse(mrs._should_run_now(now, weekday=5, hour=9, last_run_date=None))

    def test_skips_when_already_run_today(self):
        now = datetime(2026, 8, 15, 10, 0)
        self.assertFalse(
            mrs._should_run_now(now, weekday=5, hour=9, last_run_date="2026-08-15")
        )

    def test_parse_weekday(self):
        self.assertEqual(mrs.parse_weekday("sat"), 5)
        self.assertEqual(mrs.parse_weekday("3"), 3)
        self.assertEqual(mrs.parse_weekday("bogus"), 5)


if __name__ == "__main__":
    unittest.main()
