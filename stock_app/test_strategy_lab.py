import json
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import TestCase

from .models import PortfolioHolding, PortfolioTrade
from .strategy_lab import (
    BEIJING_CHAOJIA_STRATEGY_ID,
    YANGJIA_XINFA_STRATEGY_ID,
    ZHAO_LAOGE_STRATEGY_ID,
    QIAO_BANGZHU_STRATEGY_ID,
    FANG_XINXIA_STRATEGY_ID,
    run_beijing_chaoshao_strategy,
    run_yangjia_xinfa_strategy,
    run_zhao_laoge_strategy,
    run_qiao_bangzhu_strategy,
    run_fang_xinxia_strategy,
)
from .views import enrich_limit_pool_fields, _normalize_index_weight_pool


def _sample_strategy_df():
    rows = []
    start = date(2026, 5, 20)
    specs = {
        "000001.SZ": {"name": "首板优选", "base": 10.0, "latest_pct": 10.0, "prev_pct": 2.0, "mv_yi": 60},
        "000002.SZ": {"name": "连续涨停", "base": 8.0, "latest_pct": 10.0, "prev_pct": 10.0, "mv_yi": 80},
        "000003.SZ": {"name": "未涨停股", "base": 12.0, "latest_pct": 3.0, "prev_pct": 1.0, "mv_yi": 40},
        "000004.SZ": {"name": "大市值首板", "base": 15.0, "latest_pct": 10.0, "prev_pct": 1.0, "mv_yi": 800},
        "000005.SZ": {"name": "高价首板", "base": 30.0, "latest_pct": 10.0, "prev_pct": 1.0, "mv_yi": 70},
    }
    for ts_code, spec in specs.items():
        close = spec["base"]
        for i in range(25):
            trade_date = (start + timedelta(days=i)).strftime("%Y%m%d")
            pct = 0.4
            if i == 23:
                pct = spec["prev_pct"]
            if i == 24:
                pct = spec["latest_pct"]
            pre_close = close
            close = round(pre_close * (1 + pct / 100), 3)
            rows.append({
                "ts_code": ts_code,
                "trade_date": trade_date,
                "name": spec["name"],
                "industry": "测试行业",
                "open": pre_close,
                "high": close,
                "low": pre_close * 0.98,
                "close": close,
                "pre_close": pre_close,
                "pct_chg": pct,
                "amount": 120000 + i * 1000,
                "turnover_rate": 8 + i * 0.1,
                "total_mv": spec["mv_yi"] * 10000,
                "pe": 20,
                "pb": 2,
                "fd_amount": 360000 if ts_code == "000001.SZ" and i == 24 else None,
                "open_times": 1 if ts_code == "000001.SZ" and i == 24 else None,
                "first_time": "093500" if ts_code == "000001.SZ" and i == 24 else "",
                "last_time": "145700" if ts_code == "000001.SZ" and i == 24 else "",
            })
    return pd.DataFrame(rows)


def _sample_named_master_df():
    rows = []
    start = date(2026, 5, 1)
    specs = {
        "001001.SZ": {"name": "二板龙头", "industry": "机器人", "base": 12.0, "mv_yi": 90, "pcts": {37: 10.0, 39: 10.0}},
        "001002.SZ": {"name": "强势低吸", "industry": "机器人", "base": 20.0, "mv_yi": 140, "pcts": {25: 10.0, 35: 5.0, 39: -1.2}},
        "001003.SZ": {"name": "大成交趋势", "industry": "算力", "base": 45.0, "mv_yi": 600, "pcts": {20: 5.0, 30: 6.0, 39: 4.0}},
        "001004.SZ": {"name": "题材跟风", "industry": "机器人", "base": 9.0, "mv_yi": 60, "pcts": {39: 3.0}},
        "001005.SZ": {"name": "算力助攻", "industry": "算力", "base": 28.0, "mv_yi": 220, "pcts": {39: 6.0}},
    }
    for ts_code, spec in specs.items():
        close = spec["base"]
        for i in range(40):
            pct = spec["pcts"].get(i, 0.8 if i > 8 else 0.2)
            if ts_code == "001002.SZ" and 28 <= i <= 34:
                pct = 1.5
            if ts_code == "001002.SZ" and 36 <= i <= 39:
                pct = -1.0
            if ts_code == "001003.SZ" and i >= 18:
                pct = spec["pcts"].get(i, 1.2)
            pre_close = close
            close = round(pre_close * (1 + pct / 100), 3)
            amount = 300000 + i * 8000
            if ts_code == "001003.SZ":
                amount = 1800000 + i * 50000
            rows.append({
                "ts_code": ts_code,
                "trade_date": (start + timedelta(days=i)).strftime("%Y%m%d"),
                "name": spec["name"],
                "industry": spec["industry"],
                "open": pre_close,
                "high": max(close, pre_close) * 1.01,
                "low": min(close, pre_close) * 0.99,
                "close": close,
                "pre_close": pre_close,
                "pct_chg": pct,
                "amount": amount,
                "turnover_rate": 8 if ts_code != "001003.SZ" else 5,
                "total_mv": spec["mv_yi"] * 10000,
                "pe": 30,
                "pb": 3,
                "fd_amount": 800000 if pct >= 9.5 else 0,
                "open_times": 1 if pct >= 9.5 else 0,
                "first_time": "093800" if pct >= 9.5 else "",
                "last_time": "145000" if pct >= 9.5 else "",
            })
    return pd.DataFrame(rows)


class StrategyLabCoreTests(TestCase):
    def test_zhao_laoge_strategy_quantifies_second_board_leader(self):
        result = run_zhao_laoge_strategy(_sample_named_master_df(), top_n=5)
        self.assertEqual(result["strategy"]["id"], ZHAO_LAOGE_STRATEGY_ID)
        self.assertGreaterEqual(len(result["picks"]), 1)
        pick = result["picks"][0]
        self.assertIn("二板龙头确认", pick["strategy_dimensions"])
        self.assertIn("新题材主线", pick["strategy_dimensions"])
        self.assertEqual(pick["strategy_id"], ZHAO_LAOGE_STRATEGY_ID)

    def test_qiao_bangzhu_strategy_quantifies_strong_pullback(self):
        result = run_qiao_bangzhu_strategy(_sample_named_master_df(), top_n=5)
        self.assertEqual(result["strategy"]["id"], QIAO_BANGZHU_STRATEGY_ID)
        self.assertGreaterEqual(len(result["picks"]), 1)
        dims = result["picks"][0]["strategy_dimensions"]
        self.assertIn("前期强度", dims)
        self.assertIn("回踩质量", dims)
        self.assertIn("承接质量", dims)

    def test_fang_xinxia_strategy_quantifies_big_money_trend(self):
        result = run_fang_xinxia_strategy(_sample_named_master_df(), top_n=5)
        self.assertEqual(result["strategy"]["id"], FANG_XINXIA_STRATEGY_ID)
        self.assertGreaterEqual(len(result["picks"]), 1)
        dims = result["picks"][0]["strategy_dimensions"]
        self.assertIn("大资金容量", dims)
        self.assertIn("趋势龙头", dims)
        self.assertIn("资金共振", dims)

    def test_api_runs_three_new_master_strategies_with_own_defaults(self):
        cases = [
            (ZHAO_LAOGE_STRATEGY_ID, 250.0),
            (QIAO_BANGZHU_STRATEGY_ID, 500.0),
            (FANG_XINXIA_STRATEGY_ID, 2000.0),
        ]
        for strategy_id, mv_max in cases:
            with self.subTest(strategy_id=strategy_id), \
                 patch("stock_app.views.get_latest_trading_date", return_value="20260609"), \
                 patch("stock_app.views.get_real_stock_data", return_value=_sample_named_master_df()):
                res = self.client.post(
                    "/api/strategy/run/",
                    data=json.dumps({"strategy_id": strategy_id, "top_n": 5}),
                    content_type="application/json",
                )
                self.assertEqual(res.status_code, 200)
                payload = res.json()
                self.assertEqual(payload["status"], "success")
                self.assertEqual(payload["strategy"]["id"], strategy_id)
                self.assertEqual(payload["stats"]["mv_max_yi"], mv_max)

    def test_strategy_api_enriches_limit_pool_by_actual_data_trade_date(self):
        captured = {}

        def fake_enrich(df, trade_date=None, pro=None):
            captured["trade_date"] = trade_date
            return df

        with patch("stock_app.views.get_latest_trading_date", return_value="20260629"), \
             patch("stock_app.views.get_real_stock_data", return_value=_sample_named_master_df()), \
             patch("stock_app.views.enrich_limit_pool_fields", side_effect=fake_enrich):
            res = self.client.post(
                "/api/strategy/run/",
                data=json.dumps({"strategy_id": ZHAO_LAOGE_STRATEGY_ID, "top_n": 3}),
                content_type="application/json",
            )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(captured["trade_date"], _sample_named_master_df()["trade_date"].max())

    def test_yangjia_strategy_prefers_emotion_mainline_and_leader(self):
        result = run_yangjia_xinfa_strategy(_sample_strategy_df(), top_n=10, price_max=80, mv_max_yi=300)

        self.assertEqual(result["strategy"]["id"], YANGJIA_XINFA_STRATEGY_ID)
        self.assertIn("market_emotion", result["stats"])
        self.assertGreaterEqual(len(result["picks"]), 1)
        pick = result["picks"][0]
        self.assertEqual(pick["strategy_id"], YANGJIA_XINFA_STRATEGY_ID)
        self.assertIn("市场情绪周期", pick["strategy_dimensions"])
        self.assertIn("主流热点强度", pick["strategy_dimensions"])
        self.assertIn("龙头人气地位", pick["strategy_dimensions"])
        self.assertIn("市场合力承接", pick["strategy_dimensions"])
        self.assertIn("风险收益比", pick["strategy_dimensions"])
        self.assertIn("情绪周期", " ".join(pick["buy_signals"]))

    def test_api_runs_yangjia_strategy_with_own_defaults(self):
        with patch("stock_app.views.get_latest_trading_date", return_value="20260613"), \
             patch("stock_app.views.get_real_stock_data", return_value=_sample_strategy_df()):
            res = self.client.post(
                "/api/strategy/run/",
                data=json.dumps({"strategy_id": YANGJIA_XINFA_STRATEGY_ID, "top_n": 5}),
                content_type="application/json",
            )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["strategy"]["id"], YANGJIA_XINFA_STRATEGY_ID)
        self.assertEqual(payload["stats"]["mv_max_yi"], 300.0)
        self.assertEqual(payload["stats"]["price_max"], 80.0)
        self.assertIn("养家心法", payload["stats"]["filter_note"])

    def test_beijing_strategy_filters_first_board_price_and_float_mv_before_scoring(self):
        result = run_beijing_chaoshao_strategy(_sample_strategy_df(), top_n=10)

        self.assertEqual(result["strategy"]["id"], BEIJING_CHAOJIA_STRATEGY_ID)
        codes = [p["code"] for p in result["picks"]]
        self.assertIn("000001", codes)
        self.assertNotIn("000002", codes)  # previous day was already limit-up
        self.assertNotIn("000003", codes)  # latest day is not limit-up
        self.assertNotIn("000004", codes)  # default float market cap range is 20~100亿
        self.assertNotIn("000005", codes)  # default price range is 3~20

        pick = next(p for p in result["picks"] if p["code"] == "000001")
        self.assertEqual(pick["position_pct"], 10.0)
        self.assertEqual(pick["max_total_position_pct"], 60.0)
        self.assertEqual(pick["stop_loss_pct"], -5.0)
        self.assertNotIn("首板确认", pick["strategy_dimensions"])
        self.assertNotIn("低价偏好", pick["strategy_dimensions"])
        self.assertNotIn("市值偏好(20-100亿)", pick["strategy_dimensions"])
        self.assertIn("封板坚决", pick["strategy_dimensions"])
        self.assertIn("技术形态", pick["strategy_dimensions"])
        self.assertIn("股性活跃", pick["strategy_dimensions"])
        self.assertGreater(pick["fd_amount_yi"], 0)
        self.assertEqual(pick["open_times"], 1)
        self.assertEqual(pick["first_limit_time"], "093500")

    def test_strategy_uses_only_global_latest_trade_date(self):
        df = _sample_strategy_df()
        stale = df[df["ts_code"].eq("000001.SZ")].copy()
        stale["ts_code"] = "000006.SZ"
        stale["name"] = "停牌旧首板"
        stale["trade_date"] = stale["trade_date"].apply(lambda x: "20260612" if x == stale["trade_date"].max() else x)
        stale.loc[stale.index[-1], "pct_chg"] = 10.0
        result = run_beijing_chaoshao_strategy(pd.concat([df, stale], ignore_index=True), top_n=20)

        codes = [p["code"] for p in result["picks"]]
        self.assertNotIn("000006", codes)
        self.assertEqual(result["trade_date"], df["trade_date"].max())

    def test_strategy_excludes_bj_by_default_but_can_enable_it(self):
        df = _sample_strategy_df()
        bj = df[df["ts_code"].eq("000001.SZ")].copy()
        bj["ts_code"] = "920405.BJ"
        bj["name"] = "北交所首板"
        bj.loc[bj.index[-1], "pct_chg"] = 20.0
        merged = pd.concat([df, bj], ignore_index=True)

        default_result = run_beijing_chaoshao_strategy(merged, top_n=20)
        enabled_result = run_beijing_chaoshao_strategy(merged, top_n=20, exclude_bj=False)

        self.assertNotIn("920405", [p["code"] for p in default_result["picks"]])
        self.assertIn("920405", [p["code"] for p in enabled_result["picks"]])
        self.assertTrue(default_result["stats"]["exclude_bj"])

    def test_market_value_fallback_is_row_level(self):
        df = _sample_strategy_df()
        mask = df["ts_code"].eq("000001.SZ")
        df.loc[mask, "circ_mv"] = 0
        df.loc[mask, "total_mv"] = 0
        df.loc[mask, "float_mv"] = 60 * 10000
        result = run_beijing_chaoshao_strategy(df, top_n=10)

        pick = next(p for p in result["picks"] if p["code"] == "000001")
        self.assertAlmostEqual(pick["market_value"], 60.0)

    def test_api_runs_strategy_with_default_market_cap_preference(self):
        with patch("stock_app.views.get_latest_trading_date", return_value="20260613"), \
             patch("stock_app.views.get_real_stock_data", return_value=_sample_strategy_df()) as mocked_data:
            res = self.client.post(
                "/api/strategy/run/",
                data=json.dumps({"strategy_id": BEIJING_CHAOJIA_STRATEGY_ID, "top_n": 5}),
                content_type="application/json",
            )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["stats"]["mv_min_yi"], 20.0)
        self.assertEqual(payload["stats"]["mv_max_yi"], 100.0)
        self.assertEqual(payload["stats"]["price_max"], 25.0)
        self.assertIn("用于初筛", payload["stats"]["filter_note"])
        self.assertEqual(payload["strategy"]["description"].count("3~25"), 1)
        self.assertGreaterEqual(len(payload["picks"]), 1)
        mocked_data.assert_called_once()

    def test_async_strategy_task_reports_progress_and_result(self):
        with patch("stock_app.views.get_latest_trading_date", return_value="20260613"), \
             patch("stock_app.views.get_real_stock_data", return_value=_sample_strategy_df()), \
             patch("stock_app.views.enrich_limit_pool_fields", side_effect=lambda df, trade_date=None, pro=None: df):
            res = self.client.post(
                "/api/strategy/start/",
                data=json.dumps({"strategy_id": BEIJING_CHAOJIA_STRATEGY_ID, "top_n": 5}),
                content_type="application/json",
            )
            self.assertEqual(res.status_code, 202)
            task_id = res.json()["task_id"]
            payload = None
            for _ in range(30):
                progress = self.client.get(f"/api/strategy/progress/?task_id={task_id}")
                self.assertEqual(progress.status_code, 200)
                payload = progress.json()
                self.assertIn("percent", payload)
                if payload["state"] == "completed":
                    break
                time.sleep(0.1)

        self.assertIsNotNone(payload)
        self.assertEqual(payload["state"], "completed")
        self.assertEqual(payload["result"]["status"], "success")
        self.assertEqual(payload["result"]["strategy"]["id"], BEIJING_CHAOJIA_STRATEGY_ID)

    def test_strategy_holding_preserves_snapshot_and_exit_rules_when_added(self):
        result = run_beijing_chaoshao_strategy(_sample_strategy_df(), top_n=1)
        stock = result["picks"][0]
        res = self.client.post(
            "/api/portfolio/holdings/add/",
            data=json.dumps({"stock": stock, "shares": 1000, "total_capital": 100000}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        holding = PortfolioHolding.objects.get(code=stock["code"], status="holding")
        self.assertEqual(holding.strategy_id, BEIJING_CHAOJIA_STRATEGY_ID)
        self.assertEqual(holding.strategy_name, "北京炒家首板策略")
        self.assertIn("strategy_dimensions", holding.strategy_snapshot)
        self.assertEqual(holding.exit_rule_snapshot["单票硬止损"], "-5%")
        trade = PortfolioTrade.objects.get(code=stock["code"])
        self.assertEqual(trade.metadata["strategy_id"], BEIJING_CHAOJIA_STRATEGY_ID)

    def test_limit_pool_enrichment_fills_sealing_fields_and_market_cap(self):
        class FakePro:
            def limit_list_d(self, trade_date, fields):
                return pd.DataFrame([{
                    "trade_date": trade_date,
                    "ts_code": "000001.SZ",
                    "fd_amount": 500000,
                    "limit_amount": 600000,
                    "first_time": "093100",
                    "last_time": "145600",
                    "open_times": 2,
                    "up_stat": "1/1",
                    "limit_times": 1,
                    "total_mv": 660000,
                    "float_mv": 500000,
                    "turnover_ratio": 12.3,
                }])

        df = pd.DataFrame([{
            "ts_code": "000001.SZ",
            "trade_date": "20260613",
            "close": 10.0,
            "total_mv": 0,
        }])
        enriched = enrich_limit_pool_fields(df, trade_date="20260613", pro=FakePro())
        self.assertEqual(float(enriched.loc[0, "fd_amount"]), 500000)
        self.assertEqual(int(enriched.loc[0, "open_times"]), 2)
        self.assertEqual(enriched.loc[0, "first_time"], "093100")
        self.assertEqual(float(enriched.loc[0, "total_mv"]), 66)
        cache_file = Path(__file__).resolve().parent.parent / "data_cache" / "limit_pool_20260613.parquet"
        if cache_file.exists():
            cache_file.unlink()

    def test_limit_pool_enrichment_downloads_when_columns_exist_but_empty(self):
        class FakePro:
            def __init__(self):
                self.called = False

            def limit_list_d(self, trade_date, fields):
                self.called = True
                return pd.DataFrame([{
                    "trade_date": trade_date,
                    "ts_code": "000001.SZ",
                    "fd_amount": 700000,
                    "limit_amount": 800000,
                    "first_time": "094000",
                    "last_time": "144500",
                    "open_times": 1,
                    "total_mv": 660000,
                    "float_mv": 500000,
                    "turnover_ratio": 10.5,
                }])

        fake = FakePro()
        cache_file = Path(__file__).resolve().parent.parent / "data_cache" / "limit_pool_20260614.parquet"
        if cache_file.exists():
            cache_file.unlink()
        df = pd.DataFrame([{
            "ts_code": "000001.SZ",
            "trade_date": "20260614",
            "close": 10.0,
            "fd_amount": None,
            "open_times": None,
            "first_time": "",
        }])
        enriched = enrich_limit_pool_fields(df, trade_date="20260614", pro=fake)
        self.assertTrue(fake.called)
        self.assertEqual(float(enriched.loc[0, "fd_amount"]), 700000)
        self.assertEqual(int(enriched.loc[0, "open_times"]), 1)
        if cache_file.exists():
            cache_file.unlink()


class IndexWeightPoolTests(TestCase):
    def test_index_weight_pool_keeps_only_latest_trade_date(self):
        raw = pd.DataFrame({
            "index_code": ["000852.CSI"] * 4,
            "con_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
            "trade_date": ["20260331", "20260331", "20260630", "20260630"],
            "weight": [0.1, 0.2, 0.3, 0.4],
        })

        normalized = _normalize_index_weight_pool(raw, "000852.CSI")

        self.assertEqual(set(normalized["trade_date"]), {"20260630"})
        self.assertEqual(normalized["ts_code"].tolist(), ["000003.SZ", "000004.SZ"])
