from unittest.mock import patch

import pandas as pd
from django.test import TestCase, override_settings


def _snapshot(date, values):
    return pd.DataFrame(
        [
            {
                "date": date,
                "sector_name": name,
                "main_force_net": value,
                "change_pct": pct,
            }
            for name, value, pct in values
        ]
    )


class FundFlowAlgorithmTests(TestCase):
    def test_rotation_forecast_matches_source_algorithm_contract(self):
        from stock_app.fund_flow.analysis import build_fund_flow_payload

        old_energy = [value * 100000000 for value in [-10, -50, -15, -80, -20, -110, -30, -140, -40, -170, -50, -200]]
        new_ai = [value * 100000000 for value in [0, 0, 10, 50, 15, 80, 20, 110, 30, 140, 40, 170]]
        banks = [value * 100000000 for value in [25, -10, 18, -8, 14, -6, 10, -5, 9, -3, 6, -2]]
        history = pd.concat(
            [
                _snapshot(
                    f"202607{day + 1:02d}",
                    [
                        ("old_energy", old_energy[day], -1.0),
                        ("new_ai", new_ai[day], 1.0),
                        ("banks", banks[day], 0.1),
                    ],
                )
                for day in range(12)
            ],
            ignore_index=True,
        )

        payload = build_fund_flow_payload(history[history["date"] == "20260712"], history, source="cache")

        top = payload["rotation_edges"][0]
        self.assertEqual(top["source"], "old_energy")
        self.assertEqual(top["target"], "new_ai")
        self.assertEqual(top["lag_days"], 2)
        self.assertGreaterEqual(top["confidence"], 0.7)
        self.assertEqual(payload["rotation_forecast"]["strongest_route"], top)

    def test_tushare_moneyflow_aggregates_to_sector_snapshot(self):
        from stock_app.fund_flow_service import aggregate_tushare_industry_flow

        class FakePro:
            def moneyflow(self, trade_date, fields):
                return pd.DataFrame([
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": trade_date,
                        "buy_lg_amount": 10.0,
                        "sell_lg_amount": 4.0,
                        "buy_elg_amount": 3.0,
                        "sell_elg_amount": 1.0,
                        "net_mf_amount": 99.0,
                    },
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": trade_date,
                        "buy_lg_amount": 2.0,
                        "sell_lg_amount": 9.0,
                        "buy_elg_amount": 1.0,
                        "sell_elg_amount": 3.0,
                        "net_mf_amount": 88.0,
                    },
                    {
                        "ts_code": "000003.SZ",
                        "trade_date": trade_date,
                        "buy_lg_amount": 5.0,
                        "sell_lg_amount": 1.0,
                        "buy_elg_amount": 2.0,
                        "sell_elg_amount": 1.0,
                        "net_mf_amount": 77.0,
                    },
                ])

            def stock_basic(self, exchange, list_status, fields):
                return pd.DataFrame([
                    {"ts_code": "000001.SZ", "industry": "银行"},
                    {"ts_code": "000002.SZ", "industry": "银行"},
                    {"ts_code": "000003.SZ", "industry": "机器人"},
                ])

            def daily(self, trade_date, fields):
                return pd.DataFrame([
                    {"ts_code": "000001.SZ", "pct_chg": 1.0},
                    {"ts_code": "000002.SZ", "pct_chg": -2.0},
                    {"ts_code": "000003.SZ", "pct_chg": 3.0},
                ])

        frame = aggregate_tushare_industry_flow(FakePro(), "20260717")

        rows = {row["sector_name"]: row for row in frame.to_dict("records")}
        self.assertEqual(rows["银行"]["date"], "20260717")
        self.assertEqual(rows["银行"]["main_force_net"], -10000.0)
        self.assertEqual(rows["银行"]["change_pct"], -0.5)
        self.assertEqual(rows["机器人"]["main_force_net"], 50000.0)


@override_settings(ALLOWED_HOSTS=["testserver", "127.0.0.1", "localhost"])
class FundFlowModuleApiTests(TestCase):
    def test_status_returns_fund_flow_payload(self):
        payload = {
            "source": "cache",
            "date": "20260718",
            "summary": {"sector_count": 2, "total_inflow_yi": 10, "total_outflow_yi": 8, "net_flow_yi": 2},
            "top_inflow": [{"sector_name": "new_ai", "main_force_net_yi": 10.0, "change_pct": 2.1}],
            "top_outflow": [{"sector_name": "old_energy", "main_force_net_yi": -8.0, "change_pct": -1.2}],
            "rotation_edges": [],
            "rotation_forecast": {"status": "insufficient_data", "horizon_days": 3, "strongest_route": None, "edge_count": 0},
        }
        with patch("stock_app.views.get_fund_flow_payload", return_value=payload):
            response = self.client.get("/api/fund-flow/status/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["summary"]["sector_count"], 2)
        self.assertEqual(data["top_inflow"][0]["sector_name"], "new_ai")

    def test_update_starts_local_refresh_by_default(self):
        task = {"state": "running", "stage": "starting", "progress": 1}
        with patch("stock_app.views.start_fund_flow_refresh", return_value=task) as refresh, \
             patch("stock_app.views.start_fund_flow_update") as full_update:
            response = self.client.post("/api/fund-flow/update/", data="{}", content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task"]["stage"], "starting")
        refresh.assert_called_once()
        full_update.assert_not_called()

    def test_fund_flow_uses_unified_bull_bear_data_root(self):
        from stock_app.bull_bear_service import BULL_BEAR_DATA_ROOT
        from stock_app.fund_flow_service import FUND_FLOW_DATA_ROOT

        self.assertEqual(FUND_FLOW_DATA_ROOT, BULL_BEAR_DATA_ROOT)

    def test_unified_update_starts_bull_bear_and_fund_flow(self):
        from stock_app.bull_bear_service import start_unified_update

        bull_bear_task = {"state": "running", "stage": "fetch", "progress": 8}
        fund_flow_task = {"state": "running", "stage": "fetch", "progress": 8}
        with patch("stock_app.bull_bear_service.start_update", return_value=bull_bear_task) as bull_bear_update, \
             patch("stock_app.fund_flow_service.start_update", return_value=fund_flow_task) as fund_flow_update:
            task = start_unified_update("token")

        bull_bear_update.assert_called_once_with("token")
        fund_flow_update.assert_called_once()
        self.assertEqual(task["bull_bear_task"], bull_bear_task)
        self.assertEqual(task["fund_flow_task"], fund_flow_task)
        self.assertIsNone(task["fund_flow_error"])

    def test_bull_bear_full_update_downloads_fund_flow_in_same_flow(self):
        task = {"state": "running", "stage": "starting", "progress": 1}
        unified_task = {"bull_bear_task": task, "fund_flow_task": task, "fund_flow_error": None}
        with patch("stock_app.views.start_bull_bear_unified_update", return_value=unified_task) as full_update, \
             patch("stock_app.views.start_bull_bear_refresh") as refresh:
            response = self.client.post(
                "/api/bull-bear/update/",
                data='{"mode":"full_update"}',
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        full_update.assert_called_once()
        refresh.assert_not_called()
        self.assertEqual(response.json()["fund_flow_task"], task)

    def test_home_page_contains_fund_flow_markers(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        self.assertIn("navFundFlow", html)
        self.assertIn("fundFlowPanel", html)
        self.assertIn("fundFlowNetwork", html)
        self.assertIn("loadFundFlowDashboard", html)
