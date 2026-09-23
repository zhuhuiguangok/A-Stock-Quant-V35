from unittest.mock import patch

from django.test import TestCase, override_settings


@override_settings(ALLOWED_HOSTS=["testserver", "127.0.0.1", "localhost"])
class BullBearModuleApiTests(TestCase):
    def test_status_returns_dashboard_payload(self):
        payload = {
            "signal": {"cycle_score": 66.4, "phase": "偏空", "date": "2026-07-03"},
            "history": [{"trade_date": "2026-07-03", "cycle_score": 66.4, "indicator_count": 11}],
            "charts": [{"file": "cycle_position.png", "url": "/api/bull-bear/chart/cycle_position.png"}],
            "cache": [{"dataset": "stocks", "rows": 12014658, "end": "20260703"}],
            "task": {"state": "idle", "progress": 0},
            "scheduler": {"enabled": True, "interval_seconds": 3600, "state": "waiting"},
        }
        with patch("stock_app.views.get_dashboard_payload", return_value=payload):
            response = self.client.get("/api/bull-bear/status/")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["signal"]["cycle_score"], 66.4)
        self.assertEqual(len(data["cache"]), 1)
        self.assertTrue(data["scheduler"]["enabled"])

    def test_update_defaults_to_local_cache_refresh(self):
        task = {"state": "running", "stage": "starting", "progress": 1}
        with patch("stock_app.views.start_bull_bear_refresh", return_value=task) as refresh, \
             patch("stock_app.views.start_bull_bear_update") as full_update:
            response = self.client.post("/api/bull-bear/update/", data="{}", content_type="application/json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["task"]["stage"], "starting")
        refresh.assert_called_once()
        full_update.assert_not_called()

    def test_update_full_mode_uses_network_update(self):
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
