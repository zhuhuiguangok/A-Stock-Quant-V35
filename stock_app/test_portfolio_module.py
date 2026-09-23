import json

from django.test import TestCase

from .models import PortfolioAccount, PortfolioExitScan, PortfolioHolding, PortfolioTrade


class PortfolioModuleTests(TestCase):
    def _stock(self, price=10.0, position_pct=7.0):
        return {
            "code": "000001",
            "ts_code": "000001.SZ",
            "name": "平安银行",
            "industry": "银行",
            "current_price": price,
            "position_pct": position_pct,
            "position_level": "normal",
            "buy_score": 82,
            "ml_score": 68,
            "buy_signals": ["高置信度"],
        }

    def test_add_same_stock_recalculates_average_cost_and_position(self):
        res1 = self.client.post(
            "/api/portfolio/holdings/add/",
            data=json.dumps({"stock": self._stock(price=10), "shares": 700, "total_capital": 100000}),
            content_type="application/json",
        )
        self.assertEqual(res1.status_code, 200)
        res2 = self.client.post(
            "/api/portfolio/holdings/add/",
            data=json.dumps({"stock": self._stock(price=12), "shares": 300, "total_capital": 100000}),
            content_type="application/json",
        )
        self.assertEqual(res2.status_code, 200)

        holding = PortfolioHolding.objects.get(code="000001", status="holding")
        self.assertEqual(holding.shares, 1000)
        self.assertAlmostEqual(float(holding.avg_cost), 10.6, places=4)
        self.assertAlmostEqual(float(holding.cost_amount), 10600.0, places=2)
        self.assertAlmostEqual(float(holding.actual_position_pct), 10.6, places=2)
        self.assertEqual(PortfolioTrade.objects.filter(code="000001").count(), 2)
        self.assertEqual(list(PortfolioTrade.objects.values_list("side", flat=True)), ["BUY", "ADD"])

        holdings = self.client.post(
            "/api/portfolio/holdings/",
            data=json.dumps({"current_data": {"000001": {"close": 11}}}),
            content_type="application/json",
        )
        payload = holdings.json()
        self.assertEqual(payload["status"], "success")
        self.assertAlmostEqual(payload["holdings"][0]["pnl_pct"], 3.77, places=2)

    def test_exit_scan_uses_persisted_holdings_when_no_positions_payload(self):
        self.client.post(
            "/api/portfolio/holdings/add/",
            data=json.dumps({"stock": self._stock(price=10), "shares": 1000, "total_capital": 100000}),
            content_type="application/json",
        )
        res = self.client.post(
            "/api/exit-scan/",
            data=json.dumps({"current_data": {"000001": {"close": 9.0, "ma20": 10.0, "ai_score": 68}}}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["summary"]["hard_stop_count"], 1)
        self.assertEqual(PortfolioExitScan.objects.count(), 1)

    def test_close_holding_keeps_trade_audit_record(self):
        self.client.post(
            "/api/portfolio/holdings/add/",
            data=json.dumps({"stock": self._stock(price=10), "shares": 1000, "total_capital": 100000}),
            content_type="application/json",
        )
        res = self.client.post(
            "/api/portfolio/holdings/close/",
            data=json.dumps({"code": "000001", "price": 11, "trade_date": "20260627"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        holding = PortfolioHolding.objects.get(code="000001")
        self.assertEqual(holding.status, "closed")
        self.assertAlmostEqual(float(holding.realized_pnl), 1000.0, places=2)
        self.assertEqual(PortfolioTrade.objects.filter(code="000001").count(), 2)

    def test_partial_sell_reduces_shares_and_keeps_average_cost(self):
        self.client.post(
            "/api/portfolio/holdings/add/",
            data=json.dumps({"stock": self._stock(price=10), "shares": 1000, "total_capital": 100000}),
            content_type="application/json",
        )
        res = self.client.post(
            "/api/portfolio/holdings/sell/",
            data=json.dumps({"code": "000001", "shares": 500, "price": 12, "trade_date": "20260627"}),
            content_type="application/json",
        )
        self.assertEqual(res.status_code, 200)
        holding = PortfolioHolding.objects.get(code="000001")
        self.assertEqual(holding.status, "holding")
        self.assertEqual(holding.shares, 500)
        self.assertAlmostEqual(float(holding.avg_cost), 10.0, places=4)
        self.assertAlmostEqual(float(holding.cost_amount), 5000.0, places=2)
        self.assertAlmostEqual(float(holding.realized_pnl), 1000.0, places=2)
        self.assertEqual(list(PortfolioTrade.objects.values_list("side", flat=True)), ["BUY", "REDUCE"])
