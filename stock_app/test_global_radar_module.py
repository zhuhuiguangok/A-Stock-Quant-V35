# -*- coding: utf-8 -*-
"""全球雷达模块单元测试：分析层纯函数 + 服务层（mock 采集与 AI）。"""
import json
import unittest
from pathlib import Path
from unittest import mock

from django.test import TestCase

from stock_app.global_radar import analysis, service


def _series(start, drift, n=70):
    """构造带漂移的价格序列。"""
    out, v = [], float(start)
    for i in range(n):
        v *= 1 + drift
        out.append(round(v, 4))
    return out


class HeatAndScoreTest(unittest.TestCase):
    def test_chg_windows(self):
        closes = [100 + i for i in range(70)]
        self.assertEqual(analysis._chg(closes, 1), round((169 - 168) / 168 * 100, 2))
        self.assertEqual(analysis._chg(closes, 5), round((169 - 164) / 164 * 100, 2))
        self.assertIsNone(analysis._chg(closes, 200))

    def test_heat_levels(self):
        self.assertEqual(analysis._heat_level(6.0), 3)
        self.assertEqual(analysis._heat_level(2.0), 2)
        self.assertEqual(analysis._heat_level(0.5), 1)
        self.assertEqual(analysis._heat_level(-6.0), -3)
        self.assertEqual(analysis._heat_level(None), 0)

    def test_asset_heat_rows(self):
        assets = {"标普500": _series(4000, 0.001), "COMEX黄金": _series(2400, 0.002)}
        heat = analysis.build_asset_heat(assets, us10y=[4.0, 4.05, 4.1, 4.15, 4.2, 4.3], btc=None)
        names = [r["name"] for r in heat["rows"]]
        self.assertIn("标普500", names)
        self.assertIn("美债10Y收益率(%)", names)
        row = next(r for r in heat["rows"] if r["name"] == "标普500")
        self.assertIsNotNone(row["chg_1d"])

    def test_risk_appetite_bounds(self):
        assets = {"标普500": _series(4000, 0.003), "美元指数": _series(104, -0.002),
                  "COMEX黄金": _series(2400, 0.0), "WTI原油": _series(80, 0.004)}
        ra = analysis.build_risk_appetite(assets, us10y=[4.5] * 30)
        self.assertIsNotNone(ra["score"])
        self.assertTrue(0 <= ra["score"] <= 100)

    def test_risk_appetite_empty(self):
        ra = analysis.build_risk_appetite({}, us10y=None)
        self.assertIsNone(ra["score"])
        self.assertEqual(ra["label"], "数据不足")


class FearGreedTest(unittest.TestCase):
    def test_score_bounds_and_fallback_vol(self):
        fg = analysis.build_fear_greed(
            qvix=None,  # QVIX 缺失 → 已实现波动率兜底
            margin=[(f"202608{i:02d}", 20000.0 + i * 10) for i in range(1, 21)],
            breadth={"up": 3000, "down": 2000, "limit_up": 80, "amount_yi": 20000},
            assets={"沪深300": _series(4000, 0.002)},
        )
        self.assertIsNotNone(fg["score"])
        self.assertTrue(20 <= fg["score"] <= 100)
        self.assertIsNotNone(fg["realized_vol"])
        comps = {c["name"] for c in fg["components"]}
        self.assertIn("沪深300已实现波动(20日)", comps)

    def test_margin_spike_does_not_explode(self):
        fg = analysis.build_fear_greed(None, [(f"d{i}", 100.0) for i in range(20)], None, {"沪深300": _series(4000, 0)})
        self.assertTrue(0 <= fg["score"] <= 100)


class ClockTest(unittest.TestCase):
    def _macro(self, pmi, cpi):
        return {"pmi": [(f"2026{m:02d}", v) for m, v in enumerate(pmi, 1)], "cpi": [(f"2026{m:02d}", v) for m, v in enumerate(cpi, 1)]}

    def test_overheat_quadrant(self):
        macro = self._macro([49.0, 49.2, 50.1, 50.5, 51.0, 51.4], [0.2, 0.3, 0.3, 0.5, 0.8, 1.2])
        clock = analysis.build_clock_china(macro)
        self.assertTrue(clock["ok"])
        self.assertEqual(clock["quadrant"], "overheat")
        self.assertEqual(clock["asset"], "商品占优")

    def test_reflation_quadrant(self):
        macro = self._macro([51.4, 51.0, 50.5, 50.1, 49.6, 49.2], [1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
        clock = analysis.build_clock_china(macro)
        self.assertIn(clock["quadrant"], ("reflation", "recovery"))

    def test_no_data(self):
        clock = analysis.build_clock_china(None)
        self.assertFalse(clock["ok"])

    def test_us_proxy(self):
        clock = analysis.build_clock_us({"标普500": _series(4000, 0.002), "COMEX黄金": _series(2400, 0.0), "WTI原油": _series(80, 0.0)})
        self.assertTrue(clock["ok"])
        self.assertIn(clock["quadrant"], ("recovery", "overheat", "reflation", "stagflation"))


class ServiceTest(TestCase):
    def setUp(self):
        # 隔离真实缓存目录：测试的假面板/假晨报不能污染 data_cache/global_radar
        import tempfile
        from unittest import mock as _mock

        self._tmp = tempfile.TemporaryDirectory()
        from stock_app.global_radar import service as _svc
        root = Path(self._tmp.name)
        self._patchers = [
            _mock.patch.object(_svc, "GLOBAL_RADAR_ROOT", root),
            _mock.patch.object(_svc, "DASHBOARD_PATH", root / "dashboard_latest.json"),
            _mock.patch.object(_svc, "BRIEF_PATH", root / "brief_latest.json"),
        ]
        for p_ in self._patchers:
            p_.start()
        self.addCleanup(lambda: [p_.stop() for p_ in self._patchers])
        self.addCleanup(self._tmp.cleanup)

    def test_build_and_save_dashboard_with_mocks(self):
        fake = {
            "asset_heat": analysis.build_asset_heat({"标普500": _series(4000, 0.001)}, None, None),
            "risk_appetite": {"score": 55, "label": "中性摇摆", "components": []},
            "fear_greed": {"score": 60, "label": "贪婪", "components": []},
            "clock_cn": {"ok": True, "quadrant": "recovery", "quadrant_name": "复苏（Recovery）", "asset": "股票占优", "hint": "h", "region": "中国"},
            "clock_us": {"ok": True, "quadrant": "recovery", "quadrant_name": "复苏（Recovery）", "asset": "股票占优", "hint": "h", "region": "美国"},
            "meta": {"assets_ok": ["标普500"], "sources_failed": []},
        }
        with mock.patch.object(service, "build_dashboard", return_value=fake):
            service.save_dashboard(fake)
            loaded = service.read_dashboard()
            self.assertEqual(loaded["risk_appetite"]["score"], 55)
            # 事实清单可序列化且包含关键内容
            facts = service._brief_facts(fake)
            self.assertIn("全球风险偏好分 55", facts)

    def test_generate_ai_brief_mock(self):
        payload = {"risk_appetite": {"score": 55, "label": "中性"}, "fear_greed": {"score": 60, "label": "贪婪"},
                   "asset_heat": {"rows": []}, "clock_cn": {}, "clock_us": {}, "meta": {}}
        fake_ai = {"headline": "测试晨报", "points": ["要点1"], "impact": "影响", "focus": "关注", "sentiment": "bullish"}
        with mock.patch("stock_app.global_radar.ai_compat.ai_available", return_value=True), \
             mock.patch("stock_app.global_radar.ai_compat.ask_json", return_value=fake_ai):
            result = service.generate_ai_brief(payload)
        self.assertEqual(result["status"], "success")
        brief = service.read_brief()
        self.assertIsNotNone(brief)
        self.assertEqual(brief["headline"], "测试晨报")
        json.dumps(brief, ensure_ascii=False)  # 可序列化

    def test_generate_ai_brief_no_key(self):
        with mock.patch("stock_app.global_radar.ai_compat.ai_available", return_value=False):
            result = service.generate_ai_brief({"asset_heat": {"rows": []}, "meta": {}})
        self.assertEqual(result["status"], "skipped")


if __name__ == "__main__":
    unittest.main()


class RichnessTest(unittest.TestCase):
    def test_spark_and_pct_rank(self):
        closes = [100 * (1 + 0.001 * i) for i in range(70)]
        spark = analysis._spark_points(closes)
        self.assertEqual(len(spark), 20)
        self.assertTrue(all(0 <= y <= 18 for y in spark))
        rank = analysis._pct_rank(closes, 20)
        self.assertIsNotNone(rank)
        self.assertTrue(0 <= rank <= 100)
        self.assertIsNone(analysis._pct_rank([1, 2, 3], 20))

    def test_asset_heat_interpretation(self):
        # 美元 20 日强上行 → 解读出现"流动性收紧"
        dxy = [104 - i * 0.05 for i in range(70, 0, -1)]
        heat = analysis.build_asset_heat({"美元指数": dxy}, None, None)
        row = next(r for r in heat["rows"] if r["name"] == "美元指数")
        self.assertIn("美元变贵", row["interp"])
        self.assertIsNotNone(row["spark"])

    def test_transmission_chains(self):
        assets = {
            "美元指数": [104 - i * 0.09 for i in range(70, 0, -1)],  # 上行（20日约+1.7%）
            "COMEX铜": [80 * (1 + 0.002 * i) for i in range(70)],    # 上行
            "韩国KOSPI": [2500 * (1 + 0.002 * i) for i in range(70)],
            "恒生科技": [3800 * (1 + 0.003 * i) for i in range(70)],
        }
        us10y = [4.5 - i * 0.005 for i in range(60)]  # 利率下行（首日4.5→末日4.2）
        chains = analysis.build_transmission(assets, us10y)
        self.assertEqual(len(chains), 5)
        by_name = {c["name"]: c for c in chains}
        self.assertEqual(by_name["美元流动性 → 外资面"]["state_dir"], 1)
        self.assertEqual(by_name["美债利率 → 风格切换"]["state_dir"], -1)
        self.assertIn("科创/成长", by_name["美债利率 → 风格切换"]["a_directions"])

    def test_a_share_directions_with_resonance(self):
        assets = {
            "COMEX铜": [80 * (1 + 0.002 * i) for i in range(70)],
            "美元指数": [104 + i * 0.02 for i in range(70)],  # 美元走弱
        }
        us10y = [4.5 - i * 0.005 for i in range(60)]  # 利率下行（首日4.5→末日4.2）
        flow = [{"sector_name": "铜"}, {"sector_name": "有色"}]
        research = {"有色金属": 8}
        result = analysis.build_a_share_directions(assets, us10y, flow, research)
        dirs = result["directions"]
        self.assertTrue(dirs)
        self.assertIn("confirms", dirs[0])
        self.assertTrue(dirs[0]["confirms"] >= dirs[-1]["confirms"])
        # 铜上行应映射到有色方向
        names = [d["direction"] for d in dirs]
        self.assertTrue(any("有色" in n for n in names))

    def test_directions_neutral(self):
        flat = [100.0] * 70
        result = analysis.build_a_share_directions({"标普500": flat}, None, None, None)
        self.assertEqual(result["directions"], [])


class PlainLayerTest(unittest.TestCase):
    def test_decorate_adds_plain_fields(self):
        payload = analysis.decorate({
            "risk_appetite": {"score": 70},
            "fear_greed": {"score": 15},
            "clock_cn": {"ok": True, "quadrant": "overheat"},
            "clock_us": {"ok": True, "quadrant": "reflation"},
            "asset_heat": {"rows": [{"name": "COMEX黄金"}]},
        })
        self.assertEqual(payload["risk_appetite"]["emoji"], "🎉")
        self.assertEqual(payload["fear_greed"]["plain"], "冰点")
        self.assertIn("别人恐惧", payload["fear_greed"]["hint"])
        self.assertEqual(payload["clock_cn"]["season"], "夏天")
        self.assertEqual(payload["clock_us"]["season"], "冬天")
        self.assertEqual(payload["asset_heat"]["rows"][0]["emoji"], "🥇")

    def test_plain_layers_none_score(self):
        self.assertEqual(analysis.risk_plain(None)["plain"], "数据不足")
        self.assertEqual(analysis.fg_plain(None)["emoji"], "❓")
        self.assertEqual(analysis.clock_season(None)["season"], "--")
