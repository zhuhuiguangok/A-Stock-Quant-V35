"""市场数据层回归测试（Jev LLM 预测层 2026-10-28 已下线，此文件锁数据层行为）。"""
import json
from datetime import date
from unittest import mock

from django.test import SimpleTestCase

from stock_app import jev_service as js


class GanZhiTest(SimpleTestCase):
    def test_anchor_1949_10_01_is_jiazi(self):
        self.assertEqual(js.day_ganzhi(date(1949, 10, 1)), ("甲", "子", 0))

    def test_known_date_2000_01_01_is_wuwu(self):
        """万年历公认值：2000-01-01 日柱戊午。"""
        self.assertEqual(js.day_ganzhi(date(2000, 1, 1)), ("戊", "午", 54))

    def test_year_pillar_lichun_boundary(self):
        # 2024-02-03（立春前）仍属癸卯年；02-04 起甲辰年
        self.assertEqual(js.year_ganzhi(date(2024, 2, 3)), ("癸", "卯"))
        self.assertEqual(js.year_ganzhi(date(2024, 2, 4)), ("甲", "辰"))

    def test_yiji_context_complete(self):
        ctx = js.yiji_context(date(2026, 9, 21))
        for key in ("日干", "日支", "日干五行", "年柱", "上卦", "卦名"):
            self.assertTrue(ctx.get(key))


class NoLeakageTest(SimpleTestCase):
    def test_slice_as_of_blocks_future_rows(self):
        rows = [("2026010" + str(i), 3000.0 + i) for i in range(1, 10)]
        kept = js._slice_as_of(rows, date(2026, 1, 7), lookback=40)
        self.assertTrue(all(d <= "20260107" for d, _ in kept))
        self.assertEqual(kept[-1][0], "20260107")

    def test_predict_prompt_has_no_future_rows(self):
        """预测 T+1 时传入特征必须全部来自 ≤T。"""
        rows = [("2026010" + str(i), 3000.0 + i) for i in range(1, 10)]
        feats = js._market_features(js._slice_as_of(rows, date(2026, 1, 7)))
        self.assertEqual(feats["区间"], "20260101~20260107")


class ParseAndMetricsTest(SimpleTestCase):
    def test_parse_json_tolerates_wrapped_text(self):
        self.assertEqual(js._parse_json('前置说明 {"p_up": 0.6} 后缀'), {"p_up": 0.6})
        self.assertIsNone(js._parse_json(None))
        self.assertIsNone(js._parse_json("完全不是JSON"))

    def test_clamp01(self):
        self.assertEqual(js._clamp01(1.7), 1.0)
        self.assertEqual(js._clamp01(-3), 0.0)
        self.assertEqual(js._clamp01("bad"), 0.5)


class NumberProvenanceTest(SimpleTestCase):
    """问一问数字溯源校验（反金融幻觉的确定性防线）。"""

    def test_traced_numbers_pass(self):
        from stock_app.stock_qa_service import _unverifiable_numbers

        ctx = "现价 67.5，MA20=77.8，20日涨跌% -24.9，研报覆盖 1 篇"
        ans = "现价 67.5，20日跌了近 25%，MA20 在 77.8。"
        self.assertEqual(_unverifiable_numbers(ans, ctx), [])

    def test_fabricated_precision_flagged(self):
        from stock_app.stock_qa_service import _unverifiable_numbers

        ctx = "现价 67.5，20日涨跌% -24.9"
        ans = "市盈率 15.32 倍，目标价 82.46，主力流入 3.25 亿。"
        flagged = _unverifiable_numbers(ans, ctx)
        self.assertTrue(any("15.32" in f for f in flagged), flagged)
        self.assertTrue(any("82.46" in f for f in flagged), flagged)

    def test_years_exempt(self):
        from stock_app.stock_qa_service import _unverifiable_numbers

        ctx = "数据日 20261030"
        ans = "2026 年以来跌了不少，2024 年也跌过。"
        self.assertEqual(_unverifiable_numbers(ans, ctx), [])
