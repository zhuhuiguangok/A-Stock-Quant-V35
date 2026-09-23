import numpy as np
import pandas as pd
from django.test import SimpleTestCase

from stock_app.factor_engine import FACTOR_WEIGHTS_2026, calculate_rule_factors


class ShortTermFactorTests(SimpleTestCase):
    def test_five_day_candidates_are_computed_without_manual_weights(self):
        rows = []
        for i in range(8):
            rows.append(
                {
                    "ts_code": "000001.SZ",
                    "trade_date": f"202606{10 + i:02d}",
                    "open": 10 + i * 0.1,
                    "high": 10.5 + i * 0.2,
                    "low": 9.8 + i * 0.1,
                    "close": 10 + i * 0.2,
                    "vol": 1000 + i * 100,
                    "amount": 100000 + i * 15000,
                    "turnover_rate": 1.0 + i * 0.1,
                    "total_mv": 100000 + i * 1000,
                    "circ_mv": 80000 + i * 800,
                    "pe": 12.0,
                    "pb": 1.5,
                    "industry": "bank",
                }
            )
        df = pd.DataFrame(rows)

        result = calculate_rule_factors(df)

        new_factors = [
            "vol_ratio_5d",
            "amount_ratio_5d",
            "turnover_mean_5d",
            "price_position_5d",
            "range_position_5d",
            "drawdown_5d",
            "return_accel_5d",
        ]
        for factor in new_factors:
            self.assertIn(factor, result.columns)
            self.assertFalse(np.isnan(result[factor].iloc[-1]))
            self.assertNotIn(factor, FACTOR_WEIGHTS_2026)
