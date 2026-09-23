from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from stock_app import views


class BoardFilterTests(SimpleTestCase):
    def test_filters_chinext_codes_and_preserves_order(self):
        codes = ["600000.SH", "300001.SZ", "301001.SZ", "688001.SH"]

        result = views._filter_stock_codes(
            codes,
            exclude_chinext=True,
            exclude_star=False,
        )

        self.assertEqual(result, ["600000.SH", "688001.SH"])

    def test_filters_star_market_codes(self):
        codes = ["600000.SH", "688001.SH", "689001.SH", "300001.SZ"]

        result = views._filter_stock_codes(
            codes,
            exclude_chinext=False,
            exclude_star=True,
        )

        self.assertEqual(result, ["600000.SH", "300001.SZ"])

    def test_can_disable_all_board_filters(self):
        codes = ["300001.SZ", "688001.SH"]

        result = views._filter_stock_codes(
            codes,
            exclude_chinext=False,
            exclude_star=False,
        )

        self.assertEqual(result, codes)


class CachedIndexPoolTests(SimpleTestCase):
    def test_loads_latest_local_constituent_snapshot(self):
        codes = [f'{index:06d}.SZ' for index in range(700)]
        frame = pd.DataFrame({
            'ts_code': codes,
            'trade_date': ['20260814'] * len(codes),
        })
        with TemporaryDirectory() as cache_dir:
            frame.to_parquet(
                f'{cache_dir}/real_data_csi1000_20250819_20260814.parquet',
                index=False,
            )
            result = views._load_cached_index_pool('csi1000', cache_dir=cache_dir)

        self.assertEqual(len(result), 700)
        self.assertEqual(result['ts_code'].nunique(), 700)


class SelectionRequestFilterTests(SimpleTestCase):
    @patch("stock_app.views.enhance_stock_selection_v19")
    @patch("stock_app.views.get_real_stock_data")
    def test_selection_request_passes_pool_and_board_filters(
        self,
        get_real_stock_data,
        enhance_stock_selection,
    ):
        frame = pd.DataFrame(
            {
                "ts_code": ["600000.SH"],
                "trade_date": ["20260618"],
                "close": [10.0],
            }
        )
        get_real_stock_data.return_value = frame
        enhance_stock_selection.return_value = pd.DataFrame()

        response = self.client.post(
            "/api/select/",
            data={
                "max_stocks": 20,
                "stock_pool": "all",
                "exclude_chinext": False,
                "exclude_star": True,
            },
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        get_real_stock_data.assert_called_once_with(
            start_date=None,
            end_date=None,
            stock_pool="all",
            lookback_months=12,
            exclude_chinext=False,
            exclude_star=True,
        )


class BoardFilterTemplateTests(SimpleTestCase):
    def test_homepage_has_default_enabled_board_filters(self):
        response = self.client.get("/")
        content = response.content.decode("utf-8")

        self.assertContains(response, 'id="excludeChiNext"')
        self.assertContains(response, 'id="excludeStar"')
        self.assertIn('id="excludeChiNext" checked', content)
        self.assertIn('id="excludeStar" checked', content)
        self.assertIn("exclude_chinext:", content)
        self.assertIn("exclude_star:", content)
