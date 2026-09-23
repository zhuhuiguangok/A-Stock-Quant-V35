from pathlib import Path
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from stock_app.tushare_client import (
    FIXED_TUSHARE_TOKEN,
    TUSHARE_HTTP_URL,
    create_tushare_pro,
    get_tushare_token,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROXY_URL = "http://lianghua.nanyangqiankun.top"


class TushareProxyConfigurationTests(unittest.TestCase):
    def test_shared_client_patches_token_and_proxy_url(self):
        fake_pro = SimpleNamespace()
        with patch('stock_app.tushare_client.ts.set_token') as set_token, \
                patch('stock_app.tushare_client.ts.pro_api', return_value=fake_pro) as pro_api:
            result = create_tushare_pro(timeout=30)

        self.assertIs(result, fake_pro)
        set_token.assert_called_once_with(FIXED_TUSHARE_TOKEN)
        pro_api.assert_called_once_with(FIXED_TUSHARE_TOKEN, timeout=30)
        self.assertEqual(fake_pro._DataApi__token, FIXED_TUSHARE_TOKEN)
        self.assertEqual(fake_pro._DataApi__http_url, TUSHARE_HTTP_URL)

    def test_environment_cannot_replace_platform_token(self):
        with patch.dict('os.environ', {'TUSHARE_TOKEN': 'user-supplied-token'}):
            self.assertEqual(get_tushare_token(), FIXED_TUSHARE_TOKEN)

    def test_application_modules_use_shared_client(self):
        source_files = [
            PROJECT_ROOT / "stock_app" / "views.py",
            PROJECT_ROOT / "stock_app" / "fund_flow_service.py",
            PROJECT_ROOT / "stock_app" / "risk_factor_builder.py",
        ]
        direct_clients = []
        for source_file in source_files:
            lines = source_file.read_text(encoding="utf-8").splitlines()
            direct_clients.extend(
                f"{source_file.name}:{index + 1}"
                for index, line in enumerate(lines)
                if re.search(r"ts\.(?:set_token|pro_api)\(", line)
            )
        self.assertEqual(direct_clients, [])


if __name__ == "__main__":
    unittest.main()
