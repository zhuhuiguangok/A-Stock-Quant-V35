from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROXY_URL = "http://lianghua.nanyangqiankun.top"


class TushareProxyConfigurationTests(unittest.TestCase):
    def test_every_pro_api_client_sets_proxy_url(self):
        source_files = [
            PROJECT_ROOT / "stock_app" / "views.py",
            PROJECT_ROOT / "stock_app" / "risk_factor_builder.py",
        ]
        missing = []
        client_count = 0

        for source_file in source_files:
            lines = source_file.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                match = re.match(
                    r"^(?P<indent>\s*)(?P<name>_?pro)\s*=\s*ts\.pro_api\(",
                    line,
                )
                if not match:
                    continue

                client_count += 1
                expected = (
                    f"{match.group('indent')}{match.group('name')}"
                    f"._DataApi__http_url = '{PROXY_URL}'"
                )
                following_lines = lines[index + 1:index + 4]
                if expected not in following_lines:
                    missing.append(f"{source_file.name}:{index + 1}")

        self.assertEqual(client_count, 5)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
