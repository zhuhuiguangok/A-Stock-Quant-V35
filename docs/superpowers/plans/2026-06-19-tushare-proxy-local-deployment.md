# Tushare Proxy Local Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Tushare Pro client use `http://lianghua.nanyangqiankun.top`, then install and verify the Django project locally on Windows.

**Architecture:** Keep the existing direct `ts.pro_api(...)` initialization pattern. Add the proxy URL assignment immediately after each of the five client initializations, protect the behavior with a source-level regression test, and retain the existing environment-variable token flow.

**Tech Stack:** Python 3.12, Django 4.2, Tushare, unittest/pytest-compatible tests, Windows PowerShell

---

### Task 1: Add a failing proxy-coverage regression test

**Files:**
- Create: `stock_app/test_tushare_proxy.py`
- Inspect: `stock_app/views.py`
- Inspect: `stock_app/risk_factor_builder.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m unittest stock_app.test_tushare_proxy -v
```

Expected: FAIL listing all five current initialization locations as missing proxy assignments.

- [ ] **Step 3: Commit the failing test**

```powershell
git add stock_app/test_tushare_proxy.py
git commit -m "test: require proxy for every tushare client"
```

### Task 2: Configure all Tushare clients to use the proxy

**Files:**
- Modify: `stock_app/views.py:166`
- Modify: `stock_app/views.py:225`
- Modify: `stock_app/views.py:1787`
- Modify: `stock_app/views.py:3181`
- Modify: `stock_app/risk_factor_builder.py:33`
- Test: `stock_app/test_tushare_proxy.py`

- [ ] **Step 1: Add the proxy assignment after each client**

Use the corresponding variable name immediately after each initialization:

```python
pro = ts.pro_api()
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'
```

```python
_pro = ts.pro_api(timeout=20)
_pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'
```

```python
pro = ts.pro_api(timeout=30)
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'
```

- [ ] **Step 2: Run the focused test**

Run:

```powershell
python -m unittest stock_app.test_tushare_proxy -v
```

Expected: PASS, one test run with zero failures.

- [ ] **Step 3: Search for uncovered initializations**

Run:

```powershell
rg -n "ts\.pro_api|_DataApi__http_url" stock_app -g "*.py"
```

Expected: five `ts.pro_api` calls and five nearby proxy assignments.

- [ ] **Step 4: Commit the source change**

```powershell
git add stock_app/views.py stock_app/risk_factor_builder.py
git commit -m "fix: route tushare clients through proxy"
```

### Task 3: Prepare the local Python environment

**Files:**
- Read: `requirements.txt`
- Create locally, gitignored: `.env`
- Create locally, gitignored: `.venv/`

- [ ] **Step 1: Confirm the available Python version**

Run:

```powershell
python --version
```

Expected: Python 3.12.x, matching the project requirement. If the active interpreter is incompatible, locate a compatible Conda interpreter before continuing.

- [ ] **Step 2: Create an isolated virtual environment**

Run:

```powershell
python -m venv .venv
```

Expected: `.venv\Scripts\python.exe` exists.

- [ ] **Step 3: Install project dependencies**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Expected: exit code 0. If heavyweight optional ML packages fail, use the project Conda environment route from `environment.yml` rather than silently removing dependencies.

- [ ] **Step 4: Create the local token configuration**

Create `.env` with:

```dotenv
TUSHARE_TOKEN=<user-supplied-token>
```

Expected: `.env` remains excluded by `.gitignore` and is not staged.

### Task 4: Verify Django configuration and database

**Files:**
- Generated locally, gitignored: `db.sqlite3`

- [ ] **Step 1: Run Django system checks**

Run:

```powershell
.\.venv\Scripts\python.exe manage.py check
```

Expected: `System check identified no issues`.

- [ ] **Step 2: Apply database migrations**

Run:

```powershell
.\.venv\Scripts\python.exe manage.py migrate --noinput
```

Expected: exit code 0; migrations either apply successfully or report no migrations to apply.

- [ ] **Step 3: Run the complete available test suite**

Run:

```powershell
.\.venv\Scripts\python.exe manage.py test
```

Expected: exit code 0 with zero failures.

### Task 5: Verify the real proxy and local web server

**Files:**
- No tracked file changes

- [ ] **Step 1: Test a real Tushare request through the modified project pattern**

Run with `TUSHARE_TOKEN` loaded from `.env`:

```powershell
.\.venv\Scripts\python.exe -c "import os; from dotenv import load_dotenv; load_dotenv(); import tushare as ts; token=os.environ['TUSHARE_TOKEN']; pro=ts.pro_api(token); pro._DataApi__token=token; pro._DataApi__http_url='http://lianghua.nanyangqiankun.top'; df=pro.daily(ts_code='000001.SZ',start_date='20240101',end_date='20240131'); print(len(df)); print(df.head().to_string(index=False))"
```

Expected: 22 rows and non-empty daily data.

- [ ] **Step 2: Start Django locally**

Run as a hidden background process:

```powershell
Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "manage.py","runserver","127.0.0.1:8000","--noreload" -WorkingDirectory (Get-Location) -WindowStyle Hidden
```

Expected: the process listens on `127.0.0.1:8000`.

- [ ] **Step 3: Verify the homepage**

Run:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/ | Select-Object StatusCode
```

Expected: HTTP status 200.

- [ ] **Step 4: Inspect repository state**

Run:

```powershell
git status --short
```

Expected: no uncommitted tracked implementation changes; local `.env`, `.venv`, database, caches, and logs remain ignored.

