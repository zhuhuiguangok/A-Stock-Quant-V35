# 资金流与企业微信通知改造实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完善资金流监控网页，接入企业微信 Markdown 通知，并让市场状态展示明确反映数据代理范围与可信度。

**Architecture:** 保留现有资金流缓存、AkShare 联机补数和后台任务链路；新增独立的企业微信 Webhook 配置/发送层，页面配置优先、环境变量兜底。市场状态继续使用现有规则引擎，但输出样本范围、数据质量和“代理估计”标识，移除硬编码的概率式置信度表达。

**Tech Stack:** Django/Python、requests、pandas、原生 HTML/CSS/JavaScript、Django TestCase。

## Global Constraints

- 企业微信配置键固定为 `WECOM_WEBHOOK_URL` 和 `WECOM_WEBHOOK_SECRET`。
- Secret 不能回显到页面、日志或 API 响应。
- 资金流和市场状态缺数据时必须保留可解释的空状态，不能用 0 冒充真实观测。
- 页面可见字号整体上调，中文显示保持 UTF-8，不新增版本号文案。
- 不发送真实企业微信群测试消息；测试只校验 URL、签名和 Markdown payload。

---

### Task 1: 企业微信通知配置与发送

**Files:**
- Modify: `stock_app/config_v19.py`
- Modify: `stock_app/wechat_notify.py`
- Modify: `stock_app/views.py`
- Modify: `stock_app/urls.py`
- Modify: `stock_app/templates/stock_app/index.html`
- Test: `stock_app/test_wecom_notify.py`

**Interfaces:**
- `WeChatNotifier.send_stock_report(...)` continues as the caller-facing API.
- Add `WeChatNotifier.send_markdown(title: str, content: str) -> bool`.
- Add `wecom_config_api(request)` for `GET` status and `POST` save/test configuration.

- [ ] **Step 1: Write failing tests** for HMAC-SHA256 signature generation, environment-variable fallback, secret redaction, and the exact Enterprise WeChat payload shape `{"msgtype":"markdown","markdown":{"content":...}}`.
- [ ] **Step 2: Run `python -X utf8 manage.py test stock_app.test_wecom_notify -v 2` and confirm the new tests fail before implementation.
- [ ] **Step 3: Add `WECOM_WEBHOOK_URL`/`WECOM_WEBHOOK_SECRET` config defaults and a notifier path that builds `timestamp`, `nonce`, and `sign = base64(hmac_sha256(secret, f"{timestamp}\n{nonce}"))`; use page config first and environment values second.
- [ ] **Step 4: Replace Server酱/PushPlus delivery in the stock report path with the Enterprise WeChat Markdown endpoint while preserving quiet-hours and rate-limit guards.
- [ ] **Step 5: Add a CSRF-exempt configuration endpoint that saves only the webhook URL and secret to local `.env`, returns `configured`/`source`, and never returns the secret value.
- [ ] **Step 6: Replace the SendKey controls with Webhook and Secret controls plus a “测试通知” button that uses a mocked transport in tests.
- [ ] **Step 7: Run the notifier tests and `manage.py check`; expected result is all tests passing and no real outbound test message.

### Task 2: Fund-flow payload quality and API states

**Files:**
- Modify: `stock_app/fund_flow_service.py`
- Modify: `stock_app/fund_flow/analysis.py`
- Modify: `stock_app/views.py`
- Test: `stock_app/test_fund_flow_quality.py`

**Interfaces:**
- `fund_flow_status_api` keeps its existing route and adds `quality`, `sample_days`, `latest_date`, `observed_at`, and `is_estimate` fields.
- Existing `source`, `error`, `summary`, `rotation_forecast`, and relation arrays remain backward-compatible.

- [ ] **Step 1: Add failing tests for empty snapshots, one-day history, stale cache, and realtime data with valid observation dates.
- [ ] **Step 2: Run the focused tests and verify they fail on missing quality fields.
- [ ] **Step 3: Add explicit payload quality calculation: `quality='realtime'` only for a non-empty fresh fetch, `quality='cache'` for valid cached data, `quality='stale'` when the latest observation is older than the freshness threshold, and `quality='insufficient'` for too few history days.
- [ ] **Step 4: Preserve `None`/`--` semantics for unavailable values and make forecast status `insufficient_data` unless the history window is long enough for the calculation.
- [ ] **Step 5: Add source/date/sample metadata to the JSON response and run the focused fund-flow tests.

### Task 3: Market-state reliability metadata

**Files:**
- Modify: `stock_app/market_timing.py`
- Modify: `stock_app/views.py`
- Modify: `stock_app/templates/stock_app/index.html`
- Test: `stock_app/test_market_timing_quality.py`

**Interfaces:**
- `compute_market_timing(df)` continues returning the existing regime fields and adds `scope`, `sample_count`, `valid_ratio`, `signal_basis`, `data_quality`, and `confidence_basis`.

- [ ] **Step 1: Add failing tests proving the function distinguishes full-sample data from missing/zero-filled data and never claims 100% statistical confidence merely because trends are allowed.
- [ ] **Step 2: Run the focused tests and verify the current hard-coded confidence behavior is exposed as a failure.
- [ ] **Step 3: Compute latest-snapshot sample count and valid ratios before filling missing values; mark `data_quality='insufficient'` when required factors are unavailable or sample count is too small.
- [ ] **Step 4: Replace the hard-coded confidence value in the response with a bounded data-quality confidence score, and label the result as a rule-based proxy estimate for the selected stock pool.
- [ ] **Step 5: Update the market-state card to show scope, sample count, data quality, and “代理估计” wording; keep regime, return, volatility, and volume metrics visible.
- [ ] **Step 6: Run the focused market timing tests and verify JSON field names match the template.

### Task 4: Fund-flow UI scale and interaction polish

**Files:**
- Modify: `stock_app/templates/stock_app/index.html`
- Test: `stock_app/test_fund_flow_module.py`

- [ ] **Step 1: Add template assertions for Webhook/Secret controls, larger fund-flow typography hooks, quality/source badges, and stale/insufficient data text.
- [ ] **Step 2: Increase fund-flow base text to at least 15px, stat values to at least 24px, section titles to at least 18px, and relation/list text to at least 14px while preserving responsive layout.
- [ ] **Step 3: Render new quality metadata in the source/status strip, disable or clarify predictive panels when history is insufficient, and keep sector focus/filter interactions intact.
- [ ] **Step 4: Add visible loading, error, stale-cache, and empty-history states to the fund-flow dashboard and configuration form.
- [ ] **Step 5: Run template tests and inspect the page at desktop and narrow viewport widths.

### Task 5: End-to-end verification and restart

**Files:**
- Modify: `logs/runserver.restart.out.log` and `logs/runserver.restart.err.log` only through the running service; do not commit logs.

- [ ] **Step 1: Run `python -X utf8 manage.py check`, focused unit tests, `git diff --check`, and a direct API smoke test for fund-flow status/config endpoints.
- [ ] **Step 2: Restart the Django service on `127.0.0.1:18935` and verify homepage HTTP 200.
- [ ] **Step 3: Use browser automation to open the fund-flow page, verify larger text and quality labels, save a mocked configuration path without exposing Secret, and start a fund-flow refresh.
- [ ] **Step 4: Verify the browser shows completed/empty/stale states correctly and collect final server logs for errors.
