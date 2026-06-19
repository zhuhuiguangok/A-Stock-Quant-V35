# Board Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pre-analysis switches that exclude ChiNext and STAR Market stocks before market data is fetched.

**Architecture:** Introduce a small pure stock-code filtering function, pass validated request options through the existing selection endpoint into `get_real_stock_data`, and include those options in cache identity. Add two default-enabled checkboxes to the existing control panel.

**Tech Stack:** Python, Django test client, pandas, vanilla JavaScript, HTML

---

### Task 1: Test stock-code filtering

**Files:**
- Create: `stock_app/test_board_filters.py`
- Modify: `stock_app/views.py`

- [ ] Write tests for independent, combined, and disabled filters.
- [ ] Run tests and confirm failure because the helper does not exist.
- [ ] Implement `_filter_stock_codes`.
- [ ] Run focused tests and confirm pass.

### Task 2: Test request parameter propagation

**Files:**
- Modify: `stock_app/test_board_filters.py`
- Modify: `stock_app/views.py`

- [ ] Mock `get_real_stock_data` and downstream selection calls.
- [ ] POST `stock_pool`, `exclude_chinext`, and `exclude_star`.
- [ ] Confirm the test fails because the endpoint ignores these fields.
- [ ] Validate the stock pool and strict JSON booleans, then pass them to `get_real_stock_data`.
- [ ] Extend `get_real_stock_data`, filter immediately after stock pool creation, and vary the cache filename.
- [ ] Run focused tests and confirm pass.

### Task 3: Add the UI controls

**Files:**
- Modify: `stock_app/templates/stock_app/index.html`
- Modify: `stock_app/test_board_filters.py`

- [ ] Add a template test for two checked checkboxes and request field names.
- [ ] Run it and confirm failure.
- [ ] Add the two controls and request payload fields.
- [ ] Run the focused tests and confirm pass.

### Task 4: Verify the deployed application

**Files:**
- No additional tracked files

- [ ] Run `manage.py check`.
- [ ] Run `manage.py test`.
- [ ] Restart the local Django server using the nanobot environment.
- [ ] Verify the homepage returns HTTP 200 and contains both filter controls.
- [ ] Verify all new and existing tests pass.

