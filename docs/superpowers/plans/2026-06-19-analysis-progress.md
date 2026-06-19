# Analysis Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace simulated loader text with real backend analysis and training progress.

**Architecture:** Run the existing synchronous selection view in a background thread, expose an in-memory task status endpoint, and instrument key stages through a thread-local progress callback.

**Tech Stack:** Django, Python threading, vanilla JavaScript polling

---

### Task 1: Backend task API

- [ ] Write failing tests for task creation, progress lookup, completion, and unknown task.
- [ ] Add thread-safe task storage, worker execution, start endpoint, and progress endpoint.
- [ ] Keep the existing synchronous endpoint unchanged.

### Task 2: Real stage instrumentation

- [ ] Write tests for progress update behavior.
- [ ] Instrument data preparation, cache use, filtering, training decision, selection, and completion.
- [ ] Report skipped and failed training explicitly.

### Task 3: Frontend progress

- [ ] Write template assertions for progress bar and new API paths.
- [ ] Replace simulated timer with task start and one-second polling.
- [ ] Render stage, message, detail, percentage, and elapsed time.

### Task 4: Verification

- [ ] Run focused tests.
- [ ] Run Django system checks and full tests.
- [ ] Restart the local server and verify task endpoints and homepage.

