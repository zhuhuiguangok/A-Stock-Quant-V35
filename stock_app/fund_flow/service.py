import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime
from pathlib import Path

import pandas as pd

from .analysis import build_fund_flow_payload
from .core import normalize_sector_flow


class AkShareFundFlowClient:
    def __init__(self, module=None, timeout_seconds: int = 25):
        self.ak = module
        # 东方财富接口无 timeout 参数，用线程池硬超时快速失败，
        # 避免网络不通时整个更新任务挂住几分钟
        self.timeout_seconds = timeout_seconds

    def _clear_proxy(self):
        for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            os.environ.pop(key, None)
        os.environ["NO_PROXY"] = "*"
        urllib.request.getproxies = lambda: {}

    def _fetch_rank(self, indicator: str):
        if self.ak is None:
            import akshare as ak
            self.ak = ak
        return self.ak.stock_sector_fund_flow_rank(indicator=indicator, sector_type="行业资金流")

    def _fetch_rank_with_timeout(self, indicator: str):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(self._fetch_rank, indicator)
            return future.result(timeout=self.timeout_seconds)

    def sector_fund_flow(self):
        self._clear_proxy()
        base = self._fetch_rank_with_timeout("今日")

        # 【多周期升级】同时拉取 5日/10日 排行，合并出累计净流列，
        # 让"资金潜在关注"有持续性维度（失败不影响今日快照）
        merged = None
        for indicator, extra_col in (("5日", "net_5d"), ("10日", "net_10d")):
            try:
                period_frame = self._fetch_rank_with_timeout(indicator)
                normalized = normalize_sector_flow(
                    period_frame,
                    amount_candidates=[f"{indicator}主力净流入-净额", f"{indicator}主力净流入", "main_force_net"],
                )
                if normalized.empty:
                    continue
                piece = normalized[["sector_name", "main_force_net"]].rename(columns={"main_force_net": extra_col})
                merged = piece if merged is None else merged.merge(piece, on="sector_name", how="outer")
            except (FutureTimeout, Exception):
                continue
        if merged is not None:
            base = normalize_sector_flow(base).merge(merged, on="sector_name", how="left")
        return base


class FundFlowCache:
    def __init__(self, root):
        self.root = Path(root) / "cache"
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def data_path(self):
        return self.root / "fund_flow_sector.parquet"

    @property
    def metadata_path(self):
        return self.root / "fund_flow_sector.meta.json"

    def read(self):
        return pd.read_parquet(self.data_path) if self.data_path.exists() else pd.DataFrame()

    def latest(self):
        frame = self.read()
        if frame.empty or "date" not in frame:
            return pd.DataFrame()
        latest_date = frame["date"].astype(str).max()
        return frame[frame["date"].astype(str) == latest_date].copy()

    def merge_snapshot(self, frame, source):
        normalized = normalize_sector_flow(frame)
        previous = self.read()
        combined = pd.concat([previous, normalized], ignore_index=True)
        if not combined.empty:
            combined = combined.drop_duplicates(["date", "sector_name"], keep="last").sort_values(
                ["date", "sector_name"]
            )
        combined.to_parquet(self.data_path, index=False)
        meta = {
            "dataset": "sector_fund_flow",
            "source": source,
            "rows": len(combined),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "start": str(combined["date"].min()) if not combined.empty else None,
            "end": str(combined["date"].max()) if not combined.empty else None,
            "columns": list(combined.columns),
        }
        self.metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
        return combined

    def metadata(self):
        return json.loads(self.metadata_path.read_text("utf-8")) if self.metadata_path.exists() else {}


class FundFlowService:
    def __init__(self, data_root="data", client=None):
        self.data_root = Path(data_root)
        self.result_path = self.data_root / "results" / "fund_flow_latest.json"
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = FundFlowCache(data_root)
        self.client = client or AkShareFundFlowClient()

    def _read_result(self):
        if not self.result_path.exists():
            return None
        return self._with_schema_defaults(json.loads(self.result_path.read_text("utf-8")))

    @staticmethod
    def _with_schema_defaults(payload):
        payload = dict(payload)
        payload.setdefault("rotation_edges", [])
        payload.setdefault(
            "rotation_forecast",
            {"status": "insufficient_data", "horizon_days": 3, "strongest_route": None, "edge_count": 0},
        )
        payload.setdefault("attention_ranking", [])
        payload.setdefault(
            "forecast",
            {"horizon_days": 3, "inflow": [], "outflow": [], "note": "等待资金流数据积累"},
        )
        payload.setdefault(
            "market_read",
            {"headline": "等待资金流数据积累", "focus": None, "risk": None, "breadth_hint": None},
        )
        return payload

    def _write_result(self, payload):
        payload = self._with_schema_defaults(payload)
        self.result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        return payload

    def run(self, refresh=True):
        return self.dashboard(refresh=refresh)

    def dashboard(self, refresh=True):
        if not refresh:
            stored = self._read_result()
            if stored is not None:
                return stored

        source = "cache"
        error = None
        history = self.cache.read()
        snapshot = self.cache.latest()

        if refresh:
            try:
                realtime = normalize_sector_flow(self.client.sector_fund_flow())
                if not realtime.empty:
                    history = self.cache.merge_snapshot(realtime, getattr(self.client, "source_name", "AkShare"))
                    # 兜底客户端可能一次回补多日历史，快照只取最新交易日
                    latest_date = history["date"].astype(str).max()
                    snapshot = history[history["date"].astype(str) == latest_date].copy()
                    source = "realtime"
                else:
                    noop = getattr(self.client, "noop_reason", None)
                    if noop:
                        # 缓存已覆盖最近交易日，属正常跳过而非失败
                        source = "cache_fresh"
                        error = None
                    else:
                        source = "cache_fallback"
                        error = "empty realtime fund-flow response"
            except Exception as exc:
                source = "cache_fallback"
                error = str(exc)

        meta = self.cache.metadata()
        updated_at = meta.get("updated_at") or datetime.now().isoformat(timespec="seconds")
        payload = build_fund_flow_payload(snapshot, history, source=source, error=error, updated_at=updated_at)
        return self._write_result(payload)
