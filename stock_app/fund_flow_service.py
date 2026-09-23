"""Fund-flow monitor integration for the Django dashboard.

The calculation code lives in ``stock_app.fund_flow`` and is copied from the
reference project so the scoring and rotation algorithm stay identical. This
module only handles project-local data, bootstrapping and task state.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

from .bull_bear_service import BULL_BEAR_DATA_ROOT
from .fund_flow.core import normalize_sector_flow
from .fund_flow.service import AkShareFundFlowClient, FundFlowService
from .tushare_client import create_tushare_pro


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUND_FLOW_SOURCE_ROOT = Path(os.environ.get(
    "FUND_FLOW_SOURCE_ROOT",
    r"D:\codex-workspace\stock-bull-bear-tushare",
))
FUND_FLOW_DATA_ROOT = BULL_BEAR_DATA_ROOT

_LOCK = threading.Lock()
_TASK: Dict = {
    "state": "idle",
    "message": "资金流监控尚未启动",
    "stage": "idle",
    "progress": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


def _set_task(**kwargs) -> None:
    with _LOCK:
        _TASK.update(kwargs)


def get_task() -> Dict:
    with _LOCK:
        return dict(_TASK)


def _source_project_root() -> Path:
    if (FUND_FLOW_SOURCE_ROOT / "app" / "fund_flow").exists():
        return FUND_FLOW_SOURCE_ROOT
    nested = FUND_FLOW_SOURCE_ROOT / "stock-bull-bear-tushare"
    if (nested / "app" / "fund_flow").exists():
        return nested
    return FUND_FLOW_SOURCE_ROOT


def _source_data_root() -> Path:
    return Path(os.environ.get("FUND_FLOW_SOURCE_DATA_ROOT", str(_source_project_root() / "data")))


def _clear_proxy_env() -> None:
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"


def _candidate_trade_dates(max_days: int = 14) -> list[str]:
    cursor = datetime.now()
    dates = []
    for offset in range(max_days + 1):
        day = cursor - timedelta(days=offset)
        if day.weekday() < 5:
            dates.append(day.strftime("%Y%m%d"))
    return dates


def create_tushare_pro_client():
    _clear_proxy_env()
    return create_tushare_pro(timeout=30)


def aggregate_tushare_industry_flow(pro, trade_date: str) -> pd.DataFrame:
    money = pro.moneyflow(
        trade_date=trade_date,
        fields="ts_code,trade_date,buy_lg_amount,sell_lg_amount,buy_elg_amount,sell_elg_amount,net_mf_amount",
    )
    if money is None or money.empty:
        return pd.DataFrame(columns=["date", "sector_name", "main_force_net", "change_pct"])

    basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,industry")
    if basic is None or basic.empty:
        return pd.DataFrame(columns=["date", "sector_name", "main_force_net", "change_pct"])

    daily = pro.daily(trade_date=trade_date, fields="ts_code,pct_chg,amount")
    money = money.copy()
    for column in ["buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"]:
        money[column] = pd.to_numeric(money[column], errors="coerce").fillna(0.0)
    money["main_force_net"] = (
        money["buy_lg_amount"]
        + money["buy_elg_amount"]
        - money["sell_lg_amount"]
        - money["sell_elg_amount"]
    ) * 10000.0

    merged = money.merge(basic[["ts_code", "industry"]], on="ts_code", how="left")
    merged["sector_name"] = merged["industry"].fillna("").astype(str).str.strip()
    merged = merged[merged["sector_name"].ne("") & merged["sector_name"].str.lower().ne("nan")]
    if merged.empty:
        return pd.DataFrame(columns=["date", "sector_name", "main_force_net", "change_pct"])

    if daily is not None and not daily.empty:
        daily = daily.copy()
        daily["pct_chg"] = pd.to_numeric(daily["pct_chg"], errors="coerce")
        merge_cols = ["ts_code", "pct_chg"]
        if "amount" in daily.columns:
            # 行业总成交额（元）：用于净流入强度 = 主力净额 / 成交额，消除行业规模偏差
            # Tushare daily 的 amount 单位为千元
            daily["amount"] = pd.to_numeric(daily["amount"], errors="coerce")
            merge_cols.append("amount")
        merged = merged.merge(daily[merge_cols], on="ts_code", how="left")
        merged["turnover"] = merged["amount"].fillna(0) * 1000.0 if "amount" in merged.columns else 0.0
    else:
        merged["pct_chg"] = 0.0
        merged["turnover"] = 0.0

    out = (
        merged.groupby("sector_name", as_index=False)
        .agg(main_force_net=("main_force_net", "sum"), change_pct=("pct_chg", "mean"), turnover=("turnover", "sum"))
        .assign(date=str(trade_date))
    )
    out["change_pct"] = out["change_pct"].fillna(0.0).round(4)
    return normalize_sector_flow(out[["date", "sector_name", "main_force_net", "change_pct", "turnover"]], date=trade_date)


class UnifiedFundFlowClient:
    """AkShare first; Tushare moneyflow aggregation as the unified fallback."""

    def __init__(self, akshare_client=None, pro_factory=create_tushare_pro_client, backfill_days: int = 10):
        self.akshare_client = akshare_client or AkShareFundFlowClient()
        self.pro_factory = pro_factory
        self.source_name = "AkShare"
        self.backfill_days = backfill_days
        # 已缓存的交易日集合：回补时跳过，避免每小时重复拉取相同历史（省 Tushare 积分）
        self.existing_dates: set[str] = set()
        # 全部交易日均已缓存时置位，dashboard 据此把"空返回"视为"已最新"而非错误
        self.noop_reason: str | None = None

    def _tushare_backfill(self, pro):
        """【多日回补】一次拉取最近 N 个交易日的行业资金流并拼接。

        单日快照只能看当天方向；回补多日后，5日/10日持续性、
        连续同向天数等"潜在关注"算法才有输入。
        已缓存的交易日直接跳过，只补缺口。
        """
        frames = []
        last_error = None
        skipped = 0
        for trade_date in _candidate_trade_dates(max_days=self.backfill_days * 2):
            if trade_date in self.existing_dates:
                skipped += 1
            else:
                try:
                    frame = aggregate_tushare_industry_flow(pro, trade_date)
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:
                    last_error = exc
            if len(frames) + skipped >= self.backfill_days:
                break
        if not frames:
            if skipped >= self.backfill_days:
                self.noop_reason = f"最近 {skipped} 个交易日数据已在缓存，无需回补"
            elif last_error is not None:
                raise last_error
            return pd.DataFrame(columns=["date", "sector_name", "main_force_net", "change_pct"])
        self.source_name = "TushareMoneyflow"
        return pd.concat(frames, ignore_index=True)

    def sector_fund_flow(self):
        ak_error = None
        try:
            realtime = normalize_sector_flow(self.akshare_client.sector_fund_flow())
            if not realtime.empty:
                self.source_name = "AkShare"
                return realtime
        except Exception as exc:
            ak_error = exc

        pro = self.pro_factory()
        return self._tushare_backfill(pro)


def _copy_file_if_needed(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    if dst.exists() and dst.stat().st_size == src.stat().st_size and int(dst.stat().st_mtime) >= int(src.stat().st_mtime):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def bootstrap_from_source_data(force: bool = False) -> Dict:
    """Copy existing fund-flow artifacts before any dashboard/API work."""
    source_data = _source_data_root()
    if not source_data.exists():
        return {"copied": 0, "message": f"未找到源资金流数据目录: {source_data}"}

    result_dst = FUND_FLOW_DATA_ROOT / "results" / "fund_flow_latest.json"
    cache_dst = FUND_FLOW_DATA_ROOT / "cache" / "fund_flow_sector.parquet"
    if result_dst.exists() and not force:
        return {"copied": 0, "message": "主工程已有资金流结果数据"}

    copied = 0
    pairs = [
        (source_data / "results" / "fund_flow_latest.json", result_dst),
        (source_data / "cache" / "fund_flow_sector.parquet", cache_dst),
        (source_data / "cache" / "fund_flow_sector.meta.json", FUND_FLOW_DATA_ROOT / "cache" / "fund_flow_sector.meta.json"),
    ]
    for src, dst in pairs:
        if _copy_file_if_needed(src, dst):
            copied += 1

    return {"copied": copied, "message": f"已导入 {copied} 个资金流数据文件"}


def _service() -> FundFlowService:
    FUND_FLOW_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    client = UnifiedFundFlowClient()
    service = FundFlowService(FUND_FLOW_DATA_ROOT, client=client)
    # 把缓存里已有的交易日注入客户端，回补只拉缺口日期
    cached = service.cache.read()
    if not cached.empty and "date" in cached.columns:
        client.existing_dates = set(cached["date"].astype(str).unique())
    return service


def _research_resonance(attention):
    """跨模块交叉：近 7 天研报看多情绪与资金关注度共振的行业。

    资金与券商观点双重确认的板块，趋势可信度高于单一信号。
    返回 {行业: 看多研报数}，仅保留共振(看多≥2)且在资金关注度榜上的行业。
    """
    try:
        from datetime import date, timedelta

        from .models import ResearchReport
        from .research.market_utils import normalize_industry_set

        start = date.today() - timedelta(days=7)
        rows = (
            ResearchReport.objects.filter(
                status=ResearchReport.STATUS_SUMMARIZED,
                report_date__gte=start,
            )
            .select_related("summary")
            .values_list("summary__sentiment", "summary__tags")
        )
        bullish_tags = {}
        for sentiment, tags in rows:
            if sentiment != "bullish" or not tags:
                continue
            for tag in normalize_industry_set(tags):
                bullish_tags[tag] = bullish_tags.get(tag, 0) + 1
        if not bullish_tags:
            return {}
        attention_names = {item.get("sector_name") for item in (attention or []) if item.get("sector_name")}
        # 研报标签(申万类命名)与资金流行业(Tushare 命名)是两套体系，用双向子串匹配
        resonance = {}
        for tag, n in bullish_tags.items():
            if n < 2:
                continue
            for sector in attention_names:
                if tag in sector or sector in tag:
                    resonance[sector] = n
                    break
        return resonance
    except Exception:
        return {}


def get_fund_flow_payload(refresh: bool = False) -> Dict:
    bootstrap = bootstrap_from_source_data(force=False)
    payload = _service().dashboard(refresh=refresh)
    payload["bootstrap"] = bootstrap
    payload["task"] = get_task()
    payload["data_root"] = str(FUND_FLOW_DATA_ROOT)
    payload["source_root"] = str(_source_project_root())
    payload["research_resonance"] = _research_resonance(payload.get("attention_ranking"))
    return payload


def _run_refresh_from_cache() -> Dict:
    _set_task(stage="bootstrap", progress=15, message="导入源项目资金流数据")
    bootstrap_from_source_data(force=False)
    _set_task(stage="calculate", progress=55, message="基于本地资金流缓存重算监控结果")
    return _service().dashboard(refresh=False)


def _run_update() -> Dict:
    _set_task(stage="bootstrap", progress=8, message="优先导入源项目资金流缓存")
    bootstrap_from_source_data(force=False)
    _set_task(stage="fetch", progress=28, message="联机补全行业资金流快照")
    return _service().dashboard(refresh=True)


def _start_background(kind: str, runner, started_message: str) -> Dict:
    task = get_task()
    if task.get("state") == "running":
        return task

    now = datetime.now().isoformat(timespec="seconds")
    _set_task(state="running", stage="starting", progress=1, message=started_message, started_at=now, finished_at=None, error=None)

    def target() -> None:
        try:
            result = runner()
            summary = result.get("summary") or {}
            _set_task(
                state="completed",
                stage="completed",
                progress=100,
                message=f"{kind}完成: {result.get('date') or '--'} / {summary.get('sector_count', 0)} 个板块",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=result.get("error"),
            )
        except Exception as exc:  # pragma: no cover - surfaced through API
            _set_task(
                state="failed",
                stage="failed",
                progress=100,
                message=f"{kind}失败",
                finished_at=datetime.now().isoformat(timespec="seconds"),
                error=f"{exc}\n{traceback.format_exc()}",
            )

    threading.Thread(target=target, daemon=True, name=f"fund_flow_{kind}").start()
    return get_task()


def start_refresh_from_cache() -> Dict:
    return _start_background("资金流本地刷新", _run_refresh_from_cache, "准备基于本地缓存刷新资金流监控")


def start_update() -> Dict:
    return _start_background("资金流联机补数", _run_update, "准备联机补全资金流数据")


def read_local_result() -> Dict:
    path = FUND_FLOW_DATA_ROOT / "results" / "fund_flow_latest.json"
    return json.loads(path.read_text("utf-8")) if path.exists() else {}
