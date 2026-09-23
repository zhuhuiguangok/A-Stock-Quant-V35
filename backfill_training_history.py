# -*- coding: utf-8 -*-
"""一次性补拉多年期训练底仓数据（含完整牛熊周期）。

从 Tushare 按"逐交易日全市场 daily + 月度 daily_basic 快照"拉取 csi1000
股票池的长历史日线，生成与选股缓存同构的 real_data_*.parquet，
供模型重训的合并数据池（model_retrain_service._load_combined_training_data）
自动吸收，让训练数据覆盖完整牛熊周期。

用法：
    python backfill_training_history.py [年数]
    默认 4 年（2022-09 起，覆盖 2022~2024 熊市、2024-09 牛市启动、
    2025 牛市延续、2026 调整）。

说明：
    - daily 按日全市场拉取（单次约 5400 行 < Tushare 6000 行/次限制），
      再过滤到股票池，规避按 ts_code 批量拉取长区间触发的行数上限。
    - 估值/市值等 basic 字段按月度快照回填（每月第一个交易日拉一次全市场），
      使 pe/pb/total_mv 随历史月份变化，而不是用最新单一快照。
    - 成分名单为"当前 csi1000 成分回溯历史"，存在轻微幸存者偏差；
      代理积分下拿不到逐时点成分，先接受该近似。
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "stock_project.settings")

import django
django.setup()

import pandas as pd

from stock_app.tushare_client import create_tushare_pro
from stock_app.views import _filter_stock_codes, _load_csindex_pool, find_valid_basic_date

CACHE_DIR = PROJECT_ROOT / "data_cache"
BASIC_FIELDS = "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps,total_mv,circ_mv"
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,vol,amount"

OUT_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
    "turnover_rate", "volume_ratio", "pe", "pb", "ps", "total_mv", "circ_mv",
    "name", "industry",
]


def _retry(call, tries=3, wait=1.0, label=""):
    for i in range(tries):
        try:
            df = call()
            if df is not None and len(df) > 0:
                return df
            print(f"    ↺ {label} 第{i+1}次空", flush=True)
        except Exception as exc:
            print(f"    ↺ {label} 第{i+1}次失败: {exc}", flush=True)
        time.sleep(wait)
    return None


def main(years: int = 4) -> None:
    pro = create_tushare_pro(timeout=30)

    end_date = find_valid_basic_date(max_lookback=15)
    start_date = (datetime.strptime(end_date, "%Y%m%d") - timedelta(days=years * 365)).strftime("%Y%m%d")
    print(f"[1/5] 目标区间 {start_date} ~ {end_date}（{years} 年）", flush=True)

    cal = pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date,
                        fields="cal_date,is_open")
    trade_days = sorted(cal[cal["is_open"] == 1]["cal_date"].astype(str))
    print(f"    交易日 {len(trade_days)} 天", flush=True)

    print("[2/5] 加载 csi1000 股票池并过滤板块…", flush=True)
    pool = _load_csindex_pool("csi1000")
    codes = _filter_stock_codes(pool["ts_code"].unique().tolist(), exclude_chinext=True, exclude_star=True)
    code_set = set(codes)
    print(f"    成分 {len(pool)} 只 → 过滤后 {len(codes)} 只", flush=True)

    print("[3/5] 基本信息 name/industry…", flush=True)
    basic_info = _retry(lambda: pro.stock_basic(exchange="", list_status="L",
                                                fields="ts_code,name,industry"),
                        label="stock_basic")
    if basic_info is None:
        basic_info = pd.DataFrame()
    info_map = basic_info.set_index("ts_code")[["name", "industry"]].to_dict("index") if not basic_info.empty else {}

    print("[4/5] 月度 daily_basic 快照（估值随历史变化）…", flush=True)
    months = sorted({d[:6] for d in trade_days})
    basic_by_month = {}
    for m in months:
        month_days = [d for d in trade_days if d.startswith(m)]
        snap = None
        for d in month_days[:5]:  # 月内前几个交易日拿不到就顺延
            snap = _retry(lambda dd=d: pro.daily_basic(ts_code="", trade_date=dd, fields=BASIC_FIELDS),
                          tries=2, wait=0.8, label=f"daily_basic {d}")
            if snap is not None:
                snap = snap[snap["ts_code"].isin(code_set)].copy()
                snap["trade_date"] = snap["trade_date"].astype(str).str.zfill(8)
                if not snap.empty:
                    break
                snap = None
        if snap is not None:
            basic_by_month[m] = snap.drop_duplicates("ts_code")
        time.sleep(0.2)
    print(f"    月度快照 {len(basic_by_month)}/{len(months)} 个月", flush=True)

    print(f"[5/5] 逐日拉取 daily（{len(trade_days)} 天，全市场→过滤池内）…", flush=True)
    frames = []
    fail_days = []
    t0 = time.time()
    for i, d in enumerate(trade_days, 1):
        day = _retry(lambda dd=d: pro.daily(trade_date=dd, fields=DAILY_FIELDS),
                     tries=3, wait=1.2, label=f"daily {d}")
        if day is None:
            fail_days.append(d)
            continue
        day = day[day["ts_code"].isin(code_set)].copy()
        if day.empty:
            continue
        day["trade_date"] = day["trade_date"].astype(str).str.zfill(8)
        snap = basic_by_month.get(d[:6])
        if snap is not None:
            day = day.merge(snap.drop(columns=["trade_date"]), on="ts_code", how="left")
        frames.append(day)
        if i % 40 == 0 or i == len(trade_days):
            rows = sum(len(f) for f in frames)
            print(f"    {i}/{len(trade_days)} 天 | 池内累计 {rows:,} 行 | 用时 {time.time()-t0:.0f}s", flush=True)
        time.sleep(0.12)

    if not frames:
        print("❌ 一天都没拉到，放弃", flush=True)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    if info_map:
        meta = df["ts_code"].map(info_map)
        df["name"] = meta.map(lambda x: (x or {}).get("name"))
        df["industry"] = meta.map(lambda x: (x or {}).get("industry"))
    df = df[[c for c in OUT_COLUMNS if c in df.columns]]
    df = df.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"]).reset_index(drop=True)

    out = CACHE_DIR / f"real_data_csi1000_cyb1_star1_{df['trade_date'].min()}_{df['trade_date'].max()}.parquet"
    CACHE_DIR.mkdir(exist_ok=True)
    df.to_parquet(out)
    print(f"✅ 已生成 {out.name}: {len(df):,} 行 × {df.shape[1]} 列", flush=True)
    print(f"   跨度 {df['trade_date'].min()} ~ {df['trade_date'].max()}", flush=True)
    if fail_days:
        print(f"⚠️ 缺失 {len(fail_days)} 个交易日（不影响训练，已跳过）: {fail_days[:8]}…", flush=True)


if __name__ == "__main__":
    years = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    main(years)
