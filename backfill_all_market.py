# -*- coding: utf-8 -*-
"""全市场训练池历史回补：全 A 股（含北交所/创业板/科创板/ST）逐日拉取建池。

用法：python backfill_all_market.py [起始日期YYYYMMDD]
- 默认窗口 20220905 → 最新有效交易日（与 csi1000 池起点一致）
- 逐交易日全市场拉取 pro.daily(trade_date=d)（约5000-5400行<6000上限）
- 估值快照=每月首个交易日全市场 daily_basic（含当月新增股票）
- 月度落盘 real_data_allmkt_<起>_<止>.parquet，断点续传（跳过已覆盖日期）
- 含北交所与后退市股票（降低幸存者偏差）
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

import glob

import pandas as pd

from stock_app.tushare_client import create_tushare_pro

CACHE_DIR = PROJECT_ROOT / "data_cache"
ALLMKT_GLOB = str(CACHE_DIR / "real_data_allmkt_*.parquet")
DAY_FIELDS = "ts_code,trade_date,open,high,low,close,vol,amount"
BASIC_FIELDS = "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps,total_mv,circ_mv"
OUT_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
    "turnover_rate", "volume_ratio", "pe", "pb", "ps", "total_mv", "circ_mv",
    "name", "industry",
]
DEFAULT_START = "20220905"


def _retry(call, tries=3, wait=2.0, label=""):
    for i in range(tries):
        try:
            df = call()
            if df is not None and len(df) > 0:
                return df
            print(f"    ↺ {label} 第{i+1}次空", flush=True)
        except Exception as exc:
            print(f"    ↺ {label} 第{i+1}次失败: {str(exc)[:60]}", flush=True)
        time.sleep(wait)
    return None


def covered_dates() -> set:
    """已落盘的 allmkt 日期集合（断点续传）。"""
    covered = set()
    for fp in glob.glob(ALLMKT_GLOB):
        try:
            df = pd.read_parquet(fp, columns=["trade_date"])
            covered |= set(df["trade_date"].astype(str))
        except Exception:
            continue
    return covered


def main(start_date: str = None) -> None:
    pro = create_tushare_pro(timeout=30)

    covered = covered_dates()
    start = start_date or DEFAULT_START

    # 股票名称/行业（在市+退市+暂停，降低幸存者偏差）
    meta = {}
    for status in ("L", "D", "P"):
        df = _retry(lambda st=status: pro.stock_basic(exchange="", list_status=st,
                                                      fields="ts_code,name,industry"),
                    label=f"stock_basic({status})")
        if df is not None:
            for tc, nm, ind in zip(df["ts_code"], df["name"], df["industry"]):
                meta.setdefault(tc, (nm, ind))
    print(f"[1/4] 股票名称/行业字典: {len(meta)} 只（含退市）", flush=True)

    # 交易日历（锚定到最近"已完整发布"的交易日，防盘中写入半日数据）
    from stock_app.views import find_valid_basic_date

    end_date = find_valid_basic_date(max_lookback=15)
    cal = _retry(lambda: pro.trade_cal(exchange="SSE", start_date=start, end_date=end_date, is_open="1"),
                 label="trade_cal")
    if cal is None:
        print("交易日历拉取失败"); sys.exit(1)
    all_days = sorted(cal["cal_date"].astype(str))
    todo = [d for d in all_days if d not in covered]
    if not todo:
        print("✅ 全市场历史池已完整，无需回补", flush=True)
        return
    print(f"[2/4] 待回补 {len(todo)} 个交易日: {todo[0]} ~ {todo[-1]}（已覆盖 {len(covered)} 天）", flush=True)

    # 按月分组（估值快照=每月首个交易日，全市场）
    by_month: dict = {}
    for d in todo:
        by_month.setdefault(d[:6], []).append(d)

    basic_by_month: dict = {}
    for m in sorted(by_month):
        days_m = by_month[m]
        snap = _retry(lambda mm=m, dd=days_m[0]: pro.daily_basic(
            trade_date=dd, fields=BASIC_FIELDS), label=f"daily_basic {m}")
        if snap is not None:
            snap["trade_date"] = snap["trade_date"].astype(str).str.zfill(8)
            basic_by_month[m] = snap.drop_duplicates("ts_code")
        time.sleep(0.3)
    print(f"[3/4] 估值快照: {len(basic_by_month)}/{len(by_month)} 个月", flush=True)

    done_months = 0
    for m in sorted(by_month):
        days_m = by_month[m]
        frames = []
        for d in days_m:
            day = _retry(lambda dd=d: pro.daily(trade_date=dd, fields=DAY_FIELDS),
                         tries=3, wait=2.5, label=f"daily {d}")
            if day is None or day.empty:
                print(f"    ⚠️ {d} 全市场无数据（跳过）", flush=True)
                continue
            day["trade_date"] = day["trade_date"].astype(str).str.zfill(8)
            snap = basic_by_month.get(m)
            if snap is not None:
                day = day.merge(snap.drop(columns=["trade_date"]), on="ts_code", how="left")
            meta_df = pd.DataFrame(
                [(tc, meta.get(tc, ("", ""))[0], meta.get(tc, ("", ""))[1]) for tc in day["ts_code"]],
                columns=["ts_code", "name", "industry"])
            day = day.merge(meta_df, on="ts_code", how="left")
            frames.append(day[OUT_COLUMNS])
            time.sleep(0.3)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"])
        out = CACHE_DIR / f"real_data_allmkt_{df['trade_date'].min()}_{df['trade_date'].max()}.parquet"
        df.to_parquet(out)
        done_months += 1
        print(f"  ✅ {m}: {len(df):,} 行 → {out.name} ({df['trade_date'].min()}~{df['trade_date'].max()})", flush=True)

    print(f"[4/4] 完成: {done_months} 个月落盘", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
