# -*- coding: utf-8 -*-
"""训练池增量补齐：只拉指定日期之后的缺失交易日，生成小文件进合并池。

用法：python topup_training_pool.py [起始日期YYYYMMDD]
默认从 data_cache 里最新 real_data 文件的 max(trade_date) 的下一自然日补到
最新有效交易日。产出 real_data_csi1000_cyb1_star1_<start>_<end>.parquet。
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
from stock_app.views import _filter_stock_codes, _load_csindex_pool, find_valid_basic_date

CACHE_DIR = PROJECT_ROOT / "data_cache"
SECTOR_PARQUET = CACHE_DIR / "jev" / "sector_index_daily.parquet"
BASIC_FIELDS = "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,ps,total_mv,circ_mv"
DAILY_FIELDS = "ts_code,trade_date,open,high,low,close,vol,amount"
OUT_COLUMNS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount",
    "turnover_rate", "volume_ratio", "pe", "pb", "ps", "total_mv", "circ_mv",
    "name", "industry",
]


def _retry(call, tries=3, wait=1.2, label=""):
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


def topup_sector_index(pro, end_date: str) -> None:
    """补齐申万板块指数日线（板块预测 GBM/Jev 的特征源），与训练池同日刷新。"""
    if not SECTOR_PARQUET.exists():
        print("⚠️ 未找到 sector_index_daily.parquet，跳过板块指数补齐", flush=True)
        return
    old = pd.read_parquet(SECTOR_PARQUET)
    base = str(old["trade_date"].astype(str).max())
    if base >= end_date:
        print(f"✅ 板块指数已是最新 ({base})", flush=True)
        return
    df = _retry(lambda: pro.sw_daily(trade_date="", start_date=str(int(base) + 1),
                                     end_date=end_date),
                tries=3, wait=2.0, label="sw_daily")
    if df is None or df.empty:
        print(f"⚠️ sw_daily {base} 之后暂无增量", flush=True)
        return
    df["trade_date"] = df["trade_date"].astype(str).str.zfill(8)
    cols = [c for c in old.columns if c in df.columns]
    merged = pd.concat([old, df[cols]], ignore_index=True)
    merged = merged.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"])
    merged.to_parquet(SECTOR_PARQUET)
    print(f"✅ 板块指数补齐 {base} -> {merged['trade_date'].max()}: +{len(df):,} 行", flush=True)


def main(start_after: str | None = None) -> None:
    pro = create_tushare_pro(timeout=30)

    files = glob.glob(str(CACHE_DIR / "real_data_*.parquet"))
    latest = "0"
    for p in files:
        try:
            mx = str(pd.read_parquet(p, columns=["trade_date"])["trade_date"].astype(str).max())
            latest = max(latest, mx)
        except Exception:
            pass
    end_date = find_valid_basic_date(max_lookback=15)
    base = start_after or latest
    print(f"[1/4] 现有池最新日期 {latest}，最新有效交易日 {end_date}", flush=True)
    topup_sector_index(pro, end_date)
    if base >= end_date:
        print("✅ 训练池已是最新，无需补齐", flush=True)
        return

    cal = pro.trade_cal(exchange="SSE", start_date=base, end_date=end_date,
                        fields="cal_date,is_open")
    days = sorted(cal[cal["is_open"] == 1]["cal_date"].astype(str))
    days = [d for d in days if d > base]
    if not days:
        print("✅ 没有缺失交易日", flush=True)
        return
    print(f"[2/4] 待补 {len(days)} 个交易日: {days[0]} ~ {days[-1]}", flush=True)

    pool = _load_csindex_pool("csi1000")
    codes = set(_filter_stock_codes(pool["ts_code"].unique().tolist(),
                                    exclude_chinext=True, exclude_star=True))
    print(f"    股票池 {len(codes)} 只", flush=True)

    print("[3/4] 基本信息 name/industry…", flush=True)
    info = _retry(lambda: pro.stock_basic(exchange="", list_status="L",
                                          fields="ts_code,name,industry"), label="stock_basic")
    info_map = info.set_index("ts_code")[["name", "industry"]].to_dict("index") if info is not None else {}

    months = sorted({d[:6] for d in days})
    basic_by_month = {}
    for m in months:
        snap = _retry(lambda: pro.daily_basic(ts_code="", trade_date=f"{m}01", fields=BASIC_FIELDS),
                      tries=3, wait=1.0, label=f"daily_basic {m}月初")
        if snap is not None:
            snap = snap[snap["ts_code"].isin(codes)].drop_duplicates("ts_code")
            snap["trade_date"] = snap["trade_date"].astype(str).str.zfill(8)
            basic_by_month[m] = snap

    frames, failed = [], []
    for d in days:
        day = _retry(lambda dd=d: pro.daily(trade_date=dd, fields=DAILY_FIELDS),
                     tries=3, wait=1.5, label=f"daily {d}")
        if day is None:
            failed.append(d)
            continue
        day = day[day["ts_code"].isin(codes)].copy()
        if day.empty:
            continue
        day["trade_date"] = day["trade_date"].astype(str).str.zfill(8)
        snap = basic_by_month.get(d[:6])
        if snap is not None:
            day = day.merge(snap.drop(columns=["trade_date"]), on="ts_code", how="left")
        frames.append(day)
        time.sleep(0.15)

    if not frames:
        print("❌ 一天都没拉到", flush=True)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    if info_map:
        meta = df["ts_code"].map(info_map)
        df["name"] = meta.map(lambda x: (x or {}).get("name"))
        df["industry"] = meta.map(lambda x: (x or {}).get("industry"))
    df = df[[c for c in OUT_COLUMNS if c in df.columns]]
    df = df.sort_values(["ts_code", "trade_date"]).drop_duplicates(["ts_code", "trade_date"]).reset_index(drop=True)

    out = CACHE_DIR / f"real_data_csi1000_cyb1_star1_{df['trade_date'].min()}_{df['trade_date'].max()}.parquet"
    df.to_parquet(out)
    print(f"✅ 已生成 {out.name}: {len(df):,} 行 × {df.shape[1]} 列", flush=True)
    print(f"   跨度 {df['trade_date'].min()} ~ {df['trade_date'].max()}", flush=True)
    if failed:
        print(f"⚠️ 缺失交易日: {failed}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
