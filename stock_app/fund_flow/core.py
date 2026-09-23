from datetime import datetime

import pandas as pd


SECTOR_COLUMNS = ["date", "sector_name", "main_force_net", "change_pct"]
# 多周期扩展列：akshare 5日/10日排行或 Tushare 历史回补衍生出的累计净流；
# turnover 为行业当日总成交额（元），用于计算净流入强度（净额/成交额）消除行业规模偏差
SECTOR_EXTRA_COLUMNS = ["net_5d", "net_10d", "turnover"]


def parse_amount(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("，", "")
    if not text or text in {"-", "--", "nan", "None"}:
        return 0.0

    negative = text.startswith("-")
    text = text.replace("+", "").replace("-", "")
    multiplier = 1.0
    if "亿" in text:
        multiplier = 100000000.0
        text = text.replace("亿元", "").replace("亿", "")
    elif "万" in text:
        multiplier = 10000.0
        text = text.replace("万元", "").replace("万", "")
    text = text.replace("元", "").strip()
    try:
        amount = float(text) * multiplier
    except ValueError:
        return 0.0
    return -amount if negative else amount


def parse_percent(value):
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "").replace(",", "")
    if not text or text in {"-", "--", "nan", "None"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _first_existing(frame, candidates):
    return next((col for col in candidates if col in frame.columns), None)


def normalize_sector_flow(frame, date=None, amount_candidates=None):
    raw = pd.DataFrame(frame).copy()
    if raw.empty:
        return pd.DataFrame(columns=SECTOR_COLUMNS)

    date = str(date or datetime.now().strftime("%Y%m%d")).replace("-", "")
    sector_col = _first_existing(raw, ["sector_name", "名称", "板块", "行业名称"])
    amount_col = _first_existing(
        raw,
        amount_candidates
        or [
            "main_force_net",
            "主力净流入数值",
            "今日主力净流入-净额",
            "主力净流入",
            "今日主力净流入",
        ],
    )
    change_col = _first_existing(raw, ["change_pct", "今日涨跌幅", "涨跌幅", "涨跌幅%"])
    date_col = _first_existing(raw, ["date", "trade_date", "日期"])

    out = pd.DataFrame(index=raw.index)
    out["date"] = raw[date_col].astype(str).str.replace("-", "", regex=False) if date_col else date
    out["sector_name"] = raw[sector_col].astype(str).str.strip() if sector_col else raw.iloc[:, 0].astype(str)
    out["main_force_net"] = raw[amount_col].map(parse_amount) if amount_col else 0.0
    out["change_pct"] = raw[change_col].map(parse_percent) if change_col else 0.0
    # 保留多周期扩展列（存在才附加），供关注度算法使用
    for extra in SECTOR_EXTRA_COLUMNS:
        if extra in raw.columns:
            out[extra] = pd.to_numeric(raw[extra], errors="coerce")
    out = out.dropna(subset=["sector_name"])
    out = out[out["sector_name"].str.len() > 0]
    columns = SECTOR_COLUMNS + [c for c in SECTOR_EXTRA_COLUMNS if c in out.columns]
    return out[columns].drop_duplicates(["date", "sector_name"], keep="last").reset_index(drop=True)


def amount_to_yi(value):
    return round(float(value) / 100000000.0, 3)
