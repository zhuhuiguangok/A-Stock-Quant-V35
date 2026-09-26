"""全球雷达数据采集层。

每个采集函数独立兜底：任一数据源失败返回 None，不阻断整卡渲染。
行情走东财 kline（本环境已验证可达）+ 腾讯行情兜底；宏观走 Tushare 私有代理；
情绪走 akshare（不稳时降级并在 meta 里标注）。
"""
from __future__ import annotations

import logging
import threading
from datetime import date, timedelta
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# 全球资产面板：名称 → 东财 secid 候选（依次尝试，取第一个有数据的）
ASSET_SECIDS = {
    "标普500": ["100.SPX", "100.SP500"],
    "纳斯达克": ["100.NDX", "100.NASDAQ"],
    "道琼斯": ["100.DJIA"],
    "纳指100ETF": ["100.QQQ"],
    "恒生指数": ["100.HSI"],
    "恒生科技": ["100.HSTECH"],
    "日经225": ["100.N225"],
    "韩国KOSPI": ["100.KS11", "100.KOSPI"],
    "印度Sensex": ["100.SENSEX", "100.NIFTY"],
    "德国DAX": ["100.DAX", "100.GDAXI"],
    "法国CAC40": ["100.CAC", "100.FCHI"],
    "英国富时100": ["100.FTSE", "100.FT100"],
    "美元指数": ["100.UDI"],
    "COMEX黄金": ["101.GC00Y", "101.GCMAIN"],
    "WTI原油": ["102.CL00Y", "102.CLMAIN"],
    "布伦特原油": ["102.OIL00Y", "103.OIL00Y"],
    "COMEX铜": ["101.HG00Y", "101.HGMAIN"],
    "COMEX白银": ["101.SI00Y", "101.SIMAIN"],
}
# A 股指数：用于情绪、风格与全球对照
INDEX_SECIDS = {"沪深300": ["1.000300"], "上证指数": ["1.000001"], "创业板指": ["0.399006"]}

# 腾讯日 K 兜底（东财 push2his 不可用时启用；实测 2026-09-18 可达）。
# 覆盖美股三指数/港股两指数/A股三指数；外盘期货与欧洲亚洲其余指数腾讯无历史。
TENCENT_CODES = {
    "标普500": "usINX",
    "纳斯达克": "usIXIC",
    "道琼斯": "usDJI",
    "恒生指数": "hkHSI",
    "恒生科技": "hkHSTECH",
    "沪深300": "sh000300",
    "上证指数": "sh000001",
    "创业板指": "sz399006",
}
# FRED 日序列第三兜底（腾讯限流时启用；美股三指数在 FRED 有官方日收盘）。
FRED_SERIES = {"标普500": "SP500", "纳斯达克": "NASDAQCOM", "道琼斯": "DJIA"}


def _bounded(fn, seconds: float):
    """带硬超时执行——akshare 内部无超时控制，源挂死时整轮采集会被拖住。"""
    box: Dict[str, object] = {}

    def _run() -> None:
        try:
            box["r"] = fn()
        except Exception:
            box["r"] = None

    th = threading.Thread(target=_run, daemon=True)
    th.start()
    th.join(seconds)
    return box.get("r")


def _fred_series(series_id: str, days: int = 130) -> Optional[List[float]]:
    """FRED 日收盘 CSV → 收盘价序列（升序）。失败返回 None。"""
    try:
        with httpx.Client(timeout=15.0, headers=_HEADERS, trust_env=False) as client:
            r = client.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
            rows = [line.split(",") for line in r.text.strip().splitlines()[1:] if "," in line]
        closes = [float(v) for _, v in rows if v not in ("", ".")]
        return closes[-days:] if len(closes) >= 2 else None
    except Exception:
        return None


def _kline_tencent(code: str, days: int = 130) -> Optional[List[float]]:
    """腾讯日 K → 收盘价序列（升序）。失败返回 None。"""
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    try:
        with httpx.Client(timeout=12.0, headers=_HEADERS, trust_env=False) as client:
            r = client.get(url, params={"param": f"{code},day,,,{days},qfq"})
            node = (r.json().get("data") or {}).get(code) or {}
        rows = node.get("qfqday") or node.get("day") or []
        closes = []
        for row in rows:
            try:
                closes.append(float(row[2]))
            except (TypeError, ValueError, IndexError):
                continue
        return closes if len(closes) >= 2 else None
    except Exception:
        return None


def _kline_closes(secid: str, days: int = 130) -> Optional[List[float]]:
    """东财 push2his 日线 → 收盘价序列（升序）。失败返回 None。"""
    end = date.today().strftime("%Y%m%d")
    beg = (date.today() - timedelta(days=days)).strftime("%Y%m%d")
    try:
        with httpx.Client(timeout=8.0, headers=_HEADERS, trust_env=False) as client:
            r = client.get(
                "http://push2his.eastmoney.com/api/qt/stock/kline/get",
                params={
                    "secid": secid,
                    "fields1": "f1,f2,f3,f4,f5",
                    "fields2": "f51,f52,f53",
                    "klt": "101",
                    "fqt": "1",
                    "beg": beg,
                    "end": end,
                },
            )
            klines = (r.json().get("data") or {}).get("klines") or []
        if len(klines) < 2:
            return None
        return [float(k.split(",")[2]) for k in klines]
    except Exception:
        return None


def fetch_asset_closes(days: int = 130) -> Dict[str, List[float]]:
    """拉全部资产收盘序列——东财/腾讯双源互备。

    顺序：东财 secid 候选依次尝试 → 腾讯日 K → FRED CSV。
    东财连续 2 个资产全败即熔断，其余资产直接走腾讯/FRED——
    东财源挂死时每资产 4×超时会把整轮采集拖到十几分钟。
    """
    out: Dict[str, List[float]] = {}
    stats = {"east": 0, "tencent": 0, "fred": 0, "miss": 0}
    east_dead = False
    east_fail_streak = 0
    for name, candidates in {**ASSET_SECIDS, **INDEX_SECIDS}.items():
        closes, source = None, None
        if not east_dead:
            for secid in candidates:
                closes = _kline_closes(secid, days)
                if closes:
                    source = "east"
                    break
            east_fail_streak = east_fail_streak + 1 if not closes else 0
            if east_fail_streak >= 2:
                east_dead = True
                logger.warning("全球雷达: 东财源连续失败，本次运行熔断改走腾讯/FRED")
        if not closes:
            code = TENCENT_CODES.get(name)
            if code:
                closes = _kline_tencent(code, days)
                if closes:
                    source = "tencent"
        if not closes:
            series_id = FRED_SERIES.get(name)
            if series_id:
                closes = _fred_series(series_id, days)
                if closes:
                    source = "fred"
        if closes:
            out[name] = closes
            stats[source] += 1
        else:
            stats["miss"] += 1
            logger.warning("全球雷达: %s 东财/腾讯双源均无数据", name)
    logger.info("全球雷达行情源统计: %s", stats)
    return out


def fetch_us10y() -> Optional[List[float]]:
    """美债 10 年收益率序列（近 60 日）：FRED DGS10 优先 → akshare 东财源兜底。"""
    try:
        with httpx.Client(timeout=12.0, headers=_HEADERS, trust_env=False) as client:
            r = client.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "DGS10"})
            rows = [line.split(",") for line in r.text.strip().splitlines()[1:] if "," in line]
            series = [float(v) for _, v in rows if v not in ("", ".")]
            if len(series) >= 2:
                return series[-60:]
    except Exception:
        pass

    def _ak() -> Optional[List[float]]:
        import akshare as ak

        df = ak.bond_zh_us_rate(start_date="19900101")
        col = next((c for c in df.columns if "美国" in c and "10" in c), None)
        if col is not None and df is not None and not df.empty:
            series = df[col].astype(float).dropna().tolist()
            if len(series) >= 2:
                return series[-60:]
        return None

    return _bounded(_ak, 20.0)


def fetch_btc(days: int = 90) -> Optional[List[float]]:
    """BTC 收盘序列：CoinGecko 日线 → FRED CBBTCUSD 兜底。失败返回 None。"""
    try:
        with httpx.Client(timeout=12.0, trust_env=False) as client:
            r = client.get(
                "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
                params={"vs_currency": "usd", "days": str(days), "interval": "daily"},
            )
            prices = (r.json() or {}).get("prices") or []
        if len(prices) >= 2:
            return [float(p[1]) for p in prices]
    except Exception:
        pass
    try:
        with httpx.Client(timeout=12.0, headers=_HEADERS, trust_env=False) as client:
            r = client.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": "CBBTCUSD"})
            rows = [line.split(",") for line in r.text.strip().splitlines()[1:] if "," in line]
            series = [float(v) for _, v in rows if v not in ("", ".")]
            if len(series) >= 2:
                return series[-days:]
    except Exception:
        pass
    return None


def _tushare_pro(timeout: int = 30):
    from ..tushare_client import create_tushare_pro

    return create_tushare_pro(timeout=timeout)


def fetch_macro_china() -> Optional[Dict[str, List[tuple]]]:
    """中国宏观月频序列：制造业 PMI（增长轴）+ CPI 同比（通胀轴）。

    Tushare 接口按月份倒序返回（最新在前），这里翻转为时间升序。
    """
    try:
        pro = _tushare_pro()
        result: Dict[str, List[tuple]] = {}

        pmi = pro.cn_pmi(fields="MONTH,PMI010000")
        if pmi is not None and not pmi.empty:
            pmi.columns = [str(c).lower() for c in pmi.columns]  # 代理返回小写列名
            rows = []
            for _, r in pmi.head(14).iterrows():
                v = r.get("pmi010000")
                if v == v and v is not None:
                    rows.append((str(r["month"]), float(v)))
            if rows:
                result["pmi"] = list(reversed(rows))
        cpi = pro.cn_cpi(fields="month,nt_yoy")
        if cpi is not None and not cpi.empty:
            cpi.columns = [str(c).lower() for c in cpi.columns]
            rows = []
            for _, r in cpi.head(14).iterrows():
                v = r.get("nt_yoy")
                if v == v and v is not None:
                    rows.append((str(r["month"]), float(v)))
            if rows:
                result["cpi"] = list(reversed(rows))
        return result if result else None
    except Exception as exc:
        logger.warning(f"中国宏观数据拉取失败: {exc}")
        return None


def fetch_margin() -> Optional[List[tuple]]:
    """两融余额序列（近 20 交易日，亿元）：Tushare margin 汇总各交易所。

    最新交易日可能只披露了沪市（深市 T+1），单边余额约为全市场一半，
    会把 5 日变化算出 -50% 的假信号——按交易所行数过滤掉不完整日期。
    """
    try:
        pro = _tushare_pro()
        start = (date.today() - timedelta(days=30)).strftime("%Y%m%d")
        df = pro.margin(start_date=start, end_date=date.today().strftime("%Y%m%d"), fields="trade_date,rzye")
        if df is None or df.empty:
            return None
        counts = df.groupby("trade_date")["rzye"].count()
        full_dates = set(counts[counts >= 2].index)
        df = df[df["trade_date"].isin(full_dates)]
        agg = df.groupby("trade_date")["rzye"].sum() / 1e8  # 元 → 亿元
        # 转 python float（numpy 标量不可 JSON 序列化）
        return sorted((str(d), float(v)) for d, v in agg.items())
    except Exception as exc:
        logger.warning(f"两融数据拉取失败: {exc}")
        return None


def fetch_qvix() -> Optional[List[float]]:
    """50ETF 期权 QVIX 序列（A 股 VIX）。akshare 源不稳时返回 None。"""

    def _ak() -> Optional[List[float]]:
        import akshare as ak

        df = ak.index_option_50etf_qvix()
        if df is None or df.empty:
            return None
        value_col = next((c for c in df.columns if "qvix" in str(c).lower() or "波动" in str(c)), None) or df.columns[-1]
        series = df[value_col].astype(float).dropna().tolist()
        return series[-60:] if len(series) >= 2 else None

    return _bounded(_ak, 20.0)


def fetch_breadth() -> Optional[Dict]:
    """A 股广度：上涨/下跌家数、涨停数(近似)、全市场成交额。一次 daily 全量。"""
    try:
        pro = _tushare_pro()
        latest = pro.trade_cal(exchange="SSE", is_open="1", start_date=(date.today() - timedelta(days=10)).strftime("%Y%m%d"), end_date=date.today().strftime("%Y%m%d"))
        trade_date = str(sorted(latest["cal_date"])[-1])
        df = pro.daily(trade_date=trade_date, fields="ts_code,pct_chg,amount")
        if df is None or df.empty:
            return None
        codes = df["ts_code"].astype(str)
        main_board = ~codes.str.startswith(("300", "301", "688", "689", "8", "4"))
        pct = df["pct_chg"]
        up = int((pct > 0).sum())
        down = int((pct < 0).sum())
        limit_up = int(((pct >= 9.9) & main_board).sum() + ((pct >= 19.9) & ~main_board).sum())
        return {
            "trade_date": trade_date,
            "up": up,
            "down": down,
            "limit_up": limit_up,
            "amount_yi": round(float(df["amount"].sum()) * 1000.0 / 1e8, 0),  # 千元 → 亿元
        }
    except Exception as exc:
        logger.warning(f"市场广度拉取失败: {exc}")
        return None
