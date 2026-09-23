"""行情与机构权重工具。

机构权重/行业归一化移植自原项目 market_utils.py；
行情查询升级为本平台 Tushare pro（原项目 akshare 腾讯源已限流弃用）。
"""
from __future__ import annotations

import re
import threading
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Dict, Optional

import httpx


TIER1_INSTITUTIONS = [
    "中信证券", "中金公司", "中金", "华泰证券", "国泰君安", "招商证券",
    "广发证券", "申万宏源", "海通证券", "兴业证券", "国信证券", "中信建投",
    "东吴证券", "国金证券", "光大证券", "平安证券", "银河证券", "东方证券",
    "方正证券", "中泰证券", "长江证券", "安信证券", "财通证券", "浙商证券",
    "国盛证券", "华西证券", "西南证券", "民生证券", "天风证券", "东兴证券",
]


def institution_weight(institution: Optional[str]) -> float:
    if not institution:
        return 0.8
    for name in TIER1_INSTITUTIONS:
        if name in institution:
            return 1.5
    if "证券" in institution or "基金" in institution or "期货" in institution:
        return 1.0
    return 0.8


INDUSTRY_ALIASES = {
    "白酒": "食品饮料", "酒类": "食品饮料", "食品": "食品饮料", "饮料": "食品饮料", "乳制品": "食品饮料", "调味品": "食品饮料",
    "AI": "AI算力", "人工智能": "AI算力", "算力": "AI算力", "AIGC": "AI算力", "大模型": "AI算力", "光模块": "AI算力", "CPO": "AI算力", "算力租赁": "AI算力",
    "锂电": "新能源电池", "锂电池": "新能源电池", "动力电池": "新能源电池", "电池": "新能源电池", "固态电池": "新能源电池", "钠电": "新能源电池", "储能电池": "新能源电池",
    "光伏": "光伏", "太阳能": "光伏", "组件": "光伏", "硅片": "光伏",
    "风电": "风电", "海风": "风电", "海上风电": "风电",
    "半导体": "半导体", "芯片": "半导体", "晶圆": "半导体", "存储": "半导体", "存储芯片": "半导体", "EDA": "半导体", "光刻": "半导体", "先进封装": "半导体",
    "半导体设备": "半导体", "半导体材料": "半导体",
    "新能源": "新能源", "新能源车": "新能源汽车", "电动车": "新能源汽车", "汽车": "新能源汽车", "智能汽车": "新能源汽车", "智能驾驶": "新能源汽车",
    "医药": "医药生物", "创新药": "医药生物", "生物医药": "医药生物", "CXO": "医药生物", "医疗": "医药生物", "中药": "医药生物",
    "机器人": "机器人", "人形机器人": "机器人", "工业机器人": "机器人", "减速器": "机器人", "伺服": "机器人",
    "军工": "国防军工", "国防": "国防军工", "航天": "国防军工", "航空": "国防军工", "商业航天": "国防军工",
    "券商": "非银金融", "保险": "非银金融", "银行": "银行", "金融": "非银金融",
    "地产": "房地产", "房地产": "房地产", "物业": "房地产",
    "消费": "消费", "零售": "消费", "免税": "消费", "美妆": "消费", "家电": "消费",
    "电子": "电子", "消费电子": "电子", "PCB": "电子", "面板": "电子",
    "计算机": "计算机", "软件": "计算机", "信创": "计算机", "云计算": "计算机", "数据要素": "计算机",
    "传媒": "传媒", "游戏": "传媒", "影视": "传媒", "短剧": "传媒", "广告": "传媒",
    "有色": "有色金属", "金属": "有色金属", "黄金": "有色金属", "稀土": "有色金属", "铜": "有色金属", "铝": "有色金属", "锂": "有色金属",
    "煤炭": "煤炭", "电力": "公用事业", "公用事业": "公用事业",
    "化工": "基础化工", "石化": "基础化工", "石油": "石油石化",
    "机械": "机械设备", "工程机械": "机械设备", "通用设备": "机械设备", "专用设备": "机械设备",
    "建筑": "建筑装饰", "建材": "建筑材料", "水泥": "建筑材料",
    "农业": "农林牧渔", "养殖": "农林牧渔", "种植": "农林牧渔", "猪": "农林牧渔",
    "物流": "交通运输", "航运": "交通运输", "航空运输": "交通运输", "快递": "交通运输",
    "纺织服装": "纺织服饰", "服装": "纺织服饰", "纺织": "纺织服饰",
    "通信": "通信", "5G": "通信", "光通信": "通信",
    "环保": "环保", "钢铁": "钢铁", "轻工": "轻工制造", "造纸": "轻工制造",
    "社会服务": "社会服务", "旅游": "社会服务", "酒店": "社会服务", "餐饮": "社会服务",
    "培育钻石": "培育钻石", "金刚石": "培育钻石",
}


def normalize_industry(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return tag
    tag = tag.strip()
    if tag in INDUSTRY_ALIASES:
        return INDUSTRY_ALIASES[tag]
    for k, v in INDUSTRY_ALIASES.items():
        if k in tag:
            return v
    return tag


def normalize_industry_set(tags) -> set:
    return {normalize_industry(t) for t in tags if t}


TARGET_PRICE_PATTERNS = [
    r"目标价[为：:\s]*([0-9]+(?:\.[0-9]+)?)\s*元",
    r"目标价格[为：:\s]*([0-9]+(?:\.[0-9]+)?)\s*元",
    r"目标市值[为：:\s]*([0-9]+(?:\.[0-9]+)?)\s*亿",
    r"给予目标价([0-9]+(?:\.[0-9]+)?)\s*元",
    r"A股目标价([0-9]+(?:\.[0-9]+)?)\s*元",
    r"合理价值[为：:\s]*([0-9]+(?:\.[0-9]+)?)\s*元",
    r"合理估值[为：:\s]*([0-9]+(?:\.[0-9]+)?)\s*元",
]


def extract_target_price(text: str) -> Optional[str]:
    if not text:
        return None
    for pattern in TARGET_PRICE_PATTERNS:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


# ============ 行情（Tushare 升级版） ============

_price_day_cache: Dict = {"date": None, "prices": {}}
_tushare_lock = threading.Lock()


def to_ts_code(stock_code: str) -> str:
    """6 位纯数字代码 → Tushare ts_code。"""
    if not stock_code:
        return ""
    if "." in stock_code:
        return stock_code
    if stock_code.startswith(("6", "9")):
        return f"{stock_code}.SH"
    if stock_code.startswith(("4", "8")):
        return f"{stock_code}.BJ"
    return f"{stock_code}.SZ"


def _tushare_daily(ts_code: str, start: date, end: date):
    from ..tushare_client import create_tushare_pro

    with _tushare_lock:
        pro = create_tushare_pro(timeout=30)
        return pro.daily(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            fields="trade_date,close",
        )


# ═══ 每日价格快照：4 次批量调用拿全市场收盘价，之后全部走内存 ═══
# 背景：signals/晨报要对 75+ 只股票查最新价与 20 日涨跌，逐股调 Tushare
# 在全局锁下串行要 100s+。改为每天拉一次快照（最新/昨日/20日前三张全市场收盘表），
# 单股查询 O(1)。
_price_snapshot: dict = {"date": None, "latest": {}, "prev": {}, "base20": {}, "latest_date": None}
_snapshot_loading = False


def _bare_code(ts_or_code: str) -> str:
    return str(ts_or_code).split(".")[0].zfill(6)


def _load_price_snapshot() -> None:
    """按日加载全市场收盘快照（最新/上一交易日/约20个交易日前）。失败静默，保留旧快照。"""
    global _snapshot_loading
    import threading

    from datetime import date as _date, timedelta as _td

    today = _date.today()
    if _price_snapshot["date"] == today and _price_snapshot["latest"]:
        return
    if _snapshot_loading:
        return
    _snapshot_loading = True
    try:
        from ..tushare_client import create_tushare_pro

        pro = None
        with _tushare_lock:
            pro = create_tushare_pro(timeout=30)
            cal = pro.trade_cal(
                exchange="SSE", is_open="1",
                start_date=(today - _td(days=60)).strftime("%Y%m%d"),
                end_date=today.strftime("%Y%m%d"),
            )
        if cal is None or cal.empty:
            return
        dates = sorted(cal["cal_date"].astype(str))
        if len(dates) < 3:
            return

        def _fetch_day(trade_date: str) -> dict:
            with _tushare_lock:
                df = pro.daily(trade_date=trade_date, fields="ts_code,close")
            if df is None or df.empty:
                return {}
            return {_bare_code(r.ts_code): float(r.close) for r in df.itertuples(index=False)}

        # 凌晨/盘前当日日线尚未生成：从最新日往回找最近一个有数据的日子
        latest, latest_d = {}, None
        for candidate in reversed(dates[-4:]):
            latest = _fetch_day(candidate)
            if latest:
                latest_d = candidate
                break
        if not latest:
            return
        idx = dates.index(latest_d)
        prev_d = dates[idx - 1] if idx >= 1 else None
        base20_d = dates[idx - 20] if idx >= 20 else dates[0]
        prev = _fetch_day(prev_d) if prev_d else {}
        base = _fetch_day(base20_d) if base20_d and base20_d != latest_d else {}
        if latest:
            _price_snapshot.update(
                date=today, latest=latest, prev=prev, base20=base, latest_date=latest_d,
            )
    except Exception:
        pass
    finally:
        _snapshot_loading = False


def _snap_change(stock_code: str, days: int) -> Optional[float]:
    """快照版涨跌幅：days=1 用 prev，days=20 用 base20。未命中返回 None。"""
    _load_price_snapshot()
    code = _bare_code(stock_code)
    latest = _price_snapshot["latest"].get(code)
    if latest is None:
        return None
    base = (_price_snapshot["prev"] if days <= 1 else _price_snapshot["base20"]).get(code)
    if base is None or not base:
        return None
    return round((latest - base) / base * 100.0, 2)


def get_latest_price(stock_code: str) -> Optional[float]:
    """最新收盘价（当日缓存 + 全市场快照优先）。Tushare 失败时回退 StockPick 上次记录价。"""
    _load_price_snapshot()
    snap = _price_snapshot["latest"].get(_bare_code(stock_code))
    if snap is not None:
        return snap
    today = date.today()
    if _price_day_cache["date"] == today and stock_code in _price_day_cache["prices"]:
        return _price_day_cache["prices"][stock_code]

    result = None
    try:
        df = _tushare_daily(to_ts_code(stock_code), today - timedelta(days=14), today)
        if df is not None and not df.empty:
            result = round(float(df.sort_values("trade_date")["close"].iloc[-1]), 4)
    except Exception:
        result = None
    if result is None:
        try:
            from ..models import StockPick

            pick = (
                StockPick.objects.filter(stock_code=stock_code, price_at_pick__isnull=False)
                .order_by("-pick_date")
                .first()
            )
            if pick:
                result = pick.price_at_pick
        except Exception:
            pass

    if _price_day_cache["date"] != today:
        _price_day_cache.update(date=today, prices={})
    _price_day_cache["prices"][stock_code] = result
    return result


def get_price_on_or_after(stock_code: str, from_date: date, trading_days: int) -> Optional[float]:
    """从 from_date 起第 N 个交易日收盘价（用于 T+1/T+5/T+20 回填）。"""
    try:
        df = _tushare_daily(
            to_ts_code(stock_code),
            from_date,
            from_date + timedelta(days=trading_days * 2 + 15),
        )
        if df is None or df.empty:
            return None
        closes = df.sort_values("trade_date")["close"].tolist()
        if not closes:
            return None
        idx = min(trading_days, len(closes) - 1)
        return float(closes[idx])
    except Exception:
        return None


_change_day_cache: Dict = {"date": None, "changes": {}}


def get_stock_change(stock_code: str, days: int = 20) -> Optional[float]:
    """近 N 个交易日涨跌幅（原项目 akshare 限流后恒为 None，现用 Tushare 实现）。按 (code, days) 当日缓存。"""
    snap = _snap_change(stock_code, days)
    if snap is not None:
        return snap
    today = date.today()
    cache_key = (stock_code, days)
    if _change_day_cache["date"] == today and cache_key in _change_day_cache["changes"]:
        return _change_day_cache["changes"][cache_key]
    try:
        df = _tushare_daily(to_ts_code(stock_code), today - timedelta(days=days * 2 + 20), today)
        if df is None or len(df) < 2:
            return None
        closes = df.sort_values("trade_date")["close"].tolist()
        base_idx = max(0, len(closes) - 1 - days)
        result = round((closes[-1] - closes[base_idx]) / closes[base_idx] * 100, 2)
    except Exception:
        result = None
    if _change_day_cache["date"] != today:
        _change_day_cache.update(date=today, changes={})
    _change_day_cache["changes"][cache_key] = result
    return result


_names_batch_cache: Dict = {"date": None, "map": {}}


def fetch_names_batch(codes: list) -> Dict[str, str]:
    """腾讯行情接口批量查股票名称，GBK 编码，秒回不限流。按 (日, code) 缓存。"""
    result = {}
    today = date.today()
    cache_fresh = _names_batch_cache["date"] == today
    codes = [c for c in codes if c and isinstance(c, str) and len(c) == 6 and c.isdigit()]
    pending = []
    for c in codes:
        if cache_fresh and c in _names_batch_cache["map"]:
            result[c] = _names_batch_cache["map"][c]
        else:
            pending.append(c)
    if not pending:
        return result
    try:
        symbols = ",".join(
            ("sh" if c.startswith(("6", "9")) else "bj" if c.startswith(("4", "8")) else "sz") + c for c in codes
        )
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            r = client.get(f"http://qt.gtimg.cn/q={symbols}")
        r.encoding = "gbk"
        for line in r.text.split(";"):
            line = line.strip()
            if "=" not in line or '"' not in line:
                continue
            parts = line.split('"')[1].split("~")
            if len(parts) >= 3 and parts[1] and parts[2]:
                result[parts[2]] = parts[1]
        if result:
            if _names_batch_cache["date"] != today:
                _names_batch_cache.update(date=today, map={})
            _names_batch_cache["map"].update(result)
    except Exception:
        pass
    return result


_index_cache: Dict = {"date": None, "data": []}


def get_index_quotes() -> list:
    """三大指数日/周/月涨跌（东财 kline 接口，当日缓存）。"""
    today = date.today()
    if _index_cache["date"] == today and _index_cache["data"]:
        return _index_cache["data"]
    result = []
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        secid_map = {"1.000001": "上证指数", "0.399001": "深证成指", "0.399006": "创业板指"}
        with httpx.Client(timeout=10.0, headers=headers, trust_env=False) as client:
            for secid, name in secid_map.items():
                try:
                    r = client.get(
                        "http://push2his.eastmoney.com/api/qt/stock/kline/get",
                        params={
                            "secid": secid,
                            "fields1": "f1,f2,f3,f4,f5",
                            "fields2": "f51,f52,f53",
                            "klt": "101",
                            "fqt": "1",
                            "beg": (today - timedelta(days=45)).strftime("%Y%m%d"),
                            "end": today.strftime("%Y%m%d"),
                        },
                    )
                    klines = (r.json().get("data") or {}).get("klines") or []
                    if len(klines) < 2:
                        continue
                    closes = [float(k.split(",")[2]) for k in klines]
                    latest, prev = closes[-1], closes[-2]
                    base5 = closes[-6] if len(closes) >= 6 else closes[0]
                    base20 = closes[-21] if len(closes) >= 21 else closes[0]
                    result.append({
                        "symbol": secid,
                        "name": name,
                        "price": round(latest, 2),
                        "change_1d": round((latest - prev) / prev * 100, 2),
                        "change_5d": round((latest - base5) / base5 * 100, 2),
                        "change_20d": round((latest - base20) / base20 * 100, 2),
                    })
                except Exception:
                    continue
        if result:
            _index_cache.update(date=today, data=result)
    except Exception:
        pass
    return result
