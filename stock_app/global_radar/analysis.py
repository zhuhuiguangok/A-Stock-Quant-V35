"""全球雷达分析层：资产温度计 / 风险偏好分 / 宏观时钟 / A 股恐惧贪婪分。

纯函数（输入序列 → 输出 dict），不落盘不联网，便于单测。
"""
from __future__ import annotations

from typing import Dict, List, Optional


def _chg(closes: List[float], n: int) -> Optional[float]:
    """近 n 日涨跌幅（%）。样本不足返回 None。"""
    if not closes or len(closes) < n + 1:
        return None
    base, last = closes[-(n + 1)], closes[-1]
    if not base:
        return None
    return round((last - base) / base * 100.0, 2)


def _heat_level(pct: Optional[float]) -> int:
    """涨跌幅 → -3..3 热力档（红=涨绿=跌，A 股习惯）。"""
    if pct is None:
        return 0
    if pct >= 5.0:
        return 3
    if pct >= 1.5:
        return 2
    if pct >= 0.15:
        return 1
    if pct <= -5.0:
        return -3
    if pct <= -1.5:
        return -2
    if pct <= -0.15:
        return -1
    return 0


ASSET_ORDER = [
    "标普500", "纳斯达克", "道琼斯", "纳指100ETF",
    "恒生指数", "恒生科技", "日经225", "韩国KOSPI", "印度Sensex", "德国DAX", "法国CAC40", "英国富时100",
    "上证指数", "沪深300", "创业板指",
    "美元指数", "COMEX黄金", "COMEX白银", "WTI原油", "布伦特原油", "COMEX铜",
]

# 资产 20 日方向 → 一句话含义（对全球/A股的传导解读）
ASSET_INTERP = {
    "标普500": {1: "美股涨=全球胆子大，A股也跟着有底气", -1: "美股跌=全球缩手，A股情绪跟着降温"},
    "纳斯达克": {1: "美国科技股强，A股的AI芯片跟着沾光", -1: "美国科技股跌，A股科技股先怂为敬"},
    "纳指100ETF": {1: "美国科技股强，A股的AI芯片跟着沾光", -1: "美国科技股跌，A股科技股先怂为敬"},
    "道琼斯": {1: "美国老经济(银行工厂)强，周期股的好日子", -1: "美国老经济转弱"},
    "恒生指数": {1: "外资回头买中国资产了，A股港股一起嗨", -1: "外资还在观望中国资产"},
    "恒生科技": {1: "中国科技公司被重新看好", -1: "中国科技资产被抛弃"},
    "日经225": {1: "日本股市吸金，亚洲资金被分走一些", -1: "日股热度退潮"},
    "韩国KOSPI": {1: "韩国芯片股是全球半导体晴雨表，涨=芯片周期向上，A股半导体受益", -1: "芯片周期可能见顶的预警"},
    "印度Sensex": {1: "印度吸金，亚洲资金分流", -1: "新兴市场资金面改善"},
    "德国DAX": {1: "欧洲工厂开工足，中国出口链有单接", -1: "欧洲没钱买东西，中国出口承压"},
    "法国CAC40": {1: "欧洲需求回暖", -1: "欧洲需求走弱"},
    "英国富时100": {1: "欧洲需求回暖", -1: "欧洲需求走弱"},
    "上证指数": {1: "A股大盘自身在走上坡路", -1: "大盘在走下坡路"},
    "沪深300": {1: "大蓝筹(白马股)趋势向上，指数行情", -1: "白马股在调整"},
    "创业板指": {1: "成长股(小而快)当主角", -1: "成长股在挨打"},
    "美元指数": {1: "美元变贵→钱回流美国→A股外资变少，黄金反而受益", -1: "美元变便宜→钱流向新兴市场→A股人民币资产吃香"},
    "COMEX黄金": {1: "大家在买保险(避险)或抗通胀，乱世买金", -1: "恐慌退潮，钱去追风险资产了"},
    "COMEX白银": {1: "白银兼具工业+避险双属性，光伏也用它", -1: "工业需求预期转弱"},
    "WTI原油": {1: "油涨=运东西变贵(通胀)，石油股开心航空股哭", -1: "油跌=成本降了，航空物流开心，石油股哭"},
    "布伦特原油": {1: "全球通胀预期上行", -1: "全球需求预期走弱（油是全球需求体温计）"},
    "COMEX铜": {1: "铜是经济体温计：涨=全球工厂在猛干，A股有色受益", -1: "铜跌=全球工厂减速的预警"},
    "美债10Y收益率(%)": {1: "利率升→借钱贵→高估值股挨压，银行保险赚息差", -1: "利率降→借钱便宜→科技成长股起飞"},
    "比特币": {1: "币圈狂欢=全球钱多到溢出，风险偏好的极端信号", -1: "币圈寒冬=全球钱变紧"},
}

ASSET_EMOJI = {
    "标普500": "🇺🇸", "纳斯达克": "💻", "道琼斯": "🏭", "纳指100ETF": "💻",
    "恒生指数": "🇭🇰", "恒生科技": "📱", "日经225": "🇯🇵", "韩国KOSPI": "🇰🇷",
    "印度Sensex": "🇮🇳", "德国DAX": "🇩🇪", "法国CAC40": "🇫🇷", "英国富时100": "🇬🇧",
    "上证指数": "🇨🇳", "沪深300": "🀄", "创业板指": "🚀",
    "美元指数": "💵", "COMEX黄金": "🥇", "COMEX白银": "🥈", "WTI原油": "🛢️",
    "布伦特原油": "🛢️", "COMEX铜": "🔩", "美债10Y收益率(%)": "🏦", "比特币": "🪙",
}


def _spark_points(closes: List[float], n: int = 20, width: int = 60, height: int = 18) -> Optional[List[float]]:
    """近 n 日收盘 → 归一化折线坐标（SVG polyline 用）。样本不足返回 None。"""
    if not closes or len(closes) < n:
        return None
    vals = closes[-n:]
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [height / 2] * n
    return [round(height - (v - lo) / (hi - lo) * height, 1) for v in vals]


def _pct_rank(closes: List[float], window: int = 20) -> Optional[int]:
    """当前 window 日涨幅在过去全部滚动 window 日涨幅中的分位（0-100）。

    例：美元 20 日涨幅处于 92 分位 = 近半年罕见强势——给数字加上"多罕见"的语境。
    """
    if not closes or len(closes) < window * 3:
        return None
    changes = [
        (closes[i] / closes[i - window] - 1.0) * 100.0
        for i in range(window, len(closes))
    ]
    if not changes:
        return None
    current = changes[-1]
    rank = sum(1 for c in changes if c <= current) / len(changes) * 100
    return int(round(rank))


def build_asset_heat(assets: Dict[str, List[float]], us10y: Optional[List[float]], btc: Optional[List[float]]) -> Dict:
    """资产温度计：每项 1/5/20/60 日变化 + 热力档 + 迷你走势 + 含义解读 + 历史分位。"""
    rows = []
    series_map: Dict[str, List[float]] = dict(assets)
    if us10y:
        series_map["美债10Y收益率(%)"] = us10y[-60:]
    if btc:
        series_map["比特币"] = btc
    for name in ASSET_ORDER + (["美债10Y收益率(%)"] if us10y else []) + (["比特币"] if btc else []):
        closes = series_map.get(name)
        if not closes:
            continue
        chg20 = _chg(closes, 20)
        interp_map = ASSET_INTERP.get(name, {})
        direction = 1 if (chg20 or 0) > 1.0 else (-1 if (chg20 or 0) < -1.0 else 0)
        interp = interp_map.get(direction) or interp_map.get(1 if direction >= 0 else -1, "")
        rank = _pct_rank(closes, 20)
        rows.append({
            "name": name,
            "last": round(closes[-1], 2),
            "chg_1d": _chg(closes, 1),
            "chg_5d": _chg(closes, 5),
            "chg_20d": chg20,
            "chg_60d": _chg(closes, 60),
            "spark": _spark_points(closes),
            "interp": interp,
            "pct_rank_20d": rank,
        })
    for row in rows:
        row["heat_20d"] = _heat_level(row.get("chg_20d"))
        row["heat_1d"] = _heat_level(row.get("chg_1d"))
    return {"rows": rows}


def _score_from(parts: List[tuple]) -> Optional[int]:
    """[(分量值(-1..1), 权重)] → 0-100 加权分。全空返回 None。"""
    total_w = sum(w for _, w in parts)
    if total_w <= 0:
        return None
    value = sum(v * w for v, w in parts) / total_w
    return int(round((value + 1) / 2 * 100))


def _norm_pct(pct: Optional[float], span: float = 3.0) -> float:
    """涨跌幅 → -1..1（超出截断）。"""
    if pct is None:
        return 0.0
    return max(-1.0, min(1.0, pct / span))


def build_risk_appetite(assets: Dict[str, List[float]], us10y: Optional[List[float]]) -> Dict:
    """全球风险偏好分 0-100（越高越 risk-on）。

    分量：全球股指 20 日动量（正）、美元指数 5 日（负）、美债收益率 5 日（负）、
    黄金 20 日与油 20 日之差（风险厌恶 vs 通胀需求）。
    """
    parts = []
    equities = [assets[n] for n in ("标普500", "纳斯达克", "恒生指数", "日经225") if assets.get(n)]
    if equities:
        avg20 = sum(_norm_pct(_chg(e, 20)) for e in equities) / len(equities)
        parts.append((avg20, 0.40))
    dxy = assets.get("美元指数")
    if dxy:
        parts.append((-_norm_pct(_chg(dxy, 5), 1.0), 0.25))
    if us10y and len(us10y) >= 6:
        diff = us10y[-1] - us10y[-6]
        parts.append((-max(-1.0, min(1.0, diff / 0.25)), 0.20))
    gold = _norm_pct(_chg(assets.get("COMEX黄金") or [], 20))
    oil = _norm_pct(_chg(assets.get("WTI原油") or assets.get("布伦特原油") or [], 20))
    if assets.get("COMEX黄金"):
        parts.append((max(-1.0, min(1.0, oil - gold)), 0.15))

    score = _score_from(parts)
    if score is None:
        return {"score": None, "label": "数据不足", "components": []}
    label = "Risk-On 风险偏好扩张" if score >= 65 else ("Risk-Off 避险主导" if score <= 35 else "中性摇摆")
    return {
        "score": score,
        "label": label,
        "components": [
            {"name": "全球股市动量(20日)", "ok": bool(equities)},
            {"name": "美元流动性(5日)", "ok": bool(dxy)},
            {"name": "美债利率压力(5日)", "ok": bool(us10y and len(us10y) >= 6)},
            {"name": "油金比风险结构", "ok": bool(assets.get("COMEX黄金"))},
        ],
    }


def _realized_vol(closes: List[float], n: int = 20) -> Optional[float]:
    """年化已实现波动率（%）：QVIX 不可用时的波动率分量替代。"""
    if len(closes) < n + 1:
        return None
    rets = [(closes[i] / closes[i - 1] - 1.0) for i in range(len(closes) - n, len(closes))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return round(var ** 0.5 * (252 ** 0.5) * 100.0, 2)


def build_fear_greed(qvix: Optional[List[float]], margin: Optional[List[tuple]],
                     breadth: Optional[Dict], assets: Dict[str, List[float]]) -> Dict:
    """A 股恐惧贪婪分 0-100（>80 极度贪婪，<20 极度恐惧）。

    QVIX 缺失时自动用沪深300 20 日已实现波动率替代波动率分量。
    """
    parts = []
    csi = assets.get("沪深300") or assets.get("上证指数")
    vol_source = "QVIX 期权隐波"
    if qvix and len(qvix) >= 21:
        vix_ma = sum(qvix[-20:]) / 20
        ratio = max(-1.0, min(1.0, -(qvix[-1] - vix_ma) / max(vix_ma, 1) / 0.15))  # QVIX 相对均线降→贪婪
        parts.append((ratio, 0.25))
        parts.append((max(-1.0, min(1.0, (25.0 - qvix[-1]) / 10.0)), 0.15))  # 绝对水平
    elif csi:
        rv = _realized_vol(csi, 20)
        if rv is not None:
            parts.append((max(-1.0, min(1.0, (22.0 - rv) / 10.0)), 0.40))  # 波动率低→贪婪
            vol_source = "沪深300已实现波动(20日)"
    if margin and len(margin) >= 6:
        chg5 = (margin[-1][1] - margin[-6][1]) / margin[-6][1] * 100
        parts.append((max(-1.0, min(1.0, chg5 / 1.5)), 0.20))
    if breadth and breadth.get("up") is not None and breadth.get("down") is not None:
        total = breadth["up"] + breadth["down"]
        if total > 0:
            parts.append(((breadth["up"] - breadth["down"]) / total, 0.20))
    if csi:
        parts.append((_norm_pct(_chg(csi, 20)), 0.20))

    score = _score_from(parts)
    if score is None:
        return {"score": None, "label": "数据不足", "components": []}
    label = ("极度贪婪" if score >= 80 else "贪婪" if score >= 60 else
             "极度恐惧" if score <= 20 else "恐惧" if score <= 40 else "中性")
    return {
        "score": score,
        "label": label,
        "qvix": round(qvix[-1], 2) if qvix else None,
        "realized_vol": _realized_vol(csi, 20) if csi else None,
        "margin_yi": round(margin[-1][1], 0) if margin else None,
        "margin_chg5_pct": round((margin[-1][1] - margin[-6][1]) / margin[-6][1] * 100, 2) if margin and len(margin) >= 6 else None,
        "breadth": breadth,
        "index_chg20d": _chg(csi, 20) if csi else None,
        "components": [
            {"name": vol_source, "ok": bool(qvix or csi)},
            {"name": "两融余额 5 日变化", "ok": bool(margin and len(margin) >= 6)},
            {"name": "涨跌家数广度", "ok": bool(breadth and breadth.get("up") is not None)},
            {"name": "沪深300 动量", "ok": bool(csi)},
        ],
    }


CLOCK_META = {
    "reflation": {"name": "衰退（Reflation）", "asset": "债券占优", "hint": "增长回落+通胀回落，宽松预期升温，利好债券与高股息防御资产"},
    "recovery": {"name": "复苏（Recovery）", "asset": "股票占优", "hint": "增长回升+通胀回落，企业盈利改善而成本温和，股票最佳窗口"},
    "overheat": {"name": "过热（Overheat）", "asset": "商品占优", "hint": "增长回升+通胀回升，需求强劲推升物价，资源品与商品跑赢"},
    "stagflation": {"name": "滞胀（Stagflation）", "asset": "现金占优", "hint": "增长回落+通胀回升，盈利受压而政策受限，现金/短债防御"},
}


def _direction(series: List[float]) -> int:
    """序列方向：取近 3 点均值 vs 前 3 点均值，+1 上行 / -1 下行 / 0 平。"""
    if len(series) < 6:
        return 0
    recent = sum(series[-3:]) / 3
    earlier = sum(series[-6:-3]) / 3
    if earlier == 0:
        return 0
    diff = (recent - earlier) / abs(earlier)
    return 1 if diff > 0.002 else (-1 if diff < -0.002 else 0)


def build_clock_china(macro: Optional[Dict[str, List[tuple]]]) -> Dict:
    """中国宏观时钟：PMI=增长轴（相对荣枯线 50 的水平+方向），CPI 同比=通胀轴（动量方向）。"""
    if not macro or not macro.get("pmi") or not macro.get("cpi"):
        return {"ok": False, "quadrant": None, "reason": "宏观数据暂不可达"}

    pmi_vals = [v for _, v in macro["pmi"][-6:]]
    cpi_vals = [v for _, v in macro["cpi"][-6:]]
    pmi_now = pmi_vals[-1]
    growth_dir = _direction(pmi_vals)
    # 增长轴：PMI>50 且上行=强(+1)；<50 且下行=弱(-1)；其余按方向
    growth = 1 if (pmi_now >= 50 and growth_dir >= 0) or (pmi_now > 50 and growth_dir > 0) else (-1 if growth_dir < 0 else (1 if growth_dir > 0 else 0))
    infl_dir = _direction(cpi_vals)
    inflation = 1 if infl_dir > 0 else (-1 if infl_dir < 0 else 0)

    quadrant = {(1, 1): "overheat", (1, -1): "recovery", (-1, 1): "stagflation", (-1, -1): "reflation"}.get((growth, inflation))
    if quadrant is None:
        quadrant = "recovery" if growth >= 0 else "reflation"
    meta = CLOCK_META[quadrant]
    return {
        "ok": True,
        "region": "中国",
        "quadrant": quadrant,
        "quadrant_name": meta["name"],
        "asset": meta["asset"],
        "hint": meta["hint"],
        "growth_axis": growth,
        "inflation_axis": inflation,
        "pmi": round(pmi_now, 2),
        "pmi_dir": growth_dir,
        "cpi_yoy": round(cpi_vals[-1], 2),
        "cpi_dir": infl_dir,
        "pmi_month": macro["pmi"][-1][0],
        "cpi_month": macro["cpi"][-1][0],
    }


def build_clock_us(assets: Dict[str, List[float]]) -> Dict:
    """美国时钟（市场代理版）：标普 60 日动量=增长轴，黄金+油 20 日均值动量=通胀轴。"""
    spx = assets.get("标普500")
    gold = assets.get("COMEX黄金")
    oil = assets.get("WTI原油") or assets.get("布伦特原油")
    if not spx:
        return {"ok": False, "region": "美国", "quadrant": None, "reason": "美股数据缺失"}

    growth = 1 if (_chg(spx, 60) or 0) > 1.0 else (-1 if (_chg(spx, 60) or 0) < -1.0 else 0)
    inflation = 0
    if gold and oil:
        avg = (_norm_pct(_chg(gold, 20)) + _norm_pct(_chg(oil, 20))) / 2
        inflation = 1 if avg > 0.15 else (-1 if avg < -0.15 else 0)

    quadrant = {(1, 1): "overheat", (1, -1): "recovery", (-1, 1): "stagflation", (-1, -1): "reflation"}.get((growth, inflation))
    if quadrant is None:
        quadrant = "recovery" if growth >= 0 else "reflation"
    meta = CLOCK_META[quadrant]
    return {
        "ok": True,
        "region": "美国",
        "quadrant": quadrant,
        "quadrant_name": meta["name"],
        "asset": meta["asset"],
        "hint": meta["hint"],
        "growth_axis": growth,
        "inflation_axis": inflation,
        "spx_chg60d": _chg(spx, 60),
        "proxy": True,
    }


# ═══ 全球 → A 股传导链 ═══

def _trend(closes: Optional[List[float]], window: int, threshold: float = 1.0) -> int:
    """window 日变化 → +1 强 / -1 弱 / 0 平。"""
    chg = _chg(closes or [], window)
    if chg is None:
        return 0
    return 1 if chg > threshold else (-1 if chg < -threshold else 0)


def build_transmission(assets: Dict[str, List[float]], us10y: Optional[List[float]]) -> List[Dict]:
    """全球信号 → A 股的五条传导链：每条给当前状态、一句影响、关联 A 股方向。"""
    chains = []

    dxy = _trend(assets.get("美元指数"), 20, 1.0)
    chains.append({
        "name": "美元流动性 → 外资面",
        "state": {1: "美元偏强", -1: "美元偏弱", 0: "美元平稳"}[dxy],
        "state_dir": dxy,
        "impact": {1: "全球资金回流美国，A股外资供给承压，人民币资产估值受压；但黄金(美元计价)受益",
                   -1: "外资回流新兴市场窗口，人民币资产与大宗商品受益，A股流动性环境改善",
                   0: "外资面中性，焦点回到国内政策与基本面"}[dxy],
        "a_directions": ["黄金"] if dxy > 0 else (["有色", "出口链", "A股整体流动性"] if dxy < 0 else []),
    })

    us10y_t = _trend(us10y or [], 20, 1.5)
    chains.append({
        "name": "美债利率 → 风格切换",
        "state": {1: "利率上行", -1: "利率下行", 0: "利率平稳"}[us10y_t],
        "state_dir": us10y_t,
        "impact": {1: "贴现率抬升压缩高估值成长，银行保险息差受益——风格偏价值防御",
                   -1: "利率下行打开成长/科创估值弹性，久期资产受益——风格偏成长进攻",
                   0: "风格由国内利率与增量资金主导"}[us10y_t],
        "a_directions": ["银行", "保险"] if us10y_t > 0 else (["科创/成长", "TMT"] if us10y_t < 0 else []),
    })

    copper = _trend(assets.get("COMEX铜"), 20, 2.0)
    oil = _trend(assets.get("WTI原油") or assets.get("布伦特原油"), 20, 3.0)
    commodity = copper if copper != 0 else oil
    chains.append({
        "name": "大宗商品 → 通胀与周期",
        "state": {1: "商品走强", -1: "商品走弱", 0: "商品平稳"}[commodity],
        "state_dir": commodity,
        "impact": {1: "铜油齐强=全球需求回暖（复苏期信号），A股资源品受益但下游成本抬升",
                   -1: "商品走弱=全球需求预期降温，资源品承压但中下游成本改善",
                   0: "通胀与周期信号中性"}[commodity],
        "a_directions": ["有色", "煤炭", "石化"] if commodity > 0 else (["航空", "航运", "中游制造"] if commodity < 0 else []),
    })

    kospi = _trend(assets.get("韩国KOSPI"), 20, 2.0)
    spx = _trend(assets.get("标普500"), 20, 2.0)
    tech = kospi if kospi != 0 else spx
    chains.append({
        "name": "全球科技 → 产业映射",
        "state": {1: "外围科技强", -1: "外围科技弱", 0: "外围科技平稳"}[tech],
        "state_dir": tech,
        "impact": {1: "美/韩半导体与科技走强验证产业景气，A股AI算力、半导体映射受益",
                   -1: "外围科技调整压制A股科技板块风险偏好",
                   0: "产业映射信号中性"}[tech],
        "a_directions": ["AI算力", "半导体", "消费电子"] if tech > 0 else (["红利防御"] if tech < 0 else []),
    })

    hsi = _trend(assets.get("恒生指数"), 20, 2.0)
    hstech = _trend(assets.get("恒生科技"), 20, 2.0)
    cn_asset = hstech if hstech != 0 else hsi
    chains.append({
        "name": "中国资产 → 外资态度",
        "state": {1: "港股走强", -1: "港股走弱", 0: "港股平稳"}[cn_asset],
        "state_dir": cn_asset,
        "impact": {1: "港股（离岸中国资产）走强=外资对中国资产态度积极，A股联动受益",
                   -1: "港股走弱反映外资谨慎，A股增量资金依赖内资",
                   0: "外资态度中性"}[cn_asset],
        "a_directions": ["互联网", "A股核心资产"] if cn_asset > 0 else (["内需", "红利"] if cn_asset < 0 else []),
    })
    return chains


# ═══ A 股受益方向（规则打分 + 跨模块共振） ═══

# 全球信号 → A 股方向的规则权重
DIRECTION_RULES = [
    (("COMEX铜", 1), ["有色（铜铝）"], 2.0, "全球铜价走强=制造业需求验证"),
    (("COMEX铜", -1), ["中游制造"], 1.0, "铜价回落降低中游成本"),
    (("WTI原油", 1), ["石化", "油服"], 1.5, "油价上行增厚上游利润"),
    (("WTI原油", -1), ["航空", "航运"], 1.5, "油价回落改善出行成本"),
    (("美元指数", -1), ["有色", "出口链"], 1.5, "美元走弱利多大宗与新兴市场"),
    (("美元指数", 1), ["黄金"], 1.0, "强美元下的避险配置"),
    (("美债10Y收益率(%)", -1), ["科创/成长", "TMT"], 2.0, "利率下行打开成长估值"),
    (("美债10Y收益率(%)", 1), ["银行", "保险"], 1.5, "利率上行利好息差"),
    (("韩国KOSPI", 1), ["半导体", "AI算力"], 2.0, "韩股半导体领先全球周期"),
    (("标普500", 1), ["AI算力", "消费电子"], 1.0, "美股风险偏好回暖"),
    (("恒生科技", 1), ["互联网", "恒生科技链"], 1.5, "中国科技资产重估"),
    (("COMEX白银", 1), ["光伏银浆"], 1.0, "银价上行（工业+贵金属双驱动）"),
]


def build_a_share_directions(assets: Dict[str, List[float]], us10y: Optional[List[float]],
                             fund_flow_attention: Optional[List[Dict]] = None,
                             research_resonance: Optional[Dict] = None) -> Dict:
    """全球信号映射 A 股受益方向，并与资金流关注度/研报共振交叉验证。

    只有全球信号（规则分）不足以实战——叠加资金共振（资金流关注度榜在列）
    与研报共振（近7天≥2篇看多）形成三角验证，三重确认的方向置信度最高。
    """
    series_map: Dict[str, Optional[List[float]]] = dict(assets)
    if us10y:
        series_map["美债10Y收益率(%)"] = us10y

    scores: Dict[str, Dict] = {}
    for (asset, want_dir), directions, weight, reason in DIRECTION_RULES:
        closes = series_map.get(asset)
        if not closes:
            continue
        actual = _trend(closes, 20, 1.0)
        if actual != want_dir:
            continue
        for d in directions:
            entry = scores.setdefault(d, {"score": 0.0, "reasons": []})
            entry["score"] += weight
            entry["reasons"].append(reason)

    flow_names = {item.get("sector_name") for item in (fund_flow_attention or []) if item.get("sector_name")}
    resonance = research_resonance or {}

    def _match(direction: str, name_set) -> Optional[str]:
        for name in name_set:
            if name and (name in direction or direction in name or any(k in direction and k in name for k in ("有色", "银行", "保险", "半导体", "石化", "黄金", "AI", "科技", "互联网", "煤炭", "TMT", "成长"))):
                return name
        return None

    out = []
    for direction, entry in scores.items():
        flow_hit = _match(direction, flow_names)
        research_hit = _match(direction, set(resonance.keys()))
        confirms = sum(1 for x in (flow_hit, research_hit) if x)
        out.append({
            "direction": direction,
            "score": round(entry["score"], 1),
            "reasons": entry["reasons"][:2],
            "flow_resonance": flow_hit,
            "research_resonance": research_hit,
            "research_count": resonance.get(research_hit or "", 0) if research_hit else 0,
            "confirms": confirms,
            "level": "强" if confirms >= 2 else ("中" if confirms == 1 else "弱"),
        })
    out.sort(key=lambda x: (x["confirms"], x["score"]), reverse=True)
    return {
        "directions": out[:8],
        "note": "level=全球信号+资金流+研报共振的确认数（强=三重验证）",
    }


# ═══ 大白话层：分数→心情/温度，时钟→季节 ═══

RISK_PLAIN = {
    "hot": {"emoji": "🎉", "plain": "全球撒欢", "hint": "投资者胆子很大，都在追风险资产——行情好但也要留一份清醒"},
    "mid": {"emoji": "😐", "plain": "不温不火", "hint": "全球情绪中庸，没有明显的顺风或逆风，A股走自己的路"},
    "cold": {"emoji": "🛡️", "plain": "避险模式", "hint": "全球都在躲风险——通常是美元/美债在吸金，A股外资面吃紧"},
}

FG_PLAIN = {
    "very_greedy": {"emoji": "🔥", "plain": "过热", "hint": "涨疯了阶段，历史上这个位置追高容易站岗，宁可错过不可做错"},
    "greedy": {"emoji": "😊", "plain": "偏热", "hint": "多头占上风但没到疯狂，持有舒服、追高谨慎"},
    "neutral": {"emoji": "😐", "plain": "不冷不热", "hint": "多空拉锯，看方向选择，适合等信号或均衡配置"},
    "fearful": {"emoji": "😟", "plain": "偏冷", "hint": "跌的人多，情绪低但不绝望，分批低吸的观察区"},
    "very_fearful": {"emoji": "🥶", "plain": "冰点", "hint": "别人恐惧我贪婪——历史上极冷时分批买入的胜率最高"},
}


def risk_plain(score: Optional[int]) -> Dict:
    if score is None:
        return {"emoji": "❓", "plain": "数据不足", "hint": ""}
    key = "hot" if score >= 65 else ("cold" if score <= 35 else "mid")
    return RISK_PLAIN[key]


def fg_plain(score: Optional[int]) -> Dict:
    if score is None:
        return {"emoji": "❓", "plain": "数据不足", "hint": ""}
    if score >= 80:
        key = "very_greedy"
    elif score >= 60:
        key = "greedy"
    elif score >= 40:
        key = "neutral"
    elif score >= 20:
        key = "fearful"
    else:
        key = "very_fearful"
    return FG_PLAIN[key]


CLOCK_SEASON = {
    "recovery": {"season": "春天", "emoji": "🌱", "plain": "经济刚回暖、物价还没起来——历史上这是股票最甜的季节", "wear": "适合播种：股票"},
    "overheat": {"season": "夏天", "emoji": "🔥", "plain": "经济热物价也热——资源品（煤炭有色石油）最吃香", "wear": "适合穿：商品/资源股"},
    "stagflation": {"season": "秋天", "emoji": "🍂", "plain": "经济降温但物价还高——企业最难受的时段，拿现金最舒服", "wear": "适合穿：现金/短债"},
    "reflation": {"season": "冬天", "emoji": "❄️", "plain": "经济物价双降——蛰伏过冬，债券和高股息取暖", "wear": "适合穿：债券/高股息"},
}


def clock_season(quadrant: Optional[str]) -> Dict:
    if not quadrant:
        return {"season": "--", "emoji": "❓", "plain": "", "wear": ""}
    return CLOCK_SEASON.get(quadrant, {"season": "--", "emoji": "❓", "plain": "", "wear": ""})


def decorate(payload: Dict) -> Dict:
    """给面板 payload 附加大白话字段（emoji/心情/季节），就地修改并返回。"""
    ra = payload.get("risk_appetite") or {}
    ra.update(risk_plain(ra.get("score")))
    fg = payload.get("fear_greed") or {}
    fg.update(fg_plain(fg.get("score")))
    for key in ("clock_cn", "clock_us"):
        clock = payload.get(key) or {}
        clock.update(clock_season(clock.get("quadrant")))
    for row in (payload.get("asset_heat") or {}).get("rows", []):
        row["emoji"] = ASSET_EMOJI.get(row.get("name"), "📊")
    return payload
