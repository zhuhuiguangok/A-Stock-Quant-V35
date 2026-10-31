# -*- coding: utf-8 -*-
"""同状态历史概率表（Market State Probability）。

不预测方向——把当前市场状态（距250日高点回撤档 × 20日动量档）映射到
全历史上所有同状态交易日，输出其后的真实收益分布（5/20/60日）。

为什么这么做（回测结论，2026-10 验证）：
- 指数趋势状态（MA/牛熊线/双均线）对未来收益无判别力，二元清仓过滤全部跑输持有；
- 均值回归结构性成立：深回撤+阴跌状态后续继续大跌是小概率（20日跌超5%仅9%），
  而追"中回撤后的强势"是历史最差状态（60日均-3.1%）；
- 与牛熊周期回测同构：极值区域统计有效，中间地带无信号。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_CACHE = PROJECT_ROOT / "data_cache" / "jev" / "index_daily_000001sh.json"

_DD_BUCKETS = [
    ("浅回撤", lambda d: d > -5),
    ("中回撤", lambda d: -10 < d <= -5),
    ("深回撤", lambda d: -20 < d <= -10),
    ("重挫", lambda d: d <= -20),
]
_R20_BUCKETS = [
    ("强势", lambda r: r > 3),
    ("平盘", lambda r: -3 <= r <= 3),
    ("阴跌", lambda r: -8 < r <= -3),
    ("急跌", lambda r: r <= -8),
]
_HORIZONS = (5, 20, 60)


def _load_closes() -> tuple:
    try:
        closes = json.loads(INDEX_CACHE.read_text("utf-8"))
        closes = closes[-2900:]
        return [d for d, _ in closes], [c for _, c in closes]
    except Exception as exc:
        logger.warning(f"指数缓存读取失败: {exc}")
        return [], []


def _bucket_of(value: float, table) -> str:
    for label, fn in table:
        if fn(value):
            return label
    return table[-1][0]


def _percentile(sorted_vals: List[float], q: float) -> Optional[float]:
    if not sorted_vals:
        return None
    idx = min(int(len(sorted_vals) * q), len(sorted_vals) - 1)
    return round(sorted_vals[idx], 1)


def compute_state_probability(min_samples: int = 15) -> Optional[Dict]:
    """计算当前状态与全状态概率表。数据不足返回 None。"""
    days, px = _load_closes()
    n = len(px)
    if n < 320:
        return None

    def ma(i: int, k: int) -> Optional[float]:
        return sum(px[i + 1 - k:i + 1]) / k if i + 1 >= k else None

    # 全历史逐日状态
    buckets: Dict[tuple, list] = {}
    states: List[Optional[tuple]] = [None] * n
    for i in range(250, n):
        dd = (px[i] / max(px[i - 249:i + 1]) - 1) * 100
        r20 = (px[i] / px[i - 20] - 1) * 100
        dd_lab = _bucket_of(dd, _DD_BUCKETS)
        r_lab = _bucket_of(r20, _R20_BUCKETS)
        states[i] = (dd_lab, r_lab, round(dd, 1), round(r20, 1))
        entry = {"day": days[i]}
        for h in _HORIZONS:
            if i + h < n:
                entry[f"f{h}"] = (px[i + h] / px[i] - 1) * 100
        buckets.setdefault((dd_lab, r_lab), []).append(entry)

    cur = states[-1]
    if cur is None:
        return None
    dd_lab, r_lab, cur_dd, cur_r20 = cur
    cur_key = (dd_lab, r_lab)

    m20, m60, m250 = ma(n - 1, 20), ma(n - 1, 60), ma(n - 1, 250)

    def _summarize(entries: list) -> dict:
        v = [e for e in entries if "f20" in e]
        n_ = len(v)
        if n_ < min_samples:
            return {"样本": n_, "不足": True}
        out = {
            "样本": n_,
            "5日反弹概率": round(sum(1 for e in entries if "f5" in e and e["f5"] > 0) / max(1, sum(1 for e in entries if "f5" in e)) * 100),
        }
        for h in _HORIZONS:
            vals = sorted(e[f"f{h}"] for e in v if f"f{h}" in e)
            if vals:
                out[f"{h}日"] = {
                    "均": round(sum(vals) / len(vals), 1),
                    "中位": _percentile(vals, 0.5),
                    "P10": _percentile(vals, 0.1),
                    "P90": _percentile(vals, 0.9),
                    "上涨概率": round(sum(1 for x in vals if x > 0) / len(vals) * 100),
                }
        if v and "f20" in v[-1]:
            out["续跌超5%概率20日"] = round(sum(1 for e in v if "f20" in e and e["f20"] < -5) / n_ * 100)
        return out

    current_stats = _summarize(buckets.get(cur_key, []))
    # 最近8次同状态实例（只取20日窗口已走完的，供前端展示实际结果）
    done = [e for e in buckets.get(cur_key, []) if "f20" in e]
    recent = [
        {"day": e["day"], "20日实际": round(e["f20"], 1)}
        for e in done[-8:]
    ]

    # 全表（供前端折叠展示与"更差状态参考"）
    table = []
    for (d_lab, r_lab), entries in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        row = _summarize(entries)
        row.update({"回撤档": d_lab, "动量档": r_lab,
                    "是当前": d_lab == dd_lab and r_lab == r_lab})
        table.append(row)

    # 解除/恶化参考: 更深一档回撤同动量的统计
    worse = _summarize(buckets.get(("重挫", r_lab), [])) if dd_lab != "重挫" else None

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="minutes"),
        "数据日": days[-1],
        "当前状态": {
            "回撤档": dd_lab, "动量档": r_lab,
            "距250日高点%": cur_dd, "20日涨跌%": cur_r20,
            "收盘": px[-1],
            "MA20": round(m20, 0) if m20 else None,
            "MA60": round(m60, 0) if m60 else None,
            "MA250": round(m250, 0) if m250 else None,
        },
        "当前统计": current_stats,
        "最近同状态": recent,
        "全表": table,
        "若继续恶化(重挫档同动量)": worse,
        "口径说明": "统计口径=同状态历史日的真实后续收益分布，非方向预测；样本<15的桶不展示统计",
    }


if __name__ == "__main__":
    print(json.dumps(compute_state_probability(), ensure_ascii=False, indent=1)[:2000])
