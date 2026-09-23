"""从研报文本正则提取盈利预测（净利润/EPS）。移植自原项目 forecast_extractor.py。"""
from __future__ import annotations

import re
from typing import Dict, List


PROFIT_PATTERNS = [
    r"(20\d{2})\s*年[^。；，,]{0,30}?归母净利润[^0-9]{0,8}([0-9]+\.?[0-9]*)\s*亿",
    r"(20\d{2})\s*年[^。；，,]{0,30}?净利润[^0-9]{0,8}([0-9]+\.?[0-9]*)\s*亿",
    r"归母净利润[^0-9]{0,10}([0-9]+\.?[0-9]*)\s*亿",
    r"净利润[^0-9]{0,10}([0-9]+\.?[0-9]*)\s*亿",
    r"预计[^。；，,]{0,25}?盈利[^0-9]{0,8}([0-9]+\.?[0-9]*)\s*亿",
]

MULTI_YEAR_PATTERN = r"(20\d{2})\s*[-/—~]\s*(20\d{2})\s*年[^。；]{0,40}?净利润[^0-9]{0,8}([0-9]+\.?[0-9]*)"

EPS_PATTERNS = [
    r"(20\d{2})\s*年[^。；，,]{0,25}?EPS[^0-9]{0,6}([0-9]+\.?[0-9]*)\s*元",
    r"(20\d{2})\s*年[^。；，,]{0,25}?每股收益[^0-9]{0,6}([0-9]+\.?[0-9]*)\s*元",
    r"EPS[^0-9]{0,6}([0-9]+\.?[0-9]*)\s*元",
    r"每股收益[^0-9]{0,6}([0-9]+\.?[0-9]*)\s*元",
]


def extract_profit_forecasts(text: str) -> List[Dict]:
    if not text:
        return []
    text = text.replace(",", "")
    results, seen = [], set()

    for m in re.finditer(MULTI_YEAR_PATTERN, text):
        year1, value = m.group(1), m.group(3)
        try:
            v = float(value)
            if 0.01 < v < 100000 and (int(year1), v) not in seen:
                seen.add((int(year1), v))
                results.append({"year": int(year1), "profit": v, "text": m.group(0)[:80]})
        except Exception:
            continue

    for pattern in PROFIT_PATTERNS:
        for m in re.finditer(pattern, text):
            groups = m.groups()
            year, value = (groups[0], groups[1]) if len(groups) == 2 else (None, groups[0])
            try:
                v = float(value)
                if 0.01 < v < 100000:
                    y = int(year) if year else None
                    if (y, v) not in seen:
                        seen.add((y, v))
                        results.append({"year": y, "profit": v, "text": m.group(0)[:80]})
            except Exception:
                continue
    return results[:5]


def extract_eps_forecasts(text: str) -> List[Dict]:
    if not text:
        return []
    results, seen = [], set()
    for pattern in EPS_PATTERNS:
        for m in re.finditer(pattern, text):
            groups = m.groups()
            year, value = (groups[0], groups[1]) if len(groups) == 2 else (None, groups[0])
            try:
                v = float(value)
                if 0.001 < v < 1000:
                    y = int(year) if year else None
                    if (y, v) not in seen:
                        seen.add((y, v))
                        results.append({"year": y, "eps": v})
            except Exception:
                continue
    return results[:3]
