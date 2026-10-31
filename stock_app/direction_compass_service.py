# -*- coding: utf-8 -*-
"""多空合议信号（Direction Compass）：把平台全部方向性证据合成一个读数。

构成与权重（依据平台既有验证结果，2026-10-28/31 复核）：
  - GBM 主信号 40%：唯一经 2600 日 walk-forward 验证有方向优势的资产
    （选择性出手 +7.3%）；未过选择性门槛时按中性 0.5 半权计入。
  - 同状态概率 30%：当前回撤/动量状态的历史 20 日上涨概率（统计底座，
    独立机械使用不增益收益，故只作调节项）。
  - 周期位置 15%：仅极底(<20=1.0)/极顶(≥80=0.3)有验证过的极端增益，
    中间地带取当前仓位值。
  - DeepSeek 15%：历史验证无超额（-5.6%），最低权重参与合议。

输出：0-100 多空分 + 偏多/中性/偏空 标签 + 置信档（按分项一致性）。
诚实声明：这是平台各验证过的边缘证据的"合议"，不是新的超额来源；
指数日频方向的机械预测不存在（三次独立回测均证伪）。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_LATEST = PROJECT_ROOT / "data_cache" / "jev" / "forecast_latest.json"
CYCLE_PATH = PROJECT_ROOT / "data_cache" / "bull_bear" / "results" / "cycle_signal_latest.json"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text("utf-8"))
    except Exception:
        return {}


def compute_compass() -> Optional[Dict]:
    from .market_regime_service import compute_state_probability

    fc = _read_json(FORECAST_LATEST)
    gbm = fc.get("gbm") or {}
    deepseek = fc.get("deepseek") or {}
    cyc = _read_json(CYCLE_PATH)
    prob = compute_state_probability()

    if not gbm.get("p_up") and gbm.get("p_up") != 0:
        return None

    # ── GBM: 选择性出手才全权可信 ──
    gbm_p = float(gbm.get("p_up") or 0.5)
    gbm_selective = bool(gbm.get("selective"))
    gbm_eff = gbm_p if gbm_selective else 0.5

    # ── 同状态概率(20日) ──
    state_p = None
    cs = (prob or {}).get("当前统计") or {}
    state20 = cs.get("20日") or {}
    if state20.get("上涨概率") is not None:
        state_p = float(state20["上涨概率"])

    # ── 周期位置: 极端值才有方向含义 ──
    cycle_score = float(cyc.get("cycle_score") or 50)
    if cycle_score < 20:
        cycle_p = 100.0        # 极底=满仓级多头证据
    elif cycle_score >= 80:
        cycle_p = 30.0         # 极顶=偏空证据(非对称规则下只降到0.3仓)
    else:
        cycle_p = float(cyc.get("current_position") or 0.5) * 100

    # ── DeepSeek ──
    ds_p = float(deepseek.get("p_up") or 0.5)

    # 全部分项统一到 0-100 刻度再加权（GBM/DS 原始为 0-1 概率）
    score = (gbm_eff * 100 * 0.40 + (state_p if state_p is not None else 50) * 0.30
             + cycle_p * 0.15 + ds_p * 100 * 0.15)
    score = round(max(0.0, min(100.0, score)), 0)

    if score >= 58:
        label = "偏多"
    elif score <= 42:
        label = "偏空"
    else:
        label = "中性"

    # ── 置信档: 分项与最终标签的一致性 ──
    votes = [("GBM", gbm_eff), ("同状态概率", state_p), ("周期位置", cycle_p), ("DeepSeek", ds_p)]
    votes = [(n, v) for n, v in votes if v is not None]
    agree = sum(1 for _, v in votes if (v >= 50) == (score >= 50))
    confidence = {3: "高", 4: "高"}.get(agree, "中" if agree == 2 else "低")

    # 分歧提示
    dissent = [n for n, v in votes if abs(v - score) >= 15]

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="minutes"),
        "数据日": fc.get("as_of"),
        "多空分": score,
        "标签": label,
        "置信": confidence,
        "分项": {
            "GBM主信号": {"p_up": round(gbm_p, 2), "选择性出手": gbm_selective, "权重": 0.40},
            "同状态概率20日": {"p_up": round(state_p, 1) if state_p is not None else None, "权重": 0.30},
            "周期位置": {"温度": cycle_score, "阶段": cyc.get("phase"), "权重": 0.15},
            "DeepSeek": {"p_up": round(ds_p, 2), "权重": 0.15},
        },
        "分歧项": dissent,
        "合议说明": (
            "GBM选择性出手是平台唯一验证过有方向优势的信号(+7.3%)，故权重最高；"
            "同状态概率与周期极值是统计调节项；DeepSeek历史无超额仅低权参与。"
            "分项分歧大时置信自动降档。"
        ),
    }


if __name__ == "__main__":
    print(json.dumps(compute_compass(), ensure_ascii=False, indent=1))
