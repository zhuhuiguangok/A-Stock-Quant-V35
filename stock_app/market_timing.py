# -*- coding: utf-8 -*-
"""
私募级大盘择时引擎 (Market Timing)
=====================================
从 views.py 拆分出的独立择时模块。
不依赖外部指数数据，用全市场截面均值作为市场代理。

用途：
  from .market_timing import compute_market_timing, detect_market_regime

私募标准：
  - 等权平均截面收益 → 市场方向
  - 波动率均值 → 风险水平
  - 成交量比均值 → 市场参与度
  - 熊市禁止趋势信号，仅保留抄底
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_market_timing(df: pd.DataFrame, parent_logger=None) -> Dict:
    """
    私募标准大盘择时信号

    不依赖外部指数数据，用全市场截面均值作为市场代理：
    - 等权平均20日收益 → 市场方向
    - 波动率均值 → 风险水平
    - 成交量比均值 → 市场参与度

    返回:
      {
        'regime': 'bull'|'bear'|'neutral',
        'trend_allowed': bool,      # 是否允许趋势选股
        'trend_weight_pct': float,  # 趋势信号在总分中的权重(0~1)
        'market_score': float,      # 市场评分(0~100)
        'volatility': float,        # 市场波动率
      }
    """
    log = parent_logger or logger

    result = {
        'regime': 'neutral',
        'trend_allowed': True,
        'trend_weight_pct': 0.50,
        'market_score': 50.0,
        'volatility': 0.0,
    }

    # 1. 市场方向：等权平均20日收益
    if 'pmt_return_20d' in df.columns:
        _ret = pd.to_numeric(df['pmt_return_20d'], errors='coerce').fillna(0)
        if 'trade_date' in df.columns:
            _latest = df['trade_date'].max()
            _ret_snap = _ret[df['trade_date'] == _latest]
        else:
            _ret_snap = _ret
        _mean_ret = float(_ret_snap.mean()) if len(_ret_snap) > 0 else 0.0
        _median_ret = float(_ret_snap.median()) if len(_ret_snap) > 0 else 0.0
    else:
        _mean_ret = 0.0
        _median_ret = 0.0

    # 2. 波动率：20日历史波动率均值
    #    dropna 而非 fillna(0)：NaN 混入均值会把波动率拉成 0，
    #    导致 high_vol 判定永远不触发、前端显示"波动率 0%"
    if 'volat_hist_20d' in df.columns:
        _vol = pd.to_numeric(df['volat_hist_20d'], errors='coerce').dropna()
        if 'trade_date' in df.columns:
            _latest = df['trade_date'].max()
            _vol_snap = _vol[df['trade_date'] == _latest]
        else:
            _vol_snap = _vol
        result['volatility'] = float(_vol_snap.mean()) if len(_vol_snap) > 0 else 0.0
    else:
        result['volatility'] = 0.0

    # 3. 成交量比
    if 'vol_ratio_raw' in df.columns:
        _vr = pd.to_numeric(df['vol_ratio_raw'], errors='coerce').fillna(1.0)
        if 'trade_date' in df.columns:
            _latest = df['trade_date'].max()
            _vr_snap = _vr[df['trade_date'] == _latest]
        else:
            _vr_snap = _vr
        _mean_vr = float(_vr_snap.mean()) if len(_vr_snap) > 0 else 1.0
    else:
        _mean_vr = 1.0

    # 4. 小市值占比
    if 'size_score' in df.columns:
        _sz = pd.to_numeric(df['size_score'], errors='coerce').fillna(0.5)
        if 'trade_date' in df.columns:
            _latest = df['trade_date'].max()
            _sz_snap = _sz[df['trade_date'] == _latest]
        else:
            _sz_snap = _sz
        _small_cap_ratio = float((_sz_snap > 0.7).mean()) if len(_sz_snap) > 0 else 0.0
    else:
        _small_cap_ratio = 0.0

    # ═══════════════════════════════════════════════════════════════
    # 私募择时规则（基于头部私募风控实践）
    # ═══════════════════════════════════════════════════════════════
    high_vol = result['volatility'] > 0.028  # 日波动率>2.8%视为高波

    if _median_ret > 0.02 and _mean_ret > 0.01 and not high_vol:
        # 牛市：市场普涨+低波动 → 趋势选股最佳环境
        result['regime'] = 'bull'
        result['trend_allowed'] = True
        result['trend_weight_pct'] = 0.65
        result['market_score'] = min(95, 60 + (_mean_ret * 300))
    elif _median_ret < -0.02 and _mean_ret < -0.01:
        # 熊市：市场普跌 → 只允许抄底（逆势），禁止趋势追高
        result['regime'] = 'bear'
        result['trend_allowed'] = False
        result['trend_weight_pct'] = 0.0
        result['market_score'] = max(15, 40 + (_mean_ret * 200))
        log.warning("  🛑 熊市择时：趋势信号已禁止，仅保留抄底信号")
    elif high_vol:
        # 高波动：两种信号都允许但降权
        result['regime'] = 'neutral' if _mean_ret > -0.01 else 'bear'
        result['trend_allowed'] = _mean_ret > -0.01
        result['trend_weight_pct'] = 0.30 if result['trend_allowed'] else 0.0
        result['market_score'] = max(25, 45 + (_mean_ret * 150))
        if result['volatility'] > 0.035:
            log.warning(f"  ⚡ 极高波动({result['volatility']:.4f})：风控模式")
    else:
        # 震荡：信号对半
        result['regime'] = 'neutral'
        result['trend_allowed'] = True
        result['trend_weight_pct'] = 0.50
        result['market_score'] = 50.0

    if _small_cap_ratio > 0.6:
        result['regime'] = f"{result['regime']}_small_dominant"

    # 5. 置信度（2026-09-19：替换写死的 70/40 常数）
    #    三个可解释成分合成，体现"这个择时判断有多可信"：
    #    a) 方向认同度——涨/跌家数占比偏离 50% 的显著性（二项近似 z 值）
    #    b) 信噪比——截面均值收益的 t 统计量（均值/离散度×√n），tanH 映射
    #    c) 样本覆盖——截面股票数（对数饱和），样本越大越可信
    #    区间锁定 35~85：不存在 100% 置信的择时，也不给无意义的低分
    confidence = 50.0
    try:
        if 'pmt_return_20d' in df.columns:
            import math as _math
            _ret_all = pd.to_numeric(df['pmt_return_20d'], errors='coerce').dropna()
            if 'trade_date' in df.columns and not _ret_all.empty:
                _latest_c = df['trade_date'].max()
                _ret_conf = _ret_all[df['trade_date'] == _latest_c]
            else:
                _ret_conf = _ret_all
            n = int(len(_ret_conf))
            if n >= 30:
                pos_ratio = float((_ret_conf > 0).mean())
                z = abs(pos_ratio - 0.5) / (0.5 / _math.sqrt(n))
                agree = 0.5 + 0.5 * (1.0 - _math.exp(-z / 3.0))   # 0.55~1.0
                std = float(_ret_conf.std())
                t = (float(_ret_conf.mean()) / std * _math.sqrt(n)) if std > 1e-9 else 0.0
                snr = _math.tanh(abs(t) / 4.0)                     # 0~1
                cover = min(1.0, 0.4 + 0.12 * _math.log10(max(n, 1)))
                confidence = 100.0 * max(0.35, min(0.85, 0.5 * agree + 0.3 * snr + 0.2 * cover))
            else:
                confidence = 45.0  # 截面样本不足，给保守低置信
    except Exception:
        confidence = 50.0
    result['confidence'] = round(confidence, 1)

    log.info(
        f"  📊 市场择时: regime={result['regime']} | "
        f"trend={'✅' if result['trend_allowed'] else '🚫'} | "
        f"trend_w={result['trend_weight_pct']:.0%} | "
        f"市场均收益={_mean_ret:.3f} | "
        f"波动率={result['volatility']:.4f} | "
        f"评分={result['market_score']:.0f} | "
        f"置信={result['confidence']}%"
    )
    return result


def detect_market_regime(df: pd.DataFrame) -> str:
    """返回 regime 字符串（兼容旧接口）"""
    return str(compute_market_timing(df).get('regime', 'neutral'))
