#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用本地已有数据（data_cache/*.parquet）重新训练全部模型。
====================================================
- 不联网拉 Tushare，直接读本地 parquet
- 调 factor_engine.calculate_rule_factors 补齐技术因子
- 调 V19EnhancedEngine.train_all_models 训练 XGBoost / AI引擎 / 双树模型
- 训练产物落盘 models/，后端单例（stock_app.views.v19_enhanced_engine）
  下次启动时自动加载

用法：
    python retrain_with_local_data.py                 # 自动选最大可用数据
    python retrain_with_local_data.py --file <name>   # 指定 parquet 文件名
"""
import os
import sys
import glob
import argparse
import time

# ── Django 环境（views.py 依赖 django.shortcuts 等）──────────────────────
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_project.settings')
import django
django.setup()

import pandas as pd
from stock_app.views import (
    v19_enhanced_engine,
    get_real_stock_data,  # noqa: F401  (供参考，本脚本不调用)
)
from stock_app.factor_engine import calculate_rule_factors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')


def pick_best_local_data() -> str:
    """自动选择行数最多、最新的 real_data parquet。"""
    cands = glob.glob(os.path.join(CACHE_DIR, 'real_data_*.parquet'))
    if not cands:
        sys.exit('❌ data_cache 下没有 real_data_*.parquet，无法训练')
    best, best_sz = None, -1
    for p in cands:
        try:
            sz = os.path.getsize(p)
            # 优先全市场（all）数据，其次按文件大小
            if 'real_data_all_' in p:
                sz *= 2  # 权重加成
            if sz > best_sz:
                best, best_sz = p, sz
        except OSError:
            pass
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--file', default=None,
                    help='data_cache 下指定的 parquet 文件名（不含目录）')
    args = ap.parse_args()

    if args.file:
        path = os.path.join(CACHE_DIR, args.file)
        if not os.path.exists(path):
            sys.exit(f'❌ 指定文件不存在: {path}')
    else:
        path = pick_best_local_data()

    print(f'\n📦 训练数据: {os.path.basename(path)}')
    df = pd.read_parquet(path)
    print(f'   原始形状: {df.shape}')
    print(f'   交易日: {df["trade_date"].nunique() if "trade_date" in df else "?"}  '
          f'股票数: {df["ts_code"].nunique() if "ts_code" in df else "?"}')

    # ── Step 1: 补齐技术因子（rsi/macd/pmt_return/vol_ratio 等）──────────
    print('\n🔧 [1/3] 计算技术因子（calculate_rule_factors）...')
    t0 = time.time()
    df = calculate_rule_factors(df)
    print(f'   完成，耗时 {time.time()-t0:.1f}s，形状 {df.shape}')
    tech_present = [c for c in ['rsi', 'macd', 'pmt_return_5d', 'vol_ratio_5d']
                    if c in df.columns]
    print(f'   关键技术因子已就绪: {tech_present}')

    # ── Step 2: 训练全部模型 ────────────────────────────────────────────
    print('\n🧠 [2/3] 训练全部模型（XGBoost / AI引擎 / 双树模型）...')
    t0 = time.time()
    trained = v19_enhanced_engine.train_all_models(df)
    print(f'   训练 {"✅成功" if trained else "⚠️未成功"}，耗时 {time.time()-t0:.1f}s')

    # ── Step 3: 校验产物 ────────────────────────────────────────────────
    print('\n💾 [3/3] 校验 models/ 产物:')
    models_dir = os.path.join(BASE_DIR, 'models')
    expect = [
        'ai_engine_mlp.pth', 'ai_engine_transformer.pth',
        'ai_engine_gnn.pth', 'ai_engine_hgnn.pth',
        'ai_engine_smart_xgnn_nn.pth', 'ai_engine_smart_xgnn_xgb.pkl',
        'trend.pkl', 'bottom.pkl',
    ]
    now = time.time()
    for f in expect:
        p = os.path.join(models_dir, f)
        if os.path.exists(p):
            age_min = (now - os.path.getmtime(p)) / 60
            mark = '🆕' if age_min < 5 else '⏳旧'
            print(f'   {mark} {f:32s} ({os.path.getsize(p)//1024} KB, {age_min:.1f} 分钟前)')
        else:
            print(f'   ❌ {f:32s} 缺失')

    print('\n✅ 全部完成。后端重启后将自动加载新模型。')


if __name__ == '__main__':
    main()
