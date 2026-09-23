import math

import numpy as np
import pandas as pd

from .core import amount_to_yi, normalize_sector_flow


def _record(row):
    return {
        "sector_name": str(row["sector_name"]),
        "main_force_net": float(row["main_force_net"]),
        "main_force_net_yi": amount_to_yi(row["main_force_net"]),
        "change_pct": float(row.get("change_pct", 0.0)),
    }


def _summary(snapshot):
    values = snapshot["main_force_net"].astype(float)
    inflow = values[values > 0].sum()
    outflow = values[values < 0].sum()
    return {
        "sector_count": int(len(snapshot)),
        "total_inflow": float(inflow),
        "total_inflow_yi": amount_to_yi(inflow),
        "total_outflow": float(abs(outflow)),
        "total_outflow_yi": amount_to_yi(abs(outflow)),
        "net_flow": float(values.sum()),
        "net_flow_yi": amount_to_yi(values.sum()),
        "positive_count": int((values > 0).sum()),
        "negative_count": int((values < 0).sum()),
    }


def _sankey_edges(snapshot, limit=5):
    inflows = snapshot[snapshot["main_force_net"] > 0].nlargest(limit, "main_force_net")
    outflows = snapshot[snapshot["main_force_net"] < 0].nsmallest(limit, "main_force_net")
    total_in = inflows["main_force_net"].sum()
    total_out = outflows["main_force_net"].abs().sum()
    if total_in <= 0 or total_out <= 0:
        return []

    edges = []
    scale = min(total_in, total_out)
    for _, source in outflows.iterrows():
        for _, target in inflows.iterrows():
            weight = abs(source["main_force_net"]) / total_out * target["main_force_net"] / total_in * scale
            if weight > 0:
                edges.append({
                    "source": str(source["sector_name"]),
                    "target": str(target["sector_name"]),
                    "amount": float(weight),
                    "amount_yi": amount_to_yi(weight),
                })
    return sorted(edges, key=lambda item: item["amount"], reverse=True)[: limit * 2]


def _pivot_history(history, max_sectors=40):
    pivot = _pivot_history_field(history, "main_force_net", max_sectors)
    return pivot


def _pivot_history_field(history, field, max_sectors=None):
    frame = normalize_sector_flow(history)
    if frame.empty or field not in frame.columns:
        return pd.DataFrame()
    pivot = frame.pivot_table(index="date", columns="sector_name", values=field, aggfunc="last").sort_index()
    # 【性能护栏】两两配对算法（轮动/Granger/网络）是 O(N²)；110 个行业全量
    # 配对要跑数分钟。只保留资金活跃度（平均绝对净流）Top N 的行业参与配对，
    # 冷门行业本就无轮动意义，损失可忽略。
    # 注意：max_sectors=None 表示不裁剪，供关注度等多日统计使用（它覆盖全部行业）。
    if max_sectors and pivot.shape[1] > max_sectors:
        activity = pivot.abs().mean().sort_values(ascending=False)
        pivot = pivot[activity.head(max_sectors).index]
    return pivot


def _corr(a, b):
    a = pd.Series(a).astype(float)
    b = pd.Series(b).astype(float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return 0.0
    value = a.corr(b)
    return 0.0 if value != value else float(value)


def _network_edges(history, threshold=0.35):
    pivot = _pivot_history(history)
    if pivot.shape[0] < 4 or pivot.shape[1] < 2:
        return []
    edges = []
    sectors = list(pivot.columns)
    for source in sectors:
        for target in sectors:
            if source == target:
                continue
            corr = _corr(pivot[source].iloc[:-1], pivot[target].iloc[1:])
            if abs(corr) >= threshold:
                # 历史方向验证：source 当日净流入为正时，target 次日也净流入为正的比例
                src_series = pivot[source].iloc[:-1]
                tgt_next = pivot[target].iloc[1:]
                src_pos = (src_series > 0).to_numpy()
                tgt_values = tgt_next.to_numpy()
                hits, samples = None, int(src_pos.sum())
                base = None
                if samples >= 3:
                    tgt_pos_rate = float((tgt_values[src_pos] > 0).mean())
                    tgt_base = float((pivot[target] > 0).mean())
                    # 命中率需优于基准才有意义（否则只是"大家都流入"的先验）
                    hits = round(tgt_pos_rate, 3)
                    base = round(tgt_base, 3) if tgt_base == tgt_base else 0.5
                edges.append({
                    "source": source,
                    "target": target,
                    "weight": round(abs(corr), 4),
                    "correlation": round(corr, 4),
                    "lead_days": 1,
                    "hit_rate": hits,
                    "hit_base": base,
                    "samples": samples,
                })
    # 有历史胜率验证的边优先展示，纯相关性边退后
    return sorted(
        edges,
        key=lambda item: (item["hit_rate"] is not None, item["weight"]),
        reverse=True,
    )[:30]


def _clamp01(value):
    return max(0.0, min(1.0, float(value)))


def _inflow_streak_stats(series):
    """净流入持续性统计。

    返回 dict:
        streak       当前连续净流入天数（截至最新一日）
        runs         历史各连续流入段长度列表
        continue_prob 条件延续率：已连续≥2天流入时，次日继续流入的历史概率
        continue_n    该概率的样本数
    """
    vals = (series > 0).astype(int).tolist()
    streak, runs, cur = 0, [], 0
    for v in vals:
        if v:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    streak = cur
    if cur:
        runs.append(cur)
    total = hit = 0
    run = 0
    for i, v in enumerate(vals):
        run = run + 1 if v else 0
        if run >= 2 and i + 1 < len(vals):
            total += 1
            hit += 1 if vals[i + 1] else 0
    prob = round(hit / total, 3) if total >= 5 else None
    return {
        "streak": streak,
        "runs": runs,
        "continue_prob": prob,
        "continue_n": total,
    }


def _persistence_score(inflow_ratio, streak, continue_prob):
    """承接持续性分（0~1）：窗口内流入天数占比 + 当前连续段 + 历史延续率。"""
    score = 0.45 * inflow_ratio + 0.30 * min(1.0, streak / 4.0)
    score += 0.25 * (continue_prob if continue_prob is not None else 0.5)
    return _clamp01(score)


def _rotation_edges(history, max_lag=3, window=5, threshold=0.42, limit=20):
    pivot = _pivot_history(history).fillna(0.0)
    if pivot.shape[0] < max(6, window + max_lag) or pivot.shape[1] < 2:
        return []

    edges = []
    sectors = list(pivot.columns)
    for source in sectors:
        source_out = (-pivot[source].astype(float)).clip(lower=0.0)
        latest_source_pressure = float(source_out.tail(window).mean())
        if latest_source_pressure <= 0:
            continue
        source_consistency = float((source_out.tail(window) > 0).mean())

        for target in sectors:
            if source == target:
                continue
            target_in = pivot[target].astype(float).clip(lower=0.0)
            latest_target_momentum = float(target_in.tail(window).mean())
            if latest_target_momentum <= 0:
                continue
            target_consistency = float((target_in.tail(window) > 0).mean())
            # 承接方硬门槛：近 window 日至少 60% 天数净流入——只认"真在持续吸金"的板块，
            # 偶发单日脉冲的承接不算数（用户核心诉求：承接板块必须有持久性）
            if target_consistency < 0.6:
                continue
            consistency = (source_consistency + target_consistency) / 2.0
            if consistency < 0.4:
                continue
            t_stats = _inflow_streak_stats(pivot[target])
            t_persist = _persistence_score(target_consistency, t_stats["streak"], t_stats["continue_prob"])

            best = None
            for lag in range(1, max_lag + 1):
                aligned = pd.DataFrame({
                    "source_out": source_out.shift(lag),
                    "target_in": target_in,
                    "target_lag": target_in.shift(1),
                }).dropna()
                if len(aligned) < 5:
                    continue
                lag_corr = _corr(aligned["source_out"], aligned["target_in"])
                if lag_corr < threshold:
                    continue

                f_score = 0.0
                x_lag = aligned["source_out"].to_numpy(dtype=float)
                y_lag = aligned["target_lag"].to_numpy(dtype=float)
                y_now = aligned["target_in"].to_numpy(dtype=float)
                if np.std(x_lag) > 0 and np.std(y_now) > 0:
                    restricted = np.column_stack([np.ones(len(y_now)), y_lag])
                    full = np.column_stack([np.ones(len(y_now)), y_lag, x_lag])
                    restricted_rss = _rss(y_now, restricted)
                    full_rss = _rss(y_now, full)
                    denominator = full_rss / max(1, len(y_now) - full.shape[1])
                    if denominator > 0:
                        f_score = max(0.0, (restricted_rss - full_rss) / denominator)
                    elif restricted_rss > full_rss:
                        f_score = 9999.0

                source_baseline = float(source_out.mean()) or latest_source_pressure
                target_baseline = float(target_in.mean()) or latest_target_momentum
                latest_strength = _clamp01(
                    ((latest_source_pressure / max(source_baseline, 1.0)) + (latest_target_momentum / max(target_baseline, 1.0))) / 4.0
                )
                granger_strength = _clamp01(f_score / 5.0)
                confidence = _clamp01(
                    0.45 * lag_corr
                    + 0.20 * consistency
                    + 0.20 * granger_strength
                    + 0.15 * latest_strength
                )
                amount = min(latest_source_pressure, latest_target_momentum) * confidence
                candidate = {
                    "source": source,
                    "target": target,
                    "amount": float(amount),
                    "amount_yi": amount_to_yi(amount),
                    "weight": round(confidence, 4),
                    "confidence": round(confidence, 4),
                    "lag_days": lag,
                    "lag_correlation": round(float(lag_corr), 4),
                    "f_score": round(float(min(f_score, 9999.0)), 4),
                    "source_pressure_yi": amount_to_yi(latest_source_pressure),
                    "target_momentum_yi": amount_to_yi(latest_target_momentum),
                    "target_streak": t_stats["streak"],
                    "target_inflow_days": f"{int(round(target_consistency * window))}/{window}",
                    "target_continue_prob": t_stats["continue_prob"],
                    "persistence": round(t_persist, 4),
                    "final_score": round(0.6 * confidence + 0.4 * t_persist, 4),
                    "evidence": "lagged_outflow_to_inflow",
                }
                if best is None or (candidate["confidence"], -candidate["lag_days"]) > (best["confidence"], -best["lag_days"]):
                    best = candidate

            if best is not None and best["confidence"] >= threshold:
                edges.append(best)

    # 排序：综合分 = 0.6*迁移置信 + 0.4*承接持续性——持续性差的承接不再靠单日相关冲到前排
    return sorted(
        edges,
        key=lambda item: (item["final_score"], item["confidence"], item["amount"]),
        reverse=True,
    )[:limit]


def _pagerank(edges, damping=0.85, iterations=40):
    nodes = sorted({edge["source"] for edge in edges} | {edge["target"] for edge in edges})
    if not nodes:
        return []
    rank = {node: 1.0 / len(nodes) for node in nodes}
    outgoing = {node: [edge for edge in edges if edge["source"] == node] for node in nodes}
    for _ in range(iterations):
        next_rank = {node: (1 - damping) / len(nodes) for node in nodes}
        for node in nodes:
            edges_out = outgoing[node]
            if not edges_out:
                share = rank[node] / len(nodes)
                for target in nodes:
                    next_rank[target] += damping * share
                continue
            total = sum(edge["weight"] for edge in edges_out) or 1.0
            for edge in edges_out:
                next_rank[edge["target"]] += damping * rank[node] * edge["weight"] / total
        rank = next_rank
    return [
        {"sector_name": node, "pagerank": round(score, 6)}
        for node, score in sorted(rank.items(), key=lambda item: item[1], reverse=True)
    ]


def _rss(y, x):
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    return float(np.sum(residual ** 2))


def _granger_edges(history):
    pivot = _pivot_history(history)
    if pivot.shape[0] < 6 or pivot.shape[1] < 2:
        return []
    edges = []
    sectors = list(pivot.columns)
    for source in sectors:
        for target in sectors:
            if source == target:
                continue
            data = pivot[[source, target]].dropna()
            if len(data) < 6:
                continue
            x_lag = data[source].shift(1).iloc[1:].to_numpy(dtype=float)
            y_lag = data[target].shift(1).iloc[1:].to_numpy(dtype=float)
            y_now = data[target].iloc[1:].to_numpy(dtype=float)
            if np.std(x_lag) == 0 or np.std(y_now) == 0:
                continue
            restricted = np.column_stack([np.ones(len(y_now)), y_lag])
            full = np.column_stack([np.ones(len(y_now)), y_lag, x_lag])
            restricted_rss = _rss(y_now, restricted)
            full_rss = _rss(y_now, full)
            denominator = full_rss / max(1, len(y_now) - full.shape[1])
            f_score = math.inf if denominator == 0 and restricted_rss > full_rss else 0.0
            if denominator > 0:
                f_score = max(0.0, (restricted_rss - full_rss) / denominator)
            lag_corr = _corr(x_lag, y_now)
            if f_score >= 1.0 or abs(lag_corr) >= 0.4:
                edges.append({
                    "source": source,
                    "target": target,
                    "f_score": round(float(f_score if math.isfinite(f_score) else 9999.0), 4),
                    "lag_correlation": round(lag_corr, 4),
                    "lag_days": 1,
                })
    return sorted(edges, key=lambda item: (item["f_score"], abs(item["lag_correlation"])), reverse=True)[:20]


def _anomalies(snapshot, history):
    frame = normalize_sector_flow(history)
    if frame.empty:
        return []
    stats = frame.groupby("sector_name")["main_force_net"].agg(["mean", "std"]).reset_index()
    merged = snapshot.merge(stats, on="sector_name", how="left")
    merged["z_score"] = (merged["main_force_net"] - merged["mean"]) / merged["std"].replace(0, np.nan)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["z_score"])
    merged = merged[merged["z_score"].abs() >= 1.5]
    return [
        {
            **_record(row),
            "z_score": round(float(row["z_score"]), 3),
            "anomaly_type": "资金爆入" if row["z_score"] > 0 else "资金逃离",
        }
        for _, row in merged.reindex(merged["z_score"].abs().sort_values(ascending=False).index).head(10).iterrows()
    ]


# ==================== 资金关注度算法 ====================
# 目标：回答"钱现在往哪流、潜在在埋伏什么板块"。
# 不只看当日净流（噪音大），而是把三个时间维度做秩合成：
#   今日强度 38% + 5日持续 34% + 10日趋势 18% + 边际加速度 10%
# 输出 0-100 关注度：>50 资金在吸筹关注，<50 资金在派发撤离。


def _signed_streak(series):
    """从最新一天往回数连续同向天数，带符号（正=连续净流入，负=连续净流出）。"""
    streak = 0
    for value in reversed(list(series)):
        if value > 0:
            if streak >= 0:
                streak += 1
            else:
                break
        elif value < 0:
            if streak <= 0:
                streak -= 1
            else:
                break
        else:
            break
    return streak


def _multi_day_frame(latest, history):
    """整合单板块多周期净流：今日/5日/10日 + 连续同向天数 + 净流入强度 + 5日涨幅。

    5日/10日优先用快照自带的扩展列（akshare 多周期排行），
    缺失时从 Tushare 回补的历史透视表中按尾部窗口求和得到。
    净流入强度 = 净额 / 行业成交额（%），消除行业规模偏差（大市值行业天然净额大）。
    """
    pivot = _pivot_history(history, max_sectors=None)
    frame = latest[["sector_name", "main_force_net", "change_pct"]].copy()
    if "turnover" in latest.columns:
        frame["turnover"] = pd.to_numeric(latest["turnover"], errors="coerce").fillna(0.0)
    else:
        frame["turnover"] = 0.0
    frame = frame.drop_duplicates("sector_name").set_index("sector_name")

    for window, col in ((5, "net_5d"), (10, "net_10d")):
        if col in latest.columns and pd.to_numeric(latest[col], errors="coerce").notna().any():
            extra = pd.to_numeric(latest.set_index("sector_name")[col], errors="coerce")
            frame[col] = extra
        elif not pivot.empty and pivot.shape[0] >= 3:
            frame[col] = pivot.tail(window).sum()
        else:
            frame[col] = np.nan

    # 5 日累计成交额与 5 日累计涨幅（历史透视），用于窗口强度与量价背离
    turnover_pivot = _pivot_history_field(history, "turnover", max_sectors=None)
    change_pivot = _pivot_history_field(history, "change_pct", max_sectors=None)
    if not turnover_pivot.empty and turnover_pivot.shape[0] >= 3:
        frame["turnover_5d"] = turnover_pivot.tail(5).sum()
    else:
        frame["turnover_5d"] = frame["turnover"]
    if not change_pivot.empty and change_pivot.shape[0] >= 3:
        frame["chg_5d"] = change_pivot.tail(5).sum()
    else:
        frame["chg_5d"] = np.nan

    # 净流入强度（%）：主力净额占行业成交额比例；窗口强度 = 窗口净额 / 窗口成交额
    today_turnover = frame["turnover"].astype(float).replace(0, np.nan)
    frame["intensity"] = (frame["main_force_net"] / today_turnover * 100.0).round(2)
    turnover_5d = frame["turnover_5d"].astype(float).replace(0, np.nan)
    frame["intensity_5d"] = (frame["net_5d"] / turnover_5d * 100.0).round(2)

    if pivot.empty:
        frame["streak"] = np.where(frame["main_force_net"] > 0, 1, np.where(frame["main_force_net"] < 0, -1, 0))
    else:
        streaks = {sector: _signed_streak(pivot[sector]) for sector in pivot.columns}
        frame["streak"] = pd.Series(streaks)
        frame["streak"] = frame["streak"].fillna(0).astype(int)

    return frame.reset_index()


def _attention_ranking(latest, history, limit=14):
    """资金关注度评分：秩合成强度/绝对额（今日）、5日、10日净流与边际加速度。

    有成交额数据时今日项 = 0.5×净流入强度秩 + 0.5×绝对额秩，
    消除"大行业天然净额大"的规模偏差。
    """
    frame = _multi_day_frame(latest, history)
    if frame.empty:
        return []

    today = frame["main_force_net"].astype(float)
    net5 = pd.to_numeric(frame["net_5d"], errors="coerce").fillna(today)
    net10 = pd.to_numeric(frame["net_10d"], errors="coerce").fillna(net5)
    accel = today - net5 / 5.0
    intensity = pd.to_numeric(frame.get("intensity"), errors="coerce") if "intensity" in frame.columns else pd.Series(np.nan, index=frame.index)

    rank_today = today.rank(pct=True)
    if intensity.notna().sum() >= max(3, len(frame) // 3):
        rank_today = 0.5 * intensity.rank(pct=True) + 0.5 * rank_today
    rank_5 = net5.rank(pct=True)
    rank_10 = net10.rank(pct=True)
    rank_accel = accel.rank(pct=True)

    frame["attention"] = (
        0.38 * rank_today + 0.34 * rank_5 + 0.18 * rank_10 + 0.10 * rank_accel
    ) * 100.0
    frame["accel_yi"] = accel

    def _label(row):
        t, n5, n10, acc = row["main_force_net"], row["net_5d"], row["net_10d"], row["accel_yi"]
        chg = row.get("change_pct", 0.0) or 0.0
        chg5 = row.get("chg_5d")
        chg5 = chg5 if chg5 == chg5 else chg
        r10 = row.get("rank_10", 0.5)
        if t > 0 and n5 > 0 and acc > 0 and chg5 < 3.0:
            return "加速吸筹"
        if t > 0 and n5 > 0 and acc > 0:
            return "加速吸金"
        if t > 0 and n5 > 0 and chg5 < 2.0:
            return "流入吸筹"
        if t > 0 and n5 > 0:
            return "持续吸金"
        if t > 0 and n5 <= 0:
            return "边际转暖"
        if t <= 0 and n5 > 0 and (r10 >= 0.6 or chg5 >= 5.0):
            return "高位派发"
        if t < 0 and n5 <= 0:
            return "持续流出"
        return "多空拉锯"

    ranked = frame.assign(rank_10=rank_10).sort_values("attention", ascending=False).reset_index(drop=True)

    def _to_record(row):
        intensity_val = row.get("intensity")
        intensity_out = float(intensity_val) if intensity_val == intensity_val else None
        return {
            "sector_name": str(row["sector_name"]),
            "attention": int(round(float(row["attention"]))),
            "label": _label(row),
            "main_force_net_yi": amount_to_yi(row["main_force_net"]),
            "net_5d_yi": amount_to_yi(row["net_5d"]) if row["net_5d"] == row["net_5d"] else None,
            "net_10d_yi": amount_to_yi(row["net_10d"]) if row["net_10d"] == row["net_10d"] else None,
            "accel_yi": amount_to_yi(row["accel_yi"]),
            "intensity_pct": intensity_out,
            "change_pct": float(row.get("change_pct", 0.0) or 0.0),
            "streak": int(row.get("streak", 0)),
        }

    records = [_to_record(row) for _, row in ranked.head(limit).iterrows()]
    # 派发端同样值得关注：补上关注度最低的几行
    tail = ranked.tail(3).iloc[::-1] if len(ranked) > limit else ranked.tail(0)
    seen = {r["sector_name"] for r in records}
    for _, row in tail.iterrows():
        if str(row["sector_name"]) in seen:
            continue
        records.append(_to_record(row))
    return records


def _forecast_ranking(latest, history, rotation_edges, limit=8):
    """未来 1-3 日资金倾向预测榜。

    预测分（带方向）= 趋势延续(动量×连续天数) + 吸筹信号(钱进价滞涨) + 轮动带动(被验证路线指向)。
    输出净流入倾向 Top 与净流出倾向 Top，各带可读理由；样本不足时返回空。
    """
    frame = _multi_day_frame(latest, history)
    if frame.empty or len(frame) < 6:
        return []

    driven = {}
    for edge in rotation_edges or []:
        target = edge.get("target")
        confidence = float(edge.get("confidence") or edge.get("weight") or 0.0)
        source = edge.get("source")
        if target and confidence > driven.get(target, (0, None))[0]:
            driven[target] = (confidence, source)

    net5 = pd.to_numeric(frame["net_5d"], errors="coerce").fillna(0.0)
    momentum = (net5.rank(pct=True) - 0.5) * 2.0  # -1 ~ 1
    streak = frame["streak"].astype(float)
    streak_score = (streak.clip(-5, 5) / 5.0)
    intensity5 = pd.to_numeric(frame.get("intensity_5d"), errors="coerce").fillna(0.0) if "intensity_5d" in frame.columns else pd.Series(0.0, index=frame.index)

    scores = []
    for idx, row in frame.iterrows():
        t, n5 = row["main_force_net"], row["net_5d"]
        chg5 = row.get("chg_5d")
        chg5 = chg5 if chg5 == chg5 else (row.get("change_pct") or 0.0)
        reasons = []
        accumulate = 0.0
        if t > 0 and n5 > 0 and chg5 < 2.0:
            accumulate = 1.0
            reasons.append(f"钱进价滞涨(5日{chg5:+.1f}%)，疑似吸筹")
        drive_conf, drive_src = driven.get(row["sector_name"], (0.0, None))
        drive_score = max(-1.0, min(1.0, drive_conf * 2.0))
        if drive_src and abs(drive_score) > 0.2:
            reasons.append(f"轮动路线承接（{drive_src} 撤出→本板块，置信 {drive_conf:.2f}）")
        if abs(streak[idx]) >= 3:
            reasons.append(f"连续{abs(int(streak[idx]))}日{'净流入' if streak[idx] > 0 else '净流出'}")
        if abs(intensity5[idx]) >= 3.0:
            reasons.append(f"5日净流入强度 {intensity5[idx]:+.1f}%")

        raw = 0.35 * float(momentum[idx]) + 0.25 * float(streak_score[idx]) + 0.20 * accumulate + 0.20 * drive_score
        direction = "inflow" if raw > 0 else "outflow"
        scores.append({
            "sector_name": str(row["sector_name"]),
            "score": round(raw * 100.0, 1),
            "direction": direction,
            "reasons": reasons or ["资金趋势延续"],
            "main_force_net_yi": amount_to_yi(t),
            "net_5d_yi": amount_to_yi(n5) if n5 == n5 else None,
        })

    inflow_top = sorted([s for s in scores if s["direction"] == "inflow"], key=lambda x: -x["score"])[:limit]
    outflow_top = sorted([s for s in scores if s["direction"] == "outflow"], key=lambda x: x["score"])[: max(3, limit // 2)]
    return {
        "horizon_days": 3,
        "inflow": inflow_top,
        "outflow": outflow_top,
        "note": "基于趋势延续/吸筹/轮动带动的统计倾向，非投资建议",
    }


def _market_read(latest, attention, summary):
    """一句话市场解读：主线在哪、撤离方向在哪、风格如何。"""
    if not attention:
        return {"headline": "等待资金流数据积累", "focus": None, "risk": None, "breadth_hint": None}

    top = attention[0]
    focus_text = (
        f"{top['sector_name']}（关注度 {top['attention']} · 今日 {top['main_force_net_yi']:+.1f}亿"
        + (f" · 5日 {top['net_5d_yi']:+.1f}亿" if top.get("net_5d_yi") is not None else "")
        + "）"
    )

    risk = next((item for item in reversed(attention) if item.get("net_5d_yi") is not None and item["net_5d_yi"] < 0), attention[-1])
    risk_text = (
        f"{risk['sector_name']}（5日 {risk['net_5d_yi']:+.1f}亿 · {risk['label']}）"
        if risk.get("net_5d_yi") is not None else f"{risk['sector_name']}（{risk['label']}）"
    )

    sector_count = int(summary.get("sector_count", 0) or 0)
    positive = int(summary.get("positive_count", 0) or 0)
    if sector_count:
        breadth = positive / sector_count
        breadth_hint = "普涨式吸金，增量进场" if breadth >= 0.55 else ("缩圈聚焦，主线集中" if breadth >= 0.45 else "防御为主，避险情绪重")
    else:
        breadth_hint = None

    headline = f"主线聚焦 {focus_text}"
    return {"headline": headline, "focus": focus_text, "risk": risk_text, "breadth_hint": breadth_hint}


def build_fund_flow_payload(snapshot, history=None, source="realtime", error=None, updated_at=None):
    latest = normalize_sector_flow(snapshot)
    history = normalize_sector_flow(history if history is not None else latest)
    if latest.empty and not history.empty:
        latest_date = history["date"].max()
        latest = history[history["date"] == latest_date].copy()

    if latest.empty:
        return {
            "source": source,
            "error": error,
            "updated_at": updated_at,
            "date": None,
            "summary": {"sector_count": 0, "total_inflow": 0.0, "total_outflow": 0.0, "net_flow": 0.0},
            "top_inflow": [],
            "top_outflow": [],
            "anomalies": [],
            "sankey_edges": [],
            "rotation_edges": [],
            "rotation_forecast": {"status": "insufficient_data", "horizon_days": 3, "strongest_route": None, "edge_count": 0},
            "network_edges": [],
            "pagerank_nodes": [],
            "granger_edges": [],
            "attention_ranking": [],
            "forecast": {"horizon_days": 3, "inflow": [], "outflow": [], "note": "等待资金流数据积累"},
            "market_read": {"headline": "等待资金流数据积累", "focus": None, "risk": None, "breadth_hint": None},
        }

    latest = latest.sort_values("main_force_net", ascending=False).reset_index(drop=True)
    network_edges = _network_edges(history)
    rotation_edges = _rotation_edges(history)
    sankey_edges = rotation_edges or _sankey_edges(latest)
    summary = _summary(latest)
    attention = _attention_ranking(latest, history)
    forecast = _forecast_ranking(latest, history, rotation_edges)
    market_read = _market_read(latest, attention, summary)
    return {
        "source": source,
        "error": error,
        "updated_at": updated_at,
        "date": str(latest["date"].max()),
        "summary": summary,
        "top_inflow": [_record(row) for _, row in latest.head(10).iterrows()],
        "top_outflow": [_record(row) for _, row in latest.sort_values("main_force_net").head(10).iterrows()],
        "anomalies": _anomalies(latest, history),
        "sankey_edges": sankey_edges,
        "rotation_edges": rotation_edges,
        "rotation_forecast": {
            "status": "predictive" if rotation_edges else "insufficient_data",
            "horizon_days": 3,
            "strongest_route": rotation_edges[0] if rotation_edges else None,
            "edge_count": len(rotation_edges),
        },
        "network_edges": network_edges,
        "pagerank_nodes": _pagerank(rotation_edges or network_edges),
        "granger_edges": _granger_edges(history),
        "attention_ranking": attention,
        "forecast": forecast,
        "market_read": market_read,
    }
