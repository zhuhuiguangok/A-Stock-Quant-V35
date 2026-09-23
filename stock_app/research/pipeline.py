"""统一研报流水线：抓取 → AI 摘要 → 推荐 → 晨报 → 盈利预测 → 事件 → 跟踪回填。

合并自原项目 scheduler.daily_pipeline 与 admin._refresh_pipeline（原两份不一致，
admin 版缺 expectation 步骤；本版本补齐并增加盈利预测提取写入）。
"""
from __future__ import annotations

import os
from datetime import date


def run_daily_pipeline(progress=None, summarize_limit: int = None) -> dict:
    # 每轮 AI 摘要上限：env RESEARCH_SUMMARIZE_LIMIT 可调，默认 200
    if summarize_limit is None:
        summarize_limit = int(os.environ.get("RESEARCH_SUMMARIZE_LIMIT", "200"))
    def _say(msg):
        if progress:
            progress(msg)

    result = {"date": date.today().isoformat()}
    from . import services

    _say("抓取研报数据源")
    result["crawl"] = services.crawl_all(progress=progress)

    _say("AI 摘要 pending 研报")
    result["summary"] = services.summarize_pending(limit=summarize_limit, progress=progress)

    _say("生成每日推荐")
    result["recommendation"] = services.generate_daily_recommendations()

    _say("生成每日晨报")
    result["brief"] = services.generate_daily_brief(progress=progress)

    _say("提取盈利预测")
    result["forecasts"] = services.extract_forecasts_from_reports(days=7)

    _say("提取未来事件")
    from . import signals

    result["events"] = signals.extract_events_from_reports(days=7)
    _say("刷新财报披露日历")
    try:
        result["calendar_events"] = signals.refresh_calendar_events(ahead_days=45)
    except Exception as exc:  # 日历失败不阻断流水线
        result["calendar_events"] = f"failed: {exc}"

    _say("记录晨报选股买点")
    result["picks"] = services.save_picks_from_brief()

    _say("回填 T+1/T+5/T+20 收益")
    result["returns"] = services.update_pick_returns()
    return result
