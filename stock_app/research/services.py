"""研报模块写侧服务：爬虫入库 / AI 摘要 / 每日推荐 / 晨报 / 盈利预测 / 选股跟踪。

移植自原项目的 crawler_service / summary_service / recommendation_service /
brief_service / tracking_service，SQLAlchemy Session 换 Django ORM，
asyncio.gather 换 ThreadPoolExecutor。合并了原 scheduler 与 admin 两份不一致的流水线。
"""
from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Dict, List

from django.db import transaction
from django.db.models import Q

from ..models import (
    DailyBrief,
    DailyRecommendation,
    EarningsForecast,
    ResearchPreference,
    ResearchReport,
    ResearchSource,
    ResearchSummary,
    StockPick,
)
from .ai_service import AIService
from .crawlers import CRAWLER_REGISTRY
from .forecast_extractor import extract_eps_forecasts, extract_profit_forecasts
from .market_utils import (
    extract_target_price,
    get_latest_price,
    get_price_on_or_after,
    institution_weight,
    normalize_industry_set,
)

_CODE_RE = re.compile(r"^\d{6}$")


def _resolve_stock_code(report: ResearchReport) -> tuple:
    """研报主标的代码：优先报表字段，否则唯一的 6 位 target_stock。"""
    if report.stock_code:
        return report.stock_code, report.stock_name
    summary = report.summary_or_none
    if summary:
        codes = [c for c in (summary.target_stocks or []) if c and isinstance(c, str) and _CODE_RE.match(c)]
        if len(codes) == 1:
            return codes[0], report.stock_name
    return None, None


def crawl_all(progress=None, detail_budget: int = 20) -> Dict[str, dict]:
    """抓取全部注册源并入库（按 source+external_id 唯一约束去重）。

    detail_budget：每源最多补抓 N 条详情正文（PDF 解析走子进程较慢），
    其余入库为 pending，AI 摘要阶段用标题兜底；连续 3 次详情为空则熔断。
    """
    results: Dict[str, dict] = {}
    for code, crawler_cls in CRAWLER_REGISTRY.items():
        if progress:
            progress(f"抓取 {code}")
        try:
            crawler = crawler_cls()
            source, _ = ResearchSource.objects.get_or_create(
                code=code,
                defaults={"name": code, "base_url": crawler.base_url, "status": "active", "weight": 1.0},
            )
            raw_reports = crawler.fetch_list(page=1)
            saved = 0
            existing_ids = set(
                ResearchReport.objects.filter(source=source).values_list("external_id", flat=True)
            )
            existing_titles = set(
                ResearchReport.objects.filter(
                    source=source, report_date__in={r["report_date"] for r in raw_reports if r.get("report_date")}
                ).values_list("title", "report_date")
            )
            new_rows = []
            detail_used = 0
            empty_streak = 0
            for raw in raw_reports:
                if raw["external_id"] in existing_ids:
                    continue
                if (raw["title"], raw["report_date"]) in existing_titles:
                    continue
                if not raw.get("title"):
                    continue
                content_text = raw.get("content_text") or ""
                pdf_url = raw.get("pdf_url")
                if not content_text and detail_used < detail_budget and empty_streak < 3:
                    detail_used += 1
                    try:
                        detail = crawler.fetch_detail(raw["source_url"])
                        detail_text = detail.get("content_text") or ""
                        pdf_url = pdf_url or detail.get("pdf_url")
                        if detail_text:
                            content_text = detail_text
                            empty_streak = 0
                        else:
                            empty_streak += 1
                    except Exception:
                        empty_streak += 1
                new_rows.append(ResearchReport(
                    source=source,
                    external_id=raw["external_id"],
                    title=raw["title"][:500],
                    source_url=raw["source_url"][:500],
                    stock_code=(raw.get("stock_code") or "")[:16],
                    stock_name=(raw.get("stock_name") or "")[:64],
                    industry=(raw.get("industry") or "")[:64],
                    author=(raw.get("author") or "")[:128],
                    institution=(raw.get("institution") or "")[:128],
                    report_date=raw["report_date"],
                    pdf_url=(pdf_url or "")[:500],
                    content_text=content_text,
                    status=ResearchReport.STATUS_PENDING,
                ))
                saved += 1
            if new_rows:
                ResearchReport.objects.bulk_create(new_rows, ignore_conflicts=True)
            source.last_crawled_at = datetime.now()
            source.save(update_fields=["last_crawled_at"])
            results[code] = {"status": "success", "count": saved, "detail_fetched": detail_used}
        except Exception as exc:
            results[code] = {"status": "error", "message": str(exc)[:200]}
    return results


def summarize_pending(limit: int = 50, progress=None) -> dict:
    """并发（5 线程）对 pending 研报生成 AI 摘要。"""
    ai = AIService()
    pending = list(
        ResearchReport.objects.filter(status=ResearchReport.STATUS_PENDING)
        .order_by("-report_date")
        .values_list("id", "title", "content_text")[:limit]
    )
    if not pending:
        return {"processed": 0, "failed": 0, "skipped_no_key": 0 if ai.available else len(pending)}

    def _one(row):
        rid, title, content = row
        try:
            return rid, ai.summarize_report(title, content or title)
        except Exception:
            return rid, None

    results = {"processed": 0, "failed": 0, "skipped_no_key": 0}
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_one, row) for row in pending]
        done = 0
        for future in as_completed(futures):
            rid, parsed = future.result()
            done += 1
            if progress:
                progress(f"AI 摘要 {done}/{len(pending)}")
            report = ResearchReport.objects.filter(pk=rid).first()
            if not report:
                continue
            if not parsed:
                # AI 不可用/失败：保留 pending 下轮重试，不算 failed
                results["failed"] += 1
                continue
            try:
                fields = ai.parse_result(parsed)
                fields["target_price"] = fields.get("target_price") or extract_target_price(
                    (report.content_text or "")[:5000]
                ) or ""
                ResearchSummary.objects.update_or_create(report=report, defaults=fields)
                report.status = ResearchReport.STATUS_SUMMARIZED
                report.save(update_fields=["status", "updated_at"])
                results["processed"] += 1
            except Exception:
                results["failed"] += 1
    return results


def generate_daily_recommendations(recommend_date: date = None, limit: int = None) -> dict:
    """按 情绪/评级/机构权重/偏好/新鲜度 评分生成每日推荐。"""
    recommend_date = recommend_date or date.today()
    preference = ResearchPreference.get_solo()
    limit = limit or preference.daily_count or 20
    watch_stocks = set(preference.watch_stocks or [])
    watch_industries = normalize_industry_set(preference.watch_industries or [])

    reports = list(
        ResearchReport.objects.filter(
            status=ResearchReport.STATUS_SUMMARIZED,
            report_date__gte=recommend_date - timedelta(days=7),
        ).select_related("summary", "source")
    )

    scored = []
    for report in reports:
        score, reason = _calculate_score(report, watch_stocks, watch_industries)
        scored.append((report, score, reason))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:limit]

    with transaction.atomic():
        DailyRecommendation.objects.filter(recommend_date=recommend_date).delete()
        DailyRecommendation.objects.bulk_create([
            DailyRecommendation(
                report=item[0], recommend_date=recommend_date,
                score=item[1], rank=rank, reason=item[2],
            )
            for rank, item in enumerate(top, start=1)
        ])
    return {"date": recommend_date.isoformat(), "count": len(top)}


def _calculate_score(report: ResearchReport, watch_stocks: set, watch_industries: set) -> tuple:
    score = 0.0
    reasons = []
    summary = report.summary_or_none

    stock_score = 0.0
    if report.stock_code and report.stock_code in watch_stocks:
        stock_score = 1.0
        reasons.append("与你自选股匹配")
    elif summary and any(s in watch_stocks for s in (summary.target_stocks or [])):
        stock_score = 0.8
        reasons.append("涉及你关注的股票")

    industry_score = 0.0
    if summary:
        tags = normalize_industry_set(summary.tags or [])
        matched = [t for t in tags if t in watch_industries]
        if matched:
            industry_score = min(1.0, len(matched) / 3.0)
            reasons.append(f"与{matched[0]}行业相关")

    days_diff = (date.today() - report.report_date).days
    freshness = 1.0 if days_diff == 0 else 0.6 if days_diff <= 2 else 0.2
    source_weight = report.source.weight if report.source else 1.0
    inst_weight = institution_weight(report.institution)
    score = (stock_score * 0.4 + industry_score * 0.3 + freshness * 0.2 + source_weight * 0.1) * inst_weight
    return round(score, 4), "；".join(reasons) or "新发布研报"


def generate_daily_brief(brief_date: date = None, progress=None) -> dict:
    """汇总近 3 天已摘要研报，AI 生成每日晨报。"""
    brief_date = brief_date or date.today()
    ai = AIService()
    if not ai.available:
        return {"status": "skipped", "message": "未配置 AI Key（DEEPSEEK_API_KEY）"}

    reports = list(
        ResearchReport.objects.filter(
            status=ResearchReport.STATUS_SUMMARIZED,
            report_date__gte=brief_date - timedelta(days=3),
        ).select_related("summary").order_by("-report_date")[:400]
    )
    if not reports:
        return {"status": "empty", "message": "没有可聚合的研报"}

    def _quality_key(r):
        return (institution_weight(r.institution), min(len(r.content_text or ""), 5000))

    reports = sorted(reports, key=_quality_key, reverse=True)[:100]

    digest_lines = []
    sentiment_counter, tag_counter, stock_counter = Counter(), Counter(), Counter()
    for r in reports:
        s = r.summary_or_none
        if not s:
            continue
        sentiment_counter[s.sentiment or "neutral"] += 1
        tag_counter.update(s.tags or [])
        for code in s.target_stocks or []:
            if code and isinstance(code, str) and _CODE_RE.match(code):
                stock_counter[code] += 1
        if r.stock_code and r.stock_name:
            stock_counter[f"{r.stock_name}({r.stock_code})"] += 1
        digest_lines.append(
            f"《{r.title}》({r.institution or '未知机构'},评级={s.rating or '中性'})\n"
            f"摘要:{s.summary or ''}\n"
            f"要点:{' / '.join((s.investment_points or [])[:3])}"
        )

    hot_stocks = "、".join(s for s, _ in stock_counter.most_common(20)) or "无"
    digest = "\n\n".join(digest_lines)[:60000]
    report_count = len(digest_lines)
    prompt = f"""你是一名资深 A 股首席策略分析师。以下是近期 {report_count} 篇券商研报的 AI 摘要压缩包，以及高频标签和高频个股统计，请你站在全局视角，输出一份「今日投资简报」JSON：

{{
  "overview": "80 字以内，一句话概括当前市场核心矛盾与主线",
  "advice": "150 字以内，给出 2-3 条具体可操作的今日投资建议，语气专业简洁",
  "short_term": "短期(1-2周)建议：60 字以内，聚焦事件催化、资金轮动、交易性机会",
  "mid_term": "中期(1-3个月)建议：60 字以内，聚焦业绩兑现、景气度延续的板块",
  "long_term": "长期(6个月以上)建议：60 字以内，聚焦产业趋势、确定性成长的配置方向",
  "industries": ["推荐关注的行业1", "行业2", "行业3", "行业4", "行业5"],
  "stocks": [
    {{"name": "股票名称", "code": "600519", "reason": "15 字以内推荐逻辑"}},
    {{"name": "股票名称", "code": "000001", "reason": "15 字以内推荐逻辑"}},
    {{"name": "股票名称", "code": "300750", "reason": "15 字以内推荐逻辑"}},
    {{"name": "股票名称", "code": "002594", "reason": "15 字以内推荐逻辑"}},
    {{"name": "股票名称", "code": "688981", "reason": "15 字以内推荐逻辑"}},
    {{"name": "股票名称", "code": "601012", "reason": "15 字以内推荐逻辑"}},
    {{"name": "股票名称", "code": "300274", "reason": "15 字以内推荐逻辑"}},
    {{"name": "股票名称", "code": "600036", "reason": "15 字以内推荐逻辑"}}
  ]
}}

要求：
- overview 只讲最重要的市场判断，不说废话
- advice 必须具体到方向/板块，避免空话
- short_term/mid_term/long_term 三档建议要区分时间维度，短期重交易、中期重业绩、长期重产业趋势
- industries 从研报高频标签中提炼，按热度排序，不超过 5 个
- stocks 推荐 8-10 只，优先从以下研报高频提及个股中挑选：{hot_stocks}；如候选不足再从研报内容补充，每只需有具体推荐逻辑，code 无法确定时填空字符串
- 输出必须是合法 JSON

研报摘要集合：
{digest}"""

    if progress:
        progress("AI 生成每日晨报")
    result = ai.ask_json(prompt)
    if not result:
        return {"status": "error", "message": "AI 生成失败"}

    sentiment_stats = {
        "bullish": sentiment_counter.get("bullish", 0),
        "neutral": sentiment_counter.get("neutral", 0),
        "bearish": sentiment_counter.get("bearish", 0),
    }
    DailyBrief.objects.update_or_create(
        brief_date=brief_date,
        defaults={
            "overview": result.get("overview", ""),
            "advice": result.get("advice", ""),
            "short_term": result.get("short_term", ""),
            "mid_term": result.get("mid_term", ""),
            "long_term": result.get("long_term", ""),
            "industries": ai._as_list(result.get("industries")),
            "stocks": [s for s in (result.get("stocks") or []) if isinstance(s, dict)],
            "sentiment_stats": sentiment_stats,
            "ai_model": ai.deepseek_model,
        },
    )
    return {"status": "success", "date": brief_date.isoformat()}


def extract_forecasts_from_reports(days: int = 7) -> int:
    """从已摘要研报正文提取盈利预测入库（原项目缺失的写入环节）。"""
    start = date.today() - timedelta(days=days)
    reports = ResearchReport.objects.filter(
        report_date__gte=start, forecasts__isnull=True
    ).distinct()
    saved = 0
    for report in reports.only("id", "stock_code", "stock_name", "content_text", "title"):
        code, name = _resolve_stock_code(report)
        if not code:
            continue
        text = report.content_text or report.title or ""
        rows = []
        profits = extract_profit_forecasts(text)
        eps_list = extract_eps_forecasts(text)
        eps_by_year = {e["year"]: e["eps"] for e in eps_list}
        for p in profits:
            rows.append(EarningsForecast(
                report=report, stock_code=code, stock_name=(name or "")[:64],
                forecast_year=p["year"], profit_value=p["profit"],
                profit_text=(p.get("text") or "")[:128],
                eps_value=eps_by_year.get(p["year"]),
            ))
        if not rows and eps_list:
            for e in eps_list:
                rows.append(EarningsForecast(
                    report=report, stock_code=code, stock_name=(name or "")[:64],
                    forecast_year=e["year"], profit_value=None, profit_text="",
                    eps_value=e["eps"],
                ))
        if rows:
            EarningsForecast.objects.bulk_create(rows[:5])
            saved += len(rows[:5])
    return saved


def save_picks_from_brief(pick_date: date = None) -> int:
    """从晨报 stocks 列表记录选股买点。"""
    pick_date = pick_date or date.today()
    brief = DailyBrief.objects.filter(brief_date=pick_date).first()
    if not brief:
        brief = DailyBrief.objects.order_by("-brief_date").first()
        if not brief:
            return 0
        pick_date = brief.brief_date

    saved = 0
    for s in brief.stocks or []:
        code = (s.get("code") or "").strip()
        name = (s.get("name") or "").strip()
        if not code or not _CODE_RE.match(code):
            continue
        _, created = StockPick.objects.get_or_create(
            pick_date=pick_date, stock_code=code, source="daily_brief",
            defaults={"stock_name": name[:64], "price_at_pick": get_latest_price(code)},
        )
        saved += 1 if created else 0
    return saved


def update_pick_returns() -> int:
    """回填 T+1/T+5/T+20/T+60/T+120 收益（Tushare 日线）。"""
    picks = StockPick.objects.filter(price_at_pick__isnull=False)
    updated = 0
    today = date.today()
    for p in picks:
        changed = False
        days_passed = (today - p.pick_date).days
        for key, trading_days, min_days in (
            ("ret_t1", 1, 1), ("ret_t5", 5, 5), ("ret_t20", 20, 20),
            ("ret_t60", 60, 60), ("ret_t120", 120, 120),
        ):
            if getattr(p, key) is None and days_passed >= min_days:
                price = get_price_on_or_after(p.stock_code, p.pick_date, trading_days)
                if price and p.price_at_pick:
                    setattr(p, key, round((price - p.price_at_pick) / p.price_at_pick * 100, 2))
                    changed = True
        if changed:
            p.save(update_fields=["ret_t1", "ret_t5", "ret_t20", "ret_t60", "ret_t120"])
            updated += 1
    return updated
