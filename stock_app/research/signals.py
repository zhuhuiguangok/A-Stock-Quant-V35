"""研报模块读侧信号：机构共识 / 未来事件 / 盈利预期 / 选股跟踪统计。

移植自原项目 consensus_service / event_service(读) / expectation_service / tracking_service(读)。
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, timedelta

from django.db.models import Q

from ..models import (
    DailyBrief,
    DailyRecommendation,
    EarningsForecast,
    MarketEvent,
    ResearchReport,
    ResearchSource,
    ResearchSummary,
    StockPick,
)
from .market_utils import (
    fetch_names_batch,
    get_latest_price,
    get_stock_change,
    institution_weight,
    normalize_industry_set,
)

_CODE_RE = re.compile(r"^\d{6}$")


_NAME_MAP_CACHE: dict = {"date": None, "map": None}


def _stock_name_map() -> dict:
    """code→name 映射：研报表字段 + 标题正则 + 腾讯行情补缺。按日缓存（表大后全表扫描+网络补缺很贵）。"""
    from datetime import date as _date

    today = _date.today()
    if _NAME_MAP_CACHE["date"] == today and _NAME_MAP_CACHE["map"]:
        return _NAME_MAP_CACHE["map"]
    mapping = {}
    reports = ResearchReport.objects.exclude(stock_code="").exclude(stock_name="").values(
        "stock_code", "stock_name", "title"
    )
    for r in reports:
        if r["stock_code"] and r["stock_name"]:
            mapping[r["stock_code"]] = r["stock_name"]
        title = r["title"] or ""
        m = re.search(r"[-—]([一-龥A-Za-z]{2,8})[-—](\d{6})[-—]", title)
        if m and m.group(2) not in mapping:
            mapping[m.group(2)] = m.group(1)
        m2 = re.search(r"([一-龥A-Za-z]{2,8})[（(](\d{6})[)）]", title)
        if m2 and m2.group(2) not in mapping:
            mapping[m2.group(2)] = m2.group(1)
    _NAME_MAP_CACHE.update(date=today, map=mapping)
    return mapping


def get_stock_consensus(days: int = 7, limit: int = 15) -> list:
    """近 N 天机构对个股的看多共识度排行。"""
    start = date.today() - timedelta(days=days)
    reports = list(
        ResearchReport.objects.filter(
            status=ResearchReport.STATUS_SUMMARIZED, report_date__gte=start,
        ).select_related("summary")
    )

    stocks = defaultdict(lambda: {
        "stock_code": None, "stock_name": None, "report_count": 0, "buy_count": 0,
        "bullish_count": 0, "upgrade_count": 0, "first_cover_count": 0,
        "institutions": set(), "weighted_score": 0.0, "target_prices": [],
        "latest_date": None, "tags": [],
    })
    for r in reports:
        s = r.summary_or_none
        if not s:
            continue
        codes = set()
        if r.stock_code:
            codes.add((r.stock_code, r.stock_name))
        for code in s.target_stocks or []:
            if code and isinstance(code, str) and _CODE_RE.match(code):
                codes.add((code, None))
        if not codes:
            continue

        rating = s.rating or ""
        sentiment = s.sentiment or "neutral"
        is_buy = ("买入" in rating) or ("增持" in rating) or sentiment == "bullish"
        is_upgrade = "上调" in rating or "上调" in (s.summary or "")[:200]
        is_first = "首次覆盖" in (r.title or "") or "首次" in rating
        weight = institution_weight(r.institution)

        for code, name in codes:
            agg = stocks[code]
            agg["stock_code"] = code
            if name and not agg["stock_name"]:
                agg["stock_name"] = name
            agg["report_count"] += 1
            if is_buy:
                agg["buy_count"] += 1
            if sentiment == "bullish":
                agg["bullish_count"] += 1
            if is_upgrade:
                agg["upgrade_count"] += 1
            if is_first:
                agg["first_cover_count"] += 1
            if r.institution:
                agg["institutions"].add(r.institution)
            agg["weighted_score"] += weight * (2.0 if is_buy else 1.0) + (1.0 if is_upgrade else 0) + (0.8 if is_first else 0)
            if s.target_price:
                agg["target_prices"].append(s.target_price)
            if r.report_date and (agg["latest_date"] is None or r.report_date > agg["latest_date"]):
                agg["latest_date"] = r.report_date
            agg["tags"].extend(s.tags or [])

    name_map = _stock_name_map()
    missing = [c for c in stocks if c not in name_map]
    if missing:
        name_map.update(fetch_names_batch(missing[:40]))

    result = []
    for code, agg in stocks.items():
        tags = list(normalize_industry_set(agg["tags"]))
        result.append({
            "stock_code": agg["stock_code"],
            "stock_name": agg["stock_name"] or name_map.get(agg["stock_code"]) or agg["stock_code"],
            "report_count": agg["report_count"],
            "buy_count": agg["buy_count"],
            "bullish_count": agg["bullish_count"],
            "upgrade_count": agg["upgrade_count"],
            "first_cover_count": agg["first_cover_count"],
            "institution_count": len(agg["institutions"]),
            "institutions": sorted(agg["institutions"])[:5],
            "score": round(agg["weighted_score"], 1),
            "target_price": agg["target_prices"][0] if agg["target_prices"] else None,
            "latest_date": agg["latest_date"].isoformat() if agg["latest_date"] else None,
            "industries": tags[:3],
        })
    result.sort(key=lambda x: (x["score"], x["report_count"]), reverse=True)
    top = result[:limit]

    for it in top:
        chg = get_stock_change(it["stock_code"], 20)
        it["price_change_20d"] = chg
        buy_ratio = it["buy_count"] / it["report_count"] if it["report_count"] else 0
        it["divergence"] = bool(chg is not None and chg <= -5 and buy_ratio >= 0.6 and it["report_count"] >= 3)
    return top


def get_upcoming_events(days_ahead: int = 30, limit: int = 30) -> list:
    """未来事件（研报正文提取 + 业绩预告类）。"""
    today = date.today()
    events = MarketEvent.objects.filter(
        event_date__gte=today - timedelta(days=7),
        event_date__lte=today + timedelta(days=days_ahead),
    ).order_by("-event_date")
    seen, result = set(), []
    for e in events:
        key = (e.stock_code, e.event_type)
        if key in seen:
            continue
        seen.add(key)
        result.append({
            "event_date": e.event_date.isoformat(),
            "stock_code": e.stock_code,
            "stock_name": e.stock_name,
            "event_type": e.event_type,
            "title": e.title,
            "detail": e.detail,
        })
        if len(result) >= limit:
            break
    return result


def get_watch_pool_codes(days: int = 90) -> set:
    """关注池：近 N 天研报涉及个股 + 晨报推荐 + 用户自选。"""
    from ..models import ResearchPreference

    codes = set()
    start = date.today() - timedelta(days=days)
    for row in ResearchReport.objects.filter(report_date__gte=start).values_list("stock_code", "summary__target_stocks"):
        if row[0]:
            codes.add(row[0])
        for c in row[1] or []:
            if c and isinstance(c, str) and _CODE_RE.match(c):
                codes.add(c)
    for b in DailyBrief.objects.order_by("-brief_date").values_list("stocks", flat=True)[:5]:
        for s in b or []:
            code = str((s or {}).get("code") or "").strip()
            if _CODE_RE.match(code):
                codes.add(code)
    for c in ResearchPreference.get_solo().watch_stocks or []:
        if _CODE_RE.match(str(c)):
            codes.add(str(c))
    return codes


def refresh_calendar_events(ahead_days: int = 45) -> int:
    """用 Tushare 披露日历填充未来事件（财报披露日）。

    私有代理不支持 date/end_date 过滤参数，因此按关注池 ts_code 分批查询，
    本地过滤最近的报告期和未来预约披露日。
    """
    from .market_utils import to_ts_code
    from ..tushare_client import create_tushare_pro

    codes = get_watch_pool_codes()
    if not codes:
        return 0
    ts_codes = sorted({to_ts_code(c) for c in codes})
    today = date.today()
    deadline = today + timedelta(days=ahead_days)
    # 最近两个已结束的标准报告期（0331/0630/0930/1231）
    recent_periods = set()
    for back in range(1, 400):
        d = today - timedelta(days=back)
        if d.month in (3, 6, 9, 12) and d.day in (31, 30) and ((d.month, d.day) in [(3, 31), (6, 30), (9, 30), (12, 31)]):
            recent_periods.add(d.strftime("%Y%m%d"))
            if len(recent_periods) >= 2:
                break

    rows = []
    batch = 80
    pro = create_tushare_pro(timeout=30)
    for i in range(0, len(ts_codes), batch):
        part = ts_codes[i:i + batch]
        try:
            df = pro.disclosure_date(ts_code=",".join(part))
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df = df[df["end_date"].isin(recent_periods)]
        for _, r in df.iterrows():
            pre = str(r.get("pre_date") or "")[:8]
            if not pre or not pre.isdigit():
                continue
            try:
                pre_date = date(int(pre[:4]), int(pre[4:6]), int(pre[6:8]))
            except ValueError:
                continue
            if not (today <= pre_date <= deadline):
                continue
            rows.append((pre_date, r["ts_code"], str(r.get("end_date") or "")))

    saved = 0
    from .market_utils import fetch_names_batch

    name_map = fetch_names_batch([c.split(".")[0] for c in ts_codes])
    for pre_date, ts_code, period in rows:
        code = ts_code.split(".")[0]
        period_label = {"0331": "一季报", "0630": "中报", "0930": "三季报", "1231": "年报"}.get(period[4:], "财报")
        stock_name = (name_map.get(code) or "")[:64]
        _, created = MarketEvent.objects.get_or_create(
            event_date=pre_date,
            stock_code=code,
            title=f"{period_label}预约披露",
            defaults={
                "event_type": "财报披露",
                "stock_name": stock_name,
                "detail": f"报告期 {period}，预约披露日 {pre_date.isoformat()}",
            },
        )
        saved += 1 if created else 0
    return saved


def extract_events_from_reports(days: int = 7) -> int:
    """从研报正文/摘要正则提取未来事件（股东大会/解禁/财报）。

    原项目事件全靠 akshare 业绩预告（已限流），这里改为研报文本提取为主，
    导入的历史业绩预告事件仍保留在库里。
    """
    start = date.today() - timedelta(days=days)
    reports = ResearchReport.objects.filter(
        status=ResearchReport.STATUS_SUMMARIZED, report_date__gte=start,
    ).select_related("summary")

    patterns = [
        (r"股东大会[于在]?(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})", "股东大会"),
        (r"解禁[于在]?(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})", "解禁"),
        (r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})[日号]?[^。；]{0,20}?(?:发布|披露|公布)(?:年报|中报|季报|财报)", "财报披露"),
        (r"(?:年报|中报|季报|财报)[^。；]{0,15}?(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})", "财报披露"),
    ]
    saved = 0
    for report in reports:
        code, name = _resolve_code(report)
        text = " ".join(filter(None, [report.title, report.summary_or_none.summary if report.summary_or_none else "", report.content_text[:20000]]))
        for pattern, etype in patterns:
            for m in re.finditer(pattern, text):
                try:
                    event_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    continue
                if event_date < date.today() or event_date > date.today() + timedelta(days=120):
                    continue
                _, created = MarketEvent.objects.get_or_create(
                    event_date=event_date,
                    stock_code=code or "",
                    title=f"{name or report.institution or '市场'} {etype}"[:250],
                    defaults={
                        "stock_name": (name or "")[:64],
                        "event_type": etype,
                        "detail": m.group(0)[:200],
                    },
                )
                saved += 1 if created else 0
    return saved


def _resolve_code(report):
    if report.stock_code:
        return report.stock_code, report.stock_name
    summary = report.summary_or_none
    if summary:
        codes = [c for c in (summary.target_stocks or []) if c and isinstance(c, str) and _CODE_RE.match(c)]
        if len(codes) == 1:
            return codes[0], report.stock_name
    return None, None


def get_forecast_consensus(limit: int = 16) -> list:
    """盈利一致预期：多份券商预测的中位数/分歧度。"""
    current_year = date.today().year
    rows = EarningsForecast.objects.filter(profit_value__isnull=False).select_related("report")
    name_map = _stock_name_map()
    stocks = defaultdict(list)
    for r in rows:
        if r.forecast_year and r.forecast_year < current_year:
            continue
        stocks[r.stock_code].append(r)

    result = []
    for code, items in stocks.items():
        if len(items) < 3:
            continue
        values = sorted(i.profit_value for i in items if i.profit_value)
        if not values:
            continue
        median = values[len(values) // 2]
        values = [v for v in values if median / 10 <= v <= median * 10]
        if not values:
            continue
        years = [i.forecast_year for i in items if i.forecast_year]
        year = max(set(years), key=years.count) if years else None
        avg = sum(values) / len(values)
        mn, mx = min(values), max(values)
        dispersion = round((mx - mn) / avg * 100, 1) if avg else 0
        eps_vals = [i.eps_value for i in items if i.eps_value]
        result.append({
            "stock_code": code,
            "stock_name": items[0].stock_name or name_map.get(code) or code,
            "forecast_count": len(items),
            "forecast_year": year,
            "avg_profit": round(avg, 2),
            "min_profit": round(mn, 2),
            "max_profit": round(mx, 2),
            "dispersion": dispersion,
            "avg_eps": round(sum(eps_vals) / len(eps_vals), 2) if eps_vals else None,
        })
    result.sort(key=lambda x: (x["dispersion"], -x["forecast_count"]))
    return result[:limit]


def get_target_upside(limit: int = 12) -> list:
    """最新目标价 vs 现价的上行空间。"""
    summaries = ResearchSummary.objects.exclude(target_price="").select_related("report")
    name_map = _stock_name_map()
    report_map = {}
    for s in summaries:
        report = s.report
        code = None
        name = None
        if report.stock_code:
            code, name = report.stock_code, report.stock_name
        else:
            codes = [c for c in (s.target_stocks or []) if c and isinstance(c, str) and _CODE_RE.match(c)]
            if len(codes) == 1:
                code = codes[0]
        if not code:
            continue
        try:
            tp = float(s.target_price)
        except (TypeError, ValueError):
            continue
        if code not in report_map or (report.report_date and report_map[code]["date"] < report.report_date):
            report_map[code] = {"target": tp, "date": report.report_date, "name": name, "institution": report.institution}

    # 【性能护栏】只评估最近给出目标价的前 40 只（5119 篇研报后 unique 代码过百，
    # 逐股查价即使有快照也不必要；排序后取最新报告的前 40 只足够覆盖 Top12）
    ranked_codes = sorted(report_map.items(), key=lambda kv: kv[1]["date"] or "", reverse=True)[:40]
    result = []
    for code, info in ranked_codes:
        current = get_latest_price(code)
        if not current or current <= 0:
            continue
        result.append({
            "stock_code": code,
            "stock_name": info["name"] or name_map.get(code) or code,
            "target_price": info["target"],
            "current_price": round(current, 2),
            "upside": round((info["target"] - current) / current * 100, 1),
            "institution": info["institution"],
        })
    result.sort(key=lambda x: x["upside"], reverse=True)
    return result[:limit]


def get_surprises(limit: int = 12) -> list:
    """业绩预告超/低预期与研报共识的交叉信号。

    窗口 60 天：业绩预告集中在报告期前的特定几周（如 7 月中旬中报潮），
    30 天窗口在间歇期会漏掉最近一批预告。
    """
    start = date.today() - timedelta(days=60)
    events = MarketEvent.objects.filter(event_date__gte=start).order_by("-event_date")
    bullish_stocks = set()
    reports = ResearchReport.objects.filter(
        status=ResearchReport.STATUS_SUMMARIZED,
        report_date__gte=start - timedelta(days=30),
    ).select_related("summary")
    for r in reports:
        s = r.summary_or_none
        if not s:
            continue
        is_bull = ("买入" in (s.rating or "")) or ("增持" in (s.rating or "")) or s.sentiment == "bullish"
        if not is_bull:
            continue
        if r.stock_code:
            bullish_stocks.add(r.stock_code)
        for c in s.target_stocks or []:
            if c and isinstance(c, str) and _CODE_RE.match(c):
                bullish_stocks.add(c)

    positive = ["预增", "略增", "扭亏", "减亏", "续盈"]
    negative = ["预减", "略减", "首亏", "续亏"]

    seen, result = set(), []
    for e in events:
        if not e.stock_code or e.stock_code in seen:
            continue
        seen.add(e.stock_code)
        etype = None
        for t in positive:
            if t in e.title:
                etype = "beat"
                break
        if not etype:
            for t in negative:
                if t in e.title:
                    etype = "miss"
                    break
        if not etype:
            continue
        in_consensus = e.stock_code in bullish_stocks
        result.append({
            "stock_code": e.stock_code,
            "stock_name": e.stock_name,
            "event_type": etype,
            "title": e.title,
            "detail": e.detail,
            "event_date": e.event_date.isoformat(),
            "with_research_support": in_consensus,
            "signal": "超预期确认" if (etype == "beat" and in_consensus) else (
                "业绩向好" if etype == "beat" else ("预期落空风险" if in_consensus else "业绩承压")
            ),
        })
        if len(result) >= limit:
            break
    return result


def get_tracking_stats() -> dict:
    """选股跟踪：T+1/T+5/T+20/T+60/T+120 胜率与平均收益 + 至今收益。"""
    from .market_utils import get_latest_price

    picks = list(StockPick.objects.order_by("-pick_date").values()[:50])
    items = []
    for p in picks:
        ret_so_far = None
        if p["price_at_pick"]:
            latest = get_latest_price(p["stock_code"])
            if latest:
                ret_so_far = round((latest - p["price_at_pick"]) / p["price_at_pick"] * 100, 2)
        items.append({
            "pick_date": p["pick_date"].isoformat(),
            "stock_code": p["stock_code"],
            "stock_name": p["stock_name"],
            "price_at_pick": p["price_at_pick"],
            "ret_t1": p["ret_t1"],
            "ret_t5": p["ret_t5"],
            "ret_t20": p["ret_t20"],
            "ret_t60": p["ret_t60"],
            "ret_t120": p["ret_t120"],
            "ret_so_far": ret_so_far,
        })

    def hit_rate(key, pool=None):
        vals = [i[key] for i in (pool or items) if i[key] is not None]
        if not vals:
            return None
        return {
            "count": len(vals),
            "win_rate": round(len([v for v in vals if v > 0]) / len(vals) * 100, 1),
            "avg_return": round(sum(vals) / len(vals), 2),
        }

    # 「至今」排除当天 pick（买点即最新收盘，收益恒 0 会稀释统计）
    today_iso = date.today().isoformat()
    settled = [i for i in items if i["pick_date"] < today_iso]
    return {
        "t1": hit_rate("ret_t1"), "t5": hit_rate("ret_t5"), "t20": hit_rate("ret_t20"),
        "t60": hit_rate("ret_t60"), "t120": hit_rate("ret_t120"),
        "so_far": hit_rate("ret_so_far", settled),
        "picks": items[:20],
    }


def get_today_payload(limit: int = 20) -> dict:
    """今日页数据：最新晨报 + 今日推荐（无推荐时回退最新研报）。"""
    brief = DailyBrief.objects.order_by("-brief_date").first()
    brief_data = None
    if brief:
        stocks = list(brief.stocks or [])
        risk_count = 0
        for s in stocks:
            code = (s.get("code") or "").strip()
            if code and _CODE_RE.match(code):
                s["price_change_1d"] = get_stock_change(code, 1)
                chg = get_stock_change(code, 20)
            else:
                s["price_change_1d"] = None
                chg = None
            s["price_change_20d"] = chg
            s["risk"] = bool(chg is not None and chg <= -5)
            if s["risk"]:
                risk_count += 1
        brief_data = {
            "date": brief.brief_date.isoformat(),
            "overview": brief.overview,
            "advice": brief.advice,
            "short_term": brief.short_term,
            "mid_term": brief.mid_term,
            "long_term": brief.long_term,
            "industries": brief.industries or [],
            "stocks": stocks,
            "risk_count": risk_count,
            "sentiment_stats": brief.sentiment_stats or {},
            "ai_model": brief.ai_model,
        }

    recs = list(
        DailyRecommendation.objects.filter(recommend_date=date.today())
        .select_related("report__source", "report__summary")
        .order_by("rank")[:limit]
    )
    items = [_report_item(rec.report, rank=rec.rank, score=rec.score, reason=rec.reason) for rec in recs]
    if not items:
        latest = ResearchReport.objects.select_related("source", "summary").order_by("-report_date", "-id")[:limit]
        items = [_report_item(r, rank=i + 1, score=1.0, reason="最新研报") for i, r in enumerate(latest)]

    return {
        "date": date.today().isoformat(),
        "brief": brief_data,
        "items": items,
        "indices": _index_quotes_safe(),
    }


def _index_quotes_safe():
    try:
        from .market_utils import get_index_quotes

        return get_index_quotes()
    except Exception:
        return []


def _report_item(r, rank=None, score=None, reason="") -> dict:
    summary = r.summary_or_none
    return {
        "id": r.id,
        "rank": rank,
        "score": round(score, 3) if score is not None else None,
        "reason": reason,
        "title": r.title,
        "stock_code": r.stock_code,
        "stock_name": r.stock_name,
        "institution": r.institution,
        "author": r.author,
        "report_date": r.report_date.isoformat(),
        "source": r.source.code if r.source else "",
        "pdf_url": r.pdf_url,
        "source_url": r.source_url,
        "sentiment": summary.sentiment if summary else "",
        "rating": summary.rating if summary else "",
        "tags": (summary.tags if summary else []) or [],
        "summary": summary.summary if summary else "",
        "investment_points": (summary.investment_points if summary else []) or [],
        "risks": (summary.risks if summary else []) or [],
        "target_price": summary.target_price if summary else "",
    }


def list_reports(search: str = "", sentiment: str = "", source: str = "", days: int = 30, limit: int = 50, offset: int = 0) -> dict:
    """研报库：筛选 + 分页。"""
    qs = ResearchReport.objects.select_related("source", "summary").order_by("-report_date", "-id")
    if days:
        qs = qs.filter(report_date__gte=date.today() - timedelta(days=days))
    if search:
        qs = qs.filter(Q(title__icontains=search) | Q(stock_name__icontains=search) | Q(institution__icontains=search))
    if sentiment:
        qs = qs.filter(summary__sentiment=sentiment)
    if source:
        qs = qs.filter(source__code=source)
    total = qs.count()
    rows = list(qs[offset:offset + limit])
    return {"total": total, "items": [_report_item(r) for r in rows]}


def get_sources() -> list:
    return [
        {
            "code": s.code, "name": s.name, "base_url": s.base_url,
            "status": s.status, "weight": s.weight,
            "last_crawled_at": s.last_crawled_at.isoformat() if s.last_crawled_at else None,
            "report_count": s.reports.count(),
        }
        for s in ResearchSource.objects.all()
    ]
