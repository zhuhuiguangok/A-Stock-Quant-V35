"""AI 研报模块的 Django ORM 模型。

移植自原 FastAPI 项目（SQLAlchemy），差异：
- tags/risks/target_stocks/industries 等改为 JSONField（原库手动 json.dumps）
- ResearchReport 补上 (source, external_id) 唯一约束，去重靠数据库
- user_preferences 多 session 设计改为单行 ResearchPreference（本平台单用户）
"""
from django.db import models


class ResearchSource(models.Model):
    """研报爬虫数据源。"""

    name = models.CharField(max_length=64)
    code = models.CharField(max_length=32, unique=True)
    base_url = models.CharField(max_length=255)
    status = models.CharField(max_length=16, default="active")
    weight = models.FloatField(default=1.0)
    last_crawled_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_source"

    def __str__(self):
        return f"{self.name}({self.code})"


class ResearchReport(models.Model):
    """研报原文。status: pending / summarized。"""

    STATUS_PENDING = "pending"
    STATUS_SUMMARIZED = "summarized"

    source = models.ForeignKey(ResearchSource, on_delete=models.CASCADE, related_name="reports")
    external_id = models.CharField(max_length=128)
    title = models.CharField(max_length=512)
    stock_code = models.CharField(max_length=16, blank=True, default="")
    stock_name = models.CharField(max_length=64, blank=True, default="")
    industry = models.CharField(max_length=64, blank=True, default="")
    author = models.CharField(max_length=128, blank=True, default="")
    institution = models.CharField(max_length=128, blank=True, default="")
    report_date = models.DateField()
    content_text = models.TextField(blank=True, default="")
    pdf_url = models.CharField(max_length=512, blank=True, default="")
    source_url = models.CharField(max_length=512)
    status = models.CharField(max_length=16, default=STATUS_PENDING, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "research_report"
        constraints = [
            models.UniqueConstraint(fields=["source", "external_id"], name="uniq_research_report_external"),
        ]
        indexes = [
            models.Index(fields=["report_date"]),
            models.Index(fields=["stock_code"]),
            models.Index(fields=["industry"]),
        ]

    def __str__(self):
        return f"[{self.report_date}] {self.title[:30]}"

    @property
    def summary_or_none(self):
        """安全读取 OneToOne 反向关联（无摘要时返回 None 而非抛异常）。"""
        try:
            return self.summary
        except ResearchSummary.DoesNotExist:
            return None


class ResearchSummary(models.Model):
    """研报的 AI 摘要。"""

    report = models.OneToOneField(ResearchReport, on_delete=models.CASCADE, related_name="summary")
    summary = models.TextField(blank=True, default="")
    tags = models.JSONField(default=list, blank=True)
    investment_points = models.JSONField(default=list, blank=True)
    sentiment = models.CharField(max_length=16, blank=True, default="")
    target_stocks = models.JSONField(default=list, blank=True)
    rating = models.CharField(max_length=32, blank=True, default="")
    risks = models.JSONField(default=list, blank=True)
    target_price = models.CharField(max_length=64, blank=True, default="")
    ai_model = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_summary"


class DailyBrief(models.Model):
    """每日晨报。"""

    brief_date = models.DateField(unique=True)
    overview = models.TextField(blank=True, default="")
    advice = models.TextField(blank=True, default="")
    short_term = models.TextField(blank=True, default="")
    mid_term = models.TextField(blank=True, default="")
    long_term = models.TextField(blank=True, default="")
    industries = models.JSONField(default=list, blank=True)
    stocks = models.JSONField(default=list, blank=True)
    sentiment_stats = models.JSONField(default=dict, blank=True)
    ai_model = models.CharField(max_length=64, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_daily_brief"
        ordering = ["-brief_date"]


class DailyRecommendation(models.Model):
    """每日个性化推荐。"""

    report = models.ForeignKey(ResearchReport, on_delete=models.CASCADE, related_name="recommendations")
    recommend_date = models.DateField(db_index=True)
    score = models.FloatField()
    rank = models.IntegerField()
    reason = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_daily_recommendation"
        ordering = ["recommend_date", "rank"]


class StockPick(models.Model):
    """从晨报解析出的推荐个股，回填 T+1/T+5/T+20 收益检验推荐质量。"""

    pick_date = models.DateField(db_index=True)
    stock_code = models.CharField(max_length=16)
    stock_name = models.CharField(max_length=64, blank=True, default="")
    source = models.CharField(max_length=32, default="daily_brief")
    price_at_pick = models.FloatField(null=True, blank=True)
    ret_t1 = models.FloatField(null=True, blank=True)
    ret_t5 = models.FloatField(null=True, blank=True)
    ret_t20 = models.FloatField(null=True, blank=True)
    ret_t60 = models.FloatField(null=True, blank=True)
    ret_t120 = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_stock_pick"
        constraints = [
            models.UniqueConstraint(fields=["pick_date", "stock_code", "source"], name="uniq_research_pick"),
        ]


class MarketEvent(models.Model):
    """从研报提取的未来事件（股东大会/解禁/财报等）。"""

    event_date = models.DateField(db_index=True)
    stock_code = models.CharField(max_length=16, blank=True, default="")
    stock_name = models.CharField(max_length=64, blank=True, default="")
    event_type = models.CharField(max_length=32)
    title = models.CharField(max_length=255)
    detail = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_market_event"
        constraints = [
            models.UniqueConstraint(fields=["event_date", "stock_code", "title"], name="uniq_research_event"),
        ]


class EarningsForecast(models.Model):
    """从研报提取的券商盈利预测。"""

    report = models.ForeignKey(ResearchReport, on_delete=models.CASCADE, related_name="forecasts")
    stock_code = models.CharField(max_length=16, db_index=True)
    stock_name = models.CharField(max_length=64, blank=True, default="")
    forecast_year = models.IntegerField(null=True, blank=True)
    profit_value = models.FloatField(null=True, blank=True)
    profit_text = models.CharField(max_length=128, blank=True, default="")
    eps_value = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "research_earnings_forecast"


class ResearchPreference(models.Model):
    """单行用户偏好（本平台单用户，丢弃原 session_id 设计）。"""

    watch_stocks = models.JSONField(default=list, blank=True)
    watch_industries = models.JSONField(default=list, blank=True)
    daily_count = models.IntegerField(default=20)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "research_preference"

    @classmethod
    def get_solo(cls) -> "ResearchPreference":
        obj, _ = cls.objects.get_or_create(pk=1)
        return obj
