from django.db import models

from .research import (
    DailyBrief,
    DailyRecommendation,
    EarningsForecast,
    MarketEvent,
    ResearchPreference,
    ResearchReport,
    ResearchSource,
    ResearchSummary,
    StockPick,
)


class PortfolioAccount(models.Model):
    """Portfolio-level settings and capital base."""

    name = models.CharField(max_length=64, unique=True, default="default")
    total_capital = models.DecimalField(max_digits=16, decimal_places=2, default=100000)
    cash = models.DecimalField(max_digits=16, decimal_places=2, default=100000)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "portfolio_account"

    def __str__(self):
        return self.name


class PortfolioHolding(models.Model):
    """Current holding snapshot. Trade history is stored separately."""

    STATUS_HOLDING = "holding"
    STATUS_CLOSED = "closed"
    STATUS_CHOICES = [
        (STATUS_HOLDING, "Holding"),
        (STATUS_CLOSED, "Closed"),
    ]

    account = models.ForeignKey(PortfolioAccount, on_delete=models.CASCADE, related_name="holdings")
    code = models.CharField(max_length=16)
    ts_code = models.CharField(max_length=16, blank=True, default="")
    name = models.CharField(max_length=64, blank=True, default="")
    industry = models.CharField(max_length=64, blank=True, default="")

    shares = models.IntegerField(default=0)
    avg_cost = models.DecimalField(max_digits=12, decimal_places=4, default=0)
    cost_amount = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    buy_date = models.CharField(max_length=8, blank=True, default="")
    last_trade_date = models.CharField(max_length=8, blank=True, default="")
    last_trade_price = models.DecimalField(max_digits=12, decimal_places=4, default=0)
    peak_price = models.DecimalField(max_digits=12, decimal_places=4, default=0)

    target_position_pct = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    actual_position_pct = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    position_level = models.CharField(max_length=32, blank=True, default="normal")
    source = models.CharField(max_length=32, blank=True, default="")
    strategy_id = models.CharField(max_length=64, blank=True, default="")
    strategy_name = models.CharField(max_length=128, blank=True, default="")
    strategy_snapshot = models.JSONField(default=dict, blank=True)
    exit_rule_snapshot = models.JSONField(default=dict, blank=True)
    buy_score = models.DecimalField(max_digits=8, decimal_places=3, default=0)
    buy_ai_score = models.DecimalField(max_digits=8, decimal_places=3, default=0)
    buy_reason = models.JSONField(default=list, blank=True)

    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_HOLDING)
    closed_at = models.DateTimeField(null=True, blank=True)
    realized_pnl = models.DecimalField(max_digits=16, decimal_places=2, default=0)
    notes = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "portfolio_holding"
        indexes = [
            models.Index(fields=["account", "code", "status"]),
        ]

    def __str__(self):
        return f"{self.code} {self.shares}"


class PortfolioTrade(models.Model):
    """Immutable trade ledger used to reconstruct holding history."""

    SIDE_BUY = "BUY"
    SIDE_ADD = "ADD"
    SIDE_SELL = "SELL"
    SIDE_REDUCE = "REDUCE"
    SIDE_CLOSE = "CLOSE"
    SIDE_CHOICES = [
        (SIDE_BUY, "Buy"),
        (SIDE_ADD, "Add"),
        (SIDE_SELL, "Sell"),
        (SIDE_REDUCE, "Reduce"),
        (SIDE_CLOSE, "Close"),
    ]

    account = models.ForeignKey(PortfolioAccount, on_delete=models.CASCADE, related_name="trades")
    holding = models.ForeignKey(PortfolioHolding, on_delete=models.SET_NULL, null=True, blank=True, related_name="trades")
    code = models.CharField(max_length=16)
    side = models.CharField(max_length=12, choices=SIDE_CHOICES)
    trade_date = models.CharField(max_length=8)
    price = models.DecimalField(max_digits=12, decimal_places=4)
    shares = models.IntegerField()
    amount = models.DecimalField(max_digits=16, decimal_places=2)
    position_pct = models.DecimalField(max_digits=6, decimal_places=2, default=0)
    reason = models.TextField(blank=True, default="")
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "portfolio_trade"
        indexes = [
            models.Index(fields=["account", "code", "trade_date"]),
        ]


class PortfolioExitScan(models.Model):
    """Persisted exit-scan result for auditability."""

    account = models.ForeignKey(PortfolioAccount, on_delete=models.CASCADE, related_name="exit_scans")
    scan_date = models.DateTimeField(auto_now_add=True)
    current_data = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)
    summary = models.JSONField(default=dict, blank=True)

    class Meta:
        db_table = "portfolio_exit_scan"
        ordering = ["-scan_date"]
