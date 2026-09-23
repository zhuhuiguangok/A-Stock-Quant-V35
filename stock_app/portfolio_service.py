from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterable, List, Optional

from django.db import transaction
from django.utils import timezone

from .models import PortfolioAccount, PortfolioExitScan, PortfolioHolding, PortfolioTrade
from .sell_logic import ExitManager, Position


def _d(value, default="0") -> Decimal:
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(default)


def _q2(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _q4(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def normalize_code(code: str) -> str:
    return str(code or "").split(".")[0].strip()


def today_ymd() -> str:
    return timezone.localdate().strftime("%Y%m%d")


def get_default_account() -> PortfolioAccount:
    account, _ = PortfolioAccount.objects.get_or_create(
        name="default",
        defaults={"total_capital": Decimal("100000"), "cash": Decimal("100000")},
    )
    return account


def serialize_account(account: PortfolioAccount) -> dict:
    return {
        "id": account.id,
        "name": account.name,
        "total_capital": float(account.total_capital),
        "cash": float(account.cash),
    }


def serialize_holding(holding: PortfolioHolding, current_data: Optional[Dict] = None) -> dict:
    code = normalize_code(holding.code)
    cur = (current_data or {}).get(code, {}) if isinstance(current_data, dict) else {}
    current_price = _d(cur.get("close", holding.last_trade_price or holding.avg_cost))
    avg_cost = _d(holding.avg_cost)
    pnl_pct = Decimal("0")
    market_value = current_price * Decimal(holding.shares or 0)
    if avg_cost > 0:
        pnl_pct = (current_price - avg_cost) / avg_cost * Decimal("100")
    return {
        "id": holding.id,
        "code": code,
        "ts_code": holding.ts_code,
        "name": holding.name,
        "industry": holding.industry,
        "buy_price": float(holding.avg_cost),
        "avg_cost": float(holding.avg_cost),
        "buy_date": holding.buy_date,
        "last_trade_date": holding.last_trade_date,
        "last_trade_price": float(holding.last_trade_price),
        "shares": holding.shares,
        "cost_amount": float(holding.cost_amount),
        "current_price": float(current_price),
        "market_value": float(_q2(market_value)),
        "pnl_pct": float(_q2(pnl_pct)),
        "peak_price": float(holding.peak_price),
        "position_pct": float(holding.actual_position_pct),
        "target_position_pct": float(holding.target_position_pct),
        "position_level": holding.position_level,
        "source": holding.source,
        "strategy_id": holding.strategy_id,
        "strategy_name": holding.strategy_name,
        "strategy_snapshot": holding.strategy_snapshot,
        "exit_rule_snapshot": holding.exit_rule_snapshot,
        "buy_score": float(holding.buy_score),
        "buy_ai_score": float(holding.buy_ai_score),
        "buy_reason": holding.buy_reason,
        "status": holding.status,
        "realized_pnl": float(holding.realized_pnl),
    }


def _recalc_position_pct(holding: PortfolioHolding, account: PortfolioAccount) -> Decimal:
    if account.total_capital <= 0:
        return Decimal("0")
    return _q2(holding.cost_amount / account.total_capital * Decimal("100"))


@transaction.atomic
def set_account_capital(total_capital) -> PortfolioAccount:
    account = get_default_account()
    cap = _q2(_d(total_capital, account.total_capital))
    if cap <= 0:
        raise ValueError("账户总资金必须大于0")
    account.total_capital = cap
    account.cash = cap
    account.save(update_fields=["total_capital", "cash", "updated_at"])
    for holding in account.holdings.filter(status=PortfolioHolding.STATUS_HOLDING):
        holding.actual_position_pct = _recalc_position_pct(holding, account)
        holding.save(update_fields=["actual_position_pct", "updated_at"])
    return account


@transaction.atomic
def buy_or_add_from_selection(stock: dict, shares: int, total_capital=None, trade_date: Optional[str] = None) -> PortfolioHolding:
    account = set_account_capital(total_capital) if total_capital else get_default_account()
    code = normalize_code(stock.get("code") or stock.get("ts_code"))
    if not code:
        raise ValueError("缺少股票代码")
    shares = int(shares or 0)
    shares = shares // 100 * 100
    if shares <= 0:
        raise ValueError("买入股数必须至少100股，且按100股一手")
    price = _q4(_d(stock.get("current_price") or stock.get("price")))
    if price <= 0:
        raise ValueError("买入价格必须大于0")

    trade_date = trade_date or today_ymd()
    amount = _q2(price * Decimal(shares))
    target_pct = _q2(_d(stock.get("position_pct"), "0"))
    strategy_id = stock.get("strategy_id") or (stock.get("source") if stock.get("type") == "strategy" else "") or ""
    strategy_name = stock.get("strategy_name") or ""
    strategy_snapshot = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "strategy_score": stock.get("strategy_score", stock.get("buy_score", 0)),
        "strategy_dimensions": stock.get("strategy_dimensions", {}),
        "holding_days_plan": stock.get("holding_days_plan", ""),
        "entry_rule": stock.get("entry_rule", ""),
        "max_total_position_pct": stock.get("max_total_position_pct"),
        "stop_loss_pct": stock.get("stop_loss_pct"),
    }
    exit_rule_snapshot = stock.get("exit_rule") or {}
    metadata = {
        "position_level": stock.get("position_level", "normal"),
        "source": stock.get("source") or stock.get("type") or "",
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "strategy_snapshot": strategy_snapshot,
        "exit_rule_snapshot": exit_rule_snapshot,
        "buy_score": stock.get("buy_score", 0),
        "buy_ai_score": stock.get("ml_score", stock.get("buy_ai_score", 0)),
        "selection_snapshot": stock,
    }

    holding = (
        PortfolioHolding.objects
        .select_for_update()
        .filter(account=account, code=code, status=PortfolioHolding.STATUS_HOLDING)
        .first()
    )

    if holding:
        old_cost = _d(holding.cost_amount)
        old_shares = int(holding.shares or 0)
        new_shares = old_shares + shares
        new_cost = old_cost + amount
        holding.shares = new_shares
        holding.cost_amount = _q2(new_cost)
        holding.avg_cost = _q4(new_cost / Decimal(new_shares))
        holding.last_trade_date = trade_date
        holding.last_trade_price = price
        holding.peak_price = max(_d(holding.peak_price), price, _d(holding.avg_cost))
        holding.target_position_pct = target_pct or holding.target_position_pct
        holding.actual_position_pct = _recalc_position_pct(holding, account)
        holding.position_level = stock.get("position_level") or holding.position_level
        holding.buy_score = _q4(_d(stock.get("buy_score"), holding.buy_score))
        holding.buy_ai_score = _q4(_d(stock.get("ml_score", stock.get("buy_ai_score")), holding.buy_ai_score))
        holding.source = stock.get("source") or stock.get("type") or holding.source
        holding.strategy_id = strategy_id or holding.strategy_id
        holding.strategy_name = strategy_name or holding.strategy_name
        if strategy_id:
            holding.strategy_snapshot = strategy_snapshot
            holding.exit_rule_snapshot = exit_rule_snapshot
        holding.buy_reason = stock.get("buy_signals") or holding.buy_reason
        side = PortfolioTrade.SIDE_ADD
    else:
        holding = PortfolioHolding(
            account=account,
            code=code,
            ts_code=stock.get("ts_code", ""),
            name=stock.get("name", code),
            industry=stock.get("industry", ""),
            shares=shares,
            avg_cost=price,
            cost_amount=amount,
            buy_date=trade_date,
            last_trade_date=trade_date,
            last_trade_price=price,
            peak_price=price,
            target_position_pct=target_pct,
            actual_position_pct=_q2(amount / account.total_capital * Decimal("100")) if account.total_capital > 0 else Decimal("0"),
            position_level=stock.get("position_level", "normal"),
            source=stock.get("source") or stock.get("type") or "",
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            strategy_snapshot=strategy_snapshot if strategy_id else {},
            exit_rule_snapshot=exit_rule_snapshot if strategy_id else {},
            buy_score=_q4(_d(stock.get("buy_score"))),
            buy_ai_score=_q4(_d(stock.get("ml_score", stock.get("buy_ai_score")))),
            buy_reason=stock.get("buy_signals") or [],
        )
        side = PortfolioTrade.SIDE_BUY

    holding.save()
    PortfolioTrade.objects.create(
        account=account,
        holding=holding,
        code=code,
        side=side,
        trade_date=trade_date,
        price=price,
        shares=shares,
        amount=amount,
        position_pct=target_pct,
        reason="selection_add",
        metadata=metadata,
    )
    return holding


@transaction.atomic
def close_holding(code: str, price=None, trade_date: Optional[str] = None, reason: str = "manual_close") -> PortfolioHolding:
    account = get_default_account()
    code = normalize_code(code)
    holding = (
        PortfolioHolding.objects
        .select_for_update()
        .get(account=account, code=code, status=PortfolioHolding.STATUS_HOLDING)
    )
    price_d = _q4(_d(price, holding.last_trade_price or holding.avg_cost))
    amount = _q2(price_d * Decimal(holding.shares))
    pnl = _q2((price_d - holding.avg_cost) * Decimal(holding.shares))
    PortfolioTrade.objects.create(
        account=account,
        holding=holding,
        code=code,
        side=PortfolioTrade.SIDE_CLOSE,
        trade_date=trade_date or today_ymd(),
        price=price_d,
        shares=holding.shares,
        amount=amount,
        position_pct=holding.actual_position_pct,
        reason=reason,
    )
    holding.realized_pnl += pnl
    holding.status = PortfolioHolding.STATUS_CLOSED
    holding.closed_at = timezone.now()
    holding.actual_position_pct = Decimal("0")
    holding.save()
    return holding


@transaction.atomic
def sell_holding(
    code: str,
    shares: Optional[int] = None,
    price=None,
    trade_date: Optional[str] = None,
    reason: str = "manual_sell",
) -> PortfolioHolding:
    """Sell part/all of an open holding and keep average-cost ledger consistent."""
    account = get_default_account()
    code = normalize_code(code)
    holding = (
        PortfolioHolding.objects
        .select_for_update()
        .get(account=account, code=code, status=PortfolioHolding.STATUS_HOLDING)
    )
    current_shares = int(holding.shares or 0)
    sell_shares = int(shares or current_shares)
    sell_shares = min(current_shares, max(0, sell_shares // 100 * 100))
    if sell_shares <= 0:
        raise ValueError("卖出股数必须至少100股，且不能超过当前持仓")

    price_d = _q4(_d(price, holding.last_trade_price or holding.avg_cost))
    amount = _q2(price_d * Decimal(sell_shares))
    pnl = _q2((price_d - holding.avg_cost) * Decimal(sell_shares))
    side = PortfolioTrade.SIDE_CLOSE if sell_shares >= current_shares else PortfolioTrade.SIDE_REDUCE

    PortfolioTrade.objects.create(
        account=account,
        holding=holding,
        code=code,
        side=side,
        trade_date=trade_date or today_ymd(),
        price=price_d,
        shares=sell_shares,
        amount=amount,
        position_pct=holding.actual_position_pct,
        reason=reason,
    )

    remaining = current_shares - sell_shares
    holding.realized_pnl += pnl
    holding.last_trade_date = trade_date or today_ymd()
    holding.last_trade_price = price_d
    if remaining <= 0:
        holding.shares = 0
        holding.cost_amount = Decimal("0")
        holding.actual_position_pct = Decimal("0")
        holding.status = PortfolioHolding.STATUS_CLOSED
        holding.closed_at = timezone.now()
    else:
        holding.shares = remaining
        holding.cost_amount = _q2(holding.avg_cost * Decimal(remaining))
        holding.actual_position_pct = _recalc_position_pct(holding, account)
    holding.save()
    return holding


def holdings_as_exit_positions(holdings: Iterable[PortfolioHolding]) -> List[Position]:
    positions: List[Position] = []
    for h in holdings:
        pos = Position(
            code=normalize_code(h.code),
            buy_price=float(h.avg_cost),
            buy_date=h.buy_date or h.last_trade_date or today_ymd(),
            shares=int(h.shares or 0),
            buy_score=float(h.buy_score),
            buy_ai_score=float(h.buy_ai_score),
        )
        pos.peak_price = float(h.peak_price or h.avg_cost)
        positions.append(pos)
    return positions


@transaction.atomic
def run_exit_scan(current_data: Optional[Dict] = None, positions_payload: Optional[List[dict]] = None) -> dict:
    account = get_default_account()
    current_data = current_data or {}
    if positions_payload:
        positions = []
        for p in positions_payload:
            pos = Position(
                code=normalize_code(p.get("code")),
                buy_price=float(p.get("buy_price", p.get("avg_cost", 0))),
                buy_date=str(p.get("buy_date", today_ymd())),
                shares=int(p.get("shares", 0)),
                buy_score=float(p.get("buy_score", 70)),
                buy_ai_score=float(p.get("buy_ai_score", 60)),
            )
            pos.peak_price = float(p.get("peak_price", p.get("buy_price", 0)))
            positions.append(pos)
    else:
        holdings = account.holdings.filter(status=PortfolioHolding.STATUS_HOLDING)
        positions = holdings_as_exit_positions(holdings)
    if not positions:
        raise ValueError("无有效持仓数据")
    report = ExitManager().daily_scan(positions, current_data)
    PortfolioExitScan.objects.create(
        account=account,
        current_data=current_data,
        result=report,
        summary=report.get("summary", {}),
    )
    return {"status": "success", **report}
