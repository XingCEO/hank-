"""
風險管理模組
============
實現：
1. 倉位計算 (固定風險百分比)
2. 提前止損 (M5 翻轉水平突破)
3. 移動止損至保本
4. 多 R 目標管理
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import RISK_CONFIG
from data.fetcher import compute_atr
from strategy.entry import Signal, TradeDirection


@dataclass
class Position:
    """持倉狀態"""
    signal: Signal
    size: float                          # 倉位大小
    entry_price: float                   # 實際進場價
    current_sl: float                    # 當前止損價
    current_tp: float                    # 當前目標價
    entry_bar: int                       # 進場 bar
    is_open: bool = True
    pnl: float = 0.0                    # 已實現損益
    max_favorable: float = 0.0          # 最大有利行情
    max_adverse: float = 0.0            # 最大不利行情
    exit_price: float = 0.0
    exit_bar: int = 0
    exit_reason: str = ""
    breakeven_moved: bool = False       # 是否已移至保本
    r_result: float = 0.0              # 最終 R 值結果

    def __repr__(self):
        status = "OPEN" if self.is_open else "CLOSED"
        return (
            f"Position({status} {self.signal.direction.value} "
            f"entry={self.entry_price:.2f} sl={self.current_sl:.2f} "
            f"tp={self.current_tp:.2f} pnl={self.pnl:.2f})"
        )


class RiskManager:
    """風險管理器"""

    def __init__(self, initial_balance: Optional[float] = None):
        self.balance = initial_balance or RISK_CONFIG["initial_balance"]
        self.initial_balance = self.balance
        self.positions: list[Position] = []
        self.closed_positions: list[Position] = []

    def calculate_position_size(self, signal: Signal) -> float:
        """
        根據固定風險百分比計算倉位大小。
        倉位 = (帳戶 * 風險%) / 每單位風險
        """
        risk_pct = RISK_CONFIG["risk_per_trade_pct"] / 100
        risk_amount = self.balance * risk_pct
        per_unit_risk = signal.risk

        if per_unit_risk <= 0:
            return 0.0

        size = risk_amount / per_unit_risk
        return size

    def open_position(self, signal: Signal, slippage: float = 0.0) -> Position:
        """開倉"""
        size = self.calculate_position_size(signal)

        # 考慮滑點
        if signal.direction == TradeDirection.LONG:
            actual_entry = signal.entry_price + slippage
        else:
            actual_entry = signal.entry_price - slippage

        pos = Position(
            signal=signal,
            size=size,
            entry_price=actual_entry,
            current_sl=signal.stop_loss,
            current_tp=signal.take_profit,
            entry_bar=signal.bar_index,
        )

        self.positions.append(pos)
        return pos

    def update_position(
        self,
        pos: Position,
        bar_index: int,
        high: float,
        low: float,
        close: float,
        atr_value: float,
    ) -> Optional[str]:
        """
        更新持倉狀態，檢查止損/止盈。

        Returns
        -------
        str or None
            平倉原因，None 表示持倉繼續
        """
        if not pos.is_open:
            return None

        # 更新最大有利/不利行情
        if pos.signal.direction == TradeDirection.LONG:
            unrealized = high - pos.entry_price
            adverse = pos.entry_price - low
        else:
            unrealized = pos.entry_price - low
            adverse = high - pos.entry_price

        pos.max_favorable = max(pos.max_favorable, unrealized)
        pos.max_adverse = max(pos.max_adverse, adverse)

        # --- 判斷 SL/TP 命中 (同一根 bar 都觸及時，用開盤方向判斷先後) ---
        hit_sl = False
        hit_tp = False

        if pos.signal.direction == TradeDirection.LONG:
            hit_sl = low <= pos.current_sl
            hit_tp = high >= pos.current_tp
        else:
            hit_sl = high >= pos.current_sl
            hit_tp = low <= pos.current_tp

        if hit_sl and hit_tp:
            # 同一根 bar 觸及 SL 和 TP：用開盤相對進場方向判定
            # 若開盤已朝 TP 方向，假設先觸 TP；反之先觸 SL
            if pos.signal.direction == TradeDirection.LONG:
                opens_favorable = close > pos.entry_price
            else:
                opens_favorable = close < pos.entry_price
            if opens_favorable:
                return self._close_position(pos, pos.current_tp, bar_index, "止盈")
            else:
                return self._close_position(pos, pos.current_sl, bar_index, "止損")

        if hit_tp:
            return self._close_position(pos, pos.current_tp, bar_index, "止盈")

        if hit_sl:
            return self._close_position(pos, pos.current_sl, bar_index, "止損")

        # --- 移動止損至保本 + 啟動追蹤止損 ---
        risk = pos.signal.risk
        min_rr = RISK_CONFIG["breakeven_min_rr"]

        if pos.signal.direction == TradeDirection.LONG:
            # 保本觸發用 close (避免假突破 wick 提前觸發)
            close_r = (close - pos.entry_price) / risk if risk > 0 else 0

            if not pos.breakeven_moved and close_r >= min_rr:
                pos.current_sl = pos.entry_price + risk * 0.5  # 保本 + 0.5R 利潤
                pos.breakeven_moved = True

            # 追蹤止損：保本後，止損跟隨最高點回撤 1R
            if pos.breakeven_moved:
                trail_sl = high - risk * 1.0  # 從高點回撤 1R
                if trail_sl > pos.current_sl:
                    pos.current_sl = trail_sl
        else:
            close_r = (pos.entry_price - close) / risk if risk > 0 else 0

            if not pos.breakeven_moved and close_r >= min_rr:
                pos.current_sl = pos.entry_price - risk * 0.5
                pos.breakeven_moved = True

            if pos.breakeven_moved:
                trail_sl = low + risk * 1.0  # 從低點回撤 1R
                if trail_sl < pos.current_sl:
                    pos.current_sl = trail_sl

        return None

    def check_early_stop(
        self,
        pos: Position,
        m5_df: pd.DataFrame,
        bar_index: int,
    ) -> Optional[str]:
        """
        提前止損檢查 (M5 翻轉水平被實體突破)。
        在 M5 上標記進場後的翻轉水平，若被突破則手動出場。

        Parameters
        ----------
        pos : Position
            持倉
        m5_df : pd.DataFrame
            M5 K 線數據
        bar_index : int
            M5 的當前 bar index
        """
        if not pos.is_open or bar_index >= len(m5_df):
            return None

        # 在 M5 上找進場後的翻轉水平
        # 簡化: 用進場後的第一個反向 swing 作為翻轉水平
        entry_bar = pos.entry_bar

        if pos.signal.direction == TradeDirection.LONG:
            # 做多: 找進場後的最低點作為翻轉水平
            if bar_index - entry_bar < 3:
                return None

            recent_low = m5_df["low"].iloc[entry_bar:bar_index].min()

            # 當前 K 線實體突破翻轉水平 → 出場
            curr_close = m5_df["close"].iloc[bar_index]
            curr_open = m5_df["open"].iloc[bar_index]
            body_bottom = min(curr_close, curr_open)

            if body_bottom < recent_low:
                return self._close_position(
                    pos, curr_close, bar_index, "提前止損(M5翻轉突破)"
                )
        else:
            if bar_index - entry_bar < 3:
                return None

            recent_high = m5_df["high"].iloc[entry_bar:bar_index].max()

            curr_close = m5_df["close"].iloc[bar_index]
            curr_open = m5_df["open"].iloc[bar_index]
            body_top = max(curr_close, curr_open)

            if body_top > recent_high:
                return self._close_position(
                    pos, curr_close, bar_index, "提前止損(M5翻轉突破)"
                )

        return None

    def _close_position(
        self, pos: Position, exit_price: float, bar_index: int, reason: str
    ) -> str:
        """平倉"""
        pos.is_open = False
        pos.exit_price = exit_price
        pos.exit_bar = bar_index
        pos.exit_reason = reason

        if pos.signal.direction == TradeDirection.LONG:
            pos.pnl = (exit_price - pos.entry_price) * pos.size
        else:
            pos.pnl = (pos.entry_price - exit_price) * pos.size

        # 計算 R 值結果
        risk = pos.signal.risk
        if risk > 0:
            if pos.signal.direction == TradeDirection.LONG:
                pos.r_result = (exit_price - pos.entry_price) / risk
            else:
                pos.r_result = (pos.entry_price - exit_price) / risk

        self.balance += pos.pnl
        self.positions.remove(pos)
        self.closed_positions.append(pos)

        return reason

    def get_stats(self) -> dict:
        """取得績效統計"""
        if not self.closed_positions:
            return {"total_trades": 0}

        pnls = [p.pnl for p in self.closed_positions]
        r_results = [p.r_result for p in self.closed_positions]
        wins = [p for p in self.closed_positions if p.pnl > 0]
        losses = [p for p in self.closed_positions if p.pnl <= 0]

        # 最大回撤計算
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative + self.initial_balance)
        drawdown = (peak - (cumulative + self.initial_balance)) / peak
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0

        return {
            "total_trades": len(self.closed_positions),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(self.closed_positions) * 100,
            "total_pnl": sum(pnls),
            "avg_pnl": np.mean(pnls),
            "avg_r": np.mean(r_results),
            "best_trade": max(pnls),
            "worst_trade": min(pnls),
            "profit_factor": (
                abs(sum(p.pnl for p in wins)) / abs(sum(p.pnl for p in losses))
                if losses and sum(p.pnl for p in losses) != 0 else float("inf")
            ),
            "max_drawdown_pct": max_drawdown * 100,
            "final_balance": self.balance,
            "return_pct": (self.balance - self.initial_balance) / self.initial_balance * 100,
            "avg_win_r": np.mean([p.r_result for p in wins]) if wins else 0,
            "avg_loss_r": np.mean([p.r_result for p in losses]) if losses else 0,
        }
