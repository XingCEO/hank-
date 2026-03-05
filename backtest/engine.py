"""
回測引擎
========
逐 K 線遍歷，模擬完整交易流程：
1. 偵測水平
2. 過濾 (EQ + CC)
3. 產生訊號
4. 開倉 / 管理持倉
5. 記錄結果
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import BACKTEST_CONFIG, RISK_CONFIG
from data.fetcher import DataFetcher, compute_atr
from indicators.levels import detect_all_levels, Level
from indicators.eq_filter import filter_eq_levels
from indicators.cc_filter import filter_levels_with_cc
from strategy.entry import generate_signals, Signal
from strategy.risk_manager import RiskManager, Position
from strategy.mtf_analysis import analyze_mtf, MTFAnalysis, MarketBias


class BacktestEngine:
    """回測引擎核心"""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        initial_balance: Optional[float] = None,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance or RISK_CONFIG["initial_balance"]
        self.risk_manager = RiskManager(self.initial_balance)

        # 結果記錄
        self.signals: list[Signal] = []
        self.equity_curve: list[float] = []
        self.trade_log: list[dict] = []

    def run(
        self,
        df: pd.DataFrame,
        mtf_data: Optional[dict[str, pd.DataFrame]] = None,
        use_eq_filter: bool = True,
        use_cc_filter: bool = True,
        use_mtf: bool = True,
    ) -> dict:
        """
        執行回測。

        Parameters
        ----------
        df : pd.DataFrame
            主要時間框架的 OHLCV 數據 (進場用)
        mtf_data : dict
            多時間框架數據，用於方向判斷
        use_eq_filter : bool
            是否使用 EQ 過濾
        use_cc_filter : bool
            是否使用 CC 過濾
        use_mtf : bool
            是否使用多時間框架分析

        Returns
        -------
        dict
            回測結果統計
        """
        print(f"\n{'='*60}")
        print(f"  SNR 2.0 回測引擎")
        print(f"  交易對: {self.symbol}")
        print(f"  K 線數: {len(df)}")
        print(f"  期間: {df.index[0]} ~ {df.index[-1]}")
        print(f"  初始資金: ${self.initial_balance:,.2f}")
        print(f"{'='*60}\n")

        # Step 1: 偵測水平
        print("[1/5] 偵測支撐/阻力水平...")
        levels = detect_all_levels(df)

        # Step 2: EQ 過濾
        if use_eq_filter:
            print("[2/5] EQ 過濾 (新鮮水平)...")
            eq_levels = filter_eq_levels(levels, df)
        else:
            eq_levels = levels
            print("[2/5] 跳過 EQ 過濾")

        # Step 3: CC 過濾
        if use_cc_filter:
            print("[3/5] CC 過濾 (確認 K 線)...")
            cc_results = filter_levels_with_cc(eq_levels, df)
            # 有 CC 的水平 + 原始 EQ 水平都保留
            filtered_levels = list(set(
                [lv for lv, _ in cc_results] +
                eq_levels
            ))
        else:
            cc_results = None
            filtered_levels = eq_levels
            print("[3/5] 跳過 CC 過濾")

        # Step 4: 多時間框架分析
        mtf_result = None
        if use_mtf and mtf_data:
            print("[4/5] 多時間框架分析...")
            mtf_result = analyze_mtf(mtf_data)
            print(f"  {mtf_result}")
        else:
            print("[4/5] 跳過多時間框架分析")

        # Step 5: 產生訊號 & 模擬交易
        print("[5/5] 產生訊號 & 模擬交易...")
        signals = generate_signals(df, filtered_levels, cc_results)

        # 根據 MTF 過濾訊號方向
        if mtf_result and mtf_result.overall_bias != MarketBias.NEUTRAL:
            before_count = len(signals)
            signals = self._filter_by_mtf(signals, mtf_result)
            print(f"  MTF 方向過濾: {before_count} → {len(signals)} 個訊號")

        self.signals = signals

        # 模擬逐 K 線交易
        atr = compute_atr(df)
        commission = BACKTEST_CONFIG["commission_rate"]
        slippage_ratio = BACKTEST_CONFIG["slippage_atr_ratio"]

        signal_idx = 0
        for bar_idx in range(len(df)):
            # 更新現有持倉
            for pos in list(self.risk_manager.positions):
                result = self.risk_manager.update_position(
                    pos,
                    bar_idx,
                    df["high"].iloc[bar_idx],
                    df["low"].iloc[bar_idx],
                    df["close"].iloc[bar_idx],
                    atr.iloc[bar_idx],
                )
                if result:
                    self._log_trade(pos, df)

            # 檢查是否有新訊號
            while signal_idx < len(signals) and signals[signal_idx].bar_index <= bar_idx:
                sig = signals[signal_idx]
                signal_idx += 1

                if sig.bar_index != bar_idx:
                    continue

                # 不在同一方向重複開倉
                if self.risk_manager.positions:
                    continue

                # 開倉
                slippage = atr.iloc[bar_idx] * slippage_ratio
                pos = self.risk_manager.open_position(sig, slippage)

                # 扣除手續費
                commission_cost = pos.entry_price * pos.size * commission
                self.risk_manager.balance -= commission_cost

            # 記錄權益曲線
            self.equity_curve.append(self.risk_manager.balance)

        # 關閉所有未平倉部位
        for pos in list(self.risk_manager.positions):
            self.risk_manager._close_position(
                pos, df["close"].iloc[-1], len(df) - 1, "回測結束平倉"
            )
            self._log_trade(pos, df)

        # 輸出結果
        stats = self.risk_manager.get_stats()
        self._print_results(stats)

        return stats

    def _filter_by_mtf(
        self, signals: list[Signal], mtf: MTFAnalysis
    ) -> list[Signal]:
        """根據 MTF 方向過濾訊號"""
        from strategy.entry import TradeDirection

        filtered = []
        for sig in signals:
            if mtf.overall_bias == MarketBias.BULLISH:
                if sig.direction == TradeDirection.LONG:
                    filtered.append(sig)
            elif mtf.overall_bias == MarketBias.BEARISH:
                if sig.direction == TradeDirection.SHORT:
                    filtered.append(sig)
            else:
                filtered.append(sig)
        return filtered

    def _log_trade(self, pos: Position, df: pd.DataFrame):
        """記錄交易"""
        entry_time = df.index[pos.entry_bar] if pos.entry_bar < len(df) else None
        exit_time = df.index[pos.exit_bar] if pos.exit_bar < len(df) else None

        self.trade_log.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "direction": pos.signal.direction.value,
            "entry_type": pos.signal.entry_type.value,
            "entry_price": pos.entry_price,
            "exit_price": pos.exit_price,
            "stop_loss": pos.signal.stop_loss,
            "take_profit": pos.signal.take_profit,
            "size": pos.size,
            "pnl": pos.pnl,
            "r_result": pos.r_result,
            "exit_reason": pos.exit_reason,
            "level_type": pos.signal.level.level_type.value if pos.signal.level else "",
            "breakeven_moved": pos.breakeven_moved,
        })

    def _print_results(self, stats: dict):
        """列印回測結果"""
        print(f"\n{'='*60}")
        print(f"  回測結果摘要")
        print(f"{'='*60}")

        if stats["total_trades"] == 0:
            print("  沒有產生任何交易")
            return

        print(f"  總交易次數:   {stats['total_trades']}")
        print(f"  勝率:         {stats['win_rate']:.1f}%")
        print(f"  獲利交易:     {stats['winning_trades']}")
        print(f"  虧損交易:     {stats['losing_trades']}")
        print(f"  ")
        print(f"  總損益:       ${stats['total_pnl']:,.2f}")
        print(f"  平均損益:     ${stats['avg_pnl']:,.2f}")
        print(f"  平均 R 值:    {stats['avg_r']:.2f}R")
        print(f"  平均獲利 R:   {stats['avg_win_r']:.2f}R")
        print(f"  平均虧損 R:   {stats['avg_loss_r']:.2f}R")
        print(f"  ")
        print(f"  最佳交易:     ${stats['best_trade']:,.2f}")
        print(f"  最差交易:     ${stats['worst_trade']:,.2f}")
        print(f"  利潤因子:     {stats['profit_factor']:.2f}")
        print(f"  最大回撤:     {stats['max_drawdown_pct']:.1f}%")
        print(f"  ")
        print(f"  最終餘額:     ${stats['final_balance']:,.2f}")
        print(f"  總報酬率:     {stats['return_pct']:.1f}%")
        print(f"{'='*60}\n")

    def get_trade_df(self) -> pd.DataFrame:
        """取得交易記錄 DataFrame"""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)
