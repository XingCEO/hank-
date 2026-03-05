"""
回測引擎 (v2)
=============
逐 K 線遍歷，每根 bar 即時偵測水平、過濾、進場：
1. 只用「已經看到」的歷史數據偵測水平 (無 look-ahead bias)
2. CC 過濾：沒有 CC 就不進場 (除非關閉 CC 過濾)
3. EQ 過濾：只在新鮮 (未被測試) 水平進場
4. 每次只持有一個倉位
5. 水平被觸及後標記 tested，被突破後標記 broken
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional, cast

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import BACKTEST_CONFIG, RISK_CONFIG, LEVELS_CONFIG, CC_CONFIG, EQ_CONFIG, ADX_CONFIG
from data.fetcher import compute_atr
from indicators.levels import Level, LevelType, LevelSide
from strategy.entry import Signal, EntryType, TradeDirection, check_dc_entry
from strategy.risk_manager import RiskManager, Position
from strategy.mtf_analysis import analyze_mtf, MTFAnalysis, MarketBias


# ============================================================
# 即時水平偵測 (逐 bar，只看過去數據)
# ============================================================

def _detect_levels_at_bar(
    df: pd.DataFrame,
    bar_idx: int,
    atr: pd.Series,
    lookback: int,
) -> list[Level]:
    """在 bar_idx 處偵測水平，只使用 bar_idx 之前的數據。"""
    if bar_idx < lookback * 2 + 2:
        return []

    results: list[Level] = []
    local_atr = atr.iloc[bar_idx]
    if local_atr <= 0 or pd.isna(local_atr):
        return []

    # --- Classic: swing high/low 在 bar_idx - lookback 確認 ---
    pivot_bar = bar_idx - lookback
    if pivot_bar < lookback:
        return results

    # Swing High
    is_swing_high = True
    pivot_high = df["high"].iloc[pivot_bar]
    for j in range(1, lookback + 1):
        if df["high"].iloc[pivot_bar - j] >= pivot_high:
            is_swing_high = False
            break
        if df["high"].iloc[pivot_bar + j] >= pivot_high:
            is_swing_high = False
            break

    if is_swing_high:
        ts = cast(pd.Timestamp, df.index[pivot_bar]) if isinstance(df.index, pd.DatetimeIndex) else None
        results.append(Level(
            price=pivot_high,
            level_type=LevelType.CLASSIC,
            side=LevelSide.RESISTANCE,
            bar_index=pivot_bar,
            timestamp=ts,
            strength=0.7,
        ))

    # Swing Low
    is_swing_low = True
    pivot_low = df["low"].iloc[pivot_bar]
    for j in range(1, lookback + 1):
        if df["low"].iloc[pivot_bar - j] <= pivot_low:
            is_swing_low = False
            break
        if df["low"].iloc[pivot_bar + j] <= pivot_low:
            is_swing_low = False
            break

    if is_swing_low:
        ts = cast(pd.Timestamp, df.index[pivot_bar]) if isinstance(df.index, pd.DatetimeIndex) else None
        results.append(Level(
            price=pivot_low,
            level_type=LevelType.CLASSIC,
            side=LevelSide.SUPPORT,
            bar_index=pivot_bar,
            timestamp=ts,
            strength=0.7,
        ))

    # --- Breakout: 當前 bar 的強勢突破 K 線 ---
    body = abs(df["close"].iloc[bar_idx] - df["open"].iloc[bar_idx])
    bo_ratio = LEVELS_CONFIG["breakout_body_atr_ratio"]
    if body >= local_atr * bo_ratio:
        is_bull = df["close"].iloc[bar_idx] > df["open"].iloc[bar_idx]
        if is_bull:
            bo_price = min(df["close"].iloc[bar_idx], df["open"].iloc[bar_idx])
            ts = cast(pd.Timestamp, df.index[bar_idx]) if isinstance(df.index, pd.DatetimeIndex) else None
            results.append(Level(
                price=bo_price, level_type=LevelType.BREAKOUT,
                side=LevelSide.SUPPORT, bar_index=bar_idx,
                timestamp=ts, strength=0.85,
            ))
        else:
            bo_price = max(df["close"].iloc[bar_idx], df["open"].iloc[bar_idx])
            ts = cast(pd.Timestamp, df.index[bar_idx]) if isinstance(df.index, pd.DatetimeIndex) else None
            results.append(Level(
                price=bo_price, level_type=LevelType.BREAKOUT,
                side=LevelSide.RESISTANCE, bar_index=bar_idx,
                timestamp=ts, strength=0.85,
            ))

    # --- Gap: 同色 K 線之間的缺口 ---
    if bar_idx >= 2:
        gap_ratio = LEVELS_CONFIG["gap_min_atr_ratio"]
        prev_close = df["close"].iloc[bar_idx - 1]
        prev_open = df["open"].iloc[bar_idx - 1]
        curr_open_val = df["open"].iloc[bar_idx]
        curr_close_val = df["close"].iloc[bar_idx]

        prev_bull = prev_close > prev_open
        curr_bull = curr_close_val > curr_open_val

        if prev_bull and curr_bull:
            # 兩根連續多頭：gap = 當前 open - 前一根 close
            gap = curr_open_val - prev_close
            if gap >= local_atr * gap_ratio:
                gap_price = (curr_open_val + prev_close) / 2
                ts = cast(pd.Timestamp, df.index[bar_idx]) if isinstance(df.index, pd.DatetimeIndex) else None
                results.append(Level(
                    price=gap_price, level_type=LevelType.GAP,
                    side=LevelSide.SUPPORT, bar_index=bar_idx,
                    timestamp=ts, strength=0.75,
                ))
        elif not prev_bull and not curr_bull:
            # 兩根連續空頭：gap = 前一根 close - 當前 open
            gap = prev_close - curr_open_val
            if gap >= local_atr * gap_ratio:
                gap_price = (prev_close + curr_open_val) / 2
                ts = cast(pd.Timestamp, df.index[bar_idx]) if isinstance(df.index, pd.DatetimeIndex) else None
                results.append(Level(
                    price=gap_price, level_type=LevelType.GAP,
                    side=LevelSide.RESISTANCE, bar_index=bar_idx,
                    timestamp=ts, strength=0.75,
                ))

    return results


def _has_cc(
    level: Level,
    df: pd.DataFrame,
    bar_idx: int,
    atr: pd.Series,
) -> bool:
    """檢查水平附近是否有確認 K 線 (只看 bar_idx 之前的 K 線)。"""
    proximity = atr.iloc[bar_idx] * CC_CONFIG["proximity_atr_ratio"]
    min_body = atr.iloc[bar_idx] * CC_CONFIG["min_body_atr_ratio"]
    lookback = CC_CONFIG["max_lookback_bars"]

    for k in range(1, lookback + 1):
        check_bar = bar_idx - k
        if check_bar < 0:
            break

        cc_body = abs(df["close"].iloc[check_bar] - df["open"].iloc[check_bar])
        cc_bull = df["close"].iloc[check_bar] > df["open"].iloc[check_bar]

        if cc_body < min_body:
            continue

        if level.side == LevelSide.SUPPORT:
            dist = abs(df["low"].iloc[check_bar] - level.price)
            if dist <= proximity and cc_bull:
                return True
        else:
            dist = abs(df["high"].iloc[check_bar] - level.price)
            if dist <= proximity and not cc_bull:
                return True

    return False


def _is_level_fresh(
    level: Level,
    df: pd.DataFrame,
    bar_idx: int,
    atr: pd.Series,
) -> bool:
    """檢查水平是否新鮮 (從形成到現在沒被碰過)。"""
    tolerance = atr.iloc[bar_idx] * EQ_CONFIG["fresh_tolerance_atr_ratio"]

    # 從水平形成後到 bar_idx 之前，看有沒有被碰過
    for k in range(level.bar_index + 1, bar_idx):
        if k >= len(df):
            break
        if level.side == LevelSide.SUPPORT:
            if df["low"].iloc[k] <= level.price + tolerance:
                return False
        else:
            if df["high"].iloc[k] >= level.price - tolerance:
                return False
    return True


# ============================================================
# 回測引擎
# ============================================================

class BacktestEngine:
    """回測引擎核心 (v2: 逐 bar 即時偵測)"""

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        initial_balance: Optional[float] = None,
    ):
        self.symbol = symbol
        self.initial_balance = initial_balance or RISK_CONFIG["initial_balance"]
        self.risk_manager = RiskManager(self.initial_balance)

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
        執行回測 (逐 bar 即時偵測，無 look-ahead bias)。
        """
        print(f"\n{'='*60}")
        print(f"  SNR 2.0 回測引擎 v2")
        print(f"  交易對: {self.symbol}")
        print(f"  K 線數: {len(df)}")
        print(f"  期間: {df.index[0]} ~ {df.index[-1]}")
        print(f"  初始資金: ${self.initial_balance:,.2f}")
        print(f"  EQ 過濾: {'ON' if use_eq_filter else 'OFF'}")
        print(f"  CC 過濾: {'ON' if use_cc_filter else 'OFF'}")
        print(f"  MTF 分析: {'ON' if use_mtf else 'OFF'}")
        print(f"  ADX 過濾: ON (閾值={ADX_CONFIG['min_trend_threshold']})")
        print(f"{'='*60}\n")

        # 預計算 ATR + EMA 趨勢 + ADX
        atr = compute_atr(df)
        ema50 = df["close"].ewm(span=50, adjust=False).mean()
        ema200 = df["close"].ewm(span=200, adjust=False).mean()

        # ADX 計算
        adx_period = ADX_CONFIG["period"]
        high_s = df["high"]
        low_s = df["low"]
        close_s = df["close"]
        plus_dm = high_s.diff()
        minus_dm = -low_s.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
        tr = pd.concat([
            high_s - low_s,
            (high_s - close_s.shift(1)).abs(),
            (low_s - close_s.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr_adx = tr.ewm(span=adx_period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=adx_period, adjust=False).mean() / atr_adx)
        minus_di = 100 * (minus_dm.ewm(span=adx_period, adjust=False).mean() / atr_adx)
        di_diff = pd.Series(np.abs(np.asarray(plus_di) - np.asarray(minus_di)), index=df.index)
        di_sum = pd.Series(np.asarray(plus_di) + np.asarray(minus_di), index=df.index)
        di_sum_safe = di_sum.where(di_sum != 0, np.nan)
        dx = (di_diff / di_sum_safe * 100).fillna(0.0)
        adx = dx.ewm(span=adx_period, adjust=False).mean()
        commission = BACKTEST_CONFIG["commission_rate"]
        slippage_ratio = BACKTEST_CONFIG["slippage_atr_ratio"]
        lookback = LEVELS_CONFIG["swing_lookback"]

        # MTF 方向
        mtf_result = None
        mtf_bias = MarketBias.NEUTRAL
        if use_mtf and mtf_data:
            print("[MTF] 分析多時間框架方向...")
            mtf_result = analyze_mtf(mtf_data)
            mtf_bias = mtf_result.overall_bias
            print(f"  {mtf_result}")

        # 即時水平池
        active_levels: list[Level] = []
        total_signals = 0
        max_levels = 30  # 最多保留 30 個水平

        # 冷卻機制：記錄最近止損的 bar 和價格區域，避免重複進場
        cooldown_until = 0  # 冷卻到此 bar 之後才能進場
        cooldown_bars = 5  # 止損後冷卻根數
        consec_losses = 0  # 連續虧損次數

        print("[回測] 逐 K 線掃描中...")

        for bar_idx in range(lookback * 2 + 2, len(df)):
            # --- 1. 偵測新水平 (只用歷史數據) ---
            new_levels = _detect_levels_at_bar(df, bar_idx, atr, lookback)
            for lv in new_levels:
                # 去重：與現有水平太近的不加入
                local_atr = atr.iloc[bar_idx]
                merge_dist = local_atr * LEVELS_CONFIG["merge_distance_atr_ratio"]
                is_dup = False
                for existing in active_levels:
                    if abs(existing.price - lv.price) < merge_dist:
                        is_dup = True
                        break
                if not is_dup:
                    active_levels.append(lv)

            # --- 2. 更新水平狀態 ---
            bar_high = df["high"].iloc[bar_idx]
            bar_low = df["low"].iloc[bar_idx]
            local_atr = atr.iloc[bar_idx]

            bar_close = df["close"].iloc[bar_idx]

            for lv in active_levels:
                if lv.broken:
                    continue
                # 突破判定 (用收盤價，影線刺穿不算)
                if lv.side == LevelSide.SUPPORT:
                    if bar_close < lv.price - local_atr * 0.3:
                        lv.broken = True
                else:
                    if bar_close > lv.price + local_atr * 0.3:
                        lv.broken = True
                # 測試判定 (EQ) — 必須隔 5 根以上且真正碰到
                if not lv.tested:
                    if lv.side == LevelSide.SUPPORT:
                        if bar_low <= lv.price + local_atr * EQ_CONFIG["fresh_tolerance_atr_ratio"]:
                            if bar_idx > lv.bar_index + 5:
                                lv.tested = True
                    else:
                        if bar_high >= lv.price - local_atr * EQ_CONFIG["fresh_tolerance_atr_ratio"]:
                            if bar_idx > lv.bar_index + 5:
                                lv.tested = True

            # 清理：移除已突破和太舊的水平
            active_levels = [
                lv for lv in active_levels
                if not lv.broken and (bar_idx - lv.bar_index) < 500
            ]
            # 限制數量
            if len(active_levels) > max_levels:
                active_levels = active_levels[-max_levels:]

            # --- 3. 更新持倉 ---
            for pos in list(self.risk_manager.positions):
                result = self.risk_manager.update_position(
                    pos, bar_idx,
                    bar_high, bar_low,
                    df["close"].iloc[bar_idx],
                    local_atr,
                )
                if result:
                    self._log_trade(pos, df)
                    # 止損後啟動冷卻 (連虧越多冷卻越久)
                    if "止損" in result:
                        consec_losses += 1
                        actual_cooldown = cooldown_bars * min(consec_losses, 3)
                        cooldown_until = bar_idx + actual_cooldown
                    else:
                        consec_losses = 0  # 非止損出場重置

            # --- 4. 進場檢查 (只在沒有持倉時) ---
            if not self.risk_manager.positions and bar_idx > cooldown_until:
                # EMA 趨勢方向 (需要足夠間距才算有方向)
                ema_gap = abs(ema50.iloc[bar_idx] - ema200.iloc[bar_idx])
                ema_threshold = local_atr * 0.5  # EMA 間距至少 0.5 ATR
                ema_bull = ema50.iloc[bar_idx] > ema200.iloc[bar_idx] and ema_gap >= ema_threshold
                ema_bear = ema50.iloc[bar_idx] < ema200.iloc[bar_idx] and ema_gap >= ema_threshold
                # 中性 = 兩條 EMA 纏繞在一起

                sig = self._find_entry(
                    df, bar_idx, atr, active_levels,
                    use_eq_filter, use_cc_filter, mtf_bias,
                    ema_bull, ema_bear, adx,
                )
                if sig:
                    total_signals += 1
                    self.signals.append(sig)

                    slippage = local_atr * slippage_ratio
                    pos = self.risk_manager.open_position(sig, slippage)
                    commission_cost = pos.entry_price * pos.size * commission
                    self.risk_manager.balance -= commission_cost

            # 記錄權益曲線
            self.equity_curve.append(self.risk_manager.balance)

        # 關閉未平倉
        for pos in list(self.risk_manager.positions):
            self.risk_manager._close_position(
                pos, df["close"].iloc[-1], len(df) - 1, "回測結束平倉"
            )
            self._log_trade(pos, df)

        print(f"  偵測到 {len(active_levels)} 個有效水平")
        print(f"  產生 {total_signals} 個交易訊號")

        stats = self.risk_manager.get_stats()
        self._print_results(stats)

        # 診斷: 印出每筆交易
        if self.trade_log:
            print("\n[診斷] 交易明細:")
            for i, t in enumerate(self.trade_log, 1):
                direction = "多" if t["direction"] == "long" else "空"
                be = "Y" if t["breakeven_moved"] else "N"
                print(
                    f"  #{i:2d} {direction} {t['level_type']:10s} "
                    f"進={t['entry_price']:>10.2f} 出={t['exit_price']:>10.2f} "
                    f"SL={t['stop_loss']:>10.2f} TP={t['take_profit']:>10.2f} "
                    f"R={t['r_result']:+.2f} BE={be} {t['exit_reason']}"
                )
            print()

        return stats

    def _find_entry(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        atr: pd.Series,
        levels: list[Level],
        use_eq: bool,
        use_cc: bool,
        mtf_bias: MarketBias,
        ema_bull: bool = True,
        ema_bear: bool = False,
        adx: Optional[pd.Series] = None,
    ) -> Optional[Signal]:
        """在當前 bar 尋找最佳進場機會。"""
        local_atr = atr.iloc[bar_idx]
        proximity = local_atr * 0.5
        bar_high = df["high"].iloc[bar_idx]
        bar_low = df["low"].iloc[bar_idx]

        # ADX 過濾：盤整市場不進場
        adx_val: Optional[float] = None
        if adx is not None:
            adx_val = float(adx.iloc[bar_idx])
            adx_min = ADX_CONFIG["min_trend_threshold"]
            if adx_val < adx_min:
                return None  # ADX 太低，市場無方向，跳過

        best_signal: Optional[Signal] = None
        best_strength = 0.0

        for lv in levels:
            if lv.broken:
                continue
            if lv.bar_index >= bar_idx:
                continue

            # EQ 過濾：只做新鮮水平
            if use_eq and lv.tested:
                continue

            # 水平必須至少存在 3 根 K 線 (避免剛形成的水平立即進場)
            if bar_idx - lv.bar_index < 3:
                continue

            # 接近水平？
            near = False
            if lv.side == LevelSide.SUPPORT and bar_low <= lv.price + proximity:
                near = True
            elif lv.side == LevelSide.RESISTANCE and bar_high >= lv.price - proximity:
                near = True

            if not near:
                continue

            # CC 過濾：沒有 CC 就不進場
            if use_cc and not _has_cc(lv, df, bar_idx, atr):
                continue

            # MTF 方向過濾
            if mtf_bias == MarketBias.BULLISH and lv.side == LevelSide.RESISTANCE:
                continue
            if mtf_bias == MarketBias.BEARISH and lv.side == LevelSide.SUPPORT:
                continue

            # EMA 趨勢過濾：順勢交易
            # 多頭趨勢 (EMA50 > EMA200) → 只做多
            # 空頭趨勢 (EMA50 < EMA200) → 只做空
            # 中性 (EMA 纏繞) → 只做高強度水平 (breakout)
            if ema_bull and lv.side == LevelSide.RESISTANCE:
                continue  # 多頭趨勢不做空
            if ema_bear and lv.side == LevelSide.SUPPORT:
                continue  # 空頭趨勢不做多
            if not ema_bull and not ema_bear:
                # 中性市場：只做 breakout 水平 (strength >= 0.85)
                if lv.level_type != LevelType.BREAKOUT:
                    continue

            # 嘗試 DC 進場
            sig = check_dc_entry(lv, df, bar_idx, atr)
            if sig and lv.strength > best_strength:
                best_signal = sig
                best_strength = lv.strength

        return best_signal

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
            "notes": pos.signal.notes,
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
