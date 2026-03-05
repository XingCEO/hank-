"""
進場策略模組
============
實現三種進場方式：
1. DC (Double Confirmation) - 雙重確認進場
2. DE (Direct Entry) - 直接進場
3. 趨勢線進場

包含 Setup/DE 組合矩陣查表邏輯。
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import ENTRY_CONFIG, RISK_CONFIG
from data.fetcher import compute_atr, compute_body
from indicators.levels import Level, LevelType, LevelSide
from indicators.cc_filter import ConfirmationCandle, CCType


class EntryType(Enum):
    DC = "double_confirmation"
    DE = "direct_entry"
    TRENDLINE = "trendline_entry"
    REACTION = "reaction_entry"


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Signal:
    """交易訊號"""
    entry_type: EntryType
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    bar_index: int
    timestamp: Optional[pd.Timestamp] = None
    level: Optional[Level] = None
    confidence: float = 0.5
    r_multiple: float = 1.0
    notes: str = ""

    @property
    def risk(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def reward(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @property
    def rr_ratio(self) -> float:
        r = self.risk
        return self.reward / r if r > 0 else 0.0

    def __repr__(self):
        return (
            f"Signal({self.entry_type.value} {self.direction.value} "
            f"@ {self.entry_price:.2f}, SL={self.stop_loss:.2f}, "
            f"TP={self.take_profit:.2f}, RR={self.rr_ratio:.1f})"
        )


# ============================================================
# Setup / DE 組合矩陣 (策略文件的查表邏輯)
# ============================================================
# True = Direct Entry, False = Setup (需要 DC 確認)
_COMBO_MATRIX = {
    # (水平類型, DE 類型) → 是否直接進場
    (LevelType.GAP, LevelType.GAP): False,
    (LevelType.GAP, LevelType.BREAKOUT): False,
    (LevelType.GAP, LevelType.HNS): True,

    (LevelType.BREAKOUT, LevelType.GAP): False,      # BOCC 行
    (LevelType.BREAKOUT, LevelType.BREAKOUT): True,
    (LevelType.BREAKOUT, LevelType.HNS): True,

    (LevelType.CLASSIC, LevelType.GAP): False,
    (LevelType.CLASSIC, LevelType.BREAKOUT): True,
    (LevelType.CLASSIC, LevelType.HNS): False,

    (LevelType.HNS, LevelType.GAP): False,
    (LevelType.HNS, LevelType.BREAKOUT): False,
    (LevelType.HNS, LevelType.HNS): True,
}


def should_direct_entry(level_type: LevelType, de_type: LevelType) -> bool:
    """
    查表判斷是否可以直接進場。
    """
    return _COMBO_MATRIX.get((level_type, de_type), False)


# ============================================================
# DC (Double Confirmation) 進場
# ============================================================

def check_dc_entry(
    level: Level,
    df: pd.DataFrame,
    bar_index: int,
    atr: pd.Series,
) -> Optional[Signal]:
    """
    雙重確認進場：K 線收盤確認後進場。
    - BUY: 多頭蠟燭收盤於支撐之上
    - SELL: 空頭蠟燭收盤於阻力之下

    Parameters
    ----------
    level : Level
        觸發的水平
    df : pd.DataFrame
        K 線數據
    bar_index : int
        當前 bar index
    atr : pd.Series
        ATR 序列
    """
    if bar_index < 1 or bar_index >= len(df):
        return None

    price = level.price
    local_atr = atr.iloc[bar_index]
    sl_buffer = local_atr * RISK_CONFIG["sl_buffer_atr_ratio"]

    curr_close = df["close"].iloc[bar_index]
    curr_open = df["open"].iloc[bar_index]
    is_bullish = curr_close > curr_open

    if level.side == LevelSide.SUPPORT:
        # 支撐處做多: 多頭 K 線收盤在支撐上方
        if not is_bullish:
            return None
        if curr_close <= price:
            return None
        # 確認: K 線低點要接近水平
        if df["low"].iloc[bar_index] > price + local_atr * 0.5:
            return None

        entry_price = curr_close
        stop_loss = price - sl_buffer
        target_r = RISK_CONFIG["default_target_r"]
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * target_r

        return Signal(
            entry_type=EntryType.DC,
            direction=TradeDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bar_index=bar_index,
            timestamp=cast(pd.Timestamp, df.index[bar_index]) if isinstance(df.index, pd.DatetimeIndex) else None,
            level=level,
            confidence=0.65,
            r_multiple=target_r,
            notes=f"DC Long @ {level.level_type.value} support",
        )

    else:
        # 阻力處做空
        if is_bullish:
            return None
        if curr_close >= price:
            return None
        if df["high"].iloc[bar_index] < price - local_atr * 0.5:
            return None

        entry_price = curr_close
        stop_loss = price + sl_buffer
        target_r = RISK_CONFIG["default_target_r"]
        risk = stop_loss - entry_price
        take_profit = entry_price - risk * target_r

        return Signal(
            entry_type=EntryType.DC,
            direction=TradeDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bar_index=bar_index,
            timestamp=cast(pd.Timestamp, df.index[bar_index]) if isinstance(df.index, pd.DatetimeIndex) else None,
            level=level,
            confidence=0.65,
            r_multiple=target_r,
            notes=f"DC Short @ {level.level_type.value} resistance",
        )


# ============================================================
# DE (Direct Entry) 進場
# ============================================================

def check_de_entry(
    level: Level,
    df: pd.DataFrame,
    bar_index: int,
    atr: pd.Series,
    de_level_type: Optional[LevelType] = None,
) -> Optional[Signal]:
    """
    直接進場：價格觸碰水平即進場 (不需等收盤確認)。
    需要通過組合矩陣檢查。
    """
    if bar_index >= len(df):
        return None

    # 如果提供了 DE 類型，檢查組合矩陣
    if de_level_type is not None:
        if not should_direct_entry(level.level_type, de_level_type):
            return None

    price = level.price
    local_atr = atr.iloc[bar_index]
    sl_buffer = local_atr * RISK_CONFIG["sl_buffer_atr_ratio"]
    proximity = local_atr * 0.3

    if level.side == LevelSide.SUPPORT:
        # 價格觸碰支撐
        if df["low"].iloc[bar_index] > price + proximity:
            return None

        entry_price = price
        stop_loss = price - sl_buffer - local_atr * 0.2
        target_r = RISK_CONFIG["default_target_r"]
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * target_r

        return Signal(
            entry_type=EntryType.DE,
            direction=TradeDirection.LONG,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bar_index=bar_index,
            timestamp=cast(pd.Timestamp, df.index[bar_index]) if isinstance(df.index, pd.DatetimeIndex) else None,
            level=level,
            confidence=0.55,
            r_multiple=target_r,
            notes=f"DE Long @ {level.level_type.value}",
        )

    else:
        if df["high"].iloc[bar_index] < price - proximity:
            return None

        entry_price = price
        stop_loss = price + sl_buffer + local_atr * 0.2
        target_r = RISK_CONFIG["default_target_r"]
        risk = stop_loss - entry_price
        take_profit = entry_price - risk * target_r

        return Signal(
            entry_type=EntryType.DE,
            direction=TradeDirection.SHORT,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            bar_index=bar_index,
            timestamp=cast(pd.Timestamp, df.index[bar_index]) if isinstance(df.index, pd.DatetimeIndex) else None,
            level=level,
            confidence=0.55,
            r_multiple=target_r,
            notes=f"DE Short @ {level.level_type.value}",
        )


# ============================================================
# 訊號產生器 (整合所有進場邏輯)
# ============================================================

def generate_signals(
    df: pd.DataFrame,
    levels: list[Level],
    cc_results: Optional[list] = None,
) -> list[Signal]:
    """
    掃描 K 線數據，在每個水平附近檢查是否有進場訊號。

    Parameters
    ----------
    df : pd.DataFrame
        K 線數據
    levels : list[Level]
        已偵測的水平
    cc_results : list
        CC 過濾結果 [(Level, [CC, ...]), ...]

    Returns
    -------
    list[Signal]
        產生的交易訊號
    """
    atr = compute_atr(df)
    signals = []

    # 建立有 CC 確認的水平集合
    cc_level_set = set()
    if cc_results:
        for lv, _ in cc_results:
            cc_level_set.add(id(lv))

    for bar_idx in range(20, len(df)):
        for level in levels:
            # 跳過已被突破的水平
            if level.broken:
                continue

            # 跳過還沒形成的水平
            if level.bar_index >= bar_idx:
                continue

            local_atr = atr.iloc[bar_idx]
            price = level.price

            # 檢查價格是否接近水平
            bar_high = df["high"].iloc[bar_idx]
            bar_low = df["low"].iloc[bar_idx]
            proximity = local_atr * 0.5

            near_level = False
            if level.side == LevelSide.SUPPORT and bar_low <= price + proximity:
                near_level = True
            elif level.side == LevelSide.RESISTANCE and bar_high >= price - proximity:
                near_level = True

            if not near_level:
                continue

            # 有 CC → 嘗試 DC 進場
            if id(level) in cc_level_set:
                sig = check_dc_entry(level, df, bar_idx, atr)
                if sig:
                    sig.confidence += 0.1  # CC 加分
                    signals.append(sig)
                    continue

            # 嘗試 DE 進場
            sig = check_de_entry(level, df, bar_idx, atr)
            if sig:
                signals.append(sig)
                continue

            # 嘗試 DC 進場 (沒有 CC 也可以嘗試)
            sig = check_dc_entry(level, df, bar_idx, atr)
            if sig:
                signals.append(sig)

    # 去重：同一 bar 只保留最好的訊號
    signals = _deduplicate_signals(signals)

    print(f"  產生 {len(signals)} 個交易訊號")
    return signals


def _deduplicate_signals(signals: list[Signal]) -> list[Signal]:
    """同一 bar 只保留信心度最高的訊號"""
    by_bar = {}
    for sig in signals:
        key = sig.bar_index
        if key not in by_bar or sig.confidence > by_bar[key].confidence:
            by_bar[key] = sig
    return sorted(by_bar.values(), key=lambda s: s.bar_index)
