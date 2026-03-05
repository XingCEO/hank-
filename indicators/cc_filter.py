"""
CC (Confirmation Candle) 過濾器
================================
在關鍵水平附近尋找確認 K 線，增加進場信心。
四種 CC 類型：
1. Classic CC  - 水平創立或測試後的確認
2. BOCC1       - 突破前的確認
3. BOCC2       - 突破後的確認
4. HNS CC      - 頭肩形態的確認
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, cast

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import CC_CONFIG
from data.fetcher import compute_atr, compute_body
from indicators.levels import Level, LevelType, LevelSide


class CCType(Enum):
    CLASSIC = "classic_cc"
    BOCC1 = "bocc1"       # 突破前確認
    BOCC2 = "bocc2"       # 突破後確認
    HNS_CC = "hns_cc"


def _get_ts(df: pd.DataFrame, idx: int) -> Optional[pd.Timestamp]:
    """安全取得 DataFrame index 的 Timestamp"""
    if isinstance(df.index, pd.DatetimeIndex):
        return cast(pd.Timestamp, df.index[idx])
    return None


@dataclass
class ConfirmationCandle:
    """確認 K 線"""
    cc_type: CCType
    bar_index: int
    timestamp: Optional[pd.Timestamp] = None
    direction: str = "bullish"  # "bullish" or "bearish"
    strength: float = 0.5

    def __repr__(self):
        return f"CC({self.cc_type.value} {self.direction} @ bar {self.bar_index})"


def _is_strong_candle(
    df: pd.DataFrame,
    idx: int,
    atr: pd.Series,
    min_body_ratio: float,
) -> bool:
    """判斷 K 線實體是否夠大 (強力確認)"""
    body = abs(df["close"].iloc[idx] - df["open"].iloc[idx])
    return body >= atr.iloc[idx] * min_body_ratio


def find_classic_cc(
    level: Level,
    df: pd.DataFrame,
    atr: pd.Series,
) -> list[ConfirmationCandle]:
    """
    經典 CC：水平形成後，在水平附近出現方向一致的確認 K 線。
    - 支撐水平 → 找多頭 CC
    - 阻力水平 → 找空頭 CC
    """
    cfg = CC_CONFIG
    proximity = atr.iloc[level.bar_index] * cfg["proximity_atr_ratio"]
    max_lookback = cfg["max_lookback_bars"]
    min_body = cfg["min_body_atr_ratio"]

    ccs = []
    start = level.bar_index + 1
    end = min(start + max_lookback, len(df))

    for i in range(start, end):
        local_atr = atr.iloc[i]
        price = level.price

        # K 線是否在水平附近
        bar_close = df["close"].iloc[i]
        bar_open = df["open"].iloc[i]

        if level.side == LevelSide.SUPPORT:
            # 支撐: K 線低點要接近水平, 且是多頭 K 線
            if abs(df["low"].iloc[i] - price) <= proximity and bar_close > bar_open:
                if _is_strong_candle(df, i, atr, min_body):
                    ccs.append(ConfirmationCandle(
                        cc_type=CCType.CLASSIC,
                        bar_index=i,
                        timestamp=_get_ts(df, i),
                        direction="bullish",
                        strength=0.7,
                    ))
        else:
            # 阻力: K 線高點要接近水平, 且是空頭 K 線
            if abs(df["high"].iloc[i] - price) <= proximity and bar_close < bar_open:
                if _is_strong_candle(df, i, atr, min_body):
                    ccs.append(ConfirmationCandle(
                        cc_type=CCType.CLASSIC,
                        bar_index=i,
                        timestamp=_get_ts(df, i),
                        direction="bearish",
                        strength=0.7,
                    ))

    return ccs


def find_bocc(
    level: Level,
    df: pd.DataFrame,
    atr: pd.Series,
) -> list[ConfirmationCandle]:
    """
    BOCC1 (突破前確認) + BOCC2 (突破後確認)。
    只適用於 Breakout 類型的水平。

    BOCC1: 突破前，在水平另一側出現的確認 K 線 (方向可能相反)
    BOCC2: 突破後，回測水平時出現的確認 K 線
    """
    if level.level_type != LevelType.BREAKOUT:
        return []

    cfg = CC_CONFIG
    proximity = atr.iloc[level.bar_index] * cfg["proximity_atr_ratio"]
    max_lookback = cfg["max_lookback_bars"]
    min_body = cfg["min_body_atr_ratio"]

    ccs = []

    # --- BOCC1: 突破前 (往前看) ---
    pre_start = max(0, level.bar_index - max_lookback)
    for i in range(pre_start, level.bar_index):
        local_atr = atr.iloc[i]
        price = level.price
        bar_close = df["close"].iloc[i]
        bar_open = df["open"].iloc[i]

        if abs(df["close"].iloc[i] - price) <= proximity or abs(df["open"].iloc[i] - price) <= proximity:
            if _is_strong_candle(df, i, atr, min_body):
                direction = "bullish" if bar_close > bar_open else "bearish"
                ccs.append(ConfirmationCandle(
                    cc_type=CCType.BOCC1,
                    bar_index=i,
                    timestamp=_get_ts(df, i),
                    direction=direction,
                    strength=0.6,
                ))

    # --- BOCC2: 突破後 (往後看) ---
    post_start = level.bar_index + 1
    post_end = min(post_start + max_lookback, len(df))
    for i in range(post_start, post_end):
        local_atr = atr.iloc[i]
        price = level.price

        if level.side == LevelSide.SUPPORT:
            # 突破後變支撐, 回測時找多頭 CC
            if abs(df["low"].iloc[i] - price) <= proximity:
                bar_close = df["close"].iloc[i]
                bar_open = df["open"].iloc[i]
                if bar_close > bar_open and _is_strong_candle(df, i, atr, min_body):
                    ccs.append(ConfirmationCandle(
                        cc_type=CCType.BOCC2,
                        bar_index=i,
                        timestamp=_get_ts(df, i),
                        direction="bullish",
                        strength=0.8,
                    ))
        else:
            if abs(df["high"].iloc[i] - price) <= proximity:
                bar_close = df["close"].iloc[i]
                bar_open = df["open"].iloc[i]
                if bar_close < bar_open and _is_strong_candle(df, i, atr, min_body):
                    ccs.append(ConfirmationCandle(
                        cc_type=CCType.BOCC2,
                        bar_index=i,
                        timestamp=_get_ts(df, i),
                        direction="bearish",
                        strength=0.8,
                    ))

    return ccs


def find_hns_cc(
    level: Level,
    df: pd.DataFrame,
    atr: pd.Series,
) -> list[ConfirmationCandle]:
    """
    HNS CC：頭肩形態完成後 (右肩形成) 在頸線附近的確認 K 線。
    """
    if level.level_type != LevelType.HNS:
        return []

    cfg = CC_CONFIG
    proximity = atr.iloc[level.bar_index] * cfg["proximity_atr_ratio"]
    max_lookback = cfg["max_lookback_bars"]
    min_body = cfg["min_body_atr_ratio"]

    ccs = []
    start = level.bar_index + 1
    end = min(start + max_lookback, len(df))

    pattern = level.extra.get("pattern", "")

    for i in range(start, end):
        price = level.price

        if pattern == "hns_top":
            # 頂部頭肩 → 看跌, 頸線被突破後找空頭 CC
            if abs(df["high"].iloc[i] - price) <= proximity:
                bar_close = df["close"].iloc[i]
                bar_open = df["open"].iloc[i]
                if bar_close < bar_open and _is_strong_candle(df, i, atr, min_body):
                    ccs.append(ConfirmationCandle(
                        cc_type=CCType.HNS_CC,
                        bar_index=i,
                        timestamp=_get_ts(df, i),
                        direction="bearish",
                        strength=0.85,
                    ))
        elif pattern == "inverse_hns":
            # 底部反頭肩 → 看漲
            if abs(df["low"].iloc[i] - price) <= proximity:
                bar_close = df["close"].iloc[i]
                bar_open = df["open"].iloc[i]
                if bar_close > bar_open and _is_strong_candle(df, i, atr, min_body):
                    ccs.append(ConfirmationCandle(
                        cc_type=CCType.HNS_CC,
                        bar_index=i,
                        timestamp=_get_ts(df, i),
                        direction="bullish",
                        strength=0.85,
                    ))

    return ccs


def find_all_cc(
    level: Level,
    df: pd.DataFrame,
) -> list[ConfirmationCandle]:
    """
    對一個水平搜尋所有類型的 CC。
    """
    atr = compute_atr(df)
    ccs = []

    ccs.extend(find_classic_cc(level, df, atr))
    ccs.extend(find_bocc(level, df, atr))
    ccs.extend(find_hns_cc(level, df, atr))

    return ccs


def filter_levels_with_cc(
    levels: list[Level],
    df: pd.DataFrame,
) -> list[tuple[Level, list[ConfirmationCandle]]]:
    """
    為每個水平尋找 CC，回傳有 CC 確認的水平。

    Returns
    -------
    list[tuple[Level, list[ConfirmationCandle]]]
        (水平, CC 清單) 的配對
    """
    results = []

    for lv in levels:
        ccs = find_all_cc(lv, df)
        if ccs:
            # 有 CC → 強度加分
            best_cc = max(ccs, key=lambda c: c.strength)
            lv.strength = min(lv.strength + best_cc.strength * 0.2, 1.0)
            lv.extra["cc_count"] = len(ccs)
            lv.extra["cc_types"] = list(set(c.cc_type.value for c in ccs))
            results.append((lv, ccs))

    print(f"  CC 過濾: {len(levels)} 個水平 → {len(results)} 個有確認")
    return results
