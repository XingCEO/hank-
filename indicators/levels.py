"""
四大關鍵水平偵測引擎
====================
根據 SNR 2020 / SR 2.0 策略偵測：
1. Classic  - 經典支撐/阻力 (帶 gap/miss 的動態水平)
2. Breakout - 強力突破水平
3. Gap      - 同色 K 線間缺口
4. HNS      - 頭肩形態頸線
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, cast

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import LEVELS_CONFIG
from data.fetcher import compute_atr, compute_body


# ============================================================
# 資料結構
# ============================================================

class LevelType(Enum):
    CLASSIC = "classic"
    BREAKOUT = "breakout"
    GAP = "gap"
    HNS = "hns"


class LevelSide(Enum):
    SUPPORT = "support"
    RESISTANCE = "resistance"


@dataclass
class Level:
    """一個支撐/阻力水平"""
    price: float                                # 水平價格
    level_type: LevelType                       # 類型
    side: LevelSide                             # 支撐或阻力
    bar_index: int                              # 形成時的 K 線索引
    timestamp: Optional[pd.Timestamp] = None    # 形成時間
    strength: float = 1.0                       # 強度評分 (0~1)
    tested: bool = False                        # 是否已被測試過
    broken: bool = False                        # 是否已被突破
    extra: dict = field(default_factory=dict)   # 額外資訊

    def __repr__(self):
        return (
            f"Level({self.level_type.value} {self.side.value} "
            f"@ {self.price:.2f}, bar={self.bar_index}, "
            f"strength={self.strength:.2f})"
        )


def _get_timestamp(df: pd.DataFrame, idx: int) -> Optional[pd.Timestamp]:
    """安全取得 DataFrame index 的 Timestamp"""
    if isinstance(df.index, pd.DatetimeIndex):
        return cast(pd.Timestamp, df.index[idx])
    return None


# ============================================================
# Swing High / Low 偵測
# ============================================================

def find_swing_highs(df: pd.DataFrame, lookback: int = 5) -> list[int]:
    """
    找出 swing high 的 bar index。
    swing high = 該 bar 的 high 高於前後 lookback 根的 high。
    """
    highs = np.asarray(df["high"].values)
    swings = []
    for i in range(lookback, len(highs) - lookback):
        window = highs[i - lookback: i + lookback + 1]
        if highs[i] == float(np.max(window)) and int(np.sum(window == highs[i])) == 1:
            swings.append(i)
    return swings


def find_swing_lows(df: pd.DataFrame, lookback: int = 5) -> list[int]:
    """
    找出 swing low 的 bar index。
    """
    lows = np.asarray(df["low"].values)
    swings = []
    for i in range(lookback, len(lows) - lookback):
        window = lows[i - lookback: i + lookback + 1]
        if lows[i] == float(np.min(window)) and int(np.sum(window == lows[i])) == 1:
            swings.append(i)
    return swings


# ============================================================
# 1. Classic 水平偵測
# ============================================================

def detect_classic_levels(df: pd.DataFrame, atr: pd.Series) -> list[Level]:
    """
    偵測經典支撐/阻力水平。
    條件：swing high/low 處有 gap 或 miss (延遲回彈)。

    Gap: 回彈時 K 線之間有價格空隙
    Miss: K 線影線未觸及水平 (略微不到)
    """
    cfg = LEVELS_CONFIG
    lookback = cfg["swing_lookback"]
    gap_ratio = cfg["classic_gap_atr_ratio"]

    df = compute_body(df)
    levels = []

    # 偵測阻力 (swing high)
    swing_highs = find_swing_highs(df, lookback)
    for idx in swing_highs:
        price = df["high"].iloc[idx]
        local_atr = atr.iloc[idx]
        min_gap = local_atr * gap_ratio

        # 檢查後續 K 線是否有 gap/miss 式回彈
        has_reaction = False
        for j in range(idx + 1, min(idx + lookback * 2, len(df))):
            # 價格接近水平但未觸碰 (miss) 或有 gap
            if df["high"].iloc[j] >= price - min_gap and df["high"].iloc[j] < price:
                # miss: 價格幾乎碰到但差一點
                has_reaction = True
                break
            if j > idx + 1:
                gap = df["low"].iloc[j] - df["high"].iloc[j - 1]
                if gap > min_gap and df["high"].iloc[j - 1] >= price - local_atr:
                    has_reaction = True
                    break

        if has_reaction:
            levels.append(Level(
                price=price,
                level_type=LevelType.CLASSIC,
                side=LevelSide.RESISTANCE,
                bar_index=idx,
                timestamp=_get_timestamp(df, idx),
                strength=0.7,
            ))

    # 偵測支撐 (swing low)
    swing_lows = find_swing_lows(df, lookback)
    for idx in swing_lows:
        price = df["low"].iloc[idx]
        local_atr = atr.iloc[idx]
        min_gap = local_atr * gap_ratio

        has_reaction = False
        for j in range(idx + 1, min(idx + lookback * 2, len(df))):
            if df["low"].iloc[j] <= price + min_gap and df["low"].iloc[j] > price:
                has_reaction = True
                break
            if j > idx + 1:
                gap = df["low"].iloc[j - 1] - df["high"].iloc[j]
                if gap > min_gap and df["low"].iloc[j - 1] <= price + local_atr:
                    has_reaction = True
                    break

        if has_reaction:
            levels.append(Level(
                price=price,
                level_type=LevelType.CLASSIC,
                side=LevelSide.SUPPORT,
                bar_index=idx,
                timestamp=_get_timestamp(df, idx),
                strength=0.7,
            ))

    return levels


# ============================================================
# 2. Breakout 水平偵測
# ============================================================

def detect_breakout_levels(df: pd.DataFrame, atr: pd.Series) -> list[Level]:
    """
    偵測突破水平。
    條件：
    1. 強力突破 K 線 (實體 > ATR * ratio)
    2. 突破缺口兩側乾淨 (前後 N 根無干擾)
    """
    cfg = LEVELS_CONFIG
    body_ratio = cfg["breakout_body_atr_ratio"]
    clean_bars = cfg["breakout_clean_bars"]

    df = compute_body(df)
    levels = []

    for i in range(clean_bars, len(df) - clean_bars):
        local_atr = atr.iloc[i]
        body = df["body_size"].iloc[i]

        if body < local_atr * body_ratio:
            continue

        is_bull = df["is_bullish"].iloc[i]

        if is_bull:
            # 向上突破: 突破前的 K 線高點 < 突破 K 線的 body_bottom
            breakout_price = df["body_bottom"].iloc[i]
            pre_highs = df["high"].iloc[i - clean_bars: i].values
            post_lows = df["low"].iloc[i + 1: i + 1 + clean_bars].values

            # 突破前乾淨: 前面的高點都低於突破水平
            pre_clean = all(h < breakout_price + local_atr * 0.2 for h in pre_highs)
            # 突破後乾淨: 後面的低點都高於突破水平
            post_clean = (
                len(post_lows) > 0
                and all(lo > breakout_price - local_atr * 0.3 for lo in post_lows)
            )

            if pre_clean and post_clean:
                levels.append(Level(
                    price=breakout_price,
                    level_type=LevelType.BREAKOUT,
                    side=LevelSide.SUPPORT,  # 突破後變支撐
                    bar_index=i,
                    timestamp=_get_timestamp(df, i),
                    strength=0.8,
                ))
        else:
            # 向下突破
            breakout_price = df["body_top"].iloc[i]
            pre_lows = df["low"].iloc[i - clean_bars: i].values
            post_highs = df["high"].iloc[i + 1: i + 1 + clean_bars].values

            pre_clean = all(lo > breakout_price - local_atr * 0.2 for lo in pre_lows)
            post_clean = (
                len(post_highs) > 0
                and all(h < breakout_price + local_atr * 0.3 for h in post_highs)
            )

            if pre_clean and post_clean:
                levels.append(Level(
                    price=breakout_price,
                    level_type=LevelType.BREAKOUT,
                    side=LevelSide.RESISTANCE,  # 突破後變阻力
                    bar_index=i,
                    timestamp=_get_timestamp(df, i),
                    strength=0.8,
                ))

    return levels


# ============================================================
# 3. Gap 水平偵測
# ============================================================

def detect_gap_levels(df: pd.DataFrame, atr: pd.Series) -> list[Level]:
    """
    偵測缺口水平。
    條件：同色連續 K 線之間存在價格空隙。
    缺口 = 前一根 K 線的 close 與下一根 K 線的 open 之間的空間。
    """
    cfg = LEVELS_CONFIG
    min_ratio = cfg["gap_min_atr_ratio"]

    df = compute_body(df)
    levels = []

    for i in range(1, len(df)):
        local_atr = atr.iloc[i]
        min_gap = local_atr * min_ratio

        same_color = df["is_bullish"].iloc[i] == df["is_bullish"].iloc[i - 1]
        if not same_color:
            continue

        if df["is_bullish"].iloc[i]:
            # 多頭缺口: 前一根 close < 當前 open
            gap = df["open"].iloc[i] - df["close"].iloc[i - 1]
            if gap >= min_gap:
                gap_price = (df["close"].iloc[i - 1] + df["open"].iloc[i]) / 2
                levels.append(Level(
                    price=gap_price,
                    level_type=LevelType.GAP,
                    side=LevelSide.SUPPORT,
                    bar_index=i,
                    timestamp=_get_timestamp(df, i),
                    strength=0.6,
                    extra={"gap_top": df["open"].iloc[i], "gap_bottom": df["close"].iloc[i - 1]},
                ))
        else:
            # 空頭缺口: 前一根 close > 當前 open
            gap = df["close"].iloc[i - 1] - df["open"].iloc[i]
            if gap >= min_gap:
                gap_price = (df["close"].iloc[i - 1] + df["open"].iloc[i]) / 2
                levels.append(Level(
                    price=gap_price,
                    level_type=LevelType.GAP,
                    side=LevelSide.RESISTANCE,
                    bar_index=i,
                    timestamp=_get_timestamp(df, i),
                    strength=0.6,
                    extra={"gap_top": df["close"].iloc[i - 1], "gap_bottom": df["open"].iloc[i]},
                ))

    return levels


# ============================================================
# 4. HNS (Head & Shoulders) 水平偵測
# ============================================================

def detect_hns_levels(df: pd.DataFrame, atr: pd.Series) -> list[Level]:
    """
    偵測頭肩形態的頸線水平。
    看漲反轉 (inverse HNS): 在底部找 左肩低 → 頭部更低 → 右肩低
    看跌反轉 (HNS top): 在頂部找 左肩高 → 頭部更高 → 右肩高
    """
    cfg = LEVELS_CONFIG
    lookback = cfg["swing_lookback"]
    shoulder_ratio = cfg["hns_shoulder_ratio"]
    neckline_tol = cfg["hns_neckline_tolerance"]

    levels = []

    # --- HNS Top (看跌) ---
    swing_highs = find_swing_highs(df, lookback)
    for i in range(1, len(swing_highs) - 1):
        left_idx = swing_highs[i - 1]
        head_idx = swing_highs[i]
        right_idx = swing_highs[i + 1]

        left_h = df["high"].iloc[left_idx]
        head_h = df["high"].iloc[head_idx]
        right_h = df["high"].iloc[right_idx]

        local_atr = atr.iloc[head_idx]

        # 頭部必須高於兩肩
        if head_h <= left_h or head_h <= right_h:
            continue

        # 兩肩高度差不能太大
        shoulder_diff = abs(left_h - right_h)
        if shoulder_diff > local_atr * (1 - shoulder_ratio):
            continue

        # 頭部必須明顯高於肩部
        min_shoulder = min(left_h, right_h)
        if (head_h - min_shoulder) < local_atr * shoulder_ratio:
            continue

        # 計算頸線：左肩與頭部之間的低點、頭部與右肩之間的低點
        left_neck_region = df["low"].iloc[left_idx:head_idx]
        right_neck_region = df["low"].iloc[head_idx:right_idx]

        if len(left_neck_region) == 0 or len(right_neck_region) == 0:
            continue

        left_neck = left_neck_region.min()
        right_neck = right_neck_region.min()

        # 頸線兩點要接近
        if abs(left_neck - right_neck) > local_atr * neckline_tol:
            continue

        neckline_price = (left_neck + right_neck) / 2

        levels.append(Level(
            price=neckline_price,
            level_type=LevelType.HNS,
            side=LevelSide.SUPPORT,  # 頸線被突破後變阻力，但先標為支撐
            bar_index=right_idx,
            timestamp=_get_timestamp(df, right_idx),
            strength=0.9,
            extra={
                "pattern": "hns_top",
                "left_shoulder": left_idx,
                "head": head_idx,
                "right_shoulder": right_idx,
            },
        ))

    # --- Inverse HNS (看漲) ---
    swing_lows = find_swing_lows(df, lookback)
    for i in range(1, len(swing_lows) - 1):
        left_idx = swing_lows[i - 1]
        head_idx = swing_lows[i]
        right_idx = swing_lows[i + 1]

        left_l = df["low"].iloc[left_idx]
        head_l = df["low"].iloc[head_idx]
        right_l = df["low"].iloc[right_idx]

        local_atr = atr.iloc[head_idx]

        # 頭部必須低於兩肩
        if head_l >= left_l or head_l >= right_l:
            continue

        shoulder_diff = abs(left_l - right_l)
        if shoulder_diff > local_atr * (1 - shoulder_ratio):
            continue

        max_shoulder = max(left_l, right_l)
        if (max_shoulder - head_l) < local_atr * shoulder_ratio:
            continue

        left_neck_region = df["high"].iloc[left_idx:head_idx]
        right_neck_region = df["high"].iloc[head_idx:right_idx]

        if len(left_neck_region) == 0 or len(right_neck_region) == 0:
            continue

        left_neck = left_neck_region.max()
        right_neck = right_neck_region.max()

        if abs(left_neck - right_neck) > local_atr * neckline_tol:
            continue

        neckline_price = (left_neck + right_neck) / 2

        levels.append(Level(
            price=neckline_price,
            level_type=LevelType.HNS,
            side=LevelSide.RESISTANCE,
            bar_index=right_idx,
            timestamp=_get_timestamp(df, right_idx),
            strength=0.9,
            extra={
                "pattern": "inverse_hns",
                "left_shoulder": left_idx,
                "head": head_idx,
                "right_shoulder": right_idx,
            },
        ))

    return levels


# ============================================================
# 合併 & 匯出
# ============================================================

def merge_nearby_levels(levels: list[Level], atr_value: float) -> list[Level]:
    """
    合併距離太近的水平，保留強度較高的。
    """
    merge_dist = atr_value * LEVELS_CONFIG["merge_distance_atr_ratio"]

    if not levels:
        return levels

    # 按價格排序
    sorted_levels = sorted(levels, key=lambda lv: lv.price)
    merged = [sorted_levels[0]]

    for lv in sorted_levels[1:]:
        if abs(lv.price - merged[-1].price) < merge_dist:
            # 保留強度較高的
            if lv.strength > merged[-1].strength:
                merged[-1] = lv
        else:
            merged.append(lv)

    return merged


def detect_all_levels(df: pd.DataFrame) -> list[Level]:
    """
    偵測所有類型的水平並合併。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 數據

    Returns
    -------
    list[Level]
        合併後的水平清單
    """
    atr = compute_atr(df)

    classic = detect_classic_levels(df, atr)
    breakout = detect_breakout_levels(df, atr)
    gap = detect_gap_levels(df, atr)
    hns = detect_hns_levels(df, atr)

    all_levels = classic + breakout + gap + hns

    # 用最新 ATR 值做合併
    current_atr = atr.iloc[-1]
    merged = merge_nearby_levels(all_levels, current_atr)

    print(f"  偵測到水平: Classic={len(classic)}, Breakout={len(breakout)}, "
          f"Gap={len(gap)}, HNS={len(hns)} → 合併後 {len(merged)} 個")

    return merged
