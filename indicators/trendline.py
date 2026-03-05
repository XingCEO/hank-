"""
趨勢線自動擬合模組
==================
用 K 線實體自動擬合趨勢線，支援四種類型：
1. 延續型 (111)
2. 突破型 (11b2 / 1b22)
3. HNS 型
4. 隱藏 HNS 型

規則：
- 必須用 K 線實體畫線 (不能畫在影線上)
- 關鍵點距離盡量對稱
- 線條必須自然
- 時間框架必須與水平一致
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import TRENDLINE_CONFIG
from data.fetcher import compute_atr, compute_body


class TrendlineType(Enum):
    CONTINUATION = "continuation"  # 延續型 111
    BREAKOUT = "breakout"          # 突破型 11b2 / 1b22
    HNS = "hns"                    # HNS 型
    HIDDEN_HNS = "hidden_hns"     # 隱藏 HNS 型


class TrendlineDirection(Enum):
    UP = "up"
    DOWN = "down"


@dataclass
class Trendline:
    """一條趨勢線"""
    slope: float                           # 斜率 (每根 K 線的價格變化)
    intercept: float                       # 截距 (bar_index=0 的價格)
    direction: TrendlineDirection          # 方向
    touch_indices: list[int] = field(default_factory=list)  # 觸碰點的 bar index
    trendline_type: TrendlineType = TrendlineType.CONTINUATION
    r_squared: float = 0.0                # 擬合度
    strength: float = 0.0                 # 強度評分

    def price_at(self, bar_index: int) -> float:
        """計算趨勢線在某 bar 的價格"""
        return self.slope * bar_index + self.intercept

    def __repr__(self):
        return (
            f"Trendline({self.direction.value} {self.trendline_type.value}, "
            f"slope={self.slope:.4f}, touches={len(self.touch_indices)}, "
            f"R²={self.r_squared:.3f})"
        )


def _find_body_touches(
    df: pd.DataFrame,
    slope: float,
    intercept: float,
    atr: pd.Series,
    direction: TrendlineDirection,
    tolerance_ratio: float,
) -> list[int]:
    """
    找出 K 線實體觸碰趨勢線的 bar index。
    只看實體 (body_top / body_bottom)，不看影線。
    """
    touches = []

    for i in range(len(df)):
        line_price = slope * i + intercept
        tolerance = atr.iloc[i] * tolerance_ratio

        if direction == TrendlineDirection.UP:
            # 上升趨勢線：K 線實體下緣觸碰
            body_bottom = min(df["open"].iloc[i], df["close"].iloc[i])
            if abs(body_bottom - line_price) <= tolerance and body_bottom >= line_price - tolerance:
                touches.append(i)
        else:
            # 下降趨勢線：K 線實體上緣觸碰
            body_top = max(df["open"].iloc[i], df["close"].iloc[i])
            if abs(body_top - line_price) <= tolerance and body_top <= line_price + tolerance:
                touches.append(i)

    return touches


def _check_symmetry(touch_indices: list[int], max_deviation_pct: float) -> bool:
    """
    檢查觸碰點的間距是否對稱。
    """
    if len(touch_indices) < 3:
        return True  # 不夠多無法判斷對稱性

    gaps = [touch_indices[i + 1] - touch_indices[i] for i in range(len(touch_indices) - 1)]
    avg_gap = np.mean(gaps)

    if avg_gap == 0:
        return False

    deviations = [abs(g - avg_gap) / avg_gap * 100 for g in gaps]
    return bool(max(deviations) <= max_deviation_pct)


def detect_trendlines(
    df: pd.DataFrame,
    direction: Optional[TrendlineDirection] = None,
    min_bars: int = 20,
) -> list[Trendline]:
    """
    自動偵測趨勢線。

    使用滑動窗口 + 線性回歸在 K 線實體上擬合趨勢線。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 數據
    direction : TrendlineDirection or None
        只找指定方向，None = 兩個方向都找
    min_bars : int
        最小窗口大小

    Returns
    -------
    list[Trendline]
        偵測到的趨勢線
    """
    cfg = TRENDLINE_CONFIG
    min_touches = cfg["min_touches"]
    touch_tol = cfg["touch_tolerance_atr_ratio"]
    sym_max_dev = cfg["symmetry_max_deviation_pct"]

    df = compute_body(df)
    atr = compute_atr(df)
    trendlines = []

    directions_to_check = (
        [direction] if direction else
        [TrendlineDirection.UP, TrendlineDirection.DOWN]
    )

    for d in directions_to_check:
        # 選擇要擬合的 K 線邊緣
        if d == TrendlineDirection.UP:
            prices = df["body_bottom"].values  # 上升趨勢線碰實體下緣
        else:
            prices = df["body_top"].values      # 下降趨勢線碰實體上緣

        # 滑動窗口找趨勢線
        for start in range(0, len(df) - min_bars, max(1, min_bars // 4)):
            for end in range(start + min_bars, min(start + min_bars * 4, len(df)), min_bars // 2):
                window_x = np.arange(start, end)
                window_y = prices[start:end]

                if len(window_x) < min_bars:
                    continue

                # 線性回歸
                _slope, _intercept, _rval, _, _ = stats.linregress(window_x, window_y)
                slope_val: float = float(_slope)           # type: ignore[arg-type]
                intercept_val: float = float(_intercept)   # type: ignore[arg-type]
                r_squared: float = float(_rval) ** 2       # type: ignore[arg-type]

                # 基本方向檢查
                if d == TrendlineDirection.UP and slope_val <= 0:
                    continue
                if d == TrendlineDirection.DOWN and slope_val >= 0:
                    continue

                # 擬合度檢查
                if r_squared < 0.7:
                    continue

                # 找觸碰點
                touches = _find_body_touches(df, slope_val, intercept_val, atr, d, touch_tol)

                if len(touches) < min_touches:
                    continue

                # 對稱性檢查
                if not _check_symmetry(touches, sym_max_dev):
                    continue

                # 評分
                strength = (
                    r_squared * 0.3 +
                    min(len(touches) / 5, 1.0) * 0.4 +
                    (1.0 if _check_symmetry(touches, sym_max_dev * 0.5) else 0.5) * 0.3
                )

                trendlines.append(Trendline(
                    slope=slope_val,
                    intercept=intercept_val,
                    direction=d,
                    touch_indices=touches,
                    trendline_type=TrendlineType.CONTINUATION,
                    r_squared=r_squared,
                    strength=strength,
                ))

    # 去重：移除太相似的趨勢線
    trendlines = _deduplicate_trendlines(trendlines, atr)

    print(f"  偵測到 {len(trendlines)} 條趨勢線")
    return trendlines


def _deduplicate_trendlines(
    trendlines: list[Trendline],
    atr: pd.Series,
) -> list[Trendline]:
    """移除太相似的趨勢線，保留強度最高的"""
    if not trendlines:
        return trendlines

    # 按強度排序
    trendlines.sort(key=lambda t: t.strength, reverse=True)
    kept = [trendlines[0]]

    avg_atr = atr.mean()

    for tl in trendlines[1:]:
        is_duplicate = False
        for existing in kept:
            # 斜率相似 且 截距相近
            slope_diff = abs(tl.slope - existing.slope)
            intercept_diff = abs(tl.intercept - existing.intercept)

            if slope_diff < avg_atr * 0.001 and intercept_diff < avg_atr * 0.5:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(tl)

    return kept


def classify_trendline(tl: Trendline, df: pd.DataFrame) -> TrendlineType:
    """
    根據觸碰點的分布模式分類趨勢線類型。

    111:   所有觸碰在同一側 → 延續型
    11b2:  包含突破 → 突破型
    1bb11: 兩側多次測試 → HNS 型
    1b2b1: 交替觸碰 → 隱藏 HNS 型
    """
    touches = tl.touch_indices
    if len(touches) < 3:
        return TrendlineType.CONTINUATION

    # 判斷每個觸碰點的價格相對趨勢線的位置
    positions = []
    for idx in touches:
        line_price = tl.price_at(idx)
        actual_price = (df["open"].iloc[idx] + df["close"].iloc[idx]) / 2
        positions.append("above" if actual_price > line_price else "below")

    # 分析模式
    unique_positions = set(positions)

    if len(unique_positions) == 1:
        return TrendlineType.CONTINUATION

    # 檢查是否有明確的突破模式
    transitions = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i - 1])

    if transitions == 1:
        return TrendlineType.BREAKOUT
    elif transitions >= 3:
        return TrendlineType.HIDDEN_HNS
    else:
        return TrendlineType.HNS
