"""
EQ (Equilibrium) 過濾器
========================
篩選出跨時間框架「全新未被測試過」的支撐/阻力水平。
只有「新鮮」(從未被價格觸碰) 的水平才是 EQ。
"""

import pandas as pd
import numpy as np
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import EQ_CONFIG
from data.fetcher import compute_atr
from indicators.levels import Level, LevelSide


def is_level_fresh(
    level: Level,
    df: pd.DataFrame,
    atr: pd.Series,
    tolerance_ratio: Optional[float] = None,
) -> bool:
    """
    判斷一個水平是否「新鮮」(未被測試過)。

    新鮮 = 水平形成後，所有後續 K 線都沒有觸碰到該水平
    (觸碰 = 價格在 tolerance 範圍內)。

    Parameters
    ----------
    level : Level
        要檢查的水平
    df : pd.DataFrame
        K 線資料
    atr : pd.Series
        ATR 序列
    tolerance_ratio : float
        容忍距離 (佔 ATR 的比例)

    Returns
    -------
    bool
        True = 新鮮 (未測試), False = 已被測試
    """
    if tolerance_ratio is None:
        tolerance_ratio = EQ_CONFIG["fresh_tolerance_atr_ratio"]

    start_idx = level.bar_index + 1
    if start_idx >= len(df):
        return True  # 剛形成，還沒有後續 K 線

    price = level.price

    for i in range(start_idx, len(df)):
        local_atr = atr.iloc[i]
        tolerance = local_atr * tolerance_ratio

        bar_high = df["high"].iloc[i]
        bar_low = df["low"].iloc[i]

        # 如果 K 線的 high-low 範圍觸碰到水平的容忍區間
        level_top = price + tolerance
        level_bottom = price - tolerance

        if bar_low <= level_top and bar_high >= level_bottom:
            return False  # 被測試過了

    return True


def filter_eq_levels(
    levels: list[Level],
    df: pd.DataFrame,
) -> list[Level]:
    """
    過濾出符合 EQ 條件的水平 (新鮮未測試)。

    Parameters
    ----------
    levels : list[Level]
        待過濾的水平清單
    df : pd.DataFrame
        K 線資料

    Returns
    -------
    list[Level]
        通過 EQ 過濾的水平
    """
    atr = compute_atr(df)
    eq_levels = []

    for lv in levels:
        if is_level_fresh(lv, df, atr):
            lv.extra["is_eq"] = True
            lv.strength = min(lv.strength + 0.15, 1.0)  # EQ 加分
            eq_levels.append(lv)

    print(f"  EQ 過濾: {len(levels)} 個水平 → {len(eq_levels)} 個新鮮水平")
    return eq_levels


def filter_eq_multi_timeframe(
    levels: list[Level],
    mtf_data: dict[str, pd.DataFrame],
) -> list[Level]:
    """
    跨多時間框架驗證 EQ。
    水平必須在所有指定時間框架中都是新鮮的。

    Parameters
    ----------
    levels : list[Level]
        待過濾的水平清單
    mtf_data : dict[str, pd.DataFrame]
        多時間框架數據 {name: DataFrame}

    Returns
    -------
    list[Level]
        跨時間框架都新鮮的 EQ 水平
    """
    scan_tfs = EQ_CONFIG["scan_timeframes"]
    eq_levels = []

    for lv in levels:
        fresh_in_all = True
        for tf_name, tf_df in mtf_data.items():
            atr = compute_atr(tf_df)
            # 在該時間框架中找到對應的 bar index
            # 用時間戳匹配最近的 bar
            if lv.timestamp is not None and isinstance(tf_df.index, pd.DatetimeIndex):
                mask = tf_df.index <= lv.timestamp
                if mask.any():
                    equiv_idx = mask.sum() - 1
                    temp_level = Level(
                        price=lv.price,
                        level_type=lv.level_type,
                        side=lv.side,
                        bar_index=equiv_idx,
                    )
                    if not is_level_fresh(temp_level, tf_df, atr):
                        fresh_in_all = False
                        break

        if fresh_in_all:
            lv.extra["is_eq_mtf"] = True
            lv.strength = min(lv.strength + 0.2, 1.0)
            eq_levels.append(lv)

    print(f"  多時間框架 EQ: {len(levels)} → {len(eq_levels)} 個")
    return eq_levels
