"""
多時間框架分析模組
==================
實現「故事線方法」：
- 高 TF (H4) 看「拒絕」→ 判斷大方向
- 中 TF (H1) 確認趨勢
- 低 TF (M15/M5) 找「突破」確認方向，精確進場
- 不能忽略 H4 阻力 / 支撐
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.fetcher import compute_atr, compute_body
from indicators.levels import Level, LevelSide, detect_all_levels


class MarketBias(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class MTFAnalysis:
    """多時間框架分析結果"""
    high_tf_bias: MarketBias        # 高階方向
    mid_tf_bias: MarketBias         # 中階方向
    low_tf_bias: MarketBias         # 低階方向
    overall_bias: MarketBias        # 綜合方向
    confidence: float               # 信心度 0~1
    high_tf_levels: Optional[list] = None     # 高階 TF 的關鍵水平 (障礙)
    notes: Optional[list] = None              # 分析備註

    def __repr__(self):
        return (
            f"MTF(overall={self.overall_bias.value}, "
            f"conf={self.confidence:.0%}, "
            f"H={self.high_tf_bias.value} M={self.mid_tf_bias.value} "
            f"L={self.low_tf_bias.value})"
        )


def _detect_rejection(df: pd.DataFrame, level: Level, lookback: int = 5) -> bool:
    """
    檢查價格是否在水平處被拒絕 (rejection)。
    拒絕 = 價格觸碰水平後有明顯反向 K 線。
    """
    df = compute_body(df)
    price = level.price
    atr = compute_atr(df)

    end = len(df)
    start = max(0, end - lookback)

    for i in range(start, end):
        local_atr = atr.iloc[i]
        tolerance = local_atr * 0.3

        if level.side == LevelSide.RESISTANCE:
            # 阻力拒絕: 高點觸碰阻力但收盤遠低於阻力
            if df["high"].iloc[i] >= price - tolerance:
                if df["close"].iloc[i] < price - local_atr * 0.5:
                    return True
        else:
            # 支撐拒絕: 低點觸碰支撐但收盤遠高於支撐
            if df["low"].iloc[i] <= price + tolerance:
                if df["close"].iloc[i] > price + local_atr * 0.5:
                    return True

    return False


def _detect_breakout_confirmation(
    df: pd.DataFrame, level: Level, lookback: int = 5
) -> bool:
    """
    檢查價格是否突破水平並確認 (拒絕回測)。
    """
    df = compute_body(df)
    price = level.price
    atr = compute_atr(df)

    end = len(df)
    start = max(0, end - lookback)

    for i in range(start, end):
        local_atr = atr.iloc[i]

        if level.side == LevelSide.RESISTANCE:
            # 向上突破阻力: 收盤明確高於阻力
            if df["close"].iloc[i] > price + local_atr * 0.3:
                # 檢查後續是否有回測且被拒絕
                for j in range(i + 1, min(i + lookback, end)):
                    if df["low"].iloc[j] <= price + local_atr * 0.2:
                        if df["close"].iloc[j] > price:
                            return True
        else:
            # 向下突破支撐
            if df["close"].iloc[i] < price - local_atr * 0.3:
                for j in range(i + 1, min(i + lookback, end)):
                    if df["high"].iloc[j] >= price - local_atr * 0.2:
                        if df["close"].iloc[j] < price:
                            return True

    return False


def analyze_single_tf(df: pd.DataFrame) -> MarketBias:
    """
    單一時間框架的方向判斷。
    使用近期 K 線的結構判斷偏多/偏空。
    """
    if len(df) < 20:
        return MarketBias.NEUTRAL

    df = compute_body(df)
    recent = df.iloc[-20:]

    # 計算近期多空 K 線比例
    bullish_count = recent["is_bullish"].sum()
    bearish_count = len(recent) - bullish_count

    # 計算近期價格趨勢
    closes = recent["close"].values
    x = np.arange(len(closes))
    if len(x) > 1:
        slope = np.polyfit(x, closes, 1)[0]
    else:
        slope = 0

    atr = compute_atr(df)
    avg_atr = atr.iloc[-20:].mean()

    # 綜合判斷
    if slope > avg_atr * 0.02 and bullish_count > bearish_count:
        return MarketBias.BULLISH
    elif slope < -avg_atr * 0.02 and bearish_count > bullish_count:
        return MarketBias.BEARISH
    else:
        return MarketBias.NEUTRAL


def analyze_mtf(
    mtf_data: dict[str, pd.DataFrame],
) -> MTFAnalysis:
    """
    多時間框架分析。

    Parameters
    ----------
    mtf_data : dict[str, pd.DataFrame]
        {"high": H4_df, "mid": H1_df, "low": M15_df, ...}

    Returns
    -------
    MTFAnalysis
        分析結果
    """
    notes = []

    # 各 TF 方向判斷
    high_bias = MarketBias.NEUTRAL
    mid_bias = MarketBias.NEUTRAL
    low_bias = MarketBias.NEUTRAL

    if "high" in mtf_data:
        high_bias = analyze_single_tf(mtf_data["high"])
        high_levels = detect_all_levels(mtf_data["high"])
        notes.append(f"H4 方向: {high_bias.value}, 水平: {len(high_levels)} 個")
    else:
        high_levels = []

    if "mid" in mtf_data:
        mid_bias = analyze_single_tf(mtf_data["mid"])
        notes.append(f"H1 方向: {mid_bias.value}")

    if "low" in mtf_data:
        low_bias = analyze_single_tf(mtf_data["low"])
        notes.append(f"M15 方向: {low_bias.value}")

    # 綜合判斷 (高 TF 權重最大)
    bias_scores = {
        MarketBias.BULLISH: 0.0,
        MarketBias.BEARISH: 0.0,
        MarketBias.NEUTRAL: 0.0,
    }

    # 高 TF 權重 50%, 中 TF 30%, 低 TF 20%
    weights = {"high": 0.5, "mid": 0.3, "low": 0.2}
    biases = {"high": high_bias, "mid": mid_bias, "low": low_bias}

    for tf, bias in biases.items():
        bias_scores[bias] += weights[tf]

    # 決定綜合方向
    overall = max(bias_scores, key=lambda k: bias_scores[k])

    # 信心度
    confidence = bias_scores[overall]

    # 方向一致性加分
    if high_bias == mid_bias == low_bias and overall != MarketBias.NEUTRAL:
        confidence = min(confidence + 0.2, 1.0)
        notes.append("三時間框架方向一致!")

    # 高階與低階矛盾 → 降低信心
    if high_bias != MarketBias.NEUTRAL and low_bias != MarketBias.NEUTRAL and high_bias != low_bias:
        confidence *= 0.6
        notes.append("警告: 高階與低階方向矛盾")

    return MTFAnalysis(
        high_tf_bias=high_bias,
        mid_tf_bias=mid_bias,
        low_tf_bias=low_bias,
        overall_bias=overall,
        confidence=confidence,
        high_tf_levels=high_levels,
        notes=notes,
    )
