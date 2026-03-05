"""
視覺化報告模組
==============
產出：
1. K 線圖 + 水平線 + 進出場標記
2. 權益曲線
3. 交易分布圖
4. 績效統計表
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非互動模式
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.axes import Axes as MplAxes
from matplotlib.patches import FancyArrowPatch
from typing import Any, Optional
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indicators.levels import Level, LevelType, LevelSide


# 色彩配置
COLORS = {
    "background": "#1a1a2e",
    "text": "#e0e0e0",
    "grid": "#2a2a4a",
    "bullish": "#00d4aa",
    "bearish": "#ff6b6b",
    "classic": "#ffd700",
    "breakout": "#ff8c00",
    "gap": "#00bfff",
    "hns": "#ff69b4",
    "equity": "#00d4aa",
    "drawdown": "#ff6b6b",
    "long_entry": "#00ff88",
    "short_entry": "#ff4444",
    "take_profit": "#00d4aa",
    "stop_loss": "#ff6b6b",
}

LEVEL_COLORS = {
    LevelType.CLASSIC: COLORS["classic"],
    LevelType.BREAKOUT: COLORS["breakout"],
    LevelType.GAP: COLORS["gap"],
    LevelType.HNS: COLORS["hns"],
}


def _setup_dark_style(fig, axes):
    """設定深色主題"""
    fig.patch.set_facecolor(COLORS["background"])
    for ax in axes:
        ax.set_facecolor(COLORS["background"])
        ax.tick_params(colors=COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text"])
        ax.yaxis.label.set_color(COLORS["text"])
        ax.title.set_color(COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(COLORS["grid"])
        ax.grid(True, color=COLORS["grid"], alpha=0.3)


def plot_candlestick(
    ax: MplAxes,
    df: pd.DataFrame,
    max_bars: int = 200,
):
    """繪製 K 線圖"""
    data = df.tail(max_bars).copy()

    for i in range(len(data)):
        o = data["open"].iloc[i]
        c = data["close"].iloc[i]
        h = data["high"].iloc[i]
        lo = data["low"].iloc[i]

        color = COLORS["bullish"] if c >= o else COLORS["bearish"]
        body_bottom = min(o, c)
        body_height = abs(c - o)

        # 影線
        ax.plot([i, i], [lo, h], color=color, linewidth=0.5, alpha=0.8)
        # 實體
        ax.bar(i, body_height, bottom=body_bottom, width=0.6,
               color=color, edgecolor=color, alpha=0.9)

    # X 軸標籤
    step = max(1, len(data) // 10)
    ax.set_xticks(range(0, len(data), step))
    if isinstance(data.index, pd.DatetimeIndex):
        labels: list[str] = [str(data.index[i].strftime("%m/%d %H:%M")) for i in range(0, len(data), step)]
        ax.set_xticklabels(labels, rotation=45, fontsize=7)


def plot_levels(
    ax: MplAxes,
    levels: list[Level],
    df: pd.DataFrame,
    max_bars: int = 200,
):
    """在 K 線圖上繪製水平線"""
    data = df.tail(max_bars)
    offset = len(df) - max_bars if len(df) > max_bars else 0

    for lv in levels:
        if lv.bar_index < offset:
            # 水平在可視範圍之前形成，畫全寬
            x_start = 0
        else:
            x_start = lv.bar_index - offset

        if x_start >= len(data):
            continue

        color = LEVEL_COLORS.get(lv.level_type, COLORS["classic"])
        linestyle = "--" if lv.side == LevelSide.RESISTANCE else "-"
        alpha = 0.4 + lv.strength * 0.4

        ax.axhline(y=lv.price, xmin=x_start / len(data), xmax=1.0,
                    color=color, linestyle=linestyle, linewidth=0.8, alpha=alpha)

        # 標籤
        ax.annotate(
            f"{lv.level_type.value[0].upper()} {lv.price:.0f}",
            xy=(len(data) - 1, lv.price),
            fontsize=6, color=color, alpha=0.7,
            ha="right", va="bottom" if lv.side == LevelSide.SUPPORT else "top",
        )


def plot_trades(
    ax: MplAxes,
    trade_log: list[dict],
    df: pd.DataFrame,
    max_bars: int = 200,
):
    """在 K 線圖上標記進出場點"""
    offset = len(df) - max_bars if len(df) > max_bars else 0

    for trade in trade_log:
        entry_time = trade["entry_time"]
        exit_time = trade["exit_time"]

        if entry_time is None:
            continue

        # 找到對應的 bar index
        try:
            entry_idx = int(df.index.get_loc(entry_time)) - offset  # type: ignore[arg-type]
            exit_idx = int(df.index.get_loc(exit_time)) - offset if exit_time else None  # type: ignore[arg-type]
        except (KeyError, TypeError):
            continue

        if entry_idx < 0 or entry_idx >= max_bars:
            continue

        # 進場標記
        if trade["direction"] == "long":
            marker = "^"
            color = COLORS["long_entry"]
        else:
            marker = "v"
            color = COLORS["short_entry"]

        ax.scatter(entry_idx, trade["entry_price"],
                   marker=marker, color=color, s=80, zorder=5, edgecolors="white", linewidth=0.5)

        # 出場標記
        if exit_idx is not None and 0 <= exit_idx < max_bars:
            exit_color = COLORS["take_profit"] if trade["pnl"] > 0 else COLORS["stop_loss"]
            ax.scatter(exit_idx, trade["exit_price"],
                       marker="x", color=exit_color, s=60, zorder=5)


def generate_report(
    df: pd.DataFrame,
    levels: list[Level],
    trade_log: list[dict],
    equity_curve: list[float],
    stats: dict,
    output_dir: str = ".",
    filename: str = "backtest_report",
):
    """
    產出完整回測報告圖表。

    Parameters
    ----------
    df : pd.DataFrame
        K 線數據
    levels : list[Level]
        偵測到的水平
    trade_log : list[dict]
        交易記錄
    equity_curve : list[float]
        權益曲線
    stats : dict
        績效統計
    output_dir : str
        輸出目錄
    filename : str
        輸出檔名 (不含副檔名)
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    _setup_dark_style(fig, axes)

    # --- 上圖: K 線 + 水平 + 交易 ---
    ax1 = axes[0]
    ax1.set_title(f"SNR 2.0 Backtest - K Line & Levels", fontsize=12, fontweight="bold")
    plot_candlestick(ax1, df)
    plot_levels(ax1, levels, df)
    plot_trades(ax1, trade_log, df)
    ax1.set_ylabel("Price", fontsize=9)

    # --- 中圖: 權益曲線 ---
    ax2 = axes[1]
    ax2.set_title("Equity Curve", fontsize=10)
    if equity_curve:
        ax2.plot(equity_curve, color=COLORS["equity"], linewidth=1)
        ax2.fill_between(range(len(equity_curve)), equity_curve,
                          equity_curve[0], alpha=0.1, color=COLORS["equity"])
        ax2.axhline(y=equity_curve[0], color=COLORS["text"], linestyle="--",
                     linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Balance ($)", fontsize=9)

    # --- 下圖: 每筆交易 R 值 ---
    ax3 = axes[2]
    ax3.set_title("Trade R-Multiples", fontsize=10)
    if trade_log:
        r_values = [t["r_result"] for t in trade_log]
        colors_list = [COLORS["take_profit"] if r > 0 else COLORS["stop_loss"] for r in r_values]
        ax3.bar(range(len(r_values)), r_values, color=colors_list, alpha=0.8)
        ax3.axhline(y=0, color=COLORS["text"], linewidth=0.5)
        ax3.set_xlabel("Trade #", fontsize=9)
    ax3.set_ylabel("R-Multiple", fontsize=9)

    # 在圖的右上角加入統計文字
    if stats and stats.get("total_trades", 0) > 0:
        stats_text = (
            f"Trades: {stats['total_trades']}  |  "
            f"Win Rate: {stats['win_rate']:.1f}%  |  "
            f"PF: {stats['profit_factor']:.2f}  |  "
            f"Max DD: {stats['max_drawdown_pct']:.1f}%  |  "
            f"Return: {stats['return_pct']:.1f}%"
        )
        fig.text(0.5, 0.02, stats_text, ha="center", fontsize=9,
                 color=COLORS["text"], style="italic")

    plt.tight_layout(rect=(0, 0.04, 1, 1))

    # 儲存
    output_path = os.path.join(output_dir, f"{filename}.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=COLORS["background"])
    plt.close(fig)

    print(f"  報告已儲存: {output_path}")

    # 也儲存交易明細 CSV
    if trade_log:
        csv_path = os.path.join(output_dir, f"{filename}_trades.csv")
        trade_df = pd.DataFrame(trade_log)
        trade_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  交易明細: {csv_path}")

    return output_path
