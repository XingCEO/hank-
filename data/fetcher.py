"""
資料獲取模組 - 使用 ccxt 從加密貨幣交易所抓取 K 線數據
"""

import ccxt  # type: ignore[import-untyped]
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Any

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import EXCHANGE, SYMBOL, FETCH_LIMIT, TIMEFRAMES


class DataFetcher:
    """從交易所抓取 OHLCV K 線資料"""

    def __init__(self, exchange_id: str = EXCHANGE):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            "enableRateLimit": True,
        })

    def fetch_ohlcv(
        self,
        symbol: str = SYMBOL,
        timeframe: str = "1h",
        limit: int = FETCH_LIMIT,
        since: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        抓取 K 線資料並轉成 DataFrame。

        Parameters
        ----------
        symbol : str
            交易對，如 "BTC/USDT"
        timeframe : str
            時間框架，如 "1m", "5m", "15m", "30m", "1h", "4h", "1d"
        limit : int
            抓取 K 線數量上限
        since : str or None
            起始時間字串 "YYYY-MM-DD"，None 表示從最近往回算

        Returns
        -------
        pd.DataFrame
            columns: [timestamp, open, high, low, close, volume]
            index: DatetimeIndex
        """
        since_ms = None
        if since:
            since_ms = self.exchange.parse8601(since + "T00:00:00Z")

        raw = self.exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since_ms, limit=limit
        )

        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df.astype(float)
        return df

    def fetch_multi_timeframe(
        self,
        symbol: str = SYMBOL,
        timeframes: Optional[dict] = None,
        limit: int = FETCH_LIMIT,
        since: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        一次抓取多個時間框架的 K 線資料。

        Parameters
        ----------
        timeframes : dict
            key=名稱, value=ccxt timeframe 字串
            預設使用 config 的 TIMEFRAMES

        Returns
        -------
        dict[str, pd.DataFrame]
            key=timeframe 名稱, value=OHLCV DataFrame
        """
        if timeframes is None:
            timeframes = TIMEFRAMES

        data = {}
        for name, tf in timeframes.items():
            print(f"  抓取 {symbol} {tf} K 線 (最多 {limit} 根)...")
            data[name] = self.fetch_ohlcv(symbol, tf, limit, since)
            print(f"    取得 {len(data[name])} 根 K 線")

        return data


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    計算 Average True Range (ATR)。

    Parameters
    ----------
    df : pd.DataFrame
        必須包含 high, low, close 欄位
    period : int
        ATR 計算週期

    Returns
    -------
    pd.Series
        ATR 值序列
    """
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr: pd.Series = tr.rolling(window=period, min_periods=1).mean()  # type: ignore[assignment]
    return atr


def compute_body(df: pd.DataFrame) -> pd.DataFrame:
    """
    為 DataFrame 新增 K 線實體相關欄位。

    新增欄位：
    - body_top: 實體上緣
    - body_bottom: 實體下緣
    - body_size: 實體大小
    - is_bullish: 是否為多頭 K 線
    - upper_wick: 上影線長度
    - lower_wick: 下影線長度
    """
    df = df.copy()
    df["body_top"] = df[["open", "close"]].max(axis=1)
    df["body_bottom"] = df[["open", "close"]].min(axis=1)
    df["body_size"] = df["body_top"] - df["body_bottom"]
    df["is_bullish"] = df["close"] > df["open"]
    df["upper_wick"] = df["high"] - df["body_top"]
    df["lower_wick"] = df["body_bottom"] - df["low"]
    return df
