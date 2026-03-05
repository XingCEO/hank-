"""
SNR 2020 / SR 2.0 自動交易回測系統
====================================
主程式入口

使用方式：
    python main.py                          # 使用預設設定
    python main.py --symbol ETH/USDT       # 指定交易對
    python main.py --timeframe 15m         # 指定時間框架
    python main.py --no-eq                 # 停用 EQ 過濾
    python main.py --no-cc                 # 停用 CC 過濾
    python main.py --no-mtf               # 停用多時間框架
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import (
    SYMBOL, TIMEFRAMES, FETCH_LIMIT,
    RISK_CONFIG, BACKTEST_CONFIG,
)
from data.fetcher import DataFetcher
from indicators.levels import detect_all_levels
from backtest.engine import BacktestEngine
from backtest.report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="SNR 2.0 交易回測系統",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--symbol", type=str, default=SYMBOL,
        help=f"交易對 (預設: {SYMBOL})"
    )
    parser.add_argument(
        "--timeframe", type=str, default="1h",
        help="主要進場時間框架 (預設: 1h)"
    )
    parser.add_argument(
        "--limit", type=int, default=FETCH_LIMIT,
        help=f"K 線數量 (預設: {FETCH_LIMIT})"
    )
    parser.add_argument(
        "--since", type=str, default=BACKTEST_CONFIG.get("start_date"),
        help="起始日期 YYYY-MM-DD"
    )
    parser.add_argument(
        "--balance", type=float, default=RISK_CONFIG["initial_balance"],
        help=f"初始資金 (預設: {RISK_CONFIG['initial_balance']})"
    )
    parser.add_argument(
        "--no-eq", action="store_true",
        help="停用 EQ 過濾"
    )
    parser.add_argument(
        "--no-cc", action="store_true",
        help="停用 CC 過濾"
    )
    parser.add_argument(
        "--no-mtf", action="store_true",
        help="停用多時間框架分析"
    )
    parser.add_argument(
        "--no-report", action="store_true",
        help="不產出視覺化報告"
    )
    parser.add_argument(
        "--output", type=str, default=".",
        help="報告輸出目錄 (預設: 當前目錄)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(r"""
    ╔══════════════════════════════════════════════╗
    ║   SNR 2020 / SR 2.0 交易回測系統             ║
    ║   Support & Resistance Auto Trading Bot      ║
    ╚══════════════════════════════════════════════╝
    """)

    # Step 1: 抓取數據
    print("[資料] 正在從交易所抓取 K 線數據...")
    fetcher = DataFetcher()

    # 主要時間框架 (進場用)
    df = fetcher.fetch_ohlcv(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        since=args.since,
    )
    print(f"  主框架 ({args.timeframe}): {len(df)} 根 K 線")

    # 多時間框架數據
    mtf_data = None
    if not args.no_mtf:
        print("[資料] 抓取多時間框架數據...")
        mtf_timeframes = {
            "high": TIMEFRAMES["high"],
            "mid": TIMEFRAMES["mid"],
        }
        mtf_data = fetcher.fetch_multi_timeframe(
            symbol=args.symbol,
            timeframes=mtf_timeframes,
            limit=args.limit,
            since=args.since,
        )

    # Step 2: 執行回測
    engine = BacktestEngine(
        symbol=args.symbol,
        initial_balance=args.balance,
    )

    stats = engine.run(
        df=df,
        mtf_data=mtf_data,
        use_eq_filter=not args.no_eq,
        use_cc_filter=not args.no_cc,
        use_mtf=not args.no_mtf,
    )

    # Step 3: 產出報告
    if not args.no_report and stats.get("total_trades", 0) > 0:
        print("[報告] 產出視覺化報告...")
        levels = detect_all_levels(df)

        report_name = f"backtest_{args.symbol.replace('/', '_')}_{args.timeframe}"
        generate_report(
            df=df,
            levels=levels,
            trade_log=engine.trade_log,
            equity_curve=engine.equity_curve,
            stats=stats,
            output_dir=args.output,
            filename=report_name,
        )

    print("\n完成! 祝交易順利。")


if __name__ == "__main__":
    main()
