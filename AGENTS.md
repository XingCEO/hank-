# AGENTS.md - Coding Agent Instructions

## Project Overview

SNR 2020 / SR 2.0 automated trading backtest system based on Lin Jun-Hong's support/resistance strategy.
Two components: Python backtesting (12 modules) + TradingView Pine Script v6 (3 scripts).
Target market: Cryptocurrency (Binance via ccxt). Documentation language: Traditional Chinese.

## Build / Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest (CLI)
python main.py --symbol BTC/USDT --timeframe 1h --limit 1000
python main.py --symbol ETH/USDT --timeframe 15m --limit 2000 --no-mtf
python main.py --help

# Docker
docker build -t snr2-trading-bot .
docker compose up backtest          # BTC default
docker compose up backtest-eth      # ETH 15m
docker compose up backtest-raw      # No filters

# Verify all imports
python -c "from config.settings import *; from data.fetcher import DataFetcher; from indicators.levels import detect_all_levels; from indicators.eq_filter import filter_eq_levels; from indicators.cc_filter import filter_levels_with_cc; from indicators.trendline import detect_trendlines; from strategy.mtf_analysis import analyze_mtf; from strategy.entry import generate_signals; from strategy.risk_manager import RiskManager; from backtest.engine import BacktestEngine; from backtest.report import generate_report; print('OK')"
```

No test framework is configured. No pytest, no mypy, no ruff config files exist.
To validate changes, run the import verification command above and a full backtest.

## Project Structure

```
main.py                      # CLI entry (argparse)
config/settings.py           # All config constants (UPPER_SNAKE_CASE dicts)
data/fetcher.py              # ccxt OHLCV fetcher + ATR computation
indicators/levels.py         # 4 level types: Classic, Breakout, Gap, HNS
indicators/eq_filter.py      # EQ freshness filter
indicators/cc_filter.py      # CC confirmation candle filter (4 subtypes)
indicators/trendline.py      # Trendline auto-fitting via linear regression
strategy/mtf_analysis.py     # Multi-timeframe direction analysis
strategy/entry.py            # DC/DE entry + combo matrix
strategy/risk_manager.py     # Position sizing, SL/TP, breakeven
backtest/engine.py           # Backtest orchestrator
backtest/report.py           # Matplotlib visual report
tradingview/*.pine           # 3 Pine Script v6 scripts
```

## Python Code Style

### Imports

Standard library first, then third-party, then local. Every non-main module uses:
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
```
Use `from __future__ import annotations` when forward references are needed.
Suppress untyped imports: `import ccxt  # type: ignore[import-untyped]`

### Type Annotations

- Always annotate function signatures: `def func(df: pd.DataFrame, atr: float) -> list[Level]:`
- Use `Optional[X]` (not `X | None`) for optional params with `None` default
- Use lowercase generics: `list[X]`, `dict[str, X]` (Python 3.9+ style)
- Cast DataFrame index: `cast(pd.Timestamp, df.index[idx])` or use helper `_get_timestamp()`
- Use `# type: ignore[assignment]` / `# type: ignore[arg-type]` sparingly for pandas/scipy edge cases
- Import `matplotlib.axes.Axes as MplAxes` for axes type hints (not `plt.Axes`)

### Naming Conventions

| Kind | Convention | Examples |
|------|-----------|----------|
| Classes | PascalCase | `DataFetcher`, `BacktestEngine`, `Level` |
| Enums | PascalCase class, UPPER values | `LevelType.CLASSIC`, `MarketBias.BULLISH` |
| Functions | snake_case | `detect_all_levels`, `filter_eq_levels` |
| Private helpers | _leading_underscore | `_get_timestamp`, `_close_position` |
| Config constants | UPPER_SNAKE_CASE | `SYMBOL`, `LEVELS_CONFIG`, `RISK_CONFIG` |
| Config dict keys | snake_case strings | `"swing_lookback"`, `"risk_per_trade_pct"` |
| Variables | snake_case | `local_atr`, `swing_highs`, `bar_idx` |
| Files | snake_case | `risk_manager.py`, `eq_filter.py` |

### Data Structures

- Use `@dataclass` for all data containers (`Level`, `Signal`, `Position`, `Trendline`)
- Use `Enum` for categorical values (`LevelType`, `CCType`, `EntryType`, `TradeDirection`)
- Use `field(default_factory=dict)` for mutable defaults
- Define `__repr__` on dataclasses for debugging
- Use `@property` for computed fields (e.g., `Signal.risk`, `Signal.rr_ratio`)

### Error Handling

- Prefer guard clauses and early returns over try/except
- No custom exception classes; no logging module -- use `print()` with Chinese bracketed prefixes
- Example: `print("[資料] 正在從交易所抓取...")`

### Documentation

- Module/function docstrings in NumPy style with Chinese descriptions
- Section separators: `# ============================================================`
- Inline comments in Traditional Chinese

## Configuration

All parameters live in `config/settings.py` as module-level dicts. No .env files.
Parameters are expressed as ATR ratios for volatility-adaptive behavior.

## Key Dependencies

- **ccxt**: MUST be pinned `>=4.0.0,<4.5.0` (4.5.x has broken `lighter_client` import)
- **pandas**: DataFrame-centric data flow
- **scipy**: `linregress` for trendline fitting -- cast returns to `float()`
- **matplotlib**: Dark theme reports; use `MplAxes` for type hints, `tight_layout(rect=(...))` with tuple not list

## Pine Script Rules (tradingview/)

All scripts use `//@version=6`. Critical rules:

1. **ALL function calls on a single line** -- never split across lines with commas
2. **ALL ternary operators on a single line** -- `x ? a : b` never spanning multiple lines
3. **Guard empty arrays**: `if array.size(arr) > 0` before any `for i = array.size(arr) - 1 to 0`
4. **Guard bar 0**: `if bar_index > 0` before using `close[1]`, `high[1]`, etc.
5. **Clamp bar coordinates**: `line.new()` x1/x2 must be within ~5000 bars of current bar_index
6. **No local-scope historical refs**: don't use `var[1]` on variables declared inside `if`/`for` blocks
7. **Trendline lifecycle**: 2 touches = confirmed, 3 = signal (tradeable), 4+ = expired and removed

Pine naming: camelCase variables, PascalCase types, camelCase functions.
Input group labels use Chinese with `═══` decorative borders.

## Domain Knowledge

This implements a specific trading methodology. Key concepts:
- **4 level types**: Classic (swing H/L), Breakout (strong candle base), Gap (body gap), HNS (head & shoulders neckline)
- **EQ filter**: Only trade "fresh" (untested) levels
- **CC filter**: 4 confirmation candle types (Classic CC, BOCC1, BOCC2, HNS CC)
- **Entry methods**: DC (Double Confirmation -- candle close) vs DE (Direct Entry -- touch)
- **Setup/DE matrix**: BO & HNS allow Direct Entry; Classic & Gap require Setup (DC)
- **MTF hierarchy**: H4 direction -> H1 confirm -> M15/M5 precise entry
- **Risk**: 1% per trade, 1R default target, breakeven at 2R, extend to 3R
- Strategy reference document: `林俊宏交易策略.md` (read-only, source of truth)

## Git Conventions

- Branch: `main`
- Commit messages: conventional commits in English (`fix:`, `feat:`)
- Never commit `output/`, `*.png`, `*.csv`, `.env`
- Remote: `https://github.com/XingCEO/hank-.git`
