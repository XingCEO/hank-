# SNR 2020 / SR 2.0 交易回測系統

基於林俊宏 (超星集團) SNR 2020 / SR 2.0 支撐阻力交易策略的自動化回測系統。

## 功能

- **4 種水平偵測**: Classic、Breakout、Gap、Head & Shoulders (HNS)
- **EQ 過濾**: 只交易「新鮮」(未被測試) 的水平
- **CC 過濾**: 確認 K 線 (Classic CC、BOCC1、BOCC2、HNS CC)
- **多時間框架分析**: H4 判方向 → H1 確認 → M15/M5 精確進場
- **進場方式**: DC (雙重確認) + DE (直接進場)
- **風險管理**: 1% 風險/筆、1R 目標、2R 保本、3R 延伸
- **視覺化報告**: K 線圖 + 水平線 + 權益曲線 + R 值分布

## 快速開始

### 方法一：Docker (推薦)

**前置需求**: 安裝 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

```bash
# 1. 建構映像檔
docker build -t snr2-trading-bot .

# 2. 執行回測 (BTC/USDT, 1h, 1000 根 K 線)
docker run --rm -v ./output:/app/output snr2-trading-bot \
    --symbol BTC/USDT --timeframe 1h --limit 1000 --output /app/output

# 報告會輸出到 ./output/ 目錄
```

### 方法二：Docker Compose

```bash
# BTC 回測 (預設)
docker compose up backtest

# ETH 回測 (15m, 2000 根)
docker compose up backtest-eth

# 無過濾器模式 (純水平偵測)
docker compose up backtest-raw
```

報告自動輸出到 `./output/` 目錄。

### 方法三：直接執行 Python

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 執行
python main.py --symbol BTC/USDT --timeframe 1h --limit 1000
```

## 完整參數說明

```
python main.py [OPTIONS]

選項:
  --symbol TEXT       交易對 (預設: BTC/USDT)
  --timeframe TEXT    主要時間框架 (預設: 1h)
                      可選: 1m, 5m, 15m, 30m, 1h, 4h, 1d
  --limit INT         K 線數量 (預設: 1000)
  --since TEXT        起始日期 YYYY-MM-DD (預設: 2024-01-01)
  --balance FLOAT     初始資金 (預設: 10000.0)
  --no-eq             停用 EQ 過濾
  --no-cc             停用 CC 過濾
  --no-mtf            停用多時間框架分析
  --no-report         不產出視覺化報告
  --output TEXT       報告輸出目錄 (預設: 當前目錄)
```

## 常用組合範例

```bash
# BTC 15 分鐘框架，抓 2000 根 K 線
docker run --rm -v ./output:/app/output snr2-trading-bot \
    --symbol BTC/USDT --timeframe 15m --limit 2000 --output /app/output

# SOL 4 小時框架，5 萬初始資金
docker run --rm -v ./output:/app/output snr2-trading-bot \
    --symbol SOL/USDT --timeframe 4h --balance 50000 --output /app/output

# 只用水平偵測，關閉所有過濾器
docker run --rm -v ./output:/app/output snr2-trading-bot \
    --symbol ETH/USDT --timeframe 1h --no-eq --no-cc --no-mtf --output /app/output

# 指定日期範圍
docker run --rm -v ./output:/app/output snr2-trading-bot \
    --symbol BTC/USDT --timeframe 1h --since 2025-01-01 --output /app/output
```

## 輸出檔案

回測完成後會在輸出目錄產生：

| 檔案 | 說明 |
|------|------|
| `backtest_BTC_USDT_1h.png` | 視覺化報告 (K 線 + 水平 + 權益曲線) |
| `backtest_BTC_USDT_1h_trades.csv` | 交易明細 CSV |

## 專案結構

```
├── main.py                  # CLI 入口
├── config/settings.py       # 全域參數設定
├── data/fetcher.py          # ccxt 數據抓取
├── indicators/
│   ├── levels.py            # 4 種水平偵測
│   ├── eq_filter.py         # EQ 新鮮水平過濾
│   ├── cc_filter.py         # CC 確認 K 線過濾
│   └── trendline.py         # 趨勢線自動擬合
├── strategy/
│   ├── mtf_analysis.py      # 多時間框架分析
│   ├── entry.py             # DC/DE 進場邏輯
│   └── risk_manager.py      # 倉位/止損/止盈管理
├── backtest/
│   ├── engine.py            # 回測引擎
│   └── report.py            # 視覺化報告
├── tradingview/
│   ├── SNR2_Strategy.pine        # TradingView 策略
│   ├── SNR2_MTF_Dashboard.pine   # MTF 儀表板指標
│   └── SNR2_Visual_Helper.pine   # 視覺輔助指標
├── Dockerfile               # Docker 映像檔
├── docker-compose.yml       # Docker Compose 設定
└── requirements.txt         # Python 依賴
```

## TradingView 腳本

`tradingview/` 目錄包含 3 個 Pine Script v6 腳本，可直接貼到 TradingView 使用：

1. **SNR2_Strategy.pine** — 主策略 (含自動進出場)
2. **SNR2_MTF_Dashboard.pine** — 多時間框架儀表板
3. **SNR2_Visual_Helper.pine** — 視覺輔助 (水平區域、CC 標記、R/R 視覺化、7 種警報)

## 注意事項

- 數據來源為 Binance (透過 ccxt)，需要能連線到交易所 API
- ccxt 版本鎖定 `<4.5.0` (4.5.x 有 import bug)
- 回測結果僅供參考，不構成投資建議
- 初次建構 Docker 映像可能需要數分鐘 (安裝 scipy/matplotlib)
