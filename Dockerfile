# ============================================================
# SNR 2.0 Trading Bot - Docker Image
# ============================================================
# Multi-stage build for smaller image size

FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (needed for scipy/matplotlib)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================
# Production stage
# ============================================================
FROM python:3.12-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from build stage
COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

# Copy application code
COPY config/ config/
COPY data/ data/
COPY indicators/ indicators/
COPY strategy/ strategy/
COPY backtest/ backtest/
COPY main.py .
COPY requirements.txt .

# Copy TradingView scripts (for reference)
COPY tradingview/ tradingview/

# Copy strategy document
COPY 林俊宏交易策略.md .

# Create output directory for reports
RUN mkdir -p /app/output

# Default command: show help
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
