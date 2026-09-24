"""
====================================================================
PHASE 1: DATA PIPELINE — Thu thập và xử lý dữ liệu
====================================================================
Chạy: python phase1_data_pipeline.py

Input:  Yahoo Finance API
Output: data/processed/in_sample.parquet
        data/processed/out_sample.parquet
====================================================================
"""

from config import Config
from utils import save_data

import pandas as pd
from tqdm import tqdm


# ============================================================
# 1.1 — Lấy danh sách ticker Russell 1000
# ============================================================

def get_russell1000_tickers() -> list:
    """
    Lấy danh sách ticker của Russell 1000.

    TODO: Mở rộng lên 100+ tickers khi đã test xong.
    Cách lấy full list:
        - Download từ iShares IWB ETF holdings (CSV)
        - Hoặc dùng Wikipedia Russell 1000 page
    """
    # Bắt đầu với 20 tickers để test nhanh
    sample_tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ',
        'V', 'PG', 'UNH', 'HD', 'MA',
        'DIS', 'PYPL', 'NFLX', 'INTC', 'AMD',
    ]
    print(f"📋 Loaded {len(sample_tickers)} tickers (sample)")
    return sample_tickers


# ============================================================
# 1.2 — Download dữ liệu OHLCV từ Yahoo Finance
# ============================================================

def download_data(tickers: list) -> dict:
    """
    Download dữ liệu giá hàng ngày cho từng ticker.

    Returns:
        dict: {ticker: DataFrame} với columns [Open, High, Low, Close, Volume]
    """
    import yfinance as yf

    print(f"\n📥 Đang download dữ liệu cho {len(tickers)} cổ phiếu...")
    print(f"   Từ {Config.DATA_START} đến hôm nay\n")

    all_data = {}
    failed = []

    for ticker in tqdm(tickers, desc="   Downloading"):
        try:
            df = yf.download(
                ticker,
                start=Config.DATA_START,
                progress=False,
                auto_adjust=True,
            )
            if len(df) > 100:
                all_data[ticker] = df
            else:
                failed.append(ticker)
        except Exception as e:
            failed.append(ticker)
            print(f"   ❌ Lỗi {ticker}: {e}")

    print(f"\n✅ Download thành công: {len(all_data)} cổ phiếu")
    if failed:
        print(f"❌ Thất bại: {failed}")

    return all_data


# ============================================================
# 1.3 — Xử lý dữ liệu thô → DataFrame sạch
# ============================================================

def clean_data(raw_data: dict) -> pd.DataFrame:
    """
    Gộp tất cả ticker vào 1 DataFrame, xử lý missing values.

    Returns:
        DataFrame với columns: [date, ticker, Open, High, Low, Close, Volume]
    """
    print("\n🧹 Đang xử lý dữ liệu...")

    frames = []
    for ticker, df in raw_data.items():
        df = df.copy()
        df['ticker'] = ticker
        df.index.name = 'date'
        frames.append(df.reset_index())

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(['date', 'ticker']).reset_index(drop=True)

    # Forward fill missing values theo từng ticker
    combined = combined.groupby('ticker', group_keys=False).apply(
        lambda x: x.ffill()
    )

    # Drop rows vẫn còn NaN ở Close
    combined = combined.dropna(subset=['Close'])

    print(f"   ✅ Cleaned: {combined.shape[0]:,} rows, {combined['ticker'].nunique()} tickers")
    print(f"   📅 Date range: {combined['date'].min().date()} → {combined['date'].max().date()}")

    return combined


# ============================================================
# 1.4 — Chia In-sample / Out-of-sample
# ============================================================

def split_data(df: pd.DataFrame) -> tuple:
    """
    Chia dữ liệu thành 2 phần:
    - In-sample:  trước 01/01/2024 → dùng để train & backtest
    - Out-sample: từ 01/01/2024 trở đi → KHÔNG ĐƯỢC XEM cho đến cuối!

    Returns:
        tuple: (in_sample, out_sample)
    """
    cutoff = pd.Timestamp(Config.OUT_SAMPLE_START)

    in_sample = df[df['date'] < cutoff].copy()
    out_sample = df[df['date'] >= cutoff].copy()

    print(f"\n📊 Data Split:")
    print(f"   In-sample:  {in_sample['date'].min().date()} → {in_sample['date'].max().date()} ({len(in_sample):,} rows)")
    print(f"   Out-sample: {out_sample['date'].min().date()} → {out_sample['date'].max().date()} ({len(out_sample):,} rows)")
    print(f"   🔒 Out-of-sample đã tách riêng — KHÔNG ĐƯỢC XEM!")

    return in_sample, out_sample


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("📦 PHASE 1: DATA PIPELINE")
    print("=" * 60)

    # Bước 1: Lấy danh sách tickers
    tickers = get_russell1000_tickers()

    # Bước 2: Download dữ liệu
    raw_data = download_data(tickers)

    # Bước 3: Xử lý dữ liệu
    clean_df = clean_data(raw_data)

    # Bước 4: Chia in-sample / out-sample
    in_sample, out_sample = split_data(clean_df)

    # Bước 5: Lưu file
    print("\n💾 Lưu dữ liệu...")
    save_data(in_sample, f"{Config.PROCESSED_DIR}/in_sample.parquet", "In-sample")
    save_data(out_sample, f"{Config.PROCESSED_DIR}/out_sample.parquet", "Out-sample")

    print("\n✅ Phase 1 hoàn tất!")
    print("👉 Tiếp theo: python phase2_feature_store.py")
