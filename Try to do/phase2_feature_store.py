"""
====================================================================
PHASE 2: FEATURE STORE — Tạo đặc trưng (features) cho ML
====================================================================
Chạy: python phase2_feature_store.py

Input:  data/processed/in_sample.parquet  (từ Phase 1)
Output: data/features/in_sample_features.parquet
====================================================================
"""

from config import Config
from utils import load_data, save_data

import numpy as np
import pandas as pd


# ============================================================
# 2.1 — Price Features (dựa trên giá)
# ============================================================

def add_price_features(df: pd.DataFrame) -> tuple:
    """
    Tạo features từ biến động giá.

    Features:
    - return_Xd: % thay đổi giá X ngày
    - close_to_sma_X: giá / SMA(X) - 1
    - high_low_range: (high - low) / close
    """
    names = []

    # Returns nhiều timeframe
    for period in [1, 2, 3, 5, 10, 20, 60]:
        col = f'return_{period}d'
        df[col] = df.groupby('ticker')['Close'].pct_change(period)
        names.append(col)

    # Giá so với Moving Average
    for window in [10, 20, 50]:
        sma_col = f'_sma_{window}'
        ratio_col = f'close_to_sma_{window}'
        df[sma_col] = df.groupby('ticker')['Close'].transform(
            lambda x: x.rolling(window).mean()
        )
        df[ratio_col] = df['Close'] / df[sma_col] - 1
        names.append(ratio_col)

    # High-Low Range
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    names.append('high_low_range')

    print(f"   ✅ Price features: {len(names)} features")
    return df, names


# ============================================================
# 2.2 — Technical Indicators (chỉ báo kỹ thuật)
# ============================================================

def add_technical_features(df: pd.DataFrame) -> tuple:
    """
    Tạo technical indicators: RSI, MACD, Bollinger Bands.

    Gợi ý: Cài thư viện `ta` để tính tự động 100+ indicators:
        pip install ta
        import ta
    """
    names = []

    # --- RSI (14 ngày) ---
    def calc_rsi(prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = df.groupby('ticker')['Close'].transform(
        lambda x: calc_rsi(x, 14)
    )
    names.append('RSI_14')

    # --- MACD ---
    df['_ema12'] = df.groupby('ticker')['Close'].transform(lambda x: x.ewm(span=12).mean())
    df['_ema26'] = df.groupby('ticker')['Close'].transform(lambda x: x.ewm(span=26).mean())
    df['MACD'] = df['_ema12'] - df['_ema26']
    df['MACD_signal'] = df.groupby('ticker')['MACD'].transform(lambda x: x.ewm(span=9).mean())
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    names.extend(['MACD', 'MACD_signal', 'MACD_hist'])

    # --- Bollinger Bands ---
    df['_bb_mid'] = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(20).mean())
    df['_bb_std'] = df.groupby('ticker')['Close'].transform(lambda x: x.rolling(20).std())
    df['bb_position'] = (df['Close'] - (df['_bb_mid'] - 2 * df['_bb_std'])) / \
                         (4 * df['_bb_std'])  # 0~1, vị trí trong dải Bollinger
    names.append('bb_position')

    print(f"   ✅ Technical features: {len(names)} features (RSI, MACD, Bollinger)")
    return df, names


# ============================================================
# 2.3 — Volume Features
# ============================================================

def add_volume_features(df: pd.DataFrame) -> tuple:
    """Volume-based features."""
    names = []

    df['volume_ratio_20'] = df['Volume'] / df.groupby('ticker')['Volume'].transform(
        lambda x: x.rolling(20).mean()
    )
    names.append('volume_ratio_20')

    df['volume_trend_5'] = df.groupby('ticker')['Volume'].pct_change(5)
    names.append('volume_trend_5')

    print(f"   ✅ Volume features: {len(names)} features")
    return df, names


# ============================================================
# 2.4 — Volatility Features
# ============================================================

def add_volatility_features(df: pd.DataFrame) -> tuple:
    """Volatility-based features."""
    names = []

    for window in [10, 20, 60]:
        col = f'volatility_{window}d'
        df[col] = df.groupby('ticker')['return_1d'].transform(
            lambda x: x.rolling(window).std()
        )
        names.append(col)

    # Volatility ratio: biến động ngắn hạn / dài hạn
    df['vol_ratio'] = df['volatility_10d'] / df['volatility_60d']
    names.append('vol_ratio')

    print(f"   ✅ Volatility features: {len(names)} features")
    return df, names


# ============================================================
# 2.5 — Cross-sectional Features (so sánh giữa cổ phiếu cùng ngày)
# ============================================================

def add_cross_sectional_features(df: pd.DataFrame) -> tuple:
    """Ranking features: vị trí tương đối của cổ phiếu trong ngày."""
    names = []

    df['return_1d_rank'] = df.groupby('date')['return_1d'].rank(pct=True)
    names.append('return_1d_rank')

    df['volume_ratio_rank'] = df.groupby('date')['volume_ratio_20'].rank(pct=True)
    names.append('volume_ratio_rank')

    print(f"   ✅ Cross-sectional features: {len(names)} features")
    return df, names


# ============================================================
# 2.6 — Target variable (biến mục tiêu)
# ============================================================

def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo target: return NGÀY MAI.

    ⚠️ shift(-1) = lấy giá trị tương lai → CHỈ dùng làm target,
    TUYỆT ĐỐI KHÔNG dùng làm feature!
    """
    df['target_return_1d'] = df.groupby('ticker')['Close'].pct_change(1).shift(-1)
    print(f"   🎯 Target: target_return_1d (next-day return)")
    return df


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧬 PHASE 2: FEATURE STORE")
    print("=" * 60)

    # Load dữ liệu từ Phase 1
    print("\n📂 Loading data từ Phase 1...")
    df = load_data(f"{Config.PROCESSED_DIR}/in_sample.parquet", "In-sample")

    # Tạo features
    print("\n🔧 Tạo features...")
    all_feature_names = []

    df, names = add_price_features(df)
    all_feature_names.extend(names)

    df, names = add_technical_features(df)
    all_feature_names.extend(names)

    df, names = add_volume_features(df)
    all_feature_names.extend(names)

    df, names = add_volatility_features(df)
    all_feature_names.extend(names)

    df, names = add_cross_sectional_features(df)
    all_feature_names.extend(names)

    df = create_target(df)

    # Drop NaN (do rolling windows cần warmup period)
    before = len(df)
    df = df.dropna(subset=all_feature_names + ['target_return_1d'])
    after = len(df)
    print(f"\n   🧹 Dropped {before - after:,} rows NaN (rolling warmup)")

    # Drop cột tạm (bắt đầu bằng _)
    temp_cols = [c for c in df.columns if c.startswith('_')]
    df = df.drop(columns=temp_cols)

    # Tổng kết
    print(f"\n📊 Tổng cộng: {len(all_feature_names)} features")
    print(f"   {all_feature_names}")
    print(f"   Rows: {len(df):,}")

    # Lưu
    save_data(df, f"{Config.FEATURES_DIR}/in_sample_features.parquet", "Features")

    # Lưu danh sách feature names
    pd.Series(all_feature_names).to_csv(
        f"{Config.FEATURES_DIR}/feature_names.csv", index=False, header=False
    )
    print(f"   💾 Feature names → feature_names.csv")

    print("\n✅ Phase 2 hoàn tất!")
    print("👉 Tiếp theo: python phase3a_xgboost.py")
