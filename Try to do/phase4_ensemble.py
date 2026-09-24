"""
====================================================================
PHASE 4: ALPHA ENSEMBLE — Kết hợp nhiều alpha thành meta alpha
====================================================================
Chạy: python phase4_ensemble.py

Input:  alphas/alpha_xgboost.parquet      (từ Phase 3a)
        alphas/alpha_lambdamart.parquet   (từ Phase 3b, nếu có)
        alphas/alpha_lstm.parquet         (từ Phase 3c, nếu có)
Output: alphas/meta_alpha.parquet

Tự động detect những alpha nào đã tạo và combine chúng.
====================================================================
"""

from config import Config
from utils import load_data, save_data, print_metrics, calculate_metrics, \
    demean_alpha, calculate_daily_pnl

import os
import numpy as np
import pandas as pd


# ============================================================
# 4.1 — Tìm và load tất cả alpha files
# ============================================================

def discover_alphas() -> dict:
    """
    Tự động tìm các file alpha_*.parquet trong thư mục alphas/.
    Bạn không cần chạy tất cả Phase 3 — chỉ cần ít nhất 1 alpha.
    """
    alpha_dir = Config.ALPHAS_DIR
    found = {}

    candidates = {
        'alpha_xgboost':    'alpha_xgboost.parquet',
        'alpha_lambdamart': 'alpha_lambdamart.parquet',
        'alpha_lstm':       'alpha_lstm.parquet',
    }

    print("🔍 Tìm kiếm alpha files...")
    for name, filename in candidates.items():
        path = os.path.join(alpha_dir, filename)
        if os.path.exists(path):
            found[name] = path
            print(f"   ✅ Tìm thấy: {filename}")
        else:
            print(f"   ⬜ Chưa có:   {filename}")

    if not found:
        raise FileNotFoundError(
            "❌ Chưa có alpha nào! Hãy chạy ít nhất Phase 3a trước.\n"
            "   → python phase3a_xgboost.py"
        )

    print(f"\n   📊 Tổng cộng: {len(found)} alpha(s) sẵn sàng")
    return found


def load_and_merge_alphas(alpha_paths: dict) -> pd.DataFrame:
    """Load tất cả alpha files và merge vào 1 DataFrame."""
    print("\n📂 Loading & merging alphas...")

    merged = None
    alpha_cols = []

    for name, path in alpha_paths.items():
        df = pd.read_parquet(path)
        keep_cols = ['date', 'ticker', name]
        if 'target_return_1d' in df.columns:
            keep_cols.append('target_return_1d')
        df = df[keep_cols]

        if merged is None:
            merged = df
        else:
            # Merge, giữ target_return_1d nếu chưa có
            merge_on = ['date', 'ticker']
            if 'target_return_1d' in merged.columns and 'target_return_1d' in df.columns:
                df = df.drop(columns=['target_return_1d'])
            merged = merged.merge(df, on=merge_on, how='inner')

        alpha_cols.append(name)

    print(f"   ✅ Merged: {len(merged):,} rows, alpha columns: {alpha_cols}")
    return merged, alpha_cols


# ============================================================
# 4.2 — Đánh giá từng alpha riêng lẻ
# ============================================================

def evaluate_individual_alphas(df, alpha_cols):
    """Backtest nhanh từng alpha để so sánh."""
    print("\n" + "─" * 50)
    print("📊 HIỆU QUẢ TỪNG ALPHA RIÊNG LẺ")
    print("─" * 50)

    results = []
    for col in alpha_cols:
        df[f'pos_{col}'] = demean_alpha(df, col)
        pnl = calculate_daily_pnl(df, f'pos_{col}')
        metrics = calculate_metrics(pnl)
        results.append({'alpha': col, **metrics})
        print(f"\n   {col}:")
        print(f"      Sharpe: {metrics['Sharpe Ratio']}  |  Return: {metrics['Annualized Return']}  |  Drawdown: {metrics['Max Drawdown']}")

    # Correlation matrix
    if len(alpha_cols) > 1:
        print(f"\n📊 Correlation giữa các alpha:")
        corr = df[alpha_cols].corr()
        print(corr.to_string(float_format=lambda x: f'{x:.3f}'))
        print("\n   💡 Alpha tốt nên có correlation THẤP với nhau (< 0.5)")

    return results


# ============================================================
# 4.3 — Combine alphas
# ============================================================

def combine_alphas(df, alpha_cols, method='equal'):
    """
    Combine nhiều alpha thành meta alpha.

    Methods:
    - 'equal':   Trung bình đơn giản (an toàn nhất)
    - 'ivol':    Inverse volatility weighting
    - 'ridge':   ML-based weighting (Ridge Regression)
    """
    print(f"\n🔗 Combining {len(alpha_cols)} alphas (method: {method})...")

    if method == 'equal':
        df['meta_alpha'] = df[alpha_cols].mean(axis=1)
        weights = {col: 1/len(alpha_cols) for col in alpha_cols}

    elif method == 'ivol':
        vols = df[alpha_cols].std()
        w = (1 / vols) / (1 / vols).sum()
        df['meta_alpha'] = (df[alpha_cols] * w.values).sum(axis=1)
        weights = dict(zip(alpha_cols, w.values))

    elif method == 'ridge':
        from sklearn.linear_model import Ridge

        train_end = pd.Timestamp(Config.TRAIN_END)
        train_mask = df['date'] <= train_end

        model = Ridge(alpha=1.0)
        model.fit(
            df.loc[train_mask, alpha_cols],
            df.loc[train_mask, 'target_return_1d']
        )
        df['meta_alpha'] = model.predict(df[alpha_cols])
        weights = dict(zip(alpha_cols, model.coef_))

    print(f"   Weights: {weights}")
    return df, weights


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🔗 PHASE 4: ALPHA ENSEMBLE")
    print("=" * 60)

    # Tìm alpha files
    alpha_paths = discover_alphas()

    # Load & merge
    df, alpha_cols = load_and_merge_alphas(alpha_paths)

    # Đánh giá từng alpha
    evaluate_individual_alphas(df, alpha_cols)

    # Combine
    method = 'equal' if len(alpha_cols) == 1 else 'ridge'
    df, weights = combine_alphas(df, alpha_cols, method=method)

    # Đánh giá meta alpha
    df['position'] = demean_alpha(df, 'meta_alpha')
    pnl = calculate_daily_pnl(df, 'position')
    metrics = calculate_metrics(pnl)
    print_metrics(metrics, f"META ALPHA ({method})")

    # Lưu
    meta_df = df[['date', 'ticker', 'meta_alpha', 'target_return_1d']].copy()
    save_data(meta_df, f"{Config.ALPHAS_DIR}/meta_alpha.parquet", "Meta Alpha")

    print("\n✅ Phase 4 hoàn tất!")
    print("👉 Tiếp theo: python phase5_backtest.py")
