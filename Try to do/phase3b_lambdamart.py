"""
====================================================================
PHASE 3B: LAMBDAMART ALPHA — Learning-to-Rank với LightGBM
====================================================================
Chạy: python phase3b_lambdamart.py
Cài:  pip install lightgbm

Input:  data/features/in_sample_features.parquet  (từ Phase 2)
Output: models/lambdamart_model.txt
        alphas/alpha_lambdamart.parquet

LambdaMART dự đoán THỨ HẠNG (ranking) thay vì giá trị return.
Ưu điểm: ổn định hơn, không bị ảnh hưởng bởi market-wide movements.
====================================================================
"""

from config import Config
from utils import load_data, save_data

import numpy as np
import pandas as pd


def load_features():
    """Load feature data và feature names từ Phase 2."""
    df = load_data(f"{Config.FEATURES_DIR}/in_sample_features.parquet", "Features")
    feature_names = pd.read_csv(
        f"{Config.FEATURES_DIR}/feature_names.csv", header=None
    )[0].tolist()
    return df, feature_names


def prepare_ranking_data(df, feature_names):
    """
    Chuẩn bị dữ liệu cho LambdaMART.

    Khác với regression:
    - Target = ranking (0 = tệ nhất, N = tốt nhất) thay vì return
    - Cần group parameter = số cổ phiếu mỗi ngày
    """
    train_end = pd.Timestamp(Config.TRAIN_END)

    # Tạo target ranking theo từng ngày
    df['target_rank'] = df.groupby('date')['target_return_1d'].rank(method='first').astype(int)

    train = df[df['date'] <= train_end].copy()
    val = df[df['date'] > train_end].copy()

    # Group = số cổ phiếu mỗi ngày
    train_groups = train.groupby('date').size().values
    val_groups = val.groupby('date').size().values

    X_train = train[feature_names].values
    y_train = train['target_rank'].values
    X_val = val[feature_names].values
    y_val = val['target_rank'].values

    print(f"\n📊 Ranking data:")
    print(f"   Train: {len(X_train):,} samples, {len(train_groups)} groups")
    print(f"   Val:   {len(X_val):,} samples, {len(val_groups)} groups")

    return X_train, y_train, X_val, y_val, train_groups, val_groups


def train_lambdamart(X_train, y_train, X_val, y_val,
                     train_groups, val_groups):
    """
    Train LightGBM Ranker (LambdaMART).

    LambdaMART tối ưu trực tiếp cho ranking quality,
    không phải cho MSE như regression thông thường.
    """
    import lightgbm as lgb

    print("\n🤖 Training LambdaMART...")

    model = lgb.LGBMRanker(
        objective='lambdarank',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=Config.RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        group=train_groups,
        eval_set=[(X_val, y_val)],
        eval_group=[val_groups],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    print(f"   ✅ Best iteration: {model.best_iteration_}")
    return model


def generate_alpha(model, df, feature_names):
    """Predict ranking score → alpha signal."""
    print("\n🔮 Generating alpha predictions...")
    df = df.copy()
    df['alpha_lambdamart'] = model.predict(df[feature_names].values)

    alpha_df = df[['date', 'ticker', 'alpha_lambdamart', 'target_return_1d']].copy()
    return alpha_df


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 PHASE 3B: LAMBDAMART ALPHA (Learning-to-Rank)")
    print("=" * 60)

    # Load data
    df, feature_names = load_features()

    # Prepare ranking data
    X_train, y_train, X_val, y_val, train_groups, val_groups = \
        prepare_ranking_data(df, feature_names)

    # Train
    model = train_lambdamart(X_train, y_train, X_val, y_val,
                             train_groups, val_groups)

    # Save model
    model.booster_.save_model(f"{Config.MODELS_DIR}/lambdamart_model.txt")
    print(f"\n💾 Model saved → models/lambdamart_model.txt")

    # Generate alpha
    alpha_df = generate_alpha(model, df, feature_names)
    save_data(alpha_df, f"{Config.ALPHAS_DIR}/alpha_lambdamart.parquet", "Alpha LambdaMART")

    print("\n✅ Phase 3b hoàn tất!")
    print("👉 Tiếp theo:")
    print("   • python phase3c_lstm.py     (thêm alpha LSTM)")
    print("   • python phase4_ensemble.py  (combine alphas)")
