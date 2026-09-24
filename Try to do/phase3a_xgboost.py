"""
====================================================================
PHASE 3A: XGBOOST ALPHA — Train model XGBoost
====================================================================
Chạy: python phase3a_xgboost.py

Input:  data/features/in_sample_features.parquet  (từ Phase 2)
Output: models/xgboost_model.json
        alphas/alpha_xgboost.parquet
====================================================================
"""

from config import Config
from utils import load_data, save_data

import numpy as np
import pandas as pd
import joblib


def load_features():
    """Load feature data và feature names từ Phase 2."""
    df = load_data(f"{Config.FEATURES_DIR}/in_sample_features.parquet", "Features")
    feature_names = pd.read_csv(
        f"{Config.FEATURES_DIR}/feature_names.csv", header=None
    )[0].tolist()
    print(f"   📋 Features: {len(feature_names)} features")
    return df, feature_names


def split_train_val(df, feature_names):
    """
    Chia in-sample → train + validation THEO THỜI GIAN.

    ⚠️ KHÔNG random split! Phải split theo thời gian
    để tránh look-ahead bias.
    """
    train_end = pd.Timestamp(Config.TRAIN_END)

    train = df[df['date'] <= train_end]
    val = df[df['date'] > train_end]

    X_train = train[feature_names].values
    y_train = train['target_return_1d'].values
    X_val = val[feature_names].values
    y_val = val['target_return_1d'].values

    print(f"\n📊 Train/Val split (theo thời gian):")
    print(f"   Train: {len(X_train):,} samples (→ {Config.TRAIN_END})")
    print(f"   Val:   {len(X_val):,} samples ({Config.TRAIN_END} →)")

    return X_train, y_train, X_val, y_val, train, val


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Train XGBoost Regressor.

    Hyperparameters đã set ở mức an toàn (ít overfit).
    TODO: Dùng Optuna để tune hyperparameters tự động.
          pip install optuna
    """
    import xgboost as xgb

    print("\n🤖 Training XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        random_state=Config.RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # Đánh giá nhanh
    from sklearn.metrics import mean_squared_error
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    print(f"   ✅ Train RMSE: {train_rmse:.6f}")
    print(f"   ✅ Val   RMSE: {val_rmse:.6f}")

    if train_rmse < val_rmse * 0.5:
        print(f"   ⚠️ Cảnh báo: Có thể overfit (train RMSE quá thấp so với val)")

    return model


def show_feature_importance(model, feature_names, top_n=10):
    """Hiển thị top features quan trọng nhất."""
    importance = dict(zip(feature_names, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n📊 Top {top_n} Feature Importance:")
    for i, (name, score) in enumerate(sorted_imp[:top_n], 1):
        bar = '█' * int(score * 100)
        print(f"   {i:2d}. {name:.<25} {score:.4f} {bar}")


def generate_alpha(model, df, feature_names):
    """
    Dùng model để predict → tạo alpha signal.
    Prediction = kỳ vọng return ngày mai → alpha signal.
    """
    print("\n🔮 Generating alpha predictions...")
    df = df.copy()
    df['alpha_xgboost'] = model.predict(df[feature_names].values)

    # Lưu alpha: chỉ giữ cột cần thiết
    alpha_df = df[['date', 'ticker', 'alpha_xgboost', 'target_return_1d']].copy()
    return alpha_df


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🤖 PHASE 3A: XGBOOST ALPHA")
    print("=" * 60)

    # Load data
    df, feature_names = load_features()

    # Split
    X_train, y_train, X_val, y_val, train_df, val_df = split_train_val(df, feature_names)

    # Train
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # Feature importance
    show_feature_importance(model, feature_names)

    # Save model
    model.save_model(f"{Config.MODELS_DIR}/xgboost_model.json")
    print(f"\n💾 Model saved → models/xgboost_model.json")

    # Generate alpha
    alpha_df = generate_alpha(model, df, feature_names)
    save_data(alpha_df, f"{Config.ALPHAS_DIR}/alpha_xgboost.parquet", "Alpha XGBoost")

    print("\n✅ Phase 3a hoàn tất!")
    print("👉 Tiếp theo:")
    print("   • python phase3b_lambdamart.py  (thêm alpha LambdaMART)")
    print("   • python phase3c_lstm.py        (thêm alpha LSTM)")
    print("   • python phase4_ensemble.py     (nếu đủ alpha rồi)")
