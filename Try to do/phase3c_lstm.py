"""
====================================================================
PHASE 3C: LSTM ALPHA — Deep Learning với PyTorch
====================================================================
Chạy: python phase3c_lstm.py
Cài:  pip install torch

Input:  data/features/in_sample_features.parquet  (từ Phase 2)
Output: models/lstm_model.pt
        alphas/alpha_lstm.parquet

LSTM nhận chuỗi features N ngày liên tiếp → dự đoán return ngày mai.
⚠️ Khó hơn XGBoost/LambdaMART, dễ overfit. Chỉ thử khi đã xong 3a, 3b.
====================================================================
"""

from config import Config
from utils import load_data, save_data

import numpy as np
import pandas as pd


# --- Hyperparameters ---
SEQ_LENGTH = 20       # Số ngày input (chuỗi 20 ngày → predict ngày 21)
HIDDEN_SIZE = 64      # Số neurons LSTM
NUM_LAYERS = 2        # Số lớp LSTM
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001


def create_sequences(df, feature_names, seq_length=SEQ_LENGTH):
    """
    Chuyển dữ liệu bảng → sequences cho LSTM.

    Với mỗi ticker, mỗi ngày t:
        Input:  features[t-seq_length : t]  (ma trận seq_length × n_features)
        Target: return ngày t+1

    Returns:
        X: array (n_samples, seq_length, n_features)
        y: array (n_samples,)
        info: DataFrame (date, ticker) tương ứng
    """
    print(f"\n🔧 Tạo sequences (window = {seq_length} ngày)...")

    X_list, y_list, info_list = [], [], []

    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('date').reset_index(drop=True)
        features = group[feature_names].values
        targets = group['target_return_1d'].values
        dates = group['date'].values

        for i in range(seq_length, len(group)):
            X_list.append(features[i - seq_length:i])
            y_list.append(targets[i])
            info_list.append({'date': dates[i], 'ticker': ticker})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    info = pd.DataFrame(info_list)

    # Xử lý NaN/Inf
    mask = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y)
    X, y, info = X[mask], y[mask], info[mask].reset_index(drop=True)

    print(f"   ✅ Sequences: {X.shape[0]:,} samples × {X.shape[1]} steps × {X.shape[2]} features")
    return X, y, info


def train_lstm(X_train, y_train, X_val, y_val, n_features):
    """
    Build và train LSTM model.

    Architecture:
        Input (seq_length, n_features)
        → LSTM(64, 2 layers)
        → Dropout(0.3)
        → Dense(32) → ReLU
        → Dense(1)
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    # --- Model Definition ---
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=0.3)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]  # Lấy output time step cuối
            return self.fc(last_hidden).squeeze(-1)

    # --- Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🤖 Training LSTM on {device}...")

    model = LSTMModel(n_features, HIDDEN_SIZE, NUM_LAYERS).to(device)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    best_val_loss = float('inf')
    patience, patience_counter = 5, 0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        val_loss /= len(val_loader)

        # Progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:2d}/{EPOCHS} — Train Loss: {train_loss:.6f} — Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏹ Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    print(f"   ✅ Best Val Loss: {best_val_loss:.6f}")

    return model, device


def generate_alpha_lstm(model, device, X_all, info):
    """Predict với LSTM model → alpha signal."""
    import torch

    print("\n🔮 Generating alpha predictions...")
    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_all).to(device)).cpu().numpy()

    alpha_df = info.copy()
    alpha_df['alpha_lstm'] = preds
    return alpha_df


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🧠 PHASE 3C: LSTM ALPHA (Deep Learning)")
    print("=" * 60)

    # Load data
    df = load_data(f"{Config.FEATURES_DIR}/in_sample_features.parquet", "Features")
    feature_names = pd.read_csv(
        f"{Config.FEATURES_DIR}/feature_names.csv", header=None
    )[0].tolist()

    # Tạo sequences
    X, y, info = create_sequences(df, feature_names)

    # Split theo thời gian
    train_end = pd.Timestamp(Config.TRAIN_END)
    train_mask = info['date'] <= train_end
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[~train_mask], y[~train_mask]
    print(f"\n📊 Train: {len(X_train):,} — Val: {len(X_val):,}")

    # Train
    model, device = train_lstm(X_train, y_train, X_val, y_val, len(feature_names))

    # Save model
    import torch
    torch.save(model.state_dict(), f"{Config.MODELS_DIR}/lstm_model.pt")
    print(f"\n💾 Model saved → models/lstm_model.pt")

    # Generate alpha
    alpha_df = generate_alpha_lstm(model, device, X, info)

    # Merge target để phase sau dùng
    target_df = df[['date', 'ticker', 'target_return_1d']].drop_duplicates()
    alpha_df = alpha_df.merge(target_df, on=['date', 'ticker'], how='left')

    save_data(alpha_df, f"{Config.ALPHAS_DIR}/alpha_lstm.parquet", "Alpha LSTM")

    print("\n✅ Phase 3c hoàn tất!")
    print("👉 Tiếp theo: python phase4_ensemble.py")
