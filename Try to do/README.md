# 📊 Alpha Pipeline — Hướng dẫn sử dụng

## Cấu trúc dự án

```
Try to do/
├── config.py                  ← Cấu hình chung (dates, paths)
├── utils.py                   ← Hàm tiện ích (metrics, plot, I/O)
│
├── phase1_data_pipeline.py    ← Bước 1: Download & xử lý data
├── phase2_feature_store.py    ← Bước 2: Tạo features
├── phase3a_xgboost.py         ← Bước 3a: Train XGBoost
├── phase3b_lambdamart.py      ← Bước 3b: Train LambdaMART
├── phase3c_lstm.py            ← Bước 3c: Train LSTM
├── phase4_ensemble.py         ← Bước 4: Combine alphas
├── phase5_backtest.py         ← Bước 5: Backtest & đánh giá
│
├── data/                      ← Dữ liệu (tự tạo khi chạy)
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/                    ← Model đã train
├── alphas/                    ← Alpha predictions
└── results/                   ← Kết quả backtest (charts, CSV)
```

## Cách chạy — TỪNG BƯỚC MỘT

```bash
# Bước 0: Cài đặt thư viện
pip install yfinance pandas numpy scikit-learn lightgbm xgboost matplotlib tqdm

# Bước 1: Download dữ liệu
python phase1_data_pipeline.py

# Bước 2: Tạo features
python phase2_feature_store.py

# Bước 3: Train models (chọn 1 hoặc nhiều, KHÔNG BẮT BUỘC chạy hết)
python phase3a_xgboost.py
python phase3b_lambdamart.py      # cần thêm: pip install lightgbm
python phase3c_lstm.py            # cần thêm: pip install torch

# Bước 4: Combine alphas
python phase4_ensemble.py

# Bước 5: Backtest
python phase5_backtest.py
```

## Quy tắc quan trọng

1. **Chạy theo thứ tự**: Phase 1 → 2 → 3 → 4 → 5
2. **Phase 3 linh hoạt**: Bắt đầu với 3a (XGBoost) trước, thêm 3b/3c sau
3. **KHÔNG xem out-of-sample data** cho đến phase 5 cuối cùng
4. **Mỗi phase lưu output** vào thư mục tương ứng → phase sau tự load
