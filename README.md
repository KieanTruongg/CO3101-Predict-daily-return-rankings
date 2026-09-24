# 📊 Predicting Daily Ranking of Returns of Equities

> **Đồ án tổng hợp CO3101 — Trí Tuệ Nhân Tạo**  
> Đại học Bách Khoa TP.HCM | HK1 2026-2027

## 🎯 Bài toán

**Mục tiêu:** Xây dựng hệ thống AI dự đoán **thứ hạng lợi nhuận hàng ngày** (daily ranking of returns) của ~1000 cổ phiếu trên thị trường chứng khoán Mỹ.

Nói đơn giản: **mỗi ngày, hệ thống dự đoán cổ phiếu nào sẽ tăng mạnh nhất và cổ phiếu nào sẽ giảm mạnh nhất**, từ đó tạo ra chiến lược giao dịch tự động sinh lời.

Đây chính là cách các **quỹ đầu tư lượng tử (quantitative hedge funds)** lớn nhất thế giới như Renaissance Technologies, Two Sigma, WorldQuant hoạt động — họ xây dựng hàng triệu chiến lược tự động (automated alpha strategies) và kết hợp chúng lại.

### Tại sao bài toán này quan trọng?

| Khía cạnh | Chi tiết |
|-----------|----------|
| **Ứng dụng thực tế** | Đây là bài toán cốt lõi của ngành quantitative finance, quản lý hàng nghìn tỷ USD trên toàn cầu |
| **AI/ML** | Sử dụng Machine Learning (XGBoost, LightGBM, LSTM) để tìm patterns từ dữ liệu tài chính |
| **Quy mô** | ~1000 cổ phiếu × ~2500 ngày giao dịch × 50+ features = hàng triệu data points |
| **Đánh giá** | Chiến lược được đánh giá bằng Sharpe Ratio — thước đo chuẩn của ngành tài chính |

## 🏗️ Kiến trúc hệ thống

```
                    ┌───────────────┐
                    │  Market Data  │   Yahoo Finance (OHLCV)
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │ Data Pipeline │   Clean, split in/out-sample
                    └───────┬───────┘
                            ↓
                    ┌───────────────┐
                    │ Feature Store │   50+ features (price, volume, tech)
                    └───────┬───────┘
                            ↓
             ┌──────────────┼──────────────┐
             ↓              ↓              ↓
          XGBoost       LambdaMART        LSTM
          (Boosting)    (Ranking)      (Deep Learning)
             ↓              ↓              ↓
          Alpha 1        Alpha 2         Alpha 3
             └──────────────┼──────────────┘
                            ↓
                     Alpha Ensemble        Meta Alpha (ML-based)
                            ↓
                     Ranking Engine        Demean → Long-Short → PnL
                            ↓
                    ┌───────┴───────┐
                    ↓               ↓
                REST API        Database
                    ↓
                 Dashboard
                    ↑
               Monitoring
             ┌──────┴──────┐
             ↓             ↓
          Data Drift    Model Drift
             └──────┬──────┘
                    ↓
               Retraining
```

## 📖 Thuật ngữ chính

| Thuật ngữ | Giải thích |
|-----------|-----------|
| **Alpha** | Tín hiệu dự đoán cổ phiếu nào sẽ tốt/xấu hơn. Alpha dương → mua, alpha âm → bán |
| **Long-Short** | Mua cổ phiếu dự đoán tăng (long) + bán cổ phiếu dự đoán giảm (short). Tổng = 0 → không phụ thuộc thị trường |
| **Sharpe Ratio** | Lợi nhuận / Rủi ro. Sharpe > 1.0 là tốt, > 2.0 là rất tốt |
| **In-sample** | Dữ liệu trước 01/01/2024 — dùng để train & phát triển chiến lược |
| **Out-of-sample** | Dữ liệu từ 01/01/2024 — CHỈ xem 1 lần cuối cùng để kiểm tra chiến lược |
| **Meta Alpha** | Kết hợp nhiều alpha đơn lẻ → 1 alpha tổng mạnh hơn (ensemble) |
| **Russell 1000** | Chỉ số gồm 1000 công ty lớn nhất trên sàn chứng khoán Mỹ |

## 🛠️ Tech Stack

| Thành phần | Công nghệ |
|-----------|-----------|
| **Ngôn ngữ** | Python 3.10+ |
| **Dữ liệu** | Yahoo Finance (`yfinance`), `pandas`, `numpy` |
| **ML Models** | `xgboost`, `lightgbm`, `scikit-learn`, `pytorch` |
| **Visualization** | `matplotlib`, `plotly` |
| **API** | `FastAPI`, `uvicorn` |
| **Dashboard** | `Streamlit` |
| **Monitoring** | `evidently` |

## 📁 Cấu trúc thư mục

```
.
├── README.md                      ← Bạn đang đọc file này
├── description.html               ← Mô tả đồ án chi tiết (mở bằng browser)
├── workflow.html                  ← Hướng dẫn workflow + công cụ
│
├── Try to do/                     ← Code thử nghiệm (development)
│   ├── config.py                  ← Cấu hình chung
│   ├── utils.py                   ← Hàm tiện ích dùng chung
│   ├── phase1_data_pipeline.py    ← Thu thập & xử lý dữ liệu
│   ├── phase2_feature_store.py    ← Tạo 50+ features cho ML
│   ├── phase3a_xgboost.py         ← Train model XGBoost
│   ├── phase3b_lambdamart.py      ← Train model LambdaMART
│   ├── phase3c_lstm.py            ← Train model LSTM
│   ├── phase4_ensemble.py         ← Combine alphas → meta alpha
│   ├── phase5_backtest.py         ← Backtest & đánh giá chiến lược
│   └── README.md                  ← Hướng dẫn chạy từng phase
│
├── Lecture Resource/              ← Tài liệu từ giảng viên
└── Final/                         ← Bản nộp cuối cùng
```

## 🚀 Quick Start

```bash
# 1. Clone repo
git clone https://github.com/<username>/CO3101-alpha-pipeline.git
cd CO3101-alpha-pipeline

# 2. Cài đặt dependencies
pip install yfinance pandas numpy scikit-learn xgboost lightgbm matplotlib tqdm

# 3. Chạy từng bước
cd "Try to do"
python phase1_data_pipeline.py     # Download dữ liệu (~5 phút)
python phase2_feature_store.py     # Tạo features
python phase3a_xgboost.py          # Train XGBoost
python phase4_ensemble.py          # Combine alphas
python phase5_backtest.py          # Xem kết quả
```

> **Lưu ý:** Mỗi phase chạy **độc lập**. Phase sau tự động load output từ phase trước. Bạn có thể dừng ở bất kỳ phase nào, sửa code, rồi chạy lại.

## 📊 Kết quả mong đợi

| Metric | Mục tiêu | Ý nghĩa |
|--------|----------|---------|
| Sharpe Ratio | > 1.0 | Lợi nhuận bù đắp được rủi ro |
| Annualized Return | > 5% | Lợi nhuận hàng năm |
| Max Drawdown | > -15% | Mức sụt giảm tối đa chấp nhận được |
| Win Rate | > 52% | Tỷ lệ ngày có lãi |

## 📄 License

Dự án phục vụ mục đích học tập tại Đại học Bách Khoa TP.HCM.
