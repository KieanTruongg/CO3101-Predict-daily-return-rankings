"""
Cấu hình chung cho toàn bộ dự án Alpha Pipeline.
Tất cả các phase đều import từ file này.
"""

import os

# ============================================================
# THƯ MỤC GỐC = thư mục chứa file config.py này
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Cấu hình chung cho dự án."""

    # --- Market ---
    MARKET = 'US'
    UNIVERSE = 'Russell_1000'

    # --- Thời gian ---
    DATA_START = '2014-01-01'       # ≥10 năm lịch sử
    IN_SAMPLE_END = '2023-12-31'    # In-sample kết thúc
    OUT_SAMPLE_START = '2024-01-01' # ⚠️ KHÔNG ĐƯỢC XEM dữ liệu sau ngày này!
    TRAIN_END = '2022-06-30'        # Train / Validation split

    # --- Đường dẫn ---
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    FEATURES_DIR = os.path.join(DATA_DIR, 'features')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    RESULTS_DIR = os.path.join(BASE_DIR, 'results')
    ALPHAS_DIR = os.path.join(BASE_DIR, 'alphas')

    # --- Model ---
    RANDOM_SEED = 42

    @classmethod
    def make_dirs(cls):
        """Tạo tất cả thư mục cần thiết."""
        for d in [cls.RAW_DIR, cls.PROCESSED_DIR, cls.FEATURES_DIR,
                  cls.MODELS_DIR, cls.RESULTS_DIR, cls.ALPHAS_DIR]:
            os.makedirs(d, exist_ok=True)
        print("📁 Đã tạo cấu trúc thư mục:")
        print(f"   data/raw/        — dữ liệu thô")
        print(f"   data/processed/  — dữ liệu đã xử lý")
        print(f"   data/features/   — feature matrix")
        print(f"   models/          — model đã train")
        print(f"   alphas/          — alpha predictions")
        print(f"   results/         — kết quả backtest")


# Tạo thư mục khi import
Config.make_dirs()
