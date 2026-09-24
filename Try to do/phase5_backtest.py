"""
====================================================================
PHASE 5: BACKTEST — Đánh giá chiến lược cuối cùng
====================================================================
Chạy: python phase5_backtest.py

Input:  alphas/meta_alpha.parquet  (từ Phase 4)
Output: results/backtest_report.csv
        results/equity_curve.png
        results/drawdown_chart.png

Bước này:
1. Backtest meta alpha trên IN-SAMPLE (đã làm ở Phase 4)
2. (Tùy chọn) Chạy trên OUT-OF-SAMPLE — CHỈ CHẠY 1 LẦN khi đã sẵn sàng!
====================================================================
"""

from config import Config
from utils import (load_data, save_data, calculate_metrics, print_metrics,
                   plot_equity_curve, demean_alpha, calculate_daily_pnl)

import numpy as np
import pandas as pd


def backtest_in_sample():
    """Backtest meta alpha trên in-sample data."""
    print("\n📊 Backtest trên IN-SAMPLE data...")

    df = load_data(f"{Config.ALPHAS_DIR}/meta_alpha.parquet", "Meta Alpha")

    # Demean → positions
    df['position'] = demean_alpha(df, 'meta_alpha')

    # PnL hàng ngày
    daily_pnl = calculate_daily_pnl(df, 'position')

    # Metrics
    metrics = calculate_metrics(daily_pnl)
    print_metrics(metrics, "BACKTEST IN-SAMPLE")

    # Vẽ equity curve
    plot_equity_curve(
        daily_pnl,
        title='Meta Alpha — In-Sample Equity Curve',
        save_path=f"{Config.RESULTS_DIR}/equity_curve_in_sample.png"
    )

    # Lưu metrics
    pd.DataFrame([metrics]).to_csv(
        f"{Config.RESULTS_DIR}/backtest_in_sample.csv", index=False
    )
    print(f"   💾 Report saved → results/backtest_in_sample.csv")

    return daily_pnl, metrics


def backtest_out_of_sample():
    """
    ⚠️ CHỈ CHẠY KHI ĐÃ HOÀN TOÀN SẴN SÀNG!
    ⚠️ CHỈ ĐƯỢC CHẠY 1 LẦN DUY NHẤT!

    Bạn cần:
    1. Chạy Phase 2 cho out-of-sample data (tạo features)
    2. Dùng models đã train để predict
    3. Combine thành meta alpha
    4. Chạy function này

    Nếu Sharpe ratio out-of-sample gần bằng in-sample → chiến lược tốt!
    Nếu tụt mạnh → có thể đã overfit.
    """
    print("\n" + "⚠️" * 20)
    print("   ĐÂY LÀ OUT-OF-SAMPLE BACKTEST")
    print("   CHỈ CHẠY 1 LẦN KHI ĐÃ SẴN SÀNG!")
    print("⚠️" * 20)

    # TODO: Implement khi hoàn chỉnh pipeline
    # Gợi ý:
    # 1. Load out_sample.parquet
    # 2. Chạy Feature Store trên out-sample
    # 3. Predict bằng models đã train
    # 4. Combine thành meta alpha
    # 5. Backtest

    print("\n   ⏳ Chưa implement. Hoàn thành in-sample trước!")
    print("   Khi sẵn sàng, implement function này và chạy.")


def monthly_analysis(daily_pnl: pd.Series):
    """Phân tích PnL theo tháng."""
    print("\n📅 Phân tích theo tháng:")

    monthly = daily_pnl.resample('M').sum()
    positive_months = (monthly > 0).sum()
    total_months = len(monthly)

    print(f"   Tổng tháng: {total_months}")
    print(f"   Tháng dương: {positive_months} ({positive_months/total_months:.0%})")
    print(f"   Tháng âm:    {total_months - positive_months} ({(total_months - positive_months)/total_months:.0%})")

    # Top 5 tháng tốt/xấu nhất
    best = monthly.nlargest(5)
    worst = monthly.nsmallest(5)

    print(f"\n   🏆 Top 5 tháng tốt nhất:")
    for date, val in best.items():
        print(f"      {date.strftime('%Y-%m')}: {val:+.4f}")

    print(f"\n   💀 Top 5 tháng tệ nhất:")
    for date, val in worst.items():
        print(f"      {date.strftime('%Y-%m')}: {val:+.4f}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("📈 PHASE 5: BACKTEST & ĐÁNH GIÁ")
    print("=" * 60)

    # In-sample backtest
    daily_pnl, metrics = backtest_in_sample()

    # Phân tích chi tiết
    monthly_analysis(daily_pnl)

    # Đánh giá
    sharpe = float(metrics['Sharpe Ratio'])
    print("\n" + "═" * 45)
    if sharpe > 1.5:
        print("   🏆 TUYỆT VỜI! Sharpe > 1.5")
    elif sharpe > 1.0:
        print("   ✅ TỐT! Sharpe > 1.0 — đạt mục tiêu")
    elif sharpe > 0.5:
        print("   🔶 KHÁ! Sharpe > 0.5 — cần cải thiện thêm")
    else:
        print("   ⚠️ CẦN CẢI THIỆN! Sharpe < 0.5")
        print("   Gợi ý: Thêm features, thử model khác, tune hyperparameters")
    print("═" * 45)

    print("\n📝 Bước tiếp theo:")
    print("   1. Nếu Sharpe < 1.0 → quay lại Phase 2/3 cải thiện")
    print("   2. Thêm alpha mới → Phase 3 → Phase 4 → Phase 5")
    print("   3. Khi sẵn sàng → chạy out-of-sample backtest (1 LẦN DUY NHẤT)")
