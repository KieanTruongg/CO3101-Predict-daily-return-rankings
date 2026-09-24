"""
Hàm tiện ích dùng chung cho toàn bộ dự án.
Import: from utils import load_data, save_data, calculate_metrics, plot_equity_curve
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# I/O — Đọc / Ghi dữ liệu
# ============================================================

def save_data(df: pd.DataFrame, path: str, name: str = ''):
    """Lưu DataFrame ra file parquet."""
    df.to_parquet(path, index=False)
    rows = len(df)
    cols = len(df.columns)
    print(f"   💾 Đã lưu {name}: {rows:,} rows × {cols} cols → {os.path.basename(path)}")


def load_data(path: str, name: str = '') -> pd.DataFrame:
    """Đọc DataFrame từ file parquet."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ Không tìm thấy file: {path}\n"
            f"   → Hãy chạy phase trước đó trước!"
        )
    df = pd.read_parquet(path)
    print(f"   📂 Đã load {name}: {len(df):,} rows × {len(df.columns)} cols ← {os.path.basename(path)}")
    return df


# ============================================================
# METRICS — Tính toán hiệu quả
# ============================================================

def calculate_metrics(daily_pnl: pd.Series) -> dict:
    """
    Tính các metrics đánh giá hiệu quả chiến lược.

    Args:
        daily_pnl: Series PnL hàng ngày (index = date)

    Returns:
        dict với các metrics
    """
    cumulative = (1 + daily_pnl).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_years = len(daily_pnl) / 252
    annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    # Max Drawdown
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # Win rate
    win_rate = (daily_pnl > 0).mean()

    return {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Trading Days': len(daily_pnl),
    }


def print_metrics(metrics: dict, title: str = 'KẾT QUẢ'):
    """In bảng metrics đẹp ra console."""
    print(f"\n{'─' * 45}")
    print(f"  📊 {title}")
    print(f"{'─' * 45}")
    for key, value in metrics.items():
        print(f"   {key:.<30} {value}")
    print(f"{'─' * 45}")


# ============================================================
# PLOT — Vẽ biểu đồ
# ============================================================

def plot_equity_curve(daily_pnl: pd.Series, title: str = 'Equity Curve',
                      save_path: str = None):
    """
    Vẽ đường equity curve + drawdown.

    Args:
        daily_pnl: Series PnL hàng ngày
        title: Tiêu đề biểu đồ
        save_path: Đường dẫn lưu file (None = không lưu)
    """
    import matplotlib.pyplot as plt

    cumulative = (1 + daily_pnl).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1])

    # Equity curve
    axes[0].plot(cumulative.index, cumulative.values, color='#a78bfa', linewidth=1.5)
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Drawdown
    peak = cumulative.cummax()
    dd = (cumulative - peak) / peak
    axes[1].fill_between(dd.index, dd.values, 0, color='#ef4444', alpha=0.3)
    axes[1].set_ylabel('Drawdown')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   📈 Chart saved → {os.path.basename(save_path)}")

    plt.show()


# ============================================================
# ALPHA HELPERS — Hàm hỗ trợ xử lý alpha
# ============================================================

def demean_alpha(df: pd.DataFrame, alpha_col: str) -> pd.Series:
    """
    Demean alpha signal theo từng ngày → market neutral.
    Sau demean: Σ alpha = 0 mỗi ngày.
    """
    return df.groupby('date')[alpha_col].transform(lambda x: x - x.mean())


def calculate_daily_pnl(df: pd.DataFrame, position_col: str,
                        return_col: str = 'target_return_1d') -> pd.Series:
    """
    Tính PnL hàng ngày.
    PnL = Σ (position_i × return_i) cho mỗi ngày.
    """
    return (df[position_col] * df[return_col]).groupby(df['date']).sum()
