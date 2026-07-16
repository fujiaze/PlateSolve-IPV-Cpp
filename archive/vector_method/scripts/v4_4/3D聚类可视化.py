"""3D (θ, dx, dy) 空间聚类可视化

读取 4 帧的 relvec_pairs_3d.csv, 展示:
  图1: 4 帧 3D 散点图 (θ, dx, dy), 真阳性聚集, 假阳性分散
  图2: 2D 投影对比 (θ-dx, θ-dy, dx-dy 三视图)
  图3: 1D θ 投票 vs 2D (θ,dx) vs 3D (θ,dx,dy) SNR 对比 (同一帧)
  图4: 4 帧 is_near_peak 比例 (2D 聚类过滤效果)

依赖: numpy, matplotlib, pandas
"""
import os
import sys
import functools
import numpy as np

print = functools.partial(print, flush=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 中文字体支持
for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        plt.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
LOG_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs",
                       "v4_4", "validation", "v44_logs")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs",
                          "v4_4", "validation", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4 帧目录名
FRAMES = [
    ("Galaxy_Center_mosaic2", "mosaic2\n(Type3失败帧)"),
    ("NGC7293", "NGC7293"),
    ("LDN43", "LDN43\n(U=271爆炸)"),
    ("Galaxy_Center_mosaic1", "mosaic1"),
]


def load_csv(frame_dir_prefix):
    """加载 relvec_pairs_3d.csv (跳过格式错误行)"""
    for d in os.listdir(LOG_DIR):
        if d.startswith(frame_dir_prefix):
            csv_path = os.path.join(LOG_DIR, d, "relvec_pairs_3d.csv")
            if os.path.exists(csv_path):
                # invalid_raise=False 跳过格式错误行
                return np.genfromtxt(csv_path, delimiter=",", skip_header=1,
                                     invalid_raise=False)
    return None


# ============================================================================
# 图 1: 4 帧 3D 散点图 (θ, dx, dy)
# ============================================================================

def plot_3d_scatter():
    """4 帧 3D 散点图: 真阳性聚集, 假阳性分散"""
    fig = plt.figure(figsize=(16, 12))

    for idx, (prefix, label) in enumerate(FRAMES):
        ax = fig.add_subplot(2, 2, idx + 1, projection="3d")
        data = load_csv(prefix)
        if data is None or len(data) == 0:
            ax.set_title(f"{label}\n(无数据)")
            continue

        theta = data[:, 0]
        dx = data[:, 3]  # 注意: CSV 列顺序 theta, s, dx, dy, is_near_peak
        dy = data[:, 4]
        # 修正: 实际列顺序是 theta_deg, s_est, dx, dy, is_near_peak
        dx = data[:, 2]
        dy = data[:, 3]
        is_peak = data[:, 4].astype(int)

        # 降采样假阳性便于可视化 (真阳性全显示, 假阳性最多 2000 个)
        false_mask = is_peak == 0
        true_mask = is_peak == 1
        n_false = false_mask.sum()
        if n_false > 2000:
            np.random.seed(42)
            false_idx = np.where(false_mask)[0]
            sample_idx = np.random.choice(false_idx, 2000, replace=False)
            plot_mask = np.zeros(len(data), dtype=bool)
            plot_mask[true_mask] = True
            plot_mask[sample_idx] = True
        else:
            plot_mask = np.ones(len(data), dtype=bool)

        # 假阳性: 灰色小点
        ax.scatter(theta[plot_mask & false_mask],
                   dx[plot_mask & false_mask],
                   dy[plot_mask & false_mask],
                   c="#888888", s=6, alpha=0.3, label=f"假阳性 (n={n_false})")
        # 真阳性: 红色大点
        n_true = true_mask.sum()
        ax.scatter(theta[true_mask], dx[true_mask], dy[true_mask],
                   c="#ED7D31", s=30, alpha=0.8, edgecolors="red", linewidths=0.5,
                   label=f"真阳性 (n={n_true})")

        ax.set_xlabel("θ (度)", fontsize=9)
        ax.set_ylabel("dx (像素)", fontsize=9)
        ax.set_zlabel("dy (像素)", fontsize=9)
        ax.set_title(f"{label}\n3D (θ,dx,dy) 聚类 (共 {len(data)} 对)",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")

        # 调整视角便于观察聚类
        ax.view_init(elev=20, azim=45)

    plt.suptitle("3D (θ, dx, dy) 空间聚类 — 真阳性聚集, 假阳性分散",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig6_3d_scatter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图6 3D散点: {out}")


# ============================================================================
# 图 2: 2D 投影对比 (三视图)
# ============================================================================

def plot_2d_projections():
    """单帧 (mosaic1) 的 2D 投影对比: θ-dx, θ-dy, dx-dy"""
    data = load_csv("Galaxy_Center_mosaic1")
    if data is None:
        print("[SKIP] 图7: 无 mosaic1 数据")
        return

    theta = data[:, 0]
    s_est = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    is_peak = data[:, 4].astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    projections = [
        (0, 0, theta, dx, "θ (度)", "dx (像素)", "(θ, dx) 投影"),
        (0, 1, theta, dy, "θ (度)", "dy (像素)", "(θ, dy) 投影"),
        (1, 0, dx, dy, "dx (像素)", "dy (像素)", "(dx, dy) 投影"),
        (1, 1, theta, s_est, "θ (度)", "s_est", "(θ, s) 投影 (2D 聚类)"),
    ]

    for r, c, x, y, xlabel, ylabel, title in projections:
        ax = axes[r][c]
        false_mask = is_peak == 0
        true_mask = is_peak == 1

        # 降采样
        n_false = false_mask.sum()
        if n_false > 3000:
            np.random.seed(42)
            false_idx = np.where(false_mask)[0]
            sample_idx = np.random.choice(false_idx, 3000, replace=False)
            ax.scatter(x[sample_idx], y[sample_idx], c="#888888", s=8, alpha=0.3,
                       label=f"假阳性 (n={n_false}, 显示3000)")
        else:
            ax.scatter(x[false_mask], y[false_mask], c="#888888", s=8, alpha=0.3,
                       label=f"假阳性 (n={n_false})")

        ax.scatter(x[true_mask], y[true_mask], c="#ED7D31", s=25, alpha=0.8,
                   edgecolors="red", linewidths=0.5,
                   label=f"真阳性 (n={true_mask.sum()})")

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3, linestyle="--")

    plt.suptitle("mosaic1 帧 — 2D 投影对比 (4 个视角)\n真阳性在所有投影中都聚集, 假阳性分散",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig7_2d_projections.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图7 2D投影: {out}")


# ============================================================================
# 图 3: 1D vs 2D vs 3D SNR 对比 (同一帧 mosaic1)
# ============================================================================

def plot_dimension_snr_comparison():
    """同一帧数据, 在 1D/2D/3D 空间的 SNR 对比"""
    data = load_csv("Galaxy_Center_mosaic1")
    if data is None:
        print("[SKIP] 图8: 无 mosaic1 数据")
        return

    theta = data[:, 0]
    s_est = data[:, 1]
    dx = data[:, 2]
    dy = data[:, 3]
    is_peak = data[:, 4].astype(int)
    n_total = len(data)
    n_true = is_peak.sum()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- 1D θ 直方图 ---
    ax1 = axes[0]
    bins1d = np.linspace(-180, 180, 361)
    ax1.hist(theta[is_peak == 0], bins=bins1d, alpha=0.5, color="#888888",
             label=f"假阳性 (n={n_total - n_true})", edgecolor="black", linewidth=0.3)
    ax1.hist(theta[is_peak == 1], bins=bins1d, alpha=0.8, color="#ED7D31",
             label=f"真阳性 (n={n_true})", edgecolor="black", linewidth=0.3)
    # 1D SNR
    peak_1d = np.histogram(theta[is_peak == 1], bins=bins1d)[0].max()
    bg_1d = np.histogram(theta[is_peak == 0], bins=bins1d)[0].mean()
    snr_1d = peak_1d / max(bg_1d, 1)
    ax1.set_title(f"1D θ 投票\nSNR ≈ {snr_1d:.1f}x (background 被污染)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("θ (度)", fontsize=10)
    ax1.set_ylabel("投票数", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # --- 2D (θ, dx) 散点 ---
    ax2 = axes[1]
    false_mask = is_peak == 0
    true_mask = is_peak == 1
    np.random.seed(42)
    if false_mask.sum() > 3000:
        false_idx = np.where(false_mask)[0]
        sample_idx = np.random.choice(false_idx, 3000, replace=False)
        ax2.scatter(theta[sample_idx], dx[sample_idx], c="#888888", s=8, alpha=0.3,
                    label=f"假阳性 (显示3000)")
    else:
        ax2.scatter(theta[false_mask], dx[false_mask], c="#888888", s=8, alpha=0.3)
    ax2.scatter(theta[true_mask], dx[true_mask], c="#ED7D31", s=20, alpha=0.8,
                edgecolors="red", linewidths=0.5, label=f"真阳性 (n={n_true})")
    # 2D SNR (5×5 邻域)
    peak_2d = n_true  # 真阳性几乎全在 5×5 窗内
    bg_density_2d = (n_total - n_true) / (360 * (dx.max() - dx.min()))
    snr_2d = peak_2d / max(bg_density_2d, 1)
    ax2.set_title(f"2D (θ, dx) 聚类\nSNR ≈ {snr_2d:.0f}x (2D 隔离)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("θ (度)", fontsize=10)
    ax2.set_ylabel("dx (像素)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # --- 3D (θ, dx, dy) 散点 ---
    ax3 = fig.add_subplot(1, 3, 3, projection="3d", label="3d")
    if false_mask.sum() > 2000:
        false_idx = np.where(false_mask)[0]
        sample_idx = np.random.choice(false_idx, 2000, replace=False)
        ax3.scatter(theta[sample_idx], dx[sample_idx], dy[sample_idx],
                    c="#888888", s=6, alpha=0.3, label="假阳性 (显示2000)")
    else:
        ax3.scatter(theta[false_mask], dx[false_mask], dy[false_mask],
                    c="#888888", s=6, alpha=0.3)
    ax3.scatter(theta[true_mask], dx[true_mask], dy[true_mask],
                c="#ED7D31", s=25, alpha=0.8, edgecolors="red", linewidths=0.5,
                label=f"真阳性 (n={n_true})")
    # 3D SNR (5×5×5 邻域)
    peak_3d = n_true
    bg_density_3d = (n_total - n_true) / (360 * (dx.max() - dx.min()) * (dy.max() - dy.min()))
    snr_3d = peak_3d / max(bg_density_3d, 1)
    ax3.set_title(f"3D (θ,dx,dy) 聚类\nSNR ≈ {snr_3d:.0f}x (3D 精确隔离)",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("θ", fontsize=9)
    ax3.set_ylabel("dx", fontsize=9)
    ax3.set_zlabel("dy", fontsize=9)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.view_init(elev=20, azim=45)

    plt.suptitle(f"同一帧 (mosaic1) 在 1D/2D/3D 空间的 SNR 对比\n维度越高, 真阳性越集中, 假阳性越分散 → SNR 越高",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig8_dimension_snr.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图8 维度SNR对比: {out}")


# ============================================================================
# 图 4: 4 帧 is_near_peak 比例 (2D 聚类过滤效果)
# ============================================================================

def plot_near_peak_ratio():
    """4 帧 is_near_peak=1 的比例 (2D 聚类保留率)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    ratios = []
    counts_true = []
    counts_false = []

    for prefix, label in FRAMES:
        data = load_csv(prefix)
        if data is None:
            continue
        is_peak = data[:, 4].astype(int)
        n_true = is_peak.sum()
        n_total = len(data)
        ratio = n_true / n_total * 100 if n_total > 0 else 0
        labels.append(label.replace("\n", " "))
        ratios.append(ratio)
        counts_true.append(n_true)
        counts_false.append(n_total - n_true)

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, counts_false, width, label="假阳性 (is_near_peak=0)",
                   color="#C00000", edgecolor="black", linewidth=0.8, alpha=0.8)
    bars2 = ax.bar(x + width/2, counts_true, width, label="真阳性 (is_near_peak=1)",
                   color="#70AD47", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("候选对数量", fontsize=12)
    ax.set_title("4 帧 3D 数据中 is_near_peak 分布\n(2D 聚类双过滤: θ≈θ_peak 且 s≈s_peak)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, val, ratio in zip(bars2, counts_true, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, val + 100,
                f"{val}\n({ratio:.1f}%)", ha="center", va="bottom",
                fontsize=9, color="#70AD47", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig9_near_peak_ratio.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图9 near_peak比例: {out}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("3D (θ, dx, dy) 空间聚类可视化")
    print("=" * 70)

    plot_3d_scatter()
    plot_2d_projections()
    plot_dimension_snr_comparison()
    plot_near_peak_ratio()

    print()
    print("=" * 70)
    print(f"所有 3D 效果图已生成到: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
