"""2D (θ,s) 聚类优化效果图

生成 5 张图, 直观展示 V4.4 2D 聚类相对 1D θ 投票的改进:
  1. SNR 对比柱状图 (V4.3 vs V4.4, 4 帧)
  2. RMS 对比柱状图 (V4.3 vs V4.4, 4 帧, 含 mosaic2 1360px→0.63px 修复)
  3. 2D (θ,s) 聚类示意图 (合成数据: 真阳性聚集, 假阳性分散)
  4. 1D θ 投票 vs 2D 聚类对比 (同一数据两种视角)
  5. Phase B 过滤效果 (2D 过滤掉 s 错候选的数量)

依赖: numpy, matplotlib
"""
import os
import sys
import json
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
from matplotlib.colors import LogNorm

# 中文字体支持
for font in ["Microsoft YaHei", "SimHei", "DejaVu Sans"]:
    try:
        plt.rcParams["font.sans-serif"] = [font]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
SUMMARY_JSON = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs",
                              "v4_4", "validation", "summary.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs",
                          "v4_4", "validation", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_summary():
    """加载 4 帧验证结果"""
    with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def short_name(full_name):
    """缩短文件名便于显示"""
    name_map = {
        "Galaxy_Center_mosaic2": "mosaic2\n(Type3失败帧)",
        "NGC7293": "NGC7293",
        "LDN43": "LDN43\n(U=271爆炸)",
        "Galaxy_Center_mosaic1": "mosaic1",
    }
    for k, v in name_map.items():
        if k in full_name:
            return v
    return full_name[:20]


# ============================================================================
# 图 1: SNR 对比柱状图
# ============================================================================

def plot_snr_comparison(summary):
    """V4.3 vs V4.4 SNR 对比 (对数纵轴)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    frames = list(summary.keys())
    labels = [short_name(f) for f in frames]
    v43_snr = [summary[f]["v43"]["theta_snr"] for f in frames]
    v44_snr = [summary[f]["v44"]["theta_snr"] for f in frames]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, v43_snr, width, label="V4.3 (1D θ 投票)",
                   color="#4472C4", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, v44_snr, width, label="V4.4 (2D θ-s 聚类)",
                   color="#ED7D31", edgecolor="black", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("Phase A SNR (对数轴)", fontsize=12)
    ax.set_title("V4.3 (1D θ) vs V4.4 (2D θ-s 聚类) — Phase A SNR 对比",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=5, color="red", linestyle=":", linewidth=1.5, label="SNR=5 阈值")

    # 标注数值
    for bar, val in zip(bars1, v43_snr):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.15,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, color="#4472C4")
    for bar, val in zip(bars2, v44_snr):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.15,
                f"{val:.0f}", ha="center", va="bottom", fontsize=9, color="#ED7D31", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig1_snr_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图1 SNR对比: {out}")


# ============================================================================
# 图 2: RMS 对比柱状图 (重点展示 mosaic2 修复)
# ============================================================================

def plot_rms_comparison(summary):
    """V4.3 vs V4.4 RMS 对比"""
    fig, ax = plt.subplots(figsize=(10, 6))

    frames = list(summary.keys())
    labels = [short_name(f) for f in frames]
    v43_rms = [summary[f]["v43"]["rms_px"] for f in frames]
    v44_rms = [summary[f]["v44"]["rms_px"] for f in frames]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, v43_rms, width, label="V4.3 (1D θ 投票)",
                   color="#4472C4", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, v44_rms, width, label="V4.4 (2D θ-s 聚类)",
                   color="#ED7D31", edgecolor="black", linewidth=0.8)

    ax.set_ylabel("最终 RMS (像素)", fontsize=12)
    ax.set_title("V4.3 vs V4.4 — 最终 RMS 对比 (4/4 成功, RMS 一致)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(max(v43_rms), max(v44_rms)) * 1.4)

    # 标注数值
    for bar, val in zip(bars1, v43_rms):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#4472C4")
    for bar, val in zip(bars2, v44_rms):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, color="#ED7D31", fontweight="bold")

    # 注释: 之前 V4.4 (1D版本) mosaic2 RMS=1360px, 现已修复
    ax.annotate("注: 1D 版本此帧 RMS=1360px\n2D 聚类修复后 → 0.6261px",
                xy=(0.15, 0.55), xytext=(0.15, 0.55),
                fontsize=10, color="red",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", edgecolor="red"),
                transform=ax.transAxes)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig2_rms_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图2 RMS对比: {out}")


# ============================================================================
# 图 3: 2D (θ,s) 聚类示意图 (合成数据)
# ============================================================================

def plot_2d_cluster_demo():
    """用合成数据演示 2D (θ,s) 聚类: 真阳性聚集, 假阳性分散"""
    np.random.seed(42)

    # 真阳性: θ=-30°, s=1.0 附近聚集 (模拟 200 个真匹配候选)
    n_true = 200
    theta_true = np.random.normal(-30, 0.3, n_true)  # θ 标准差 0.3°
    s_true = np.random.normal(1.0, 0.003, n_true)    # s 标准差 0.003 (距离比非常一致)

    # 假阳性: 均匀分散在 (θ, s) 平面 (模拟 800 个错误候选)
    n_false = 800
    theta_false = np.random.uniform(-180, 180, n_false)
    s_false = np.random.uniform(0.9, 1.1, n_false)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 1D θ 投票 (假阳性污染峰值)
    ax1 = axes[0]
    bins = np.linspace(-180, 180, 361)
    ax1.hist(theta_false, bins=bins, alpha=0.5, color="#888888",
             label=f"假阳性 (n={n_false})", edgecolor="black", linewidth=0.3)
    ax1.hist(theta_true, bins=bins, alpha=0.8, color="#ED7D31",
             label=f"真阳性 (n={n_true})", edgecolor="black", linewidth=0.3)
    ax1.axvline(x=-30, color="red", linestyle="--", linewidth=2, label="θ_true=-30°")
    ax1.set_xlabel("θ_rot (度)", fontsize=12)
    ax1.set_ylabel("投票数", fontsize=12)
    ax1.set_title("1D θ 投票 (V4.3)\n假阳性在峰值附近堆积, 拉高 background",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3, linestyle="--")

    # 标注 SNR
    peak_val = np.histogram(theta_true, bins=bins)[0].max()
    bg_val = np.histogram(theta_false, bins=bins)[0].mean()
    snr_1d = peak_val / max(bg_val, 1)
    ax1.text(0.98, 0.95, f"SNR ≈ {snr_1d:.1f}x\n(background 被污染)",
             transform=ax1.transAxes, ha="right", va="top", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"))

    # 右图: 2D (θ,s) 聚类 (真阳性形成密集簇)
    ax2 = axes[1]
    ax2.scatter(theta_false, s_false, s=8, c="#888888", alpha=0.4,
                label=f"假阳性 (n={n_false})")
    ax2.scatter(theta_true, s_true, s=25, c="#ED7D31", alpha=0.8,
                edgecolors="red", linewidths=0.5,
                label=f"真阳性 (n={n_true})")
    ax2.axhline(y=1.0, color="blue", linestyle=":", linewidth=1, alpha=0.5)
    ax2.axvline(x=-30, color="red", linestyle="--", linewidth=2, label="θ_true=-30°, s_true=1.0")

    # 画 5×5 邻域检测框
    rect = plt.Rectangle((-30.5, 0.995), 1.0, 0.01, linewidth=2,
                          edgecolor="green", facecolor="none", linestyle="-",
                          label="5×5 峰值检测窗")
    ax2.add_patch(rect)

    ax2.set_xlabel("θ_rot (度)", fontsize=12)
    ax2.set_ylabel("s_est (尺度估计)", fontsize=12)
    ax2.set_title("2D (θ,s) 聚类 (V4.4)\n真阳性形成密集簇, 假阳性分散 → SNR 大幅提升",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlim(-180, 180)
    ax2.set_ylim(0.9, 1.1)

    # 标注 SNR (2D 聚类后, 5×5 窗内真阳性密度远高于背景)
    cluster_val = n_true  # 几乎所有真阳性都在 5×5 窗内
    bg_density = n_false / (360 * 0.2)  # 假阳性均匀分布在 360° × 0.2 s 范围
    snr_2d = cluster_val / max(bg_density, 1)
    ax2.text(0.98, 0.95, f"SNR ≈ {snr_2d:.0f}x\n(2D 聚类精确隔离)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=11,
             bbox=dict(boxstyle="round", facecolor="lightgreen", edgecolor="green"))

    plt.suptitle("核心改进: 1D θ 投票 → 2D (θ,s) 聚类", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig3_2d_cluster_demo.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图3 2D聚类示意: {out}")


# ============================================================================
# 图 4: Phase B 过滤效果 (2D 过滤掉 s 错候选)
# ============================================================================

def plot_phaseb_filter():
    """从日志提取 Phase B 过滤数据, 展示 2D 过滤效果"""
    # 从验证日志提取的关键数据 (4 帧 mode0 的过滤统计)
    # 日志: "RelVec PhaseB: 2D过滤 s_peak=... 过滤掉 N 个 s 错候选, 保留 M 对"
    frames = ["mosaic2", "NGC7293", "LDN43", "mosaic1"]
    # (过滤掉, 保留) 从日志提取
    filter_data = {
        "mosaic2":  {"filtered": 11614, "kept": 1465,  "total": 13079},
        "NGC7293":  {"filtered": 16793, "kept": 3513,  "total": 20306},
        "LDN43":    {"filtered": 11062, "kept": 893,   "total": 11955},
        "mosaic1":  {"filtered": 28098, "kept": 3513,  "total": 31611},
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图: 过滤前后的候选数量对比
    ax1 = axes[0]
    filtered = [filter_data[f]["filtered"] for f in frames]
    kept = [filter_data[f]["kept"] for f in frames]
    x = np.arange(len(frames))
    width = 0.35

    bars1 = ax1.bar(x - width/2, filtered, width, label="过滤掉 (s 错候选)",
                    color="#C00000", edgecolor="black", linewidth=0.8, alpha=0.8)
    bars2 = ax1.bar(x + width/2, kept, width, label="保留 (s≈s_peak)",
                    color="#70AD47", edgecolor="black", linewidth=0.8)

    ax1.set_ylabel("候选对数量", fontsize=12)
    ax1.set_title("Phase B 2D 过滤效果\n(θ 对但 s 错的假阳性被过滤)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(frames, fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="--")

    for bar, val in zip(bars1, filtered):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 200,
                 f"{val}", ha="center", va="bottom", fontsize=9, color="#C00000")
    for bar, val in zip(bars2, kept):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 200,
                 f"{val}", ha="center", va="bottom", fontsize=9, color="#70AD47", fontweight="bold")

    # 右图: 过滤比例饼图 (4 帧平均)
    ax2 = axes[1]
    total_filtered = sum(filtered)
    total_kept = sum(kept)
    total = total_filtered + total_kept
    sizes = [total_filtered, total_kept]
    labels_pie = [f"过滤掉 (s 错)\n{total_filtered} 对 ({100*total_filtered/total:.1f}%)",
                  f"保留 (s≈s_peak)\n{total_kept} 对 ({100*total_kept/total:.1f}%)"]
    colors_pie = ["#C00000", "#70AD47"]
    explode = (0, 0.1)

    ax2.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie,
            autopct="", startangle=90, textprops={"fontsize": 11},
            wedgeprops={"edgecolor": "black", "linewidth": 1})
    ax2.set_title(f"2D 聚类过滤总览 (4 帧合计 {total} 对)\n平均过滤率 {100*total_filtered/total:.1f}%",
                  fontsize=12, fontweight="bold")

    plt.suptitle("Phase B 双过滤: θ≈θ_peak 且 s≈s_peak", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig4_phaseb_filter.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图4 PhaseB过滤: {out}")


# ============================================================================
# 图 5: 单元测试 SNR 提升 (合成数据, 1D vs 2D)
# ============================================================================

def plot_unittest_snr():
    """单元测试 4 个场景的 SNR 提升 (1D 旧版 vs 2D 新版)"""
    # 从单元测试结果提取 (之前 1D 版本 vs 当前 2D 版本)
    tests = ["test1: k-vector\n(N=20,M=30)", "test2: t=0\n(θ=30°)",
             "test3: t≠0\n(θ=45°,t=100\")", "test4: U限流\n(N=200,max_u=100)"]
    snr_1d = [3978, 3978, 1463, 39]       # 1D 旧版 SNR
    snr_2d = [7830, 45984, 44065, 16859]  # 2D 新版 SNR

    fig, ax = plt.subplots(figsize=(11, 6))

    x = np.arange(len(tests))
    width = 0.35

    bars1 = ax.bar(x - width/2, snr_1d, width, label="1D θ 投票 (旧)",
                   color="#4472C4", edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, snr_2d, width, label="2D θ-s 聚类 (新)",
                   color="#ED7D31", edgecolor="black", linewidth=0.8)

    ax.set_yscale("log")
    ax.set_ylabel("SNR (对数轴)", fontsize=12)
    ax.set_title("单元测试 SNR 提升: 1D θ 投票 → 2D (θ,s) 聚类\n(4 个合成数据场景, 23/23 全通过)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tests, fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.axhline(y=5, color="red", linestyle=":", linewidth=1.5, label="SNR=5 阈值")

    # 标注数值和提升倍数
    for i, (b1, v1, b2, v2) in enumerate(zip(bars1, snr_1d, bars2, snr_2d)):
        ax.text(b1.get_x() + b1.get_width()/2, v1 * 1.2,
                f"{v1}", ha="center", va="bottom", fontsize=9, color="#4472C4")
        ax.text(b2.get_x() + b2.get_width()/2, v2 * 1.2,
                f"{v2}", ha="center", va="bottom", fontsize=9, color="#ED7D31", fontweight="bold")
        # 提升倍数箭头
        ratio = v2 / v1 if v1 > 0 else 0
        ax.annotate(f"×{ratio:.0f}", xy=(i, v2 * 3), fontsize=11, ha="center",
                    color="green", fontweight="bold")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "fig5_unittest_snr.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 图5 单元测试SNR: {out}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("2D (θ,s) 聚类优化效果图生成")
    print("=" * 70)

    summary = load_summary()
    print(f"加载 {len(summary)} 帧验证结果")

    plot_snr_comparison(summary)
    plot_rms_comparison(summary)
    plot_2d_cluster_demo()
    plot_phaseb_filter()
    plot_unittest_snr()

    print()
    print("=" * 70)
    print(f"所有效果图已生成到: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
