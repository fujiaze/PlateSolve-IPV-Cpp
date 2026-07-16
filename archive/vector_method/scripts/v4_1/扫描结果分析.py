"""V4.1 扫描结果分析 — 星点数量 N_total 最优值选择

功能: 读取 sweep_all.json, 按 N_total 分组统计成功率/RMS/耗时/10px命中率,
      绘制曲线图, 推荐最优 N_total。

用途: V4.1 版本参数整定的决策依据

用法: python 扫描结果分析.py [sweep_all.json路径]
      默认: lib/plate_solve/logs/v4/sweep/sweep_all.json
"""
import os
import sys
import json
import math
import traceback
from collections import defaultdict, OrderedDict

import numpy as np

# ============================================================================
# UTF-8 编码初始化（Windows GBK 兼容）
# ============================================================================
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# matplotlib 配置（Agg 后端 + 中文字体）
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 路径初始化
# ============================================================================
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
SWEEP_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "sweep")
DEFAULT_SWEEP_JSON = os.path.join(SWEEP_DIR, "sweep_all.json")

# 输出文件
OUT_CSV = os.path.join(SWEEP_DIR, "sweep_summary.csv")
OUT_PNG = os.path.join(SWEEP_DIR, "sweep_curves.png")
OUT_REPORT = os.path.join(SWEEP_DIR, "最优N推荐报告.md")

# 已知滤镜列表（用于分组分析排序）
KNOWN_FILTERS = ['Lum', 'Red', 'Green', 'Blue', 'H-alpha', 'OIII', 'SII']

# 成功率目标阈值
SUCCESS_RATE_TARGET = 99.0


# ============================================================================
# 数据加载
# ============================================================================
def load_sweep_results(json_path):
    """加载 sweep_all.json, 返回 list of dict

    Args:
        json_path: sweep_all.json 路径

    Returns:
        list of dict, 每条记录包含 filename/n_total/status/rms_px 等字段。
        失败时返回 None。
    """
    if not os.path.exists(json_path):
        print(f"[错误] 扫描结果文件不存在: {json_path}")
        print(f"       请先运行 参数扫描实验.py 生成扫描数据。")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[错误] 读取扫描结果失败: {e}")
        return None

    if not isinstance(data, list) or len(data) == 0:
        print(f"[错误] 扫描结果为空或格式不正确: {json_path}")
        print(f"       请确认扫描实验已产生数据。")
        return None

    return data


# ============================================================================
# 统计工具
# ============================================================================
def _safe_float(v, default=0.0):
    """安全转 float, None/异常返回 default"""
    if v is None:
        return default
    try:
        f = float(v)
        if not math.isfinite(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _median(values):
    """计算中位数, 空列表返回 None"""
    if not values:
        return None
    return float(np.median(values))


def _percentile(values, q):
    """计算分位数, 空列表返回 None"""
    if not values:
        return None
    return float(np.percentile(values, q))


def compute_n_stats(records):
    """对一组记录（同一 N_total）计算统计指标

    Args:
        records: list of dict, 同一 N_total 的所有记录

    Returns:
        dict: 统计指标
    """
    n_total_count = len(records)
    success_records = [r for r in records if r.get('status') == 'success']
    n_success = len(success_records)
    success_rate = 100.0 * n_success / n_total_count if n_total_count > 0 else 0.0

    # 成功帧的指标
    rms_list = [_safe_float(r.get('rms_px')) for r in success_records if r.get('rms_px') is not None]
    time_list = [_safe_float(r.get('solve_time_s')) for r in success_records if r.get('solve_time_s') is not None]
    pct10_list = [_safe_float(r.get('pct_10px')) for r in success_records if r.get('pct_10px') is not None]
    matched_list = [_safe_float(r.get('matched_count')) for r in success_records if r.get('matched_count') is not None]
    pct5_list = [_safe_float(r.get('pct_5px')) for r in success_records if r.get('pct_5px') is not None]
    n_detected_list = [_safe_float(r.get('n_detected')) for r in success_records if r.get('n_detected') is not None]
    n_saturated_list = [_safe_float(r.get('n_saturated')) for r in success_records if r.get('n_saturated') is not None]

    # 失败原因分布
    fail_reasons = defaultdict(int)
    for r in records:
        if r.get('status') != 'success':
            fail_reasons[r.get('status', 'unknown')] += 1

    return {
        'n_total_value': records[0].get('n_total') if records else None,
        'n_records': n_total_count,
        'n_success': n_success,
        'n_fail': n_total_count - n_success,
        'success_rate': success_rate,
        'med_rms': _median(rms_list),
        'p25_rms': _percentile(rms_list, 25),
        'p75_rms': _percentile(rms_list, 75),
        'med_solve_time': _median(time_list),
        'med_pct_10px': _median(pct10_list),
        'med_pct_5px': _median(pct5_list),
        'med_matched': _median(matched_list),
        'med_n_detected': _median(n_detected_list),
        'med_n_saturated': _median(n_saturated_list),
        'fail_reasons': dict(fail_reasons),
    }


def group_by_n_total(data):
    """按 n_total 分组, 返回 OrderedDict{n_total: [records]}, 按 n_total 升序"""
    groups = defaultdict(list)
    for r in data:
        n = r.get('n_total')
        if n is None:
            continue
        groups[int(n)].append(r)
    return OrderedDict(sorted(groups.items()))


# ============================================================================
# 最优 N 选择
# ============================================================================
def select_best_n(stats_by_n):
    """根据选择标准选出最优 N_total

    优先级:
        1. 成功率最高（目标 ≥99%）
        2. 中位 RMS 最低
        3. 中位耗时最低（次要）
        4. 中位 10px 命中率最高

    Args:
        stats_by_n: OrderedDict {n_total: stats_dict}

    Returns:
        (best_n, best_stats, reason_text)
    """
    if not stats_by_n:
        return None, None, "无数据"

    # 1) 成功率最高优先
    max_sr = max(s['success_rate'] for s in stats_by_n.values())
    candidates = {n: s for n, s in stats_by_n.items() if s['success_rate'] >= max_sr - 1e-9}

    reason_lines = []
    reason_lines.append(f"第一步: 成功率最高 = {max_sr:.2f}%, 候选 N = {sorted(candidates.keys())}")

    # 2) 在成功率并列的候选中, 选 RMS 最低（忽略 RMS 为 None 的）
    cand_with_rms = {n: s for n, s in candidates.items() if s['med_rms'] is not None}
    if cand_with_rms:
        min_rms = min(s['med_rms'] for s in cand_with_rms.values())
        candidates = {n: s for n, s in cand_with_rms.items() if s['med_rms'] <= min_rms + 1e-9}
        reason_lines.append(f"第二步: 中位 RMS 最低 = {min_rms:.4f}px, 候选 N = {sorted(candidates.keys())}")

    # 3) 耗时最低（次要）
    cand_with_time = {n: s for n, s in candidates.items() if s['med_solve_time'] is not None}
    if cand_with_time:
        min_time = min(s['med_solve_time'] for s in cand_with_time.values())
        candidates = {n: s for n, s in cand_with_time.items() if s['med_solve_time'] <= min_time + 1e-9}
        reason_lines.append(f"第三步: 中位耗时最低 = {min_time:.3f}s, 候选 N = {sorted(candidates.keys())}")

    # 4) 10px 命中率最高
    cand_with_pct = {n: s for n, s in candidates.items() if s['med_pct_10px'] is not None}
    if cand_with_pct:
        max_pct = max(s['med_pct_10px'] for s in cand_with_pct.values())
        candidates = {n: s for n, s in cand_with_pct.items() if s['med_pct_10px'] >= max_pct - 1e-9}
        reason_lines.append(f"第四步: 10px 命中率最高 = {max_pct:.2f}%, 候选 N = {sorted(candidates.keys())}")

    # 若仍并列, 取最小 N（更快、更省内存）
    best_n = min(candidates.keys())
    best_stats = candidates[best_n]
    reason_lines.append(f"最终: 选取最小 N = {best_n}（并列时取小值, 计算更快）")

    return best_n, best_stats, "\n".join(reason_lines)


# ============================================================================
# CSV 输出
# ============================================================================
def write_summary_csv(stats_by_n, csv_path):
    """输出 sweep_summary.csv, 每行一个 N_total"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    headers = [
        'N_total', '总帧数', '成功帧数', '失败帧数', '成功率(%)',
        '中位RMS(px)', 'RMS_P25(px)', 'RMS_P75(px)',
        '中位耗时(s)', '中位10px命中率(%)', '中位5px命中率(%)',
        '中位匹配对数', '中位检测星数', '中位饱和星数', '失败原因分布'
    ]

    import csv
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for n, s in stats_by_n.items():
            fail_str = '; '.join(f'{k}:{v}' for k, v in sorted(s['fail_reasons'].items(), key=lambda x: -x[1])) if s['fail_reasons'] else ''
            writer.writerow([
                n,
                s['n_records'],
                s['n_success'],
                s['n_fail'],
                f"{s['success_rate']:.2f}",
                f"{s['med_rms']:.4f}" if s['med_rms'] is not None else '',
                f"{s['p25_rms']:.4f}" if s['p25_rms'] is not None else '',
                f"{s['p75_rms']:.4f}" if s['p75_rms'] is not None else '',
                f"{s['med_solve_time']:.3f}" if s['med_solve_time'] is not None else '',
                f"{s['med_pct_10px']:.2f}" if s['med_pct_10px'] is not None else '',
                f"{s['med_pct_5px']:.2f}" if s['med_pct_5px'] is not None else '',
                f"{s['med_matched']:.1f}" if s['med_matched'] is not None else '',
                f"{s['med_n_detected']:.0f}" if s['med_n_detected'] is not None else '',
                f"{s['med_n_saturated']:.0f}" if s['med_n_saturated'] is not None else '',
                fail_str,
            ])

    print(f"[输出] CSV: {csv_path}")


# ============================================================================
# 绘图: sweep_curves.png
# ============================================================================
def plot_curves(stats_by_n, png_path, best_n=None):
    """绘制 4 子图曲线: 成功率/RMS/耗时/10px命中率

    Args:
        stats_by_n: OrderedDict {n_total: stats}
        png_path: 输出 PNG 路径
        best_n: 最优 N_total（用于在图上标记）
    """
    if not stats_by_n:
        print("[跳过] 无数据, 不绘制曲线图")
        return

    ns = sorted(stats_by_n.keys())
    sr = [stats_by_n[n]['success_rate'] for n in ns]
    med_rms = [stats_by_n[n]['med_rms'] for n in ns]
    p25_rms = [stats_by_n[n]['p25_rms'] for n in ns]
    p75_rms = [stats_by_n[n]['p75_rms'] for n in ns]
    med_time = [stats_by_n[n]['med_solve_time'] for n in ns]
    med_pct10 = [stats_by_n[n]['med_pct_10px'] for n in ns]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('V4.1 星点数量 N_total 扫描结果', fontsize=15, fontweight='bold')

    def _mark_best(ax, xs, ys, best_n_val, higher_is_better=True):
        """在最优点画红圈 + 标注"""
        if best_n_val not in xs:
            return
        idx = xs.index(best_n_val)
        yv = ys[idx]
        if yv is None:
            return
        ax.scatter([best_n_val], [yv], s=180, facecolors='none',
                   edgecolors='red', linewidths=2.5, zorder=5)
        ax.annotate(f'最优 N={best_n_val}\n({yv:.2f})',
                    xy=(best_n_val, yv),
                    xytext=(15, 15), textcoords='offset points',
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.2, alpha=0.7))

    # ── 子图1: 成功率 ──
    ax = axes[0, 0]
    ax.plot(ns, sr, 'o-', color='#1f77b4', linewidth=2, markersize=7)
    ax.axhline(y=SUCCESS_RATE_TARGET, color='green', linestyle='--', alpha=0.6,
               label=f'目标 {SUCCESS_RATE_TARGET}%')
    ax.set_xlabel('N_total (星点数)')
    ax.set_ylabel('成功率 (%)')
    ax.set_title('成功率 vs N_total')
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    _mark_best(ax, ns, sr, best_n, higher_is_better=True)

    # ── 子图2: 中位 RMS + P25/P75 误差带 ──
    ax = axes[0, 1]
    rms_valid = [(n, v) for n, v in zip(ns, med_rms) if v is not None]
    if rms_valid:
        xs_v, ys_v = zip(*rms_valid)
        p25_v = [stats_by_n[n]['p25_rms'] for n in xs_v]
        p75_v = [stats_by_n[n]['p75_rms'] for n in xs_v]
        ax.plot(xs_v, ys_v, 'o-', color='#d62728', linewidth=2, markersize=7, label='中位 RMS')
        ax.fill_between(xs_v, p25_v, p75_v, color='#d62728', alpha=0.15, label='P25-P75 带')
        if best_n is not None and best_n in xs_v:
            _mark_best(ax, list(xs_v), list(ys_v), best_n, higher_is_better=False)
    ax.set_xlabel('N_total (星点数)')
    ax.set_ylabel('RMS (px)')
    ax.set_title('中位 RMS vs N_total (含 P25/P75 带)')
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # ── 子图3: 中位耗时 ──
    ax = axes[1, 0]
    time_valid = [(n, v) for n, v in zip(ns, med_time) if v is not None]
    if time_valid:
        xs_v, ys_v = zip(*time_valid)
        ax.plot(xs_v, ys_v, 's-', color='#2ca02c', linewidth=2, markersize=7)
        if best_n is not None and best_n in xs_v:
            _mark_best(ax, list(xs_v), list(ys_v), best_n, higher_is_better=False)
    ax.set_xlabel('N_total (星点数)')
    ax.set_ylabel('求解耗时 (s)')
    ax.set_title('中位求解耗时 vs N_total')
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.grid(True, alpha=0.3)

    # ── 子图4: 中位 10px 命中率 ──
    ax = axes[1, 1]
    pct_valid = [(n, v) for n, v in zip(ns, med_pct10) if v is not None]
    if pct_valid:
        xs_v, ys_v = zip(*pct_valid)
        ax.plot(xs_v, ys_v, '^-', color='#ff7f0e', linewidth=2, markersize=8)
        if best_n is not None and best_n in xs_v:
            _mark_best(ax, list(xs_v), list(ys_v), best_n, higher_is_better=True)
    ax.set_xlabel('N_total (星点数)')
    ax.set_ylabel('10px 命中率 (%)')
    ax.set_title('中位 10px 命中率 vs N_total')
    ax.set_xscale('log')
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[输出] PNG: {png_path}")


# ============================================================================
# 分组分析
# ============================================================================
def analyze_by_filter(data):
    """按滤镜分组分析每个 N_total 的成功率

    Returns:
        dict: {filter: {n_total: stats}}
    """
    by_filter = defaultdict(list)
    for r in data:
        flt = r.get('filter', 'unknown')
        by_filter[flt].append(r)

    result = OrderedDict()
    # 已知滤镜优先, 其余按字母序
    filter_order = KNOWN_FILTERS + sorted([f for f in by_filter.keys() if f not in KNOWN_FILTERS])
    for flt in filter_order:
        if flt not in by_filter:
            continue
        stats_n = group_by_n_total(by_filter[flt])
        if not stats_n:
            continue
        result[flt] = OrderedDict()
        for n, recs in stats_n.items():
            result[flt][n] = compute_n_stats(recs)
    return result


def analyze_by_target(data):
    """按目标分组分析每个 N_total 的成功率

    Returns:
        dict: {target: {n_total: stats}}
    """
    by_target = defaultdict(list)
    for r in data:
        tgt = r.get('target', 'unknown')
        by_target[tgt].append(r)

    result = OrderedDict()
    for tgt in sorted(by_target.keys()):
        stats_n = group_by_n_total(by_target[tgt])
        if not stats_n:
            continue
        result[tgt] = OrderedDict()
        for n, recs in stats_n.items():
            result[tgt][n] = compute_n_stats(recs)
    return result


def analyze_always_fail_frames(data):
    """分析在所有 N 值下都失败的帧

    Returns:
        (always_fail_frames, frame_status_map)
        - always_fail_frames: list of (filename, target, filter, reasons_set)
        - frame_status_map: {filename: {n_total: status}}
    """
    # 收集每帧每个 N 的状态
    frame_n_status = defaultdict(dict)
    frame_meta = {}
    for r in data:
        fn = r.get('filename')
        n = r.get('n_total')
        if fn is None or n is None:
            continue
        frame_n_status[fn][int(n)] = r.get('status', 'unknown')
        if fn not in frame_meta:
            frame_meta[fn] = {
                'target': r.get('target', 'unknown'),
                'filter': r.get('filter', 'unknown'),
                'exposure_s': r.get('exposure_s', 0),
            }

    always_fail = []
    for fn, n_map in frame_n_status.items():
        if not n_map:
            continue
        # 该帧至少跑过一个 N, 且全部失败
        if all(st != 'success' for st in n_map.values()):
            reasons = set(n_map.values())
            always_fail.append((fn, frame_meta[fn], reasons, len(n_map)))

    return always_fail, frame_n_status


def analyze_stability(data):
    """稳定性分析: 同一帧在不同 N 值下 RMS 的变化幅度

    Returns:
        dict: {filename: {n_total: rms_px}}（仅成功帧）
        全局稳定性指标: rms_std_by_frame (list)
    """
    frame_n_rms = defaultdict(dict)
    for r in data:
        if r.get('status') != 'success':
            continue
        fn = r.get('filename')
        n = r.get('n_total')
        rms = r.get('rms_px')
        if fn is None or n is None or rms is None:
            continue
        frame_n_rms[fn][int(n)] = _safe_float(rms)

    # 仅保留有 ≥2 个 N 成功的帧, 计算 RMS 标准差
    rms_std_by_frame = []
    rms_range_by_frame = []
    for fn, n_map in frame_n_rms.items():
        if len(n_map) < 2:
            continue
        vals = list(n_map.values())
        rms_std_by_frame.append(float(np.std(vals, ddof=0)))
        rms_range_by_frame.append(max(vals) - min(vals))

    return frame_n_rms, rms_std_by_frame, rms_range_by_frame


def analyze_saturated_vs_n(data):
    """饱和星数影响分析: n_saturated 与最优 N 的关系

    将帧按 n_saturated 分桶, 统计每桶在不同 N 下的成功率。

    Returns:
        dict: {bucket_label: {n_total: (n_total_count, n_success, success_rate)}}
    """
    # 按 n_saturated 分桶
    def _bucket(ns):
        if ns is None:
            return 'unknown'
        ns = int(ns)
        if ns < 50:
            return '<50'
        elif ns < 100:
            return '50-100'
        elif ns < 200:
            return '100-200'
        elif ns < 500:
            return '200-500'
        else:
            return '>=500'

    by_bucket = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # bucket -> n_total -> [total, success]
    for r in data:
        ns = r.get('n_saturated')
        n = r.get('n_total')
        if n is None:
            continue
        bk = _bucket(ns)
        by_bucket[bk][int(n)][0] += 1
        if r.get('status') == 'success':
            by_bucket[bk][int(n)][1] += 1

    # 排序桶
    bucket_order = ['<50', '50-100', '100-200', '200-500', '>=500', 'unknown']
    result = OrderedDict()
    for bk in bucket_order:
        if bk not in by_bucket:
            continue
        result[bk] = OrderedDict()
        for n in sorted(by_bucket[bk].keys()):
            total, succ = by_bucket[bk][n]
            sr = 100.0 * succ / total if total > 0 else 0.0
            result[bk][n] = (total, succ, sr)
    return result


# ============================================================================
# 报告输出: 最优N推荐报告.md
# ============================================================================
def fmt_num(v, fmt='{:.4f}'):
    """格式化数值, None 返回 '-'"""
    if v is None:
        return '-'
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return '-'


def write_report(stats_by_n, best_n, best_stats, selection_reason,
                 filter_stats, target_stats, always_fail, frame_n_status,
                 stability_data, saturated_data, data, report_path):
    """输出最优N推荐报告.md"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    lines = []
    L = lines.append

    L("# V4.1 星点数量 N_total 最优值推荐报告")
    L("")
    L(f"**生成依据**: `sweep_all.json` ({len(data)} 条记录)")
    L(f"**扫描 N_total 列表**: {sorted(stats_by_n.keys())}")
    L(f"**成功率目标**: ≥ {SUCCESS_RATE_TARGET}%")
    L("")

    # ── 1. 推荐结论 ──
    L("## 1. 推荐结论")
    L("")
    if best_n is not None:
        L(f"### 🎯 推荐最优 N_total = **{best_n}**")
        L("")
        L(f"- 成功率: **{best_stats['success_rate']:.2f}%** "
          f"({best_stats['n_success']}/{best_stats['n_records']})")
        L(f"- 中位 RMS: **{fmt_num(best_stats['med_rms'], '{:.4f}')} px** "
          f"(P25={fmt_num(best_stats['p25_rms'])}, P75={fmt_num(best_stats['p75_rms'])})")
        L(f"- 中位求解耗时: **{fmt_num(best_stats['med_solve_time'], '{:.3f}')} s**")
        L(f"- 中位 10px 命中率: **{fmt_num(best_stats['med_pct_10px'], '{:.2f}')}%**")
        L(f"- 中位匹配对数: **{fmt_num(best_stats['med_matched'], '{:.1f}')}**")
        L(f"- 中位检测星数: **{fmt_num(best_stats['med_n_detected'], '{:.0f}')}**")
        L(f"- 中位饱和星数: **{fmt_num(best_stats['med_n_saturated'], '{:.0f}')}**")
        L("")
        L("### 选择过程")
        L("")
        L("```")
        L(selection_reason)
        L("```")
        L("")

        # 是否达到目标
        if best_stats['success_rate'] >= SUCCESS_RATE_TARGET:
            L(f"✅ 达到成功率目标 (≥{SUCCESS_RATE_TARGET}%)")
        else:
            L(f"⚠️ 当前最优成功率 {best_stats['success_rate']:.2f}% 未达到目标 {SUCCESS_RATE_TARGET}%")
            L(f"   可能原因: 扫描未完成 / 部分帧本身无解 / 数据集特殊")
        L("")
    else:
        L("⚠️ 无法推荐: 无有效扫描数据")
        L("")

    # ── 2. 各 N_total 统计表 ──
    L("## 2. 各 N_total 统计表")
    L("")
    L("| N_total | 总帧数 | 成功 | 失败 | 成功率(%) | 中位RMS(px) | P25 RMS | P75 RMS | 中位耗时(s) | 中位10px(%) | 中位匹配数 | 中位检测星 | 中位饱和星 |")
    L("|---------|--------|------|------|-----------|-------------|---------|---------|-------------|-------------|-----------|-----------|-----------|")
    for n, s in stats_by_n.items():
        L(f"| {n} | {s['n_records']} | {s['n_success']} | {s['n_fail']} | "
          f"{s['success_rate']:.2f} | {fmt_num(s['med_rms'])} | {fmt_num(s['p25_rms'])} | {fmt_num(s['p75_rms'])} | "
          f"{fmt_num(s['med_solve_time'], '{:.3f}')} | {fmt_num(s['med_pct_10px'], '{:.2f}')} | "
          f"{fmt_num(s['med_matched'], '{:.1f}')} | {fmt_num(s['med_n_detected'], '{:.0f}')} | "
          f"{fmt_num(s['med_n_saturated'], '{:.0f}')} |")
    L("")

    # 失败原因分布
    L("### 失败原因分布")
    L("")
    L("| N_total | 失败原因 | 数量 |")
    L("|---------|----------|------|")
    for n, s in stats_by_n.items():
        if not s['fail_reasons']:
            L(f"| {n} | (无失败) | 0 |")
        else:
            for reason, cnt in sorted(s['fail_reasons'].items(), key=lambda x: -x[1]):
                L(f"| {n} | {reason} | {cnt} |")
    L("")

    # ── 3. 按滤镜分组 ──
    L("## 3. 按滤镜分组分析")
    L("")
    L("不同滤镜（Lum/Red/Green/Blue/H-alpha/OIII/SII）的星点密度和饱和特性不同, 最优 N 可能不同。")
    L("")
    for flt, n_stats in filter_stats.items():
        L(f"### 滤镜: {flt}")
        L("")
        L("| N_total | 总帧数 | 成功 | 成功率(%) | 中位RMS(px) | 中位耗时(s) | 中位10px(%) | 中位饱和星 |")
        L("|---------|--------|------|-----------|-------------|-------------|-------------|-----------|")
        for n, s in n_stats.items():
            L(f"| {n} | {s['n_records']} | {s['n_success']} | {s['success_rate']:.2f} | "
              f"{fmt_num(s['med_rms'])} | {fmt_num(s['med_solve_time'], '{:.3f}')} | "
              f"{fmt_num(s['med_pct_10px'], '{:.2f}')} | {fmt_num(s['med_n_saturated'], '{:.0f}')} |")
        # 该滤镜最优 N
        best_n_f, best_s_f, _ = select_best_n(n_stats)
        if best_n_f is not None:
            L("")
            L(f"**该滤镜最优 N = {best_n_f}** "
              f"(成功率 {best_s_f['success_rate']:.2f}%, "
              f"RMS {fmt_num(best_s_f['med_rms'])} px)")
        L("")

    # ── 4. 按目标分组 ──
    L("## 4. 按目标分组分析")
    L("")
    L("| 目标 | N_total | 总帧数 | 成功 | 成功率(%) | 中位RMS(px) | 中位10px(%) |")
    L("|------|---------|--------|------|-----------|-------------|-------------|")
    for tgt, n_stats in target_stats.items():
        # 取该目标成功率最高（并列取最小N）
        best_n_t, best_s_t, _ = select_best_n(n_stats)
        for n, s in n_stats.items():
            mark = ' ⭐' if (best_n_t is not None and n == best_n_t) else ''
            L(f"| {tgt} | {n}{mark} | {s['n_records']} | {s['n_success']} | "
              f"{s['success_rate']:.2f} | {fmt_num(s['med_rms'])} | "
              f"{fmt_num(s['med_pct_10px'], '{:.2f}')} |")
    L("")

    # ── 5. 失败帧分析 ──
    L("## 5. 失败帧分析")
    L("")
    if always_fail:
        L(f"### 在所有 N 值下都失败的帧: **{len(always_fail)}** 个")
        L("")
        L("| 文件名 | 目标 | 滤镜 | 曝光(s) | 失败 N 数 | 失败原因集合 |")
        L("|--------|------|------|---------|-----------|--------------|")
        for fn, meta, reasons, n_fail_count in always_fail:
            L(f"| {fn} | {meta['target']} | {meta['filter']} | {meta['exposure_s']} | "
              f"{n_fail_count} | {', '.join(sorted(reasons))} |")
        L("")
        L("> 这些帧在所有 N 值下均失败, 可能原因:")
        L("> - 图像本身问题（无星、严重失焦、运动模糊）")
        L("> - OBJCTRA/OBJCTDEC 缺失或错误")
        L("> - 星点检测阈值过高, 检测星数不足")
        L("> - 视场极端（如目标位于天区边缘, Gaia 数据稀疏）")
        L("")
    else:
        L("✅ 没有在所有 N 值下都失败的帧。")
        L("")

    # 部分失败（仅某些 N 失败）
    partial_fail = []
    for fn, n_map in frame_n_status.items():
        n_total_n = len(n_map)
        n_fail = sum(1 for st in n_map.values() if st != 'success')
        if 0 < n_fail < n_total_n:
            partial_fail.append((fn, n_fail, n_total_n, n_map))

    if partial_fail:
        L(f"### 部分失败帧（仅某些 N 值失败）: **{len(partial_fail)}** 个")
        L("")
        L("| 文件名 | 失败 N 数 / 总 N 数 | 失败的 N 值及原因 |")
        L("|--------|---------------------|--------------------|")
        for fn, n_fail, n_total_n, n_map in partial_fail[:30]:  # 最多列 30 个
            fail_detail = ', '.join(f'N{n}={st}' for n, st in sorted(n_map.items()) if st != 'success')
            L(f"| {fn} | {n_fail}/{n_total_n} | {fail_detail} |")
        if len(partial_fail) > 30:
            L(f"| ... | ... | (共 {len(partial_fail)} 个, 仅显示前 30) |")
        L("")

    # ── 6. 饱和星数影响 ──
    L("## 6. 饱和星数影响分析")
    L("")
    L("按 n_saturated 分桶, 分析不同饱和星数水平下各 N_total 的成功率。")
    L("（饱和星多的帧可能需要更大 N 以提供足够非饱和星点）")
    L("")
    L("| 饱和星数桶 | N_total | 总帧数 | 成功 | 成功率(%) |")
    L("|------------|---------|--------|------|-----------|")
    for bk, n_stats in saturated_data.items():
        for n, (total, succ, sr) in n_stats.items():
            L(f"| {bk} | {n} | {total} | {succ} | {sr:.2f} |")
    L("")

    # ── 7. 稳定性分析 ──
    L("## 7. 稳定性分析")
    L("")
    L("同一帧在不同 N 值下 RMS 的变化幅度（仅含 ≥2 个 N 成功的帧）。")
    L("")
    frame_n_rms, rms_std_list, rms_range_list = stability_data
    if rms_std_list:
        med_std = float(np.median(rms_std_list))
        max_std = float(np.max(rms_std_list))
        med_range = float(np.median(rms_range_list))
        max_range = float(np.max(rms_range_list))
        L(f"- 中位 RMS 标准差: **{med_std:.4f} px**")
        L(f"- 最大 RMS 标准差: **{max_std:.4f} px**")
        L(f"- 中位 RMS 极差 (max-min): **{med_range:.4f} px**")
        L(f"- 最大 RMS 极差: **{max_range:.4f} px**")
        L("")
        if med_std < 0.1:
            L("✅ 同帧 RMS 跨 N 值变化小, 求解稳定。")
        elif med_std < 0.5:
            L("⚠️ 同帧 RMS 跨 N 值有一定变化, 建议关注。")
        else:
            L("❌ 同帧 RMS 跨 N 值变化较大, N 选择对结果有显著影响。")
        L("")

        # 列出最不稳定的 5 个帧
        unstable = []
        for fn, n_map in frame_n_rms.items():
            if len(n_map) < 2:
                continue
            vals = list(n_map.values())
            std = float(np.std(vals, ddof=0))
            unstable.append((fn, std, min(vals), max(vals), sorted(n_map.keys())))
        unstable.sort(key=lambda x: -x[1])
        L("#### 最不稳定的 5 个帧（RMS 标准差最大）:")
        L("")
        L("| 文件名 | RMS 标准差 | 最小 RMS | 最大 RMS | 涉及 N 值 |")
        L("|--------|-----------|----------|----------|-----------|")
        for fn, std, vmin, vmax, ns in unstable[:5]:
            L(f"| {fn} | {std:.4f} | {vmin:.4f} | {vmax:.4f} | {ns} |")
        L("")
    else:
        L("无足够数据（同一帧至少 2 个 N 成功）进行稳定性分析。")
        L("")

    # ── 8. 数据完整性提示 ──
    L("## 8. 数据完整性")
    L("")
    expected_n_values = sorted(stats_by_n.keys())
    L(f"- 扫描覆盖的 N_total 值: {expected_n_values}")
    for n, s in stats_by_n.items():
        L(f"  - N={n}: {s['n_records']} 条记录")
    L("")
    L("> 注: 若扫描仍在进行, 部分统计可能不完整。建议扫描全部 633 帧完成后重新运行本脚本。")
    L("")

    L("---")
    L(f"*报告由 `扫描结果分析.py` 自动生成*")
    L("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"[输出] 报告: {report_path}")


# ============================================================================
# 主流程
# ============================================================================
def main():
    # 解析命令行参数
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        json_path = DEFAULT_SWEEP_JSON

    print(f"=== V4.1 扫描结果分析 ===")
    print(f"输入: {json_path}")
    print(f"输出目录: {SWEEP_DIR}")
    print()

    # 1. 加载数据
    data = load_sweep_results(json_path)
    if data is None:
        sys.exit(1)

    print(f"[加载] 共 {len(data)} 条记录")

    # 2. 按 N_total 分组统计
    stats_by_n = group_by_n_total(data)
    if not stats_by_n:
        print("[错误] 无法按 n_total 分组, 数据可能缺少 n_total 字段")
        sys.exit(1)

    print(f"[分组] N_total 值: {sorted(stats_by_n.keys())}")
    for n, recs in stats_by_n.items():
        print(f"  N={n}: {len(recs)} 条")
    print()

    # 计算每个 N 的统计
    for n in stats_by_n:
        stats_by_n[n] = compute_n_stats(stats_by_n[n])

    # 3. 选择最优 N
    best_n, best_stats, reason = select_best_n(stats_by_n)
    if best_n is not None:
        print(f"[最优] N_total = {best_n} "
              f"(成功率 {best_stats['success_rate']:.2f}%, "
              f"RMS {fmt_num(best_stats['med_rms'])} px, "
              f"耗时 {fmt_num(best_stats['med_solve_time'], '{:.3f}')} s)")
    print()

    # 4. 输出 CSV
    write_summary_csv(stats_by_n, OUT_CSV)

    # 5. 绘制曲线图
    plot_curves(stats_by_n, OUT_PNG, best_n=best_n)

    # 6. 额外分析
    print("[分析] 按滤镜分组...")
    filter_stats = analyze_by_filter(data)
    print(f"  覆盖滤镜: {list(filter_stats.keys())}")

    print("[分析] 按目标分组...")
    target_stats = analyze_by_target(data)
    print(f"  覆盖目标: {list(target_stats.keys())}")

    print("[分析] 失败帧分析...")
    always_fail, frame_n_status = analyze_always_fail_frames(data)
    print(f"  全 N 失败帧: {len(always_fail)} 个")

    print("[分析] 稳定性分析...")
    stability_data = analyze_stability(data)
    frame_n_rms, rms_std_list, rms_range_list = stability_data
    if rms_std_list:
        print(f"  可分析帧数: {len(rms_std_list)}, 中位 RMS 标准差: {float(np.median(rms_std_list)):.4f} px")

    print("[分析] 饱和星数影响...")
    saturated_data = analyze_saturated_vs_n(data)
    print(f"  饱和桶: {list(saturated_data.keys())}")
    print()

    # 7. 输出报告
    write_report(stats_by_n, best_n, best_stats, reason,
                 filter_stats, target_stats, always_fail, frame_n_status,
                 stability_data, saturated_data, data, OUT_REPORT)

    print()
    print("=== 分析完成 ===")
    print(f"  CSV : {OUT_CSV}")
    print(f"  PNG : {OUT_PNG}")
    print(f"  报告: {OUT_REPORT}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"[致命错误] {e}")
        traceback.print_exc()
        sys.exit(2)
