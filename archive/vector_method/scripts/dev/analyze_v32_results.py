"""V3.2全量测试结果分析"""
import csv, sys
from collections import defaultdict
import numpy as np

csv_path = sys.argv[1] if len(sys.argv) > 1 else r"F:\Astro dev\Astro CS Normalization Database\v32_robustness_test_results.csv"

with open(csv_path, encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

n = len(rows)
ok = [r for r in rows if r['success'] == 'True']
fail = [r for r in rows if r['success'] != 'True']

print(f'总帧数: {n}')
print(f'成功: {len(ok)} ({len(ok)/n*100:.1f}%)')
print(f'失败: {len(fail)} ({len(fail)/n*100:.1f}%)')

# 按望远镜统计
by_tel = defaultdict(lambda: {'total': 0, 'ok': 0, 'rms': [], 'solve_t': []})
for r in rows:
    tel = r['telescope'] or 'unknown'
    by_tel[tel]['total'] += 1
    if r['success'] == 'True':
        by_tel[tel]['ok'] += 1
        by_tel[tel]['rms'].append(float(r['rms_px']))
        by_tel[tel]['solve_t'].append(float(r['t_solve_s']))

print('\n按望远镜统计:')
print(f'  {"望远镜":<10} {"总数":>5} {"成功":>5} {"成功率":>7} {"中位RMS":>8} {"中位耗时":>8}')
for tel in sorted(by_tel.keys()):
    d = by_tel[tel]
    med_rms = np.median(d['rms']) if d['rms'] else 0
    med_t = np.median(d['solve_t']) if d['solve_t'] else 0
    pct = d['ok'] / d['total'] * 100 if d['total'] else 0
    print(f'  {tel:<10} {d["total"]:>5} {d["ok"]:>5} {pct:>6.1f}% {med_rms:>7.3f}px {med_t:>7.2f}s')

# 按滤镜统计
by_filt = defaultdict(lambda: {'total': 0, 'ok': 0, 'rms': []})
for r in rows:
    filt = r['filter_name'] or 'unknown'
    by_filt[filt]['total'] += 1
    if r['success'] == 'True':
        by_filt[filt]['ok'] += 1
        by_filt[filt]['rms'].append(float(r['rms_px']))

print('\n按滤镜统计:')
print(f'  {"滤镜":<10} {"总数":>5} {"成功":>5} {"成功率":>7} {"中位RMS":>8}')
for filt in ['Lum', 'Red', 'Green', 'Blue', 'H-alpha', 'Oiii', 'Sii', 'unknown']:
    if filt not in by_filt:
        continue
    d = by_filt[filt]
    med_rms = np.median(d['rms']) if d['rms'] else 0
    pct = d['ok'] / d['total'] * 100 if d['total'] else 0
    print(f'  {filt:<10} {d["total"]:>5} {d["ok"]:>5} {pct:>6.1f}% {med_rms:>7.3f}px')

# RMS分布
if ok:
    rms = [float(r['rms_px']) for r in ok]
    solve_t = [float(r['t_solve_s']) for r in ok]
    print(f'\nRMS分布: 中位={np.median(rms):.3f} 均值={np.mean(rms):.3f} P75={np.percentile(rms, 75):.3f} max={np.max(rms):.3f}')
    print(f'解析耗时: 中位={np.median(solve_t):.2f}s 均值={np.mean(solve_t):.2f}s P75={np.percentile(solve_t, 75):.2f}s max={np.max(solve_t):.2f}s')

# 失败原因
reasons = defaultdict(int)
for r in fail:
    reason = r['fail_reason'][:50]
    reasons[reason] += 1
print('\n失败原因:')
for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f'  {reason}: {cnt}帧')

# 失败帧列表（按望远镜/滤镜分组）
fail_by_group = defaultdict(list)
for r in fail:
    key = f"{r['telescope']}/{r['filter_name']}"
    fail_by_group[key].append(r['filename'][:50])

print('\n失败帧按组:')
for key in sorted(fail_by_group.keys()):
    fnames = fail_by_group[key]
    print(f'  {key}: {len(fnames)}帧')
