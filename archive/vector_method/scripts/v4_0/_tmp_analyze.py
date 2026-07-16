"""临时分析脚本：统计50帧批量测试结果"""
import json, statistics, os
from collections import defaultdict

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
json_path = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "batch_test",
                         "batch_test_50frames_seed42.json")

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

n = len(data)
success = [d for d in data if d.get('status') == 'success']
fail_solve = [d for d in data if d.get('status') == 'fail_solve']
errors = [d for d in data if d.get('status', '').startswith('error')]
fail_no = [d for d in data if d.get('status') == 'fail_no_objctra']

print(f"总帧数: {n}")
print(f"成功: {len(success)} ({100*len(success)/n:.1f}%)")
print(f"fail_solve: {len(fail_solve)}")
print(f"error: {len(errors)}")
print(f"fail_no_objctra: {len(fail_no)}")
print()

if success:
    rms = [d['rms_px'] for d in success if 'rms_px' in d]
    pct10 = [d['pct_10px'] for d in success if 'pct_10px' in d]
    pct5 = [d['pct_5px'] for d in success if 'pct_5px' in d]
    pct3 = [d['pct_3px'] for d in success if 'pct_3px' in d]
    solve_t = [d['solve_time_s'] for d in success if 'solve_time_s' in d]
    total_t = [d['total_time_s'] for d in success if 'total_time_s' in d]
    matched = [d['matched_count'] for d in success if 'matched_count' in d]

    print("=== 成功帧统计 ===")
    print(f"中位RMS: {statistics.median(rms):.3f} px")
    print(f"中位10px命中率: {statistics.median(pct10):.1f}%")
    print(f"中位5px命中率: {statistics.median(pct5):.1f}%")
    print(f"中位3px命中率: {statistics.median(pct3):.1f}%")
    print(f"中位求解耗时: {statistics.median(solve_t):.3f}s")
    print(f"中位总耗时: {statistics.median(total_t):.2f}s")
    print(f"中位匹配对数: {statistics.median(matched):.0f}")
    print()

    # 按目标分组统计
    by_target = defaultdict(list)
    for d in success:
        by_target[d.get('target', '?')].append(d)

    print("=== 按目标分组 ===")
    for t in sorted(by_target.keys()):
        frames = by_target[t]
        r = [f['rms_px'] for f in frames if 'rms_px' in f]
        p10 = [f['pct_10px'] for f in frames if 'pct_10px' in f]
        print(f"  {t}: {len(frames)}帧, 中位RMS={statistics.median(r):.2f}px, 中位10px={statistics.median(p10):.1f}%")
    print()

    # 高RMS帧(>10px)
    print("=== 高RMS帧(>10px) ===")
    high_rms = sorted([d for d in success if d.get('rms_px', 0) > 10], key=lambda x: -x['rms_px'])
    for d in high_rms:
        fn = d["filename"]
        print(f"  {fn}: RMS={d['rms_px']:.1f}px 10px={d.get('pct_10px', 0)}% matched={d.get('matched_count', 0)}")
    print()

    # 低10px命中率帧(<70%)
    print("=== 低10px命中率帧(<70%) ===")
    low_pct = sorted([d for d in success if d.get('pct_10px', 100) < 70], key=lambda x: x.get('pct_10px', 0))
    for d in low_pct:
        fn = d["filename"]
        print(f"  {fn}: 10px={d.get('pct_10px', 0)}% RMS={d['rms_px']:.1f}px matched={d.get('matched_count', 0)}")
    print()

    # 失败帧详情
    print("=== 失败帧详情 ===")
    for d in fail_solve + errors:
        fn = d["filename"]
        print(f"  {fn}: {d['status']}")
