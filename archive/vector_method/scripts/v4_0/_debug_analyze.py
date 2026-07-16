"""Debug分析: 成功vs失败帧特征对比"""
import json, numpy as np

r = json.load(open(r'f:\Astro dev\Astro CS Normalization Database\lib\plate_solve\logs\v4\batch_test\batch_test_50frames_seed42.json', 'r', encoding='utf-8'))
succ = [x for x in r if x['status'] == 'success']
fail = [x for x in r if x['status'] == 'fail_solve']
skip = [x for x in r if x['status'] == 'skip_no_wcs_header']

print('=== 成功帧特征 ===')
print(f'  Dec范围: {min(x["fits_center_dec"] for x in succ):.2f} ~ {max(x["fits_center_dec"] for x in succ):.2f}')
print(f'  RA范围: {min(x["fits_center_ra"] for x in succ):.2f} ~ {max(x["fits_center_ra"] for x in succ):.2f}')
print(f'  饱和星: 中位={np.median([x["n_saturated"] for x in succ]):.0f} min={min(x["n_saturated"] for x in succ)} max={max(x["n_saturated"] for x in succ)}')
print(f'  检测星: 中位={np.median([x["n_detected"] for x in succ]):.0f} min={min(x["n_detected"] for x in succ)} max={max(x["n_detected"] for x in succ)}')
print(f'  焦距: {sorted(set(x["focallen"] for x in succ))}')
tg_s = {}
for x in succ:
    tg_s.setdefault(x['target'], [0, 0])
    tg_s[x['target']][0] += 1
print(f'  目标(成功/总数): ')
for tg in sorted(tg_s):
    total = sum(1 for x in r if x.get('target') == tg and x['status'] != 'skip_no_wcs_header')
    print(f'    {tg}: {tg_s[tg][0]}/{total}')

print()
print('=== 失败帧特征(fail_solve) ===')
print(f'  Dec范围: {min(x["fits_center_dec"] for x in fail):.2f} ~ {max(x["fits_center_dec"] for x in fail):.2f}')
print(f'  RA范围: {min(x["fits_center_ra"] for x in fail):.2f} ~ {max(x["fits_center_ra"] for x in fail):.2f}')
print(f'  饱和星: 中位={np.median([x["n_saturated"] for x in fail]):.0f} min={min(x["n_saturated"] for x in fail)} max={max(x["n_saturated"] for x in fail)}')
print(f'  检测星: 中位={np.median([x["n_detected"] for x in fail]):.0f} min={min(x["n_detected"] for x in fail)} max={max(x["n_detected"] for x in fail)}')
print(f'  焦距: {sorted(set(x["focallen"] for x in fail))}')

print()
print('=== 失败帧列表(Dec排序) ===')
for x in sorted(fail, key=lambda x: x['fits_center_dec']):
    print(f'  {x["filename"][:55]:55s} Dec={x["fits_center_dec"]:7.2f} sat={x["n_saturated"]:4d} det={x["n_detected"]:5d} fl={x["focallen"]:.0f} {x["filter"]}')

print()
print('=== 无WCS头帧 ===')
for x in skip[:5]:
    print(f'  {x["filename"][:60]}')
print(f'  ...共{len(skip)}帧')
