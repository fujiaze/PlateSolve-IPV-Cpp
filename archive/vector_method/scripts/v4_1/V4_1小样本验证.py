"""V4.1 小样本验证 — 6帧(3失败+3成功)验证不对称选星策略

功能: 对6个帧运行V4.1求解(img_n_target=50, gaia_density_ratio=1.5),
      对比V4.0默认(n_img_total=250)的效果

用途: 验证V4.1不对称选星策略是否能恢复失败帧且不退化成功帧
"""
import os, sys, math, time, json, traceback
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_0"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import logging
logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("V4.1小样本验证")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_1_cpp import VectorMatchV4_1Cpp
from vector_match_v4_cpp import VectorMatchV4Cpp
from vector_match_v2 import GaiaClientPy

def _parse_ra_hms(s):
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m/60.0 + sec/3600.0) * 15.0
    return float(s)

def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith('-'):
        sign = -1.0
        s = s[1:]
    elif s.startswith('+'):
        s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m/60.0 + sec/3600.0)
    return float(s)

def solve_frame(fits_path, solver, label, output_dir):
    """对单帧运行求解, 返回结果dict"""
    base = os.path.basename(fits_path)
    print(f"\n{'='*70}")
    print(f"  [{label}] {base}")
    print(f"{'='*70}")

    result_info = {'filename': base, 'label': label}
    t_start = time.time()

    try:
        # 1. 读取
        reader = ImageReader()
        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = img.metadata.observation.focallen
        ps = img.metadata.observation.xpixsz
        s0 = 206.265 * ps / fl
        exptime = getattr(img.metadata.observation, 'exptime', 1.0)

        kws = img.keywords
        kw_dict = {k.name.upper(): k.value for k in kws}
        obj_ra_str = kw_dict.get('OBJCTRA') or kw_dict.get('RA')
        obj_dec_str = kw_dict.get('OBJCTDEC') or kw_dict.get('DEC')
        cra0 = _parse_ra_hms(obj_ra_str)
        cdec0 = _parse_dec_dms(obj_dec_str)

        # 2. 检测
        detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        det = detector.detect_ex(img.data)
        n_detected = len(det.x)
        n_saturated = int(np.sum(det.saturated))
        print(f"  检测: n_detected={n_detected}, n_saturated={n_saturated}")

        wcs_json = os.path.join(output_dir, f"wcs_{label}_{base}.json")

        # 判断solver类型
        is_v41 = isinstance(solver, VectorMatchV4_1Cpp)
        version = "V4.1" if is_v41 else "V4.0"

        t_solve = time.time()

        if is_v41:
            # V4.1: 使用不对称选星参数
            result = solver.solve(
                np.array(det.x, np.float64), np.array(det.y, np.float64),
                np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
                cra0, cdec0, fl, ps, w, h,
                wcs_out=wcs_json,
                exptime=exptime,
                img_n_target=50,
                gaia_density_ratio=1.5,
                gaia_query_radius_factor=0.55,
                verbose=False,
            )
        else:
            # V4.0: 使用默认参数
            result = solver.solve(
                np.array(det.x, np.float64), np.array(det.y, np.float64),
                np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
                cra0, cdec0, fl, ps, w, h,
                wcs_out=wcs_json,
                exptime=exptime,
                verbose=False,
            )

        solve_time = time.time() - t_solve

        if result:
            print(f"  {version}结果: 成功 matched={result.matched_count} RMS={result.rms_px:.4f}px "
                  f"scale={result.scale_arcsec_px:.4f}\"/px 耗时={solve_time:.2f}s")
            result_info.update({
                'success': True,
                'matched': result.matched_count,
                'rms_px': result.rms_px,
                'scale': result.scale_arcsec_px,
                'rotation': result.rotation_deg,
                'solve_time': round(solve_time, 2),
            })
        else:
            print(f"  {version}结果: 失败 耗时={solve_time:.2f}s")
            result_info.update({
                'success': False,
                'matched': 0,
                'rms_px': -1,
                'solve_time': round(solve_time, 2),
            })

        result_info['n_detected'] = n_detected
        result_info['n_saturated'] = n_saturated

    except Exception as e:
        result_info['error'] = str(e)
        result_info['success'] = False
        print(f"  异常: {e}")
        traceback.print_exc()

    return result_info

# ========== 主流程 ==========
output_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_1", "small_sample")
os.makedirs(output_dir, exist_ok=True)

frames = [
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@042902-1200S-Oiii.fts", "失败"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@025221-1200S-Oiii.fts", "成功"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic2_flying_dutchman-20250205@062533-180S-Lum.fts", "失败"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic2_flying_dutchman-20250205@062145-180S-Lum.fts", "成功"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic1_flying_dutchman-20250206@054309-180S-Green.fts", "失败"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic1_flying_dutchman-20250206@055047-180S-Blue.fts", "成功"),
]

# 创建两个solver
print("创建V4.0 solver...")
solver_v40 = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
print("创建V4.1 solver...")
solver_v41 = VectorMatchV4_1Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)

results = []

# 对每个帧分别运行V4.0和V4.1
for fits_path, label in frames:
    base = os.path.basename(fits_path)
    print(f"\n{'#'*70}")
    print(f"# 帧: {base} ({label})")
    print(f"{'#'*70}")

    # V4.0
    r40 = solve_frame(fits_path, solver_v40, f"V4.0_{label}", output_dir)
    r40['version'] = 'V4.0'
    results.append(r40)

    # V4.1
    r41 = solve_frame(fits_path, solver_v41, f"V4.1_{label}", output_dir)
    r41['version'] = 'V4.1'
    results.append(r41)

solver_v40.close()
solver_v41.close()

# 保存结果
results_path = os.path.join(output_dir, "v41_small_sample_results.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 汇总表
print(f"\n{'='*100}")
print(f"  V4.1 小样本验证汇总")
print(f"{'='*100}")
print(f"{'帧':<55} {'标签':<6} {'版本':<6} {'结果':<6} {'matched':<10} {'RMS(px)':<10} {'耗时(s)':<10}")
print(f"{'-'*100}")
for r in results:
    success_str = '✓成功' if r.get('success') else '✗失败'
    matched = r.get('matched', 0)
    rms = f"{r.get('rms_px', -1):.2f}" if r.get('rms_px', -1) >= 0 else "N/A"
    t = r.get('solve_time', 0)
    print(f"{r['filename'][:55]:<55} {r.get('label',''):<6} {r.get('version',''):<6} {success_str:<6} {matched:<10} {rms:<10} {t:<10}")

# 对比分析
print(f"\n{'='*100}")
print(f"  对比分析")
print(f"{'='*100}")
for i in range(0, len(results), 2):
    r40 = results[i]
    r41 = results[i+1]
    base = r40['filename']
    label = r40.get('label', '')

    v40_ok = r40.get('success', False)
    v41_ok = r41.get('success', False)
    v40_m = r40.get('matched', 0)
    v41_m = r41.get('matched', 0)
    v40_rms = r40.get('rms_px', -1)
    v41_rms = r41.get('rms_px', -1)

    if not v40_ok and v41_ok:
        status = "✓ 恢复成功"
    elif v40_ok and not v41_ok:
        status = "✗ 退化!"
    elif v40_ok and v41_ok:
        if v41_m >= v40_m * 0.8:
            status = f"✓ 都成功 (匹配{v40_m}→{v41_m})"
        else:
            status = f"⚠ 都成功但匹配下降 ({v40_m}→{v41_m})"
    else:
        status = "✗ 都失败"

    print(f"  {base[:50]:<50} [{label}] {status}")

print(f"\n结果保存: {results_path}")
