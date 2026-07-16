"""V4.1 新选星策略实验 — 饱和为主 + Gaia 1.5x密度

用户方案:
  图像侧: 饱和>50颗→只用饱和; 否则饱和+非饱和亮星补足到50颗
  Gaia侧: 圆形查询(直径=1.1×FOV对角线), 面密度=1.5×图像面密度

对比: V4.0默认(n_img_total=250, Phase 0密度匹配)
"""
import os, sys, math, time, json
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_0"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("新选星策略实验")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_cpp import VectorMatchV4Cpp
from vector_match_v2 import GaiaClientPy
from WCS重投影校核V4 import sky_to_pixel_wcs

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

def new_strategy_select_stars(det, n_img_target=50, sat_threshold=50):
    """新选星策略: 饱和>50→只用饱和; 否则饱和+非饱和亮星补足到50
    
    Returns: (sel_idx, sel_reason)
    """
    sat_mask = np.array(det.saturated, dtype=bool)
    sat_idx = np.where(sat_mask)[0]
    n_sat = len(sat_idx)
    
    if n_sat > sat_threshold:
        # 只用饱和星
        return sat_idx, f"只用饱和星({n_sat}颗>阈值{sat_threshold})"
    
    # 饱和 + 非饱和亮星补足
    flux_arr = np.array(det.flux)
    non_sat_idx = np.where(~sat_mask)[0]
    # 非饱和星按flux降序
    if len(non_sat_idx) > 0:
        non_sat_sorted = non_sat_idx[np.argsort(-flux_arr[non_sat_idx])]
    else:
        non_sat_sorted = np.array([], dtype=np.int64)
    
    n_needed = max(0, n_img_target - n_sat)
    top_non_sat = non_sat_sorted[:n_needed]
    sel_idx = np.concatenate([sat_idx, top_non_sat])
    return sel_idx, f"饱和{n_sat}+非饱和{n_needed}={len(sel_idx)}颗"

def new_strategy_query_gaia(gaia_client, cra0, cdec0, w, h, s0, n_img, density_ratio=1.5):
    """新Gaia查询策略: 圆形区域, 面密度=1.5×图像面密度
    
    Args:
        gaia_client: GaiaClientPy实例
        cra0, cdec0: 指向中心
        w, h: 图像宽高
        s0: 角秒/像素
        n_img: 图像侧星点数
        density_ratio: Gaia密度/图像密度
        
    Returns: (ra_arr, dec_arr, mag_arr, query_radius_deg, m_lim, n_gaia_target)
    """
    # FOV对角线(度)
    fov_diag_deg = math.sqrt(w*w + h*h) * s0 / 3600.0
    # 查询半径: 直径=1.1×FOV对角线, 半径=0.55×FOV对角线
    query_radius_deg = fov_diag_deg * 0.55
    
    # 图像面密度 (颗/平方度)
    img_area_deg2 = w * h * s0 * s0 / (3600.0 * 3600.0)
    rho_img = n_img / img_area_deg2 if img_area_deg2 > 0 else 0
    
    # 目标Gaia星数 = 1.5 × 图像密度 × 查询圆面积
    query_area_deg2 = math.pi * query_radius_deg * query_radius_deg
    n_gaia_target = int(density_ratio * rho_img * query_area_deg2)
    n_gaia_target = max(n_gaia_target, 50)  # 下限50
    
    # 迭代极限星等: 从亮到暗, 找到最接近n_gaia_target的星等
    # 先用较亮的星等查询, 不够就降低星等
    best_result = None
    best_diff = float('inf')
    
    for mag_try in [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]:
        ra_t, dec_t, mag_t = gaia_client.cone_search(cra0, cdec0, query_radius_deg, mag_try)
        n_gaia = len(ra_t)
        diff = abs(n_gaia - n_gaia_target)
        if diff < best_diff:
            best_diff = diff
            best_result = (np.array(ra_t, dtype=np.float64), 
                          np.array(dec_t, dtype=np.float64),
                          np.array(mag_t, dtype=np.float64),
                          mag_try)
        if n_gaia >= n_gaia_target:
            break
    
    ra_arr, dec_arr, mag_arr, m_lim = best_result
    return ra_arr, dec_arr, mag_arr, query_radius_deg, m_lim, n_gaia_target

def run_experiment_for_frame(fits_path, solver, gaia_client, output_dir):
    """对单帧运行新策略实验"""
    import traceback
    base = os.path.basename(fits_path)
    print(f"\n{'='*70}")
    print(f"  {base}")
    print(f"{'='*70}")
    
    result_info = {'filename': base}
    
    try:
        # 1. 读取
        reader = ImageReader()
        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = img.metadata.observation.focallen
        ps = img.metadata.observation.xpixsz
        s0 = 206.265 * ps / fl
        exptime = getattr(img.metadata.observation, 'exptime', 1.0)
        if exptime is None or exptime == 0:
            exptime = getattr(img.metadata.calibration, 'exptime', 1.0)
        
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
        
        # ========== 策略A: 新选星策略 ==========
        print(f"\n  --- 策略A: 新选星策略(饱和>50→只用饱和; 否则饱和+非饱和补50) ---")
        sel_idx, reason = new_strategy_select_stars(det, n_img_target=50, sat_threshold=50)
        n_img_new = len(sel_idx)
        print(f"  选星: {reason}, 共{n_img_new}颗")
        
        # Gaia查询(新策略) — 仅用于统计, 实际求解由V4.0内部完成
        ra_arr, dec_arr, mag_arr, q_r_deg, m_lim, n_target = new_strategy_query_gaia(
            gaia_client, cra0, cdec0, w, h, s0, n_img_new, density_ratio=1.5)
        n_gaia_new = len(ra_arr)
        print(f"  Gaia: 查询半径={q_r_deg:.3f}°, 目标星数={n_target}, 实际={n_gaia_new}, m_lim={m_lim}")
        
        # 求解(新策略)
        # 用V4.0 solver, 但n_img_total传入新值, 让solver内部用其选星逻辑
        # 但V4.0的选星逻辑是"饱和全选+非饱和补足到n_img_total", 与新策略一致(只是n_img_total从250改为50)
        # 当饱和>50时, V4.0会全选饱和星(因为max(n_img_total, n_saturated))
        wcs_json_new = os.path.join(output_dir, f"wcs_new_{base}.json")
        t_new = time.time()
        result_new = solver.solve(
            np.array(det.x, np.float64), np.array(det.y, np.float64),
            np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
            cra0, cdec0, fl, ps, w, h,
            wcs_out=wcs_json_new,
            exptime=exptime,
            n_img_total=max(n_img_new, n_saturated),  # 关键: 新策略的n_img_total
            verbose=False,
        )
        t_new_dur = time.time() - t_new
        print(f"  结果: {'成功' if result_new else '失败'}, 耗时={t_new_dur:.2f}s")
        if result_new:
            print(f"    matched={result_new.matched_count}, RMS={result_new.rms_px:.4f}px, "
                  f"scale={result_new.scale_arcsec_px:.4f}\"/px, rot={result_new.rotation_deg:.2f}°")
        
        # ========== 策略B: V4.0默认(对比基准) ==========
        print(f"\n  --- 策略B: V4.0默认(n_img_total=250) ---")
        wcs_json_old = os.path.join(output_dir, f"wcs_old_{base}.json")
        t_old = time.time()
        result_old = solver.solve(
            np.array(det.x, np.float64), np.array(det.y, np.float64),
            np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
            cra0, cdec0, fl, ps, w, h,
            wcs_out=wcs_json_old,
            exptime=exptime,
            n_img_total=250,
            verbose=False,
        )
        t_old_dur = time.time() - t_old
        print(f"  结果: {'成功' if result_old else '失败'}, 耗时={t_old_dur:.2f}s")
        if result_old:
            print(f"    matched={result_old.matched_count}, RMS={result_old.rms_px:.4f}px, "
                  f"scale={result_old.scale_arcsec_px:.4f}\"/px, rot={result_old.rotation_deg:.2f}°")
        
        # ========== 策略C: 新策略 + n_img=100 ==========
        print(f"\n  --- 策略C: n_img=100 + Gaia 1.5x密度 ---")
        sel_idx_c, reason_c = new_strategy_select_stars(det, n_img_target=100, sat_threshold=50)
        n_img_c = len(sel_idx_c)
        ra_c, dec_c, mag_c, q_r_c, m_lim_c, n_target_c = new_strategy_query_gaia(
            gaia_client, cra0, cdec0, w, h, s0, n_img_c, density_ratio=1.5)
        print(f"  选星: {reason_c}, 共{n_img_c}颗; Gaia={len(ra_c)}颗(m_lim={m_lim_c})")
        wcs_json_c = os.path.join(output_dir, f"wcs_c100_{base}.json")
        t_c = time.time()
        result_c = solver.solve(
            np.array(det.x, np.float64), np.array(det.y, np.float64),
            np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
            cra0, cdec0, fl, ps, w, h,
            wcs_out=wcs_json_c,
            exptime=exptime,
            n_img_total=max(n_img_c, n_saturated),
            verbose=False,
        )
        t_c_dur = time.time() - t_c
        print(f"  结果: {'成功' if result_c else '失败'}, 耗时={t_c_dur:.2f}s")
        if result_c:
            print(f"    matched={result_c.matched_count}, RMS={result_c.rms_px:.4f}px, "
                  f"scale={result_c.scale_arcsec_px:.4f}\"/px, rot={result_c.rotation_deg:.2f}°")
        
        result_info.update({
            'n_detected': n_detected, 'n_saturated': n_saturated,
            'new_n_img': n_img_new, 'new_n_gaia': n_gaia_new, 'new_m_lim': m_lim,
            'new_success': bool(result_new), 'new_matched': result_new.matched_count if result_new else 0,
            'new_rms': result_new.rms_px if result_new else -1,
            'new_time': round(t_new_dur, 2),
            'old_success': bool(result_old), 'old_matched': result_old.matched_count if result_old else 0,
            'old_rms': result_old.rms_px if result_old else -1,
            'old_time': round(t_old_dur, 2),
            'c100_n_img': n_img_c, 'c100_n_gaia': len(ra_c),
            'c100_success': bool(result_c), 'c100_matched': result_c.matched_count if result_c else 0,
            'c100_rms': result_c.rms_px if result_c else -1,
            'c100_time': round(t_c_dur, 2),
        })
        
    except Exception as e:
        result_info['error'] = str(e)
        print(f"  异常: {e}")
        traceback.print_exc()
    
    return result_info

# ========== 主流程 ==========
output_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "experiments")
os.makedirs(output_dir, exist_ok=True)

frames = [
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@042902-1200S-Oiii.fts", "失败", "NGC55"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@025221-1200S-Oiii.fts", "成功", "NGC55"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic2_flying_dutchman-20250205@062533-180S-Lum.fts", "失败", "Victory"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic2_flying_dutchman-20250205@062145-180S-Lum.fts", "成功", "Victory"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic1_flying_dutchman-20250206@054309-180S-Green.fts", "失败", "Victory"),
    (r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic1_flying_dutchman-20250206@055047-180S-Blue.fts", "成功", "Victory"),
]

solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
gaia_client = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"))

results = []
for fits_path, label, target in frames:
    r = run_experiment_for_frame(fits_path, solver, gaia_client, output_dir)
    r['label'] = label
    r['target'] = target
    results.append(r)

solver.close()

# 保存结果
results_path = os.path.join(output_dir, "new_strategy_results.json")
with open(results_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 打印汇总表
print(f"\n{'='*100}")
print(f"  实验汇总")
print(f"{'='*100}")
print(f"{'帧':<55} {'标签':<6} {'策略A(新50)':<20} {'策略B(V4.0-250)':<20} {'策略C(新100)':<20}")
print(f"{'-'*100}")
for r in results:
    a = f"{'✓' if r.get('new_success') else '✗'} m={r.get('new_matched',0)} rms={r.get('new_rms',-1):.2f}" if r.get('new_rms',-1)>=0 else f"{'✓' if r.get('new_success') else '✗'} m={r.get('new_matched',0)}"
    b = f"{'✓' if r.get('old_success') else '✗'} m={r.get('old_matched',0)} rms={r.get('old_rms',-1):.2f}" if r.get('old_rms',-1)>=0 else f"{'✓' if r.get('old_success') else '✗'} m={r.get('old_matched',0)}"
    c = f"{'✓' if r.get('c100_success') else '✗'} m={r.get('c100_matched',0)} rms={r.get('c100_rms',-1):.2f}" if r.get('c100_rms',-1)>=0 else f"{'✓' if r.get('c100_success') else '✗'} m={r.get('c100_matched',0)}"
    print(f"{r['filename'][:55]:<55} {r['label']:<6} {a:<20} {b:<20} {c:<20}")
print(f"\n结果保存: {results_path}")
