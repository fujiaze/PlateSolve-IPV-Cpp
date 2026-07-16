"""V3.5 SIP拟合精度根因诊断脚本
功能: 诊断C++内部SIP RMS(1.716px)与Python验证RMS(2.676px)差异的根本原因
分析项:
  1. C++内部RMS复现 — 在匹配对上计算残差
  2. 亮度/距离分层残差 — 区分质心误差与SIP外推
  3. Phase D原始CD vs Siril修正CD — 哪个更准
  4. SIP过拟合检测 — 训练集vs测试集
  5. 残差方向/径向模式 — 系统性偏差
用法:
  python diag_sip_root_cause.py
"""
# 全局强制UTF-8编码
import sys, os
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

import json, math, logging, numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("diag_sip")

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy, bisection_mag_limit, gnomonic_forward, _DEGTORAD
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from star_detector import StarDetector, SDetParamsPy
from scipy.spatial import cKDTree


# ============================================================================
# 工具函数
# ============================================================================

def load_wcs_json(json_path):
    """读取V3.5输出的WCS JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return {
        'CD': np.array(d['CD'], dtype=np.float64),
        'CRVAL': np.array(d['CRVAL'], dtype=np.float64),
        'CRPIX': np.array(d['CRPIX'], dtype=np.float64),
        'SIP_A': np.array(d['SIP_A'], dtype=np.float64).reshape(6, 6),
        'SIP_B': np.array(d['SIP_B'], dtype=np.float64).reshape(6, 6),
        'RMS_PX': float(d.get('RMS_PX', 0)),
        'SIP_ORDER': int(d.get('SIP_ORDER', 5)),
    }


def compute_cd_from_solve(s, theta_rad, flip_mode, s0, cos_dec):
    """从V3.5 solve参数计算Phase D原始CD矩阵"""
    fx = (flip_mode == 1 or flip_mode == 3)
    fy = (flip_mode == 2 or flip_mode == 3)
    sign_x = -1.0 if fx else 1.0
    sign_y = -1.0 if fy else 1.0
    ct = math.cos(theta_rad)
    st = math.sin(theta_rad)
    s0_over_s_3600 = s0 / (s * 3600.0)
    cd = np.array([
        [sign_x * s0_over_s_3600 * ct / cos_dec, -sign_x * s0_over_s_3600 * st / cos_dec],
        [-sign_y * s0_over_s_3600 * st, -sign_y * s0_over_s_3600 * ct]
    ])
    return cd


def wcs_linear_inverse(ra_src, dec_src, cd, crval, crpix):
    """线性WCS逆投影(无SIP): sky → pixel"""
    cdet = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    if abs(cdet) < 1e-30:
        return np.full(len(ra_src), np.nan), np.full(len(ra_src), np.nan)
    cd_inv = np.array([[cd[1, 1], -cd[0, 1]], [-cd[1, 0], cd[0, 0]]]) / cdet
    dra = ra_src - crval[0]
    ddec = dec_src - crval[1]
    xi = cd_inv[0, 0] * dra + cd_inv[0, 1] * ddec
    eta = cd_inv[1, 0] * dra + cd_inv[1, 1] * ddec
    x = xi + crpix[0]
    y = eta + crpix[1]
    return x, y


def wcs_sip_inverse(ra_src, dec_src, wcs):
    """标准WCS-SIP逆投影: sky → pixel"""
    cd = wcs['CD']
    crval = wcs['CRVAL']
    crpix = wcs['CRPIX']
    sip_A = wcs['SIP_A']
    sip_B = wcs['SIP_B']
    sip_order = wcs['SIP_ORDER']

    cdet = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    if abs(cdet) < 1e-30:
        return np.full(len(ra_src), np.nan), np.full(len(ra_src), np.nan)
    cd_inv = np.array([[cd[1, 1], -cd[0, 1]], [-cd[1, 0], cd[0, 0]]]) / cdet

    dra = ra_src - crval[0]
    ddec = dec_src - crval[1]
    xi_prime = cd_inv[0, 0] * dra + cd_inv[0, 1] * ddec
    eta_prime = cd_inv[1, 0] * dra + cd_inv[1, 1] * ddec

    # 粗筛帧内
    x_lin = xi_prime + crpix[0]
    y_lin = eta_prime + crpix[1]
    margin = 500
    prelim = (x_lin > -margin) & (x_lin < 10000 + margin) & \
             (y_lin > -margin) & (y_lin < 10000 + margin)

    xi = xi_prime[prelim].copy()
    eta = eta_prime[prelim].copy()
    xi_prime_s = xi_prime[prelim].copy()
    eta_prime_s = eta_prime[prelim].copy()

    # 非零SIP项
    max_order = min(sip_order, 6) if sip_order > 0 else 0
    sip_terms = []
    if max_order >= 2:
        for p in range(max_order + 1):
            for q in range(max_order + 1):
                if p + q < 2 or p + q > max_order:
                    continue
                if p >= 6 or q >= 6:
                    continue
                a_c = sip_A[p, q]
                b_c = sip_B[p, q]
                if abs(a_c) > 1e-30 or abs(b_c) > 1e-30:
                    sip_terms.append((p, q, a_c, b_c))

    # 迭代求解
    for iteration in range(30):
        sip_dx = np.zeros_like(xi)
        sip_dy = np.zeros_like(eta)
        for p, q, a_c, b_c in sip_terms:
            xi_c = np.clip(xi, -1e4, 1e4)
            eta_c = np.clip(eta, -1e4, 1e4)
            term = xi_c ** p * eta_c ** q
            term = np.where(np.isfinite(term), term, 0.0)
            sip_dx += a_c * term
            sip_dy += b_c * term
        xi_new = xi_prime_s - sip_dx
        eta_new = eta_prime_s - sip_dy
        max_dx = np.max(np.abs(xi_new - xi))
        max_dy = np.max(np.abs(eta_new - eta))
        if max_dx < 1e-6 and max_dy < 1e-6:
            break
        xi = xi_new
        eta = eta_new

    x_pix = np.full(len(ra_src), np.nan)
    y_pix = np.full(len(ra_src), np.nan)
    x_pix[prelim] = xi + crpix[0]
    y_pix[prelim] = eta + crpix[1]
    return x_pix, y_pix


def match_and_compute_rms(x_gaia, y_gaia, x_det, y_det, w, h,
                          max_dist_px=5.0, mag_arr=None, r_arr=None):
    """匹配Gaia投影点与检测星，计算残差统计

    返回: dict with keys: n_matched, rms_px, mean_px, median_px,
          matched_dists, matched_mag, matched_r, matched_dx, matched_dy
    """
    valid_det = np.isfinite(x_det) & np.isfinite(y_det)
    if valid_det.sum() < 2:
        return None
    det_coords = np.column_stack([x_det[valid_det], y_det[valid_det]])
    tree = cKDTree(det_coords)

    valid_gaia = np.isfinite(x_gaia) & np.isfinite(y_gaia) & \
                 (x_gaia > 0) & (x_gaia < w) & (y_gaia > 0) & (y_gaia < h)
    if valid_gaia.sum() < 2:
        return None

    gaia_coords = np.column_stack([x_gaia[valid_gaia], y_gaia[valid_gaia]])
    dists, idxs = tree.query(gaia_coords, k=1)

    matched = dists < max_dist_px
    if matched.sum() < 2:
        return None

    match_dists = dists[matched]
    # 计算dx/dy方向残差
    match_gaia = gaia_coords[matched]
    match_det = det_coords[idxs[matched]]
    dx = match_det[:, 0] - match_gaia[:, 0]
    dy = match_det[:, 1] - match_gaia[:, 1]

    result = {
        'n_matched': int(matched.sum()),
        'rms_px': float(np.sqrt(np.mean(match_dists ** 2))),
        'mean_px': float(np.mean(match_dists)),
        'median_px': float(np.median(match_dists)),
        'p90_px': float(np.percentile(match_dists, 90)),
        'matched_dists': match_dists,
        'matched_dx': dx,
        'matched_dy': dy,
    }

    # 附加星等和距离信息
    if mag_arr is not None:
        gaia_idx_in_frame = np.where(valid_gaia)[0]
        result['matched_mag'] = mag_arr[gaia_idx_in_frame[matched]]
    if r_arr is not None:
        gaia_idx_in_frame = np.where(valid_gaia)[0]
        result['matched_r'] = r_arr[gaia_idx_in_frame[matched]]

    return result


def print_section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_rms_table(stats, label=""):
    """打印RMS统计表"""
    if stats is None:
        print(f"  {label}: 匹配对不足")
        return
    print(f"  {label}: n={stats['n_matched']}, "
          f"RMS={stats['rms_px']:.3f}px, "
          f"均值={stats['mean_px']:.3f}px, "
          f"中位={stats['median_px']:.3f}px, "
          f"P90={stats['p90_px']:.3f}px")


def print_layered_rms(stats, s0, label=""):
    """按亮度/距离分层打印RMS"""
    if stats is None:
        return
    dists = stats['matched_dists']

    # 按亮度分层
    if 'matched_mag' in stats:
        mag = stats['matched_mag']
        print(f"\n  {label} 按亮度分层:")
        for mag_lo, mag_hi, tag in [(0, 13, '极亮'), (13, 15, '亮'), (15, 16, '中'), (16, 18, '暗'), (18, 99, '极暗')]:
            mask = (mag >= mag_lo) & (mag < mag_hi)
            if mask.sum() >= 2:
                rms = np.sqrt(np.mean(dists[mask] ** 2))
                print(f"    mag=[{mag_lo},{mag_hi}) {tag}: n={mask.sum()}, RMS={rms:.3f}px ({rms*s0:.3f}\")")

    # 按距中心距离分层
    if 'matched_r' in stats:
        r = stats['matched_r']
        print(f"\n  {label} 按距中心距离分层:")
        for r_lo, r_hi in [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]:
            mask = (r >= r_lo) & (r < r_hi)
            if mask.sum() >= 2:
                rms = np.sqrt(np.mean(dists[mask] ** 2))
                print(f"    r=[{r_lo},{r_hi}): n={mask.sum()}, RMS={rms:.3f}px ({rms*s0:.3f}\")")


# ============================================================================
# 主诊断流程
# ============================================================================

def main():
    print_section("V3.5 SIP拟合精度根因诊断")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- 1. 读取FITS图像 ----
    print_section("1. 读取FITS图像")
    reader = ImageReader()
    fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights",
        "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    cra0 = img.metadata.wcs.crval1
    cdec0 = img.metadata.wcs.crval2
    s0 = 206.265 * ps / fl
    print(f"  图像: {w}x{h}, fl={fl}mm, ps={ps}um, s0={s0:.4f}\"/px")
    print(f"  初始WCS: RA={cra0:.6f}°, Dec={cdec0:.6f}°")

    # ---- 2. 读取WCS JSON ----
    print_section("2. 读取WCS JSON (Siril修正CD)")
    wcs_json_path = os.path.join(PROJECT_ROOT, "vm35_wcs_output.json")
    wcs = load_wcs_json(wcs_json_path)
    cd_siril = wcs['CD']
    crval_siril = wcs['CRVAL']
    crpix = wcs['CRPIX']
    sip_A = wcs['SIP_A']
    sip_B = wcs['SIP_B']
    sip_order = wcs['SIP_ORDER']
    sip_rms = wcs['RMS_PX']
    print(f"  CD(Siril): [[{cd_siril[0,0]:.10e}, {cd_siril[0,1]:.10e}],")
    print(f"              [{cd_siril[1,0]:.10e}, {cd_siril[1,1]:.10e}]]")
    print(f"  CRVAL: [{crval_siril[0]:.10f}, {crval_siril[1]:.10f}]")
    print(f"  CRPIX: [{crpix[0]:.3f}, {crpix[1]:.3f}]")
    print(f"  SIP阶数: {sip_order}, SIP RMS: {sip_rms:.6f} px ({sip_rms*s0:.6f}\")")

    # Siril CD提取的像素尺度和旋转角
    scale_x_s = math.sqrt(cd_siril[0, 0]**2 + cd_siril[1, 0]**2) * 3600
    scale_y_s = math.sqrt(cd_siril[0, 1]**2 + cd_siril[1, 1]**2) * 3600
    rot_x_s = math.degrees(math.atan2(cd_siril[1, 0], cd_siril[0, 0]))
    print(f"  Siril CD像素尺度: X={scale_x_s:.4f}\"/px, Y={scale_y_s:.4f}\"/px")
    print(f"  Siril CD旋转角: θ_x={rot_x_s:.2f}°")
    print(f"  与s0偏差: X={abs(scale_x_s-s0)/s0*100:.2f}%, Y={abs(scale_y_s-s0)/s0*100:.2f}%")

    # ---- 3. 运行V3.5 solve ----
    print_section("3. 运行V3.5 solve")
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    img_x = np.array(det.x, dtype=np.float64)
    img_y = np.array(det.y, dtype=np.float64)
    img_flux = np.array(det.flux, dtype=np.float64)
    img_saturated = np.array(det.saturated, dtype=np.int32)
    print(f"  检测星数: {len(img_x)} (正常={det.normal_count}, 饱和={det.saturated_count})")

    # 运行solve
    vm = VectorMatchV35Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    result = vm.solve(img_x, img_y, img_flux, img_saturated,
                      cra0, cdec0, fl, ps, w, h)
    vm.close()

    if result is None:
        print("  [错误] V3.5 solve失败!")
        return

    print(f"  solve结果:")
    print(f"    s={result.solve_s:.6f}, θ={result.rotation_deg:.4f}°")
    print(f"    n_inliers={result.matched_count}, rms={result.rms_px:.6f}px ({result.rms_arcsec:.6f}\")")
    print(f"    flip_mode={result.flip_mode}")
    print(f"    center: RA={result.center_ra:.10f}, Dec={result.center_dec:.10f}")
    print(f"    SIP RMS(C++内部): {result.sip_rms_px:.6f}px")

    # ---- 4. 计算Phase D原始CD ----
    print_section("4. Phase D原始CD vs Siril修正CD")
    theta_rad = math.radians(result.rotation_deg)
    cos_dec = math.cos(result.center_dec * _DEGTORAD)
    cd_phased = compute_cd_from_solve(result.solve_s, theta_rad, result.flip_mode, s0, cos_dec)
    crval_phased = np.array([result.center_ra, result.center_dec])

    print(f"  CD(Phase D): [[{cd_phased[0,0]:.10e}, {cd_phased[0,1]:.10e}],")
    print(f"                [{cd_phased[1,0]:.10e}, {cd_phased[1,1]:.10e}]]")
    print(f"  CRVAL(Phase D): [{crval_phased[0]:.10f}, {crval_phased[1]:.10f}]")

    # CD差异
    cd_diff = cd_siril - cd_phased
    cd_diff_rel = cd_diff / (np.abs(cd_phased) + 1e-30) * 100
    crval_diff = crval_siril - crval_phased
    dra_diff = 3600 * crval_diff[0] * cos_dec
    ddec_diff = 3600 * crval_diff[1]
    print(f"\n  CD差异(相对):")
    print(f"    [[{cd_diff_rel[0,0]:.4f}%, {cd_diff_rel[0,1]:.4f}%],")
    print(f"     [{cd_diff_rel[1,0]:.4f}%, {cd_diff_rel[1,1]:.4f}%]]")
    print(f"  CRVAL差异: ΔRA={dra_diff:.3f}\", ΔDec={ddec_diff:.3f}\"")

    # Phase D CD提取的像素尺度和旋转角
    scale_x_p = math.sqrt(cd_phased[0, 0]**2 + cd_phased[1, 0]**2) * 3600
    scale_y_p = math.sqrt(cd_phased[0, 1]**2 + cd_phased[1, 1]**2) * 3600
    rot_x_p = math.degrees(math.atan2(cd_phased[1, 0], cd_phased[0, 0]))
    print(f"\n  Phase D CD像素尺度: X={scale_x_p:.4f}\"/px, Y={scale_y_p:.4f}\"/px")
    print(f"  Phase D CD旋转角: θ_x={rot_x_p:.2f}°")
    print(f"  与s0偏差: X={abs(scale_x_p-s0)/s0*100:.2f}%, Y={abs(scale_y_p-s0)/s0*100:.2f}%")

    # ---- 5. 查询Gaia星表 ----
    print_section("5. 查询Gaia星表")
    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    fov_deg = math.sqrt(w * w + h * h) * s0 / 3600.0
    _, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        gaia, crval_siril[0], crval_siril[1], max(0.8, fov_deg * 1.2), 2000)
    gaia.close()
    cat_ra = np.array(cat_ra, dtype=np.float64)
    cat_dec = np.array(cat_dec, dtype=np.float64)
    cat_mag = np.array(cat_mag, dtype=np.float64)
    print(f"  Gaia查询: {M}颗 (极限星等={cat_mag.max():.1f})")

    # 取前N颗亮星用于投影验证
    N_gaia = min(1500, M)
    idx_bright = np.argsort(cat_mag)[:N_gaia]
    ra_src = cat_ra[idx_bright]
    dec_src = cat_dec[idx_bright]
    mag_src = cat_mag[idx_bright]
    print(f"  取前{N_gaia}亮星用于投影验证")

    # ---- 6. 核心诊断: 四种投影方式对比 ----
    print_section("6. 核心诊断: 四种投影方式残差对比")

    # 方式A: Phase D原始CD + 线性投影
    x_pd, y_pd = wcs_linear_inverse(ra_src, dec_src, cd_phased, crval_phased, crpix)

    # 方式B: Siril CD + 线性投影(无SIP)
    x_siril_lin, y_siril_lin = wcs_linear_inverse(ra_src, dec_src, cd_siril, crval_siril, crpix)

    # 方式C: Siril CD + SIP投影
    x_siril_sip, y_siril_sip = wcs_sip_inverse(ra_src, dec_src, wcs)

    # 方式D: Phase D CD + SIP投影(用Phase D CD但Siril SIP系数 — 不标准，仅作参考)
    # 注: SIP系数是针对Siril CD拟合的，不能直接用于Phase D CD
    # 跳过此方式，因为SIP系数与CD耦合

    # 计算距CRPIX的距离
    r_from_center_pd = np.sqrt((x_pd - crpix[0])**2 + (y_pd - crpix[1])**2)
    r_from_center_sip = np.sqrt((x_siril_sip - crpix[0])**2 + (y_siril_sip - crpix[1])**2)

    # 匹配检测星
    print(f"\n  检测星数: {len(img_x)}")

    stats_pd = match_and_compute_rms(x_pd, y_pd, img_x, img_y, w, h,
                                      max_dist_px=5.0, mag_arr=mag_src, r_arr=r_from_center_pd)
    stats_siril_lin = match_and_compute_rms(x_siril_lin, y_siril_lin, img_x, img_y, w, h,
                                             max_dist_px=5.0, mag_arr=mag_src, r_arr=r_from_center_pd)
    stats_siril_sip = match_and_compute_rms(x_siril_sip, y_siril_sip, img_x, img_y, w, h,
                                             max_dist_px=5.0, mag_arr=mag_src, r_arr=r_from_center_sip)

    print(f"\n  --- 整体RMS对比 ---")
    print_rms_table(stats_pd, "A. Phase D CD + 线性")
    print_rms_table(stats_siril_lin, "B. Siril CD + 线性")
    print_rms_table(stats_siril_sip, "C. Siril CD + SIP")
    print(f"  D. C++内部SIP RMS: {sip_rms:.3f}px (训练集)")

    # SIP改善量
    if stats_siril_lin and stats_siril_sip:
        improvement = (stats_siril_lin['rms_px'] - stats_siril_sip['rms_px']) / stats_siril_lin['rms_px'] * 100
        print(f"\n  SIP改善: {improvement:.1f}% (Siril线性→Siril SIP)")

    # ---- 7. 亮度/距离分层分析 ----
    print_section("7. 亮度/距离分层残差分析")

    print("\n  --- A. Phase D CD + 线性 ---")
    print_layered_rms(stats_pd, s0, "Phase D 线性")

    print("\n  --- B. Siril CD + 线性 ---")
    print_layered_rms(stats_siril_lin, s0, "Siril 线性")

    print("\n  --- C. Siril CD + SIP ---")
    print_layered_rms(stats_siril_sip, s0, "Siril SIP")

    # ---- 8. 关键对比: Phase D vs Siril ----
    print_section("8. 关键对比: Phase D原始CD vs Siril修正CD")

    if stats_pd and stats_siril_lin:
        print(f"\n  线性投影RMS:")
        print(f"    Phase D: {stats_pd['rms_px']:.3f}px")
        print(f"    Siril:   {stats_siril_lin['rms_px']:.3f}px")
        diff = stats_siril_lin['rms_px'] - stats_pd['rms_px']
        if diff > 0:
            print(f"    → Phase D更优 (差{diff:.3f}px)")
        else:
            print(f"    → Siril更优 (差{-diff:.3f}px)")

    if stats_pd and stats_siril_sip:
        print(f"\n  Phase D线性 vs Siril SIP:")
        print(f"    Phase D线性: {stats_pd['rms_px']:.3f}px")
        print(f"    Siril SIP:   {stats_siril_sip['rms_px']:.3f}px")
        diff = stats_siril_sip['rms_px'] - stats_pd['rms_px']
        if diff > 0:
            print(f"    → Phase D线性已优于Siril SIP! (差{diff:.3f}px)")
            print(f"    → SIP修正引入了额外误差!")
        else:
            print(f"    → Siril SIP优于Phase D线性 (差{-diff:.3f}px)")

    # ---- 9. 仅亮星+中心区域分析 ----
    print_section("9. 仅亮星(mag<16) + 中心区域(r<1500px) 分析")

    for label, x_g, y_g, r_arr in [
        ("Phase D 线性", x_pd, y_pd, r_from_center_pd),
        ("Siril SIP", x_siril_sip, y_siril_sip, r_from_center_sip),
    ]:
        # 亮星 + 中心区域
        bright_center = (mag_src < 16) & np.isfinite(x_g) & np.isfinite(y_g) & \
                        (x_g > 0) & (x_g < w) & (y_g > 0) & (y_g < h) & \
                        (r_arr < 1500)
        n_bc = bright_center.sum()
        if n_bc >= 2:
            stats_bc = match_and_compute_rms(
                x_g[bright_center], y_g[bright_center], img_x, img_y, w, h,
                max_dist_px=5.0)
            print_rms_table(stats_bc, f"{label} (mag<16, r<1500)")
        else:
            print(f"  {label} (mag<16, r<1500): 亮星+中心区域星数不足 ({n_bc})")

    # ---- 10. C++内部RMS复现 ----
    print_section("10. C++内部RMS复现 (在solve匹配对上计算)")

    # 重建C++内部的匹配过程:
    # C++使用 _build_image_vectors 选出的亮星构建U向量组
    # 然后用Phase D的变换参数投影Gaia星，匹配到U向量组
    # SIP RMS是在这些匹配对上计算的

    from vector_match_v2 import _build_image_vectors, _build_catalog_vectors, _apply_flip, _apply_similarity

    U, N_img, n_sat, _ = _build_image_vectors(img_x, img_y, img_flux, img_saturated, s0, w, h)
    print(f"  C++图像向量组: N_img={N_img} (饱和星={n_sat})")

    # 查询Gaia (与C++相同的查询)
    gaia2 = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    Ngaia = math.ceil(1.5 * n_sat) if n_sat >= 50 else 200
    maglim2, M2, cat_ra2, cat_dec2, cat_mag2 = bisection_mag_limit(
        gaia2, cra0, cdec0, fov_deg * 1.2 / 2.0, Ngaia)
    gaia2.close()
    cat_ra2 = np.array(cat_ra2, dtype=np.float64)
    cat_dec2 = np.array(cat_dec2, dtype=np.float64)
    cat_mag2 = np.array(cat_mag2, dtype=np.float64)
    print(f"  C++ Gaia查询: M={M2}, 极限星等={maglim2:.2f}")

    W = _build_catalog_vectors(cat_ra2, cat_dec2, cra0, cdec0)
    Wf = _apply_flip(W, result.flip_mode)

    # 用Phase D变换参数投影Gaia星
    Wt = _apply_similarity(Wf, result.solve_s, theta_rad, result.solve_tx, result.solve_ty)

    # 匹配: U → Wt (1对1互斥)
    tree_wt = cKDTree(Wt)
    dists_uw, idxs_uw = tree_wt.query(U, k=1)

    # 用s0作为内点阈值(与C++ Phase B一致)
    tau = 1.0 * s0
    inlier_mask = dists_uw < tau
    n_inliers = inlier_mask.sum()

    if n_inliers >= 2:
        # 计算C++内部RMS (在匹配对上)
        diffs = U[inlier_mask] - Wt[idxs_uw[inlier_mask]]
        rms_cpp_internal = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
        rms_cpp_px = rms_cpp_internal / s0
        print(f"  C++内部匹配对: n_inliers={n_inliers}")
        print(f"  C++内部RMS(角秒): {rms_cpp_internal:.3f}\"")
        print(f"  C++内部RMS(像素): {rms_cpp_px:.3f}px")
        print(f"  C++报告SIP RMS:  {sip_rms:.3f}px")
        print(f"  差异: {abs(rms_cpp_px - sip_rms):.3f}px")

    # ---- 11. SIP过拟合检测 ----
    print_section("11. SIP过拟合检测: 训练集 vs 测试集")

    # 训练集: C++匹配对中的检测星
    # 测试集: 所有检测星(不限于C++匹配对)

    # 将C++匹配对中的检测星转换为像素坐标
    cx, cy = w / 2.0, h / 2.0
    # U向量组中的星就是C++用于匹配的星
    u_x_px = U[:, 0] / s0 + cx
    u_y_px = -U[:, 1] / s0 + cy

    # 用Siril SIP投影Gaia星表，匹配到C++匹配星
    # 先用Siril SIP投影所有Gaia亮星
    x_gaia_sip, y_gaia_sip = wcs_sip_inverse(ra_src, dec_src, wcs)

    # C++匹配星在检测星中的索引 — 需要重建
    # _build_image_vectors选出的星的像素坐标
    if n_sat >= 50:
        sel_mask = img_saturated.astype(bool)
    else:
        mask_sat = img_saturated.astype(bool)
        n_normal = 100 - n_sat
        normal_idx = np.where(~mask_sat)[0]
        if len(normal_idx) > 0 and n_normal > 0:
            sorted_idx = normal_idx[np.argsort(-img_flux[normal_idx])]
            top_normal = sorted_idx[:n_normal]
            sel_mask = np.zeros(len(img_x), dtype=bool)
            sel_mask[np.where(mask_sat)[0]] = True
            sel_mask[top_normal] = True
        else:
            sel_mask = img_saturated.astype(bool)

    sel_x = img_x[sel_mask]
    sel_y = img_y[sel_mask]
    sel_flux = img_flux[sel_mask]

    # 训练集: C++匹配星
    train_tree = cKDTree(np.column_stack([sel_x, sel_y]))
    valid_gaia_sip = np.isfinite(x_gaia_sip) & (x_gaia_sip > 0) & (x_gaia_sip < w) & \
                     (y_gaia_sip > 0) & (y_gaia_sip < h)
    if valid_gaia_sip.sum() >= 2:
        gaia_sip_coords = np.column_stack([x_gaia_sip[valid_gaia_sip], y_gaia_sip[valid_gaia_sip]])
        dists_train, _ = train_tree.query(gaia_sip_coords, k=1)
        matched_train = dists_train < 5.0
        if matched_train.sum() >= 2:
            rms_train = np.sqrt(np.mean(dists_train[matched_train] ** 2))
            print(f"  训练集(C++匹配星): n={matched_train.sum()}, RMS={rms_train:.3f}px")

    # 测试集: 所有检测星
    all_tree = cKDTree(np.column_stack([img_x, img_y]))
    if valid_gaia_sip.sum() >= 2:
        dists_test, _ = all_tree.query(gaia_sip_coords, k=1)
        matched_test = dists_test < 5.0
        if matched_test.sum() >= 2:
            rms_test = np.sqrt(np.mean(dists_test[matched_test] ** 2))
            print(f"  测试集(所有检测星): n={matched_test.sum()}, RMS={rms_test:.3f}px")

        if matched_train.sum() >= 2 and matched_test.sum() >= 2:
            overfit_ratio = rms_test / rms_train
            print(f"\n  过拟合比: 测试/训练 = {overfit_ratio:.2f}")
            if overfit_ratio > 1.3:
                print(f"  [警告] 过拟合比>1.3, SIP可能过拟合训练集!")
            elif overfit_ratio > 1.1:
                print(f"  [注意] 过拟合比>1.1, SIP可能轻微过拟合")
            else:
                print(f"  [正常] 过拟合比接近1.0, SIP泛化良好")

    # ---- 12. 残差方向/径向模式分析 ----
    print_section("12. 残差方向/径向模式分析")

    for label, stats in [("Siril SIP", stats_siril_sip), ("Phase D 线性", stats_pd)]:
        if stats is None:
            continue
        dx = stats['matched_dx']
        dy = stats['matched_dy']
        dists = stats['matched_dists']

        print(f"\n  --- {label} ---")
        print(f"  残差方向统计:")
        print(f"    dx均值={np.mean(dx):.3f}px, dy均值={np.mean(dy):.3f}px")
        print(f"    dx标准差={np.std(dx):.3f}px, dy标准差={np.std(dy):.3f}px")

        # 径向/切向分解
        if 'matched_r' in stats:
            r = stats['matched_r']
            # 径向方向: 从中心到投影点
            # 需要投影点的坐标来计算径向方向
            # 简化: 用残差与径向距离的相关性
            if len(r) >= 10:
                # 残差幅度与径向距离的相关性
                corr = np.corrcoef(r, dists)[0, 1]
                print(f"    残差与径向距离相关系数: {corr:.3f}")
                if abs(corr) > 0.3:
                    print(f"    [注意] 残差与径向距离显著相关, 可能存在系统性径向偏差")

    # ---- 13. 中心区域线性残差诊断 ----
    print_section("13. 中心区域(r<500px)线性残差诊断")

    for label, x_g, y_g, cd_used, crval_used in [
        ("Phase D", x_pd, y_pd, cd_phased, crval_phased),
        ("Siril", x_siril_lin, y_siril_lin, cd_siril, crval_siril),
    ]:
        center_mask = np.isfinite(x_g) & np.isfinite(y_g) & \
                      (x_g > 0) & (x_g < w) & (y_g > 0) & (y_g < h) & \
                      (np.sqrt((x_g - crpix[0])**2 + (y_g - crpix[1])**2) < 500)
        n_center = center_mask.sum()
        if n_center >= 2:
            stats_center = match_and_compute_rms(
                x_g[center_mask], y_g[center_mask], img_x, img_y, w, h,
                max_dist_px=5.0, mag_arr=mag_src[center_mask])
            print_rms_table(stats_center, f"{label} 中心r<500px 线性")

            # 中心区域应该几乎没有畸变，残差主要来自质心误差和CD精度
            if stats_center:
                print(f"    → 中心区域线性残差={stats_center['rms_px']:.3f}px, "
                      f"这大致是质心误差+CD误差的下限")

    # ---- 14. 综合诊断结论 ----
    print_section("14. 综合诊断结论")

    print(f"\n  已知数据:")
    print(f"    C++内部SIP RMS: {sip_rms:.3f}px (基于~200匹配对)")
    if stats_siril_sip:
        print(f"    Python验证SIP RMS: {stats_siril_sip['rms_px']:.3f}px (基于检测星匹配)")
    if stats_pd:
        print(f"    Phase D线性RMS: {stats_pd['rms_px']:.3f}px")
    if stats_siril_lin:
        print(f"    Siril线性RMS: {stats_siril_lin['rms_px']:.3f}px")

    print(f"\n  差异来源分析:")

    # 差异1: 训练集vs测试集
    if stats_siril_sip:
        gap = stats_siril_sip['rms_px'] - sip_rms
        print(f"    1. 训练集vs测试集差距: {gap:.3f}px")
        print(f"       C++ RMS在~200个匹配对上计算(训练集)")
        print(f"       Python RMS在所有检测星上计算(测试集)")
        print(f"       差距={gap:.3f}px, 占总差异的{gap/(stats_siril_sip['rms_px']-sip_rms+1e-10)*100:.0f}%")

    # 差异2: CD矩阵差异
    if stats_pd and stats_siril_lin:
        cd_impact = stats_siril_lin['rms_px'] - stats_pd['rms_px']
        print(f"    2. CD矩阵差异影响: {cd_impact:.3f}px")
        if cd_impact > 0:
            print(f"       Siril CD比Phase D CD差{cd_impact:.3f}px")
            print(f"       → Siril SIP拟合过程中CD修正方向错误!")
        else:
            print(f"       Siril CD比Phase D CD好{-cd_impact:.3f}px")

    # 差异3: SIP修正效果
    if stats_siril_lin and stats_siril_sip:
        sip_effect = stats_siril_lin['rms_px'] - stats_siril_sip['rms_px']
        print(f"    3. SIP修正效果: {sip_effect:.3f}px ({sip_effect/stats_siril_lin['rms_px']*100:.1f}%)")
        if sip_effect < 0:
            print(f"       [严重] SIP修正反而使RMS变差! SIP过拟合!")

    # 差异4: 中心区域基线
    print(f"\n  根因判断:")
    print(f"    如果中心区域(r<500)线性残差≈1.8px, 则:")
    print(f"    - 质心误差+CD误差的基线≈1.8px")
    print(f"    - SIP最多只能修正径向畸变部分")
    print(f"    - 如果SIP改善仅10%, 说明径向畸变本身很小")
    print(f"    - 主要误差来源是质心精度和CD精度, 而非畸变")

    print(f"\n{'=' * 70}")
    print(f"  诊断完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
