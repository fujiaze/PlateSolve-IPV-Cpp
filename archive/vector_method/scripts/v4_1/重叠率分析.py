"""计算V4.1选出的250颗图像星与Gaia星的真实匹配重叠率

功能:
    对NGC55 Oiii的失败帧和成功帧，分别计算V4.1选出的250颗图像星
    与Gaia查询到的265颗星之间的真实匹配重叠率。

方法:
    1. 读取FITS → 星点检测 → V4.1选250颗
    2. 用已有WCS或近似WCS将250颗图像星投影到天球坐标(RA,DEC)
    3. 用Gaia cone_search查询该区域星表，取最亮265颗
    4. 用KDTree匹配：角距离<阈值(3"/10"/30"/60")的视为匹配
    5. 统计重叠率 = 匹配数/250

依赖:
    - astro_image_io (FITS读取)
    - star_detector (星点检测)
    - vector_match_v2 (GaiaClientPy, gnomonic_inverse)
    - vector_match_v4_cpp (VectorMatchV4Cpp, density_match_query)
    - numpy, scipy
"""

import os, sys, math, json, time
import numpy as np
from scipy.spatial import cKDTree

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy, gnomonic_forward, gnomonic_inverse
from vector_match_v4_cpp import VectorMatchV4Cpp, density_match_query


# ============================================================================
# WCS投影工具函数
# ============================================================================

def pixel_to_sky_wcs(x, y, cd, crval, crpix, sip_A=None, sip_B=None, sip_order=0):
    """标准WCS+SIP逆投影: 像素坐标 → 天球坐标(RA, Dec度)

    Args:
        x, y: 像素坐标数组
        cd: 2x2 CD矩阵
        crval: [crval1, crval2] 参考天球坐标(度)
        crpix: [crpix1, crpix2] 参考像素坐标
        sip_A, sip_B: SIP畸变系数 6x6
        sip_order: SIP多项式阶数
    Returns:
        (ra_deg, dec_deg): 天球坐标数组(度)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    dx = x - crpix[0]
    dy = y - crpix[1]

    # SIP改正
    if sip_order >= 2 and sip_A is not None and sip_B is not None:
        fdx = dx.copy()
        fdy = dy.copy()
        for p in range(2, sip_order + 1):
            for i in range(p + 1):
                j = p - i
                fdx += sip_A[i][j] * dx**i * dy**j
                fdy += sip_B[i][j] * dx**i * dy**j
    else:
        fdx = dx
        fdy = dy

    # CD矩阵 → (xi, eta) 度
    xi = cd[0, 0] * fdx + cd[0, 1] * fdy
    eta = cd[1, 0] * fdx + cd[1, 1] * fdy

    # TAN逆投影: (xi, eta)度 → (RA, Dec)度
    # xi/eta 即为gnomonic投影中的偏移（度单位）
    ra0 = crval[0]
    dec0 = crval[1]

    # 使用gnomonic_inverse（输入角秒）
    xi_asec = xi * 3600.0
    eta_asec = eta * 3600.0
    ra_out, dec_out = gnomonic_inverse(xi_asec, eta_asec, ra0, dec0)

    return ra_out, dec_out


def sky_to_pixel_wcs(ra, dec, cd_inv, crval, crpix, sip_ap=None, sip_bp=None, sip_order=0):
    """标准WCS+SIP正投影: 天球坐标(RA, Dec度) → 像素坐标

    Args:
        ra, dec: 天球坐标数组(度)
        cd_inv: CD逆矩阵 2x2
        crval: [crval1, crval2]
        crpix: [crpix1, crpix2]
        sip_ap, sip_bp: SIP AP/BP系数(逆变换) 6x6
        sip_order: SIP阶数
    Returns:
        (x, y): 像素坐标数组
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)

    # gnomonic正投影: (RA,Dec) → (xi, eta) 角秒
    xi_asec, eta_asec, _ = gnomonic_forward(ra, dec, crval[0], crval[1])

    # 角秒 → 度
    xi = xi_asec / 3600.0
    eta = eta_asec / 3600.0

    # CD逆矩阵 → (dx, dy)
    dx = cd_inv[0, 0] * xi + cd_inv[0, 1] * eta
    dy = cd_inv[1, 0] * xi + cd_inv[1, 1] * eta

    # SIP AP/BP改正
    if sip_order >= 2 and sip_ap is not None and sip_bp is not None:
        fdx = dx.copy()
        fdy = dy.copy()
        for p in range(2, sip_order + 1):
            for i in range(p + 1):
                j = p - i
                fdx += sip_ap[i][j] * dx**i * dy**j
                fdy += sip_bp[i][j] * dx**i * dy**j
        dx = fdx
        dy = fdy

    x = dx + crpix[0]
    y = dy + crpix[1]
    return x, y


def angular_distance_deg(ra1, dec1, ra2, dec2):
    """计算两组天球坐标之间的角距(度), 精确球面余弦公式"""
    ra1 = np.radians(np.asarray(ra1, dtype=np.float64))
    dec1 = np.radians(np.asarray(dec1, dtype=np.float64))
    ra2 = np.radians(np.asarray(ra2, dtype=np.float64))
    dec2 = np.radians(np.asarray(dec2, dtype=np.float64))
    # Vincenty公式（比haversine更精确）
    dra = ra1 - ra2
    num = np.sqrt((np.cos(dec2) * np.sin(dra))**2 +
                  (np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(dra))**2)
    den = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(dra)
    return np.degrees(np.arctan2(num, den))


# ============================================================================
# RA/DEC解析
# ============================================================================

def parse_ra_hms(s):
    """解析RA格式 'HH MM SS.SSS' → 度"""
    parts = str(s).strip().split()
    return (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15


def parse_dec_dms(s):
    """解析DEC格式 'sDD MM SS.SSS' → 度"""
    s = str(s).strip()
    sign = -1 if s.startswith('-') else 1
    s = s.lstrip('+-').strip()
    parts = s.split()
    return sign * (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600)


# ============================================================================
# V4.1选星逻辑
# ============================================================================

def v41_select_stars(det, n_total=250):
    """V4.1选星: 全部饱和星 + flux最高的非饱和星补足至n_total

    Args:
        det: StarDetectionResult
        n_total: 目标总数(默认250)
    Returns:
        sel_idx: 选中星点在det中的索引
        sel_sat: 选中星点的饱和标记
    """
    sat_mask = np.array(det.saturated, dtype=bool)
    sat_idx = np.where(sat_mask)[0]
    non_sat_idx = np.where(~sat_mask)[0]
    flux = np.array(det.flux, dtype=np.float64)

    non_sat_sorted = non_sat_idx[np.argsort(-flux[non_sat_idx])]
    n_needed = max(0, n_total - len(sat_idx))
    top_non_sat = non_sat_sorted[:n_needed]
    sel_idx = np.concatenate([sat_idx, top_non_sat])
    sel_sat = sat_mask[sel_idx]
    return sel_idx, sel_sat


# ============================================================================
# 主分析函数
# ============================================================================

def analyze_frame(fits_path, wcs_json_path=None, use_v4_solve=False):
    """分析单帧的重叠率

    Args:
        fits_path: FITS文件路径
        wcs_json_path: 已有WCS JSON路径(优先使用)
        use_v4_solve: 是否运行V4.0求解
    Returns:
        dict: 分析结果
    """
    basename = os.path.basename(fits_path)
    print(f"\n{'='*70}")
    print(f"帧: {basename}")
    print(f"{'='*70}")

    # 1. 读取FITS
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    exptime = img.metadata.calibration.exptime
    s0 = 206.265 * ps / fl
    print(f"  图像: {w}×{h}, fl={fl}mm, ps={ps}μm, s0={s0:.4f}\"/px")

    # 2. 读取FITS头获取初始指向
    kw_dict = {}
    for kw in img.keywords:
        kw_dict[kw.name.upper()] = kw.value

    obj_ra_str = kw_dict.get('OBJCTRA') or kw_dict.get('RA')
    obj_dec_str = kw_dict.get('OBJCTDEC') or kw_dict.get('DEC')
    cra0 = parse_ra_hms(obj_ra_str)
    cdec0 = parse_dec_dms(obj_dec_str)
    print(f"  指向: RA={cra0:.6f}°, Dec={cdec0:.6f}°")

    # 3. 星点检测
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    n_detected = len(det.x)
    sat_mask_all = np.array(det.saturated, dtype=bool)
    n_saturated = int(np.sum(sat_mask_all))
    flux_all = np.array(det.flux, dtype=np.float64)
    print(f"  检测: {n_detected}颗, 饱和{n_saturated}颗, 正常{n_detected - n_saturated}颗")

    # 4. V4.1选星(250颗)
    sel_idx, sel_sat = v41_select_stars(det, n_total=250)
    sel_x = np.array(det.x)[sel_idx].astype(np.float64)
    sel_y = np.array(det.y)[sel_idx].astype(np.float64)
    sel_flux = flux_all[sel_idx]
    N_sel = len(sel_idx)
    print(f"  选取: {N_sel}颗 (饱和{np.sum(sel_sat)}, 非饱和{np.sum(~sel_sat)})")

    # ========================================================================
    # 5. 获取WCS — 优先使用已有WCS JSON，否则尝试V4.0求解，最后用近似WCS
    # ========================================================================
    wcs_source = "unknown"
    if wcs_json_path and os.path.exists(wcs_json_path):
        with open(wcs_json_path, 'r', encoding='utf-8') as f:
            wcs_data = json.load(f)
        cd = np.array(wcs_data['CD'], dtype=np.float64).reshape(2, 2)
        crval = np.array(wcs_data['CRVAL'], dtype=np.float64)
        crpix = np.array(wcs_data['CRPIX'], dtype=np.float64)
        sip_A = np.array(wcs_data['SIP_A'], dtype=np.float64).reshape(6, 6)
        sip_B = np.array(wcs_data['SIP_B'], dtype=np.float64).reshape(6, 6)
        sip_order = int(wcs_data.get('SIP_ORDER', 0))
        rms_px = float(wcs_data.get('RMS_PX', -1))
        wcs_source = f"已有WCS JSON (RMS={rms_px:.2f}px)"
        print(f"  WCS: {wcs_source}")
        print(f"    CRVAL=({crval[0]:.6f}, {crval[1]:.6f}), CRPIX=({crpix[0]:.1f}, {crpix[1]:.1f})")
        print(f"    CD=({cd[0,0]:.6e}, {cd[0,1]:.6e}, {cd[1,0]:.6e}, {cd[1,1]:.6e})")
    elif use_v4_solve:
        # 尝试V4.0求解
        solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
        result = solver.solve(
            sel_x, sel_y, sel_flux,
            np.array(det.saturated)[sel_idx].astype(np.int32),
            cra0, cdec0, fl, ps, w, h,
            exptime=exptime, verbose=True,
        )
        solver.close()
        if result and result.cd is not None:
            cd = result.cd
            crval = result.crval
            crpix = result.crpix
            sip_A = result.sip_A
            sip_B = result.sip_B
            sip_order = result.sip_order
            wcs_source = f"V4.0求解成功 (RMS={result.rms_px:.2f}px)"
            print(f"  WCS: {wcs_source}")
        else:
            # 近似WCS
            cd = np.array([[-s0 / 3600, 0], [0, s0 / 3600]], dtype=np.float64)
            crval = np.array([cra0, cdec0], dtype=np.float64)
            crpix = np.array([w / 2.0, h / 2.0], dtype=np.float64)
            sip_A = np.zeros((6, 6), dtype=np.float64)
            sip_B = np.zeros((6, 6), dtype=np.float64)
            sip_order = 0
            wcs_source = "近似WCS (OBJCTRA/DEC + 单位CD)"
            print(f"  WCS: {wcs_source}")
    else:
        # 直接用近似WCS
        cd = np.array([[-s0 / 3600, 0], [0, s0 / 3600]], dtype=np.float64)
        crval = np.array([cra0, cdec0], dtype=np.float64)
        crpix = np.array([w / 2.0, h / 2.0], dtype=np.float64)
        sip_A = np.zeros((6, 6), dtype=np.float64)
        sip_B = np.zeros((6, 6), dtype=np.float64)
        sip_order = 0
        wcs_source = "近似WCS (OBJCTRA/DEC + 单位CD)"
        print(f"  WCS: {wcs_source}")

    # ========================================================================
    # 6. 投影250颗图像星到天球坐标
    # ========================================================================
    img_ra, img_dec = pixel_to_sky_wcs(sel_x, sel_y, cd, crval, crpix, sip_A, sip_B, sip_order)
    print(f"\n  图像星天球坐标范围:")
    print(f"    RA: [{img_ra.min():.6f}, {img_ra.max():.6f}]°")
    print(f"    Dec: [{img_dec.min():.6f}, {img_dec.max():.6f}]°")

    # ========================================================================
    # 7. 查询Gaia星表 — 用与V4.1 Phase 0相同的密度匹配查询
    # ========================================================================
    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
    fov_diag_deg = math.sqrt(w * w + h * h) * s0 / 3600.0

    # 方法1: 用density_match_query获取与V4.1完全一致的Gaia星表
    dm = density_match_query(
        gaia, cra0, cdec0, fov_diag_deg, N_sel,
        k_match=1.5, query_radius_factor=1.0,
        m_lim_step=0.5, m_lim_max_iter=8,
        density_tolerance=0.1,
        focal_length_mm=fl, exposure_time_s=exptime,
    )
    gaia_ra_all = np.array(dm['ra'])
    gaia_dec_all = np.array(dm['dec'])
    gaia_mag_all = np.array(dm['mag'])
    n_gaia_all = len(gaia_ra_all)
    print(f"\n  Gaia密度匹配查询: {n_gaia_all}颗 (m_lim={dm['m_lim_final']:.2f}, iter={dm['m_lim_iterations']})")
    print(f"    n_target={dm['n_target']} (k_match=1.5 × {N_sel})")

    # FOV内筛选 + 取最亮265颗（V4.1的n_target就是375，但M实际取决于FOV内数量）
    # V4.1 Phase A用 n_target = k_match × n_img_bright = 1.5 × 250 = 375
    # 但实际FOV内的Gaia星数M取决于密度匹配的结果
    # 这里我们取最亮的 265 颗(与题目一致)和 n_target 颗分别分析

    # gnomonic投影筛选FOV内
    xi_g, eta_g, valid_g = gnomonic_forward(gaia_ra_all, gaia_dec_all, cra0, cdec0)
    fov_half_w = w / 2 * s0
    fov_half_h = h / 2 * s0
    in_fov = valid_g & (np.abs(xi_g) < fov_half_w) & (np.abs(eta_g) < fov_half_h)
    fov_idx = np.where(in_fov)[0]
    n_gaia_fov = len(fov_idx)
    print(f"  FOV内Gaia星: {n_gaia_fov}颗")

    # 取FOV内最亮的265颗（题目指定）
    if n_gaia_fov > 0:
        fov_mag = gaia_mag_all[fov_idx]
        bright_265_order = np.argsort(fov_mag)[:min(265, n_gaia_fov)]
        idx_265 = fov_idx[bright_265_order]
        gaia_ra_265 = gaia_ra_all[idx_265]
        gaia_dec_265 = gaia_dec_all[idx_265]
        gaia_mag_265 = gaia_mag_all[idx_265]
        n_265 = len(idx_265)
        print(f"  Gaia最亮{n_265}颗: mag范围 [{gaia_mag_265.min():.2f}, {gaia_mag_265.max():.2f}]")

        # 也取n_target颗（V4.1实际用的数量）
        n_target = dm['n_target']
        bright_target_order = np.argsort(fov_mag)[:min(n_target, n_gaia_fov)]
        idx_target = fov_idx[bright_target_order]
        gaia_ra_target = gaia_ra_all[idx_target]
        gaia_dec_target = gaia_dec_all[idx_target]
        gaia_mag_target = gaia_mag_all[idx_target]
        print(f"  Gaia V4.1目标{n_target}颗: mag范围 [{gaia_mag_target.min():.2f}, {gaia_mag_target.max():.2f}]")
    else:
        print(f"  警告: FOV内无Gaia星!")
        gaia.close()
        return None

    # 也查询更大范围(mag<20)确保不遗漏
    gaia_ra_20, gaia_dec_20, gaia_mag_20 = gaia.cone_search(
        cra0, cdec0, fov_diag_deg * 0.7, 20.0
    )
    gaia_ra_20 = np.array(gaia_ra_20)
    gaia_dec_20 = np.array(gaia_dec_20)
    gaia_mag_20 = np.array(gaia_mag_20)
    n_gaia_20 = len(gaia_ra_20)
    print(f"  Gaia mag<20查询: {n_gaia_20}颗")
    gaia.close()

    # ========================================================================
    # 8. KDTree匹配 — 图像星投影天球 vs Gaia星天球
    # ========================================================================
    # 使用精确角距匹配（而非近似平面距离）
    print(f"\n  {'='*50}")
    print(f"  真实匹配重叠率分析 (WCS来源: {wcs_source})")
    print(f"  {'='*50}")

    # 用角距做匹配 — 对每颗图像星找最近的Gaia星
    results = {}
    for gaia_label, gaia_ra_sub, gaia_dec_sub, gaia_mag_sub in [
        ("Gaia最亮265颗", gaia_ra_265, gaia_dec_265, gaia_mag_265),
        ("Gaia V4.1目标颗", gaia_ra_target, gaia_dec_target, gaia_mag_target),
    ]:
        n_gaia_sub = len(gaia_ra_sub)
        if n_gaia_sub == 0:
            continue

        # 精确角距匹配
        dists_deg = np.zeros(N_sel)
        nearest_idx = np.zeros(N_sel, dtype=int)
        for i in range(N_sel):
            d = angular_distance_deg(
                np.full(n_gaia_sub, img_ra[i]),
                np.full(n_gaia_sub, img_dec[i]),
                gaia_ra_sub, gaia_dec_sub,
            )
            nearest_idx[i] = np.argmin(d)
            dists_deg[i] = d[nearest_idx[i]]

        dists_arcsec = dists_deg * 3600.0

        # 不同阈值下的匹配统计
        matched_3as = int(np.sum(dists_arcsec < 3.0))
        matched_5as = int(np.sum(dists_arcsec < 5.0))
        matched_10as = int(np.sum(dists_arcsec < 10.0))
        matched_30as = int(np.sum(dists_arcsec < 30.0))
        matched_60as = int(np.sum(dists_arcsec < 60.0))
        matched_120as = int(np.sum(dists_arcsec < 120.0))

        print(f"\n  --- {gaia_label} ({n_gaia_sub}颗) vs 图像星 ({N_sel}颗) ---")
        print(f"  <3\":  {matched_3as:4d} ({100*matched_3as/N_sel:.1f}%)")
        print(f"  <5\":  {matched_5as:4d} ({100*matched_5as/N_sel:.1f}%)")
        print(f"  <10\": {matched_10as:4d} ({100*matched_10as/N_sel:.1f}%)")
        print(f"  <30\": {matched_30as:4d} ({100*matched_30as/N_sel:.1f}%)")
        print(f"  <60\": {matched_60as:4d} ({100*matched_60as/N_sel:.1f}%)")
        print(f"  <120\": {matched_120as:4d} ({100*matched_120as/N_sel:.1f}%)")

        # 距离分布统计
        print(f"\n  距离分布 (角秒):")
        for pct in [10, 25, 50, 75, 90]:
            print(f"    P{pct}: {np.percentile(dists_arcsec, pct):.2f}\"")
        print(f"    均值: {np.mean(dists_arcsec):.2f}\"")
        print(f"    中位数: {np.median(dists_arcsec):.2f}\"")

        # 饱和/非饱和分类统计
        for sat_label, sat_m in [("饱和星", sel_sat), ("非饱和星", ~sel_sat)]:
            n_sub = int(np.sum(sat_m))
            if n_sub == 0:
                continue
            d_sub = dists_arcsec[sat_m]
            m3 = int(np.sum(d_sub < 3.0))
            m10 = int(np.sum(d_sub < 10.0))
            m60 = int(np.sum(d_sub < 60.0))
            print(f"    {sat_label}({n_sub}颗): <3\"={m3}({100*m3/n_sub:.1f}%), "
                  f"<10\"={m10}({100*m10/n_sub:.1f}%), <60\"={m60}({100*m60/n_sub:.1f}%)")

        results[gaia_label] = {
            'matched_3as': matched_3as, 'matched_5as': matched_5as,
            'matched_10as': matched_10as, 'matched_30as': matched_30as,
            'matched_60as': matched_60as, 'matched_120as': matched_120as,
            'dists_arcsec': dists_arcsec,
        }

    # ========================================================================
    # 9. 对比: 全部检测星的重叠率（不限于250颗）
    # ========================================================================
    print(f"\n  --- 全部{n_detected