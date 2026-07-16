"""V4.0 WCS重投影校核 — 用解析出的WCS把Gaia亮星投影到图像上供人工校核

功能:
    读取FITS图像 → 星点检测 → V4.0求解WCS → 查询Gaia亮星
    → 标准WCS-SIP逆投影 → 红色十字标记 → 输出PNG供人工校核

用途:
    校核V4.0解析出的WCS是否正确: 红色十字应精准落在图像星点上

用法:
    python WCS重投影校核V4.py [FITS路径] [输出PNG路径]
    不带参数则使用默认测试帧 M20_T2 Red
"""
import os, sys, math, json, logging
import numpy as np

# ============================================================================
# 路径初始化
# ============================================================================
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("WCS重投影校核V4")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_cpp import VectorMatchV4Cpp
from vector_match_v2 import GaiaClientPy


# ============================================================================
# 标准WCS-SIP逆投影: 天球坐标(RA,Dec) → 像素坐标(x,y)
# ============================================================================
def sky_to_pixel_wcs(ra_arr, dec_arr, cd, crval, crpix, sip_A, sip_B, sip_order):
    """标准 WCS-SIP 逆投影公式 (FITS WCS兼容)

    算法:
        1. 天球偏移: ξ = (RA - CRVAL) * cos(Dec), η = Dec - CRVAL
           (标准WCS: 中间世界坐标 = CD矩阵 · 像素偏移)
        2. CD逆矩阵线性投影: (Δx, Δy) = CD⁻¹ · (ξ, η)
        3. SIP迭代修正: ξ_corr = ξ' - Σ A_pq · ξ^p · η^q (15次迭代或收敛)
        4. 像素坐标: x = ξ_corr + CRPIX_x, y = η_corr + CRPIX_y

    Args:
        ra_arr, dec_arr: 天球坐标数组(度)
        cd: 2x2 CD矩阵(标准FITS WCS格式, 度/pixel)
        crval: [CRVAL1, CRVAL2] 投影中心(度)
        crpix: [CRPIX1, CRPIX2] 参考像素
        sip_A, sip_B: 6x6 SIP系数矩阵
        sip_order: SIP阶数(0=无SIP, 2=二阶, ...)

    Returns:
        (x_pix, y_pix): 像素坐标数组
    """
    ra = np.asarray(ra_arr, dtype=np.float64)
    dec = np.asarray(dec_arr, dtype=np.float64)

    # CD逆矩阵
    cdet = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    if abs(cdet) < 1e-30:
        cdet = 1e-30
    cd_inv = np.array([[cd[1, 1], -cd[0, 1]],
                       [-cd[1, 0], cd[0, 0]]]) / cdet

    # 天球偏移(标准WCS: ξ=(RA-RA0)·cos(Dec0), η=Dec-Dec0)
    cos_d = math.cos(crval[1] * math.pi / 180.0)
    dra = (ra - crval[0]) * cos_d
    ddec = dec - crval[1]

    # 线性投影
    xi_prime = cd_inv[0, 0] * dra + cd_inv[0, 1] * ddec
    eta_prime = cd_inv[1, 0] * dra + cd_inv[1, 1] * ddec

    # 粗筛帧内星点(带margin)
    margin = 500
    prelim = (xi_prime + crpix[0] > -margin) & (xi_prime + crpix[0] < margin + 10000) & \
             (eta_prime + crpix[1] > -margin) & (eta_prime + crpix[1] < margin + 10000)

    # SIP迭代修正
    if sip_order >= 2:
        xi = xi_prime[prelim].copy()
        eta = eta_prime[prelim].copy()
        xi_prime_s = xi_prime[prelim].copy()
        eta_prime_s = eta_prime[prelim].copy()

        # 预计算非零SIP项
        sip_terms = []
        for p in range(sip_order + 1):
            for q in range(sip_order + 1):
                if p + q < 2 or p + q > sip_order:
                    continue
                a_c = sip_A[p, q] if p < sip_A.shape[0] and q < sip_A.shape[1] else 0.0
                b_c = sip_B[p, q] if p < sip_B.shape[0] and q < sip_B.shape[1] else 0.0
                if abs(a_c) > 1e-30 or abs(b_c) > 1e-30:
                    sip_terms.append((p, q, a_c, b_c))

        for _ in range(15):
            sip_dx = np.zeros_like(xi)
            sip_dy = np.zeros_like(eta)
            for p, q, a_c, b_c in sip_terms:
                # 限制ξ/η范围避免高阶外推爆炸
                xc = np.clip(xi, -5e3, 5e3)
                yc = np.clip(eta, -5e3, 5e3)
                term = xc ** p * yc ** q
                term = np.where(np.isfinite(term), term, 0.0)
                sip_dx += a_c * term
                sip_dy += b_c * term
            xi_new = xi_prime_s - sip_dx
            eta_new = eta_prime_s - sip_dy
            if np.max(np.abs(xi_new - xi)) < 1e-6 and np.max(np.abs(eta_new - eta)) < 1e-6:
                break
            xi, eta = xi_new, eta_new

        x_pix = np.full(len(ra), np.nan)
        y_pix = np.full(len(ra), np.nan)
        x_pix[prelim] = xi + crpix[0]
        y_pix[prelim] = eta + crpix[1]
    else:
        x_pix = xi_prime + crpix[0]
        y_pix = eta_prime + crpix[1]

    return x_pix, y_pix


# ============================================================================
# 主流程
# ============================================================================
def reproject_verify(fits_path, output_png=None, n_bright=1000):
    """对单帧FITS做V4.0解析+WCS重投影校核

    Args:
        fits_path: FITS图像路径
        output_png: 输出PNG路径(None则自动命名)
        n_bright: 投影的Gaia亮星数(默认1000)
    """
    logger.info(f"=== WCS重投影校核 V4.0 ===")
    logger.info(f"输入: {fits_path}")

    # ── 1. 读取FITS图像 ──
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    cra0 = img.metadata.wcs.crval1
    cdec0 = img.metadata.wcs.crval2
    s0 = 206.265 * ps / fl
    logger.info(f"图像: {w}x{h}  焦距={fl}mm  像元={ps}um  s0={s0:.4f}\"/px")
    logger.info(f"FITS头中心: RA={cra0:.6f}° Dec={cdec0:.6f}°")

    # ── 2. 星点检测 ──
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    logger.info(f"检测到星点: {len(det.x)}颗 (饱和{int(np.sum(det.saturated))}颗)")

    # ── 3. V4.0求解WCS ──
    wcs_json = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4",
                            f"wcs_{os.path.basename(fits_path)}.json")
    os.makedirs(os.path.dirname(wcs_json), exist_ok=True)

    solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
    result = solver.solve(
        np.array(det.x, np.float64), np.array(det.y, np.float64),
        np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
        cra0, cdec0, fl, ps, w, h,
        wcs_out=wcs_json,
        exptime=getattr(img.metadata.observation, 'exptime', 1.0),
    )
    solver.close()

    if not result:
        logger.error("V4.0求解失败, 无法重投影")
        return None

    logger.info(f"V4.0求解成功: mode={result.flip_mode} matched={result.matched_count} "
                f"RMS={result.rms_px:.3f}px s={result.scale_arcsec_px:.4f}\"/px "
                f"θ={result.rotation_deg:.2f}°")

    # ── 4. 读取WCS参数 ──
    with open(wcs_json, 'r', encoding='utf-8') as f:
        wcs = json.load(f)
    cd = np.array(wcs['CD'], dtype=np.float64).reshape(2, 2)
    crval = np.array(wcs['CRVAL'], dtype=np.float64)
    crpix = np.array(wcs['CRPIX'], dtype=np.float64)
    sip_A = np.array(wcs['SIP_A'], dtype=np.float64).reshape(6, 6)
    sip_B = np.array(wcs['SIP_B'], dtype=np.float64).reshape(6, 6)
    sip_order = int(wcs.get('SIP_ORDER', 4))
    rms_px = float(wcs['RMS_PX'])
    logger.info(f"WCS: CD={cd.tolist()}")
    logger.info(f"     CRVAL={crval.tolist()}  CRPIX={crpix.tolist()}  SIP_ORDER={sip_order}  RMS={rms_px:.3f}px")

    # ── 5. 查询Gaia亮星 ──
    fov_deg = math.sqrt(w * w + h * h) * s0 / 3600.0
    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    # 查询半径取FOV对角线的0.7倍(确保覆盖帧内)
    query_radius = max(fov_deg * 0.7, 1.0)
    ra_t, dec_t, mag_t = gaia.cone_search(crval[0], crval[1], query_radius, 22.0)
    gaia.close()

    ra_all = np.array(ra_t, dtype=np.float64)
    dec_all = np.array(dec_t, dtype=np.float64)
    mag_all = np.array(mag_t, dtype=np.float64)
    logger.info(f"Gaia查询: {len(ra_all)}颗 (半径={query_radius:.2f}°)")

    # 取最亮的n_bright颗
    if len(ra_all) > n_bright:
        idx_bright = np.argsort(mag_all)[:n_bright]
        ra_src = ra_all[idx_bright]
        dec_src = dec_all[idx_bright]
        mag_src = mag_all[idx_bright]
    else:
        ra_src = ra_all
        dec_src = dec_all
        mag_src = mag_all

    # ── 6. WCS-SIP逆投影 ──
    x_pix, y_pix = sky_to_pixel_wcs(
        ra_src, dec_src, cd, crval, crpix, sip_A, sip_B, sip_order)

    # 筛选帧内星点
    in_frame = np.isfinite(x_pix) & (x_pix > 0) & (x_pix < w) & (y_pix > 0) & (y_pix < h)
    x_in = x_pix[in_frame]
    y_in = y_pix[in_frame]
    mag_in = mag_src[in_frame]
    logger.info(f"投影: {int(np.sum(in_frame))}颗在帧内 (共{len(ra_src)}颗)")

    # ── 7. 投影质量诊断 ──
    from scipy.spatial import cKDTree
    det_tree = cKDTree(np.column_stack([det.x, det.y]))
    dists, _ = det_tree.query(np.column_stack([x_in, y_in]))
    n_3px = int(np.sum(dists < 3))
    n_5px = int(np.sum(dists < 5))
    n_10px = int(np.sum(dists < 10))
    n_test = len(x_in)
    logger.info(f"投影质量(帧内{n_test}颗Gaia星): "
                f"3px内={n_3px}({100*n_3px/max(n_test,1):.1f}%)  "
                f"5px内={n_5px}({100*n_5px/max(n_test,1):.1f}%)  "
                f"10px内={n_10px}({100*n_10px/max(n_test,1):.1f}%)")

    # ── 8. 渲染输出PNG ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = img.data.astype(np.float32)
    dd = data[data > 0]
    lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) > 1 else (0, 1)
    img_s = np.clip((data - lo) / max(hi - lo, 1), 0, 1)

    DPI = 100
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")

    # 红色十字标记Gaia投影位置
    ax.scatter(x_in, y_in, marker="+", color="#FF0000", s=80,
               linewidths=2, alpha=0.9, label=f"Gaia {n_bright} bright")

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    if output_png is None:
        output_png = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4",
                                  f"reproject_{os.path.basename(fits_path).replace('.fts', '.png')}")
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    fig.savefig(output_png, dpi=DPI, pad_inches=0)
    plt.close(fig)
    logger.info(f"输出: {output_png}")

    # 返回诊断结果
    return {
        'fits_path': fits_path,
        'output_png': output_png,
        'wcs_json': wcs_json,
        'matched': result.matched_count,
        'rms_px': rms_px,
        'sip_order': sip_order,
        'flip_mode': result.flip_mode,
        'n_in_frame': int(np.sum(in_frame)),
        'n_3px': n_3px, 'n_5px': n_5px, 'n_10px': n_10px,
        'pct_3px': 100 * n_3px / max(n_test, 1),
        'pct_5px': 100 * n_5px / max(n_test, 1),
        'pct_10px': 100 * n_10px / max(n_test, 1),
    }


# ============================================================================
# 命令行入口
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) >= 2:
        fits_path = sys.argv[1]
    else:
        # 默认测试帧
        fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights",
                                 "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts")

    output_png = sys.argv[2] if len(sys.argv) >= 3 else None

    info = reproject_verify(fits_path, output_png)
    if info:
        print(f"\n=== 校核结果 ===")
        print(f"帧: {info['fits_path']}")
        print(f"PNG: {info['output_png']}")
        print(f"匹配对: {info['matched']}  RMS: {info['rms_px']:.3f}px  SIP阶: {info['sip_order']}")
        print(f"翻转模式: {info['flip_mode']}  帧内Gaia星: {info['n_in_frame']}")
        print(f"投影质量: 3px内={info['n_3px']}({info['pct_3px']:.1f}%)  "
              f"5px内={info['n_5px']}({info['pct_5px']:.1f}%)  "
              f"10px内={info['n_10px']}({info['pct_10px']:.1f}%)")
        print(f"\n请打开PNG人工校核: 红色十字应精准落在图像星点上")
