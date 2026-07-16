"""V3.5 WCS-SIP渲染验证脚本
功能: 读取V3.5输出的WCS JSON文件，用标准WCS-SIP逆投影将Gaia亮星投影到图像上，
      生成验证图并输出定量指标（投影残差统计、帧内星数等），用于检验WCS和SIP拟合精度。
用法:
  python verify_wcs_v35.py [FITS文件路径] [--json WCS_JSON路径] [--out 输出PNG路径]
  若不指定--json，自动使用与FITS同目录下的vm35_wcs_output.json
  若不指定FITS，默认使用M20 Red测试帧
"""
import os, sys, math, json, argparse, numpy as np, logging

logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("verify_v35")

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy, bisection_mag_limit


def load_wcs_json(json_path):
    """读取V3.5输出的WCS JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    wcs = {
        'CD': np.array(d['CD'], dtype=np.float64),
        'CRVAL': np.array(d['CRVAL'], dtype=np.float64),
        'CRPIX': np.array(d['CRPIX'], dtype=np.float64),
        'SIP_A': np.array(d['SIP_A'], dtype=np.float64).reshape(6, 6),
        'SIP_B': np.array(d['SIP_B'], dtype=np.float64).reshape(6, 6),
        'RMS_PX': float(d.get('RMS_PX', 0)),
        'SIP_ORDER': int(d.get('SIP_ORDER', 5)),
    }
    return wcs


def wcs_sip_inverse_project(ra_src, dec_src, wcs):
    """标准WCS-SIP逆投影: sky(α,δ) → pixel(x,y)

    步骤:
      1. Δα = α - CRVAL1, Δδ = δ - CRVAL2
      2. [ξ', η'] = CD⁻¹ · [Δα, Δδ]
      3. 迭代求解: ξ = ξ' - ΣA_pq·ξ^p·η^q, η = η' - ΣB_pq·ξ^p·η^q
      4. x = ξ + CRPIX1, y = η + CRPIX2
    """
    cd = wcs['CD']
    crval = wcs['CRVAL']
    crpix = wcs['CRPIX']
    sip_A = wcs['SIP_A']
    sip_B = wcs['SIP_B']
    sip_order = wcs['SIP_ORDER']

    # CD逆矩阵
    cdet = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    if abs(cdet) < 1e-30:
        logger.error("CD矩阵行列式接近零，无法求逆")
        return np.full(len(ra_src), np.nan), np.full(len(ra_src), np.nan)
    cd_inv = np.array([[cd[1, 1], -cd[0, 1]], [-cd[1, 0], cd[0, 0]]]) / cdet

    # 天球→中间坐标 [ξ', η']
    dra = ra_src - crval[0]
    ddec = dec_src - crval[1]
    xi_prime = cd_inv[0, 0] * dra + cd_inv[0, 1] * ddec
    eta_prime = cd_inv[1, 0] * dra + cd_inv[1, 1] * ddec

    # 粗筛帧内星点（避免远离帧的星点SIP高次幂溢出）
    x_lin = xi_prime + crpix[0]
    y_lin = eta_prime + crpix[1]
    margin = 500
    prelim = (x_lin > -margin) & (x_lin < 10000 + margin) & \
             (y_lin > -margin) & (y_lin < 10000 + margin)

    xi = xi_prime[prelim].copy()
    eta = eta_prime[prelim].copy()
    xi_prime_s = xi_prime[prelim].copy()
    eta_prime_s = eta_prime[prelim].copy()

    # 预计算非零SIP项（根据SIP_ORDER过滤阶数）
    max_order = min(sip_order, 6) if sip_order > 0 else 0
    sip_terms = []
    if max_order >= 2:
        for p in range(max_order + 1):
            for q in range(max_order + 1):
                if p + q < 2 or p + q > max_order: continue
                if p >= 6 or q >= 6: continue
                a_c = sip_A[p, q]
                b_c = sip_B[p, q]
                if abs(a_c) > 1e-30 or abs(b_c) > 1e-30:
                    sip_terms.append((p, q, a_c, b_c))

    logger.info(f"SIP阶数={sip_order}, 非零项={len(sip_terms)}")

    # 迭代求解 ξ, η
    for iteration in range(30):
        sip_dx = np.zeros_like(xi)
        sip_dy = np.zeros_like(eta)
        for p, q, a_c, b_c in sip_terms:
            # 防止高次幂溢出：clamp ξ/η到合理范围
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
            logger.info(f"SIP迭代收敛: {iteration + 1}次")
            break
        xi = xi_new
        eta = eta_new

    x_pix = np.full(len(ra_src), np.nan)
    y_pix = np.full(len(ra_src), np.nan)
    x_pix[prelim] = xi + crpix[0]
    y_pix[prelim] = eta + crpix[1]

    return x_pix, y_pix


def compute_residual_stats(x_gaia, y_gaia, x_det, y_det, s0, max_match_dist_px=5.0):
    """计算Gaia投影点与检测星点之间的匹配残差统计"""
    from scipy.spatial import cKDTree
    # 构建检测星点KD树
    valid_det = np.isfinite(x_det) & np.isfinite(y_det)
    if valid_det.sum() < 2:
        return None
    det_coords = np.column_stack([x_det[valid_det], y_det[valid_det]])
    tree = cKDTree(det_coords)

    # 对每个Gaia投影点找最近邻
    valid_gaia = np.isfinite(x_gaia) & np.isfinite(y_gaia)
    if valid_gaia.sum() < 2:
        return None
    gaia_coords = np.column_stack([x_gaia[valid_gaia], y_gaia[valid_gaia]])
    dists, idxs = tree.query(gaia_coords, k=1)

    # 只保留距离小于阈值的匹配对
    matched = dists < max_match_dist_px
    if matched.sum() < 2:
        return None

    match_dists = dists[matched]
    stats = {
        'n_matched': int(matched.sum()),
        'mean_px': float(np.mean(match_dists)),
        'median_px': float(np.median(match_dists)),
        'rms_px': float(np.sqrt(np.mean(match_dists ** 2))),
        'p90_px': float(np.percentile(match_dists, 90)),
        'max_px': float(np.max(match_dists)),
        'mean_arcsec': float(np.mean(match_dists) * s0),
        'rms_arcsec': float(np.sqrt(np.mean(match_dists ** 2)) * s0),
    }
    return stats


def render_verification(img_data, w, h, x_gaia, y_gaia, out_path, title=""):
    """渲染验证图：图像+Gaia投影十字标"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = img_data.astype(np.float32)
    dd = data[data > 0]
    lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) > 1 else (0, 1)
    img_s = np.clip((data - lo) / max(hi - lo, 1), 0, 1)

    DPI = 100
    fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")

    # 帧内Gaia星
    bt = np.isfinite(x_gaia) & (x_gaia > 0) & (x_gaia < w) & (y_gaia > 0) & (y_gaia < h)
    if bt.sum() > 0:
        ax.scatter(x_gaia[bt], y_gaia[bt], marker="+", color="#FF0000",
                   s=80, linewidths=2.5, alpha=0.9)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    if title:
        fig.text(0.5, 0.98, title, ha='center', va='top', fontsize=14,
                 color='yellow', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    fig.savefig(out_path, dpi=DPI, pad_inches=0)
    plt.close(fig)
    logger.info(f"验证图已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="V3.5 WCS-SIP渲染验证")
    parser.add_argument("fits", nargs="?", default=None, help="FITS文件路径")
    parser.add_argument("--json", default=None, help="WCS JSON文件路径")
    parser.add_argument("--out", default=None, help="输出PNG路径")
    parser.add_argument("--n_gaia", type=int, default=1000, help="投影Gaia亮星数(默认1000)")
    parser.add_argument("--no_render", action="store_true", help="只输出统计不渲染")
    args = parser.parse_args()

    # 默认测试帧
    default_fname = "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
    if args.fits:
        fits_path = args.fits
    else:
        fits_path = os.path.join(PROJECT_ROOT, "testdata", "lights", default_fname)
        logger.info(f"使用默认测试帧: {default_fname}")

    # 读取FITS
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    cra0 = img.metadata.wcs.crval1
    cdec0 = img.metadata.wcs.crval2
    s0 = 206.265 * ps / fl
    logger.info(f"图像: {w}x{h}, fl={fl}mm, ps={ps}um, s0={s0:.4f}\"/px")
    logger.info(f"初始WCS: RA={cra0:.6f}°, Dec={cdec0:.6f}°")

    # WCS JSON路径
    wcs_json = args.json
    if wcs_json is None:
        wcs_json = os.path.join(PROJECT_ROOT, "vm35_wcs_output.json")
    if not os.path.exists(wcs_json):
        logger.error(f"WCS JSON不存在: {wcs_json}")
        logger.error("请先运行V3.5 solve生成WCS JSON，或用--json指定路径")
        return
    logger.info(f"读取WCS JSON: {wcs_json}")

    # 读取WCS参数
    wcs = load_wcs_json(wcs_json)
    cd = wcs['CD']
    crval = wcs['CRVAL']
    crpix = wcs['CRPIX']
    sip_order = wcs['SIP_ORDER']

    print("\n" + "=" * 60)
    print("V3.5 WCS-SIP 验证报告")
    print("=" * 60)
    print(f"CD矩阵: [[{cd[0,0]:.10e}, {cd[0,1]:.10e}],")
    print(f"          [{cd[1,0]:.10e}, {cd[1,1]:.10e}]]")
    print(f"CRVAL: [{crval[0]:.10f}, {crval[1]:.10f}]")
    print(f"CRPIX: [{crpix[0]:.3f}, {crpix[1]:.3f}]")
    print(f"SIP阶数: {sip_order}")
    print(f"SIP RMS: {wcs['RMS_PX']:.6f} px ({wcs['RMS_PX']*s0:.6f}\")")

    # CD矩阵基本检查
    cdet = cd[0, 0] * cd[1, 1] - cd[0, 1] * cd[1, 0]
    print(f"\nCD行列式: {cdet:.10e}")
    if abs(cdet) < 1e-20:
        print("  [警告] CD行列式接近零，WCS可能无效!")

    # 从CD矩阵提取像素尺度
    scale_x = math.sqrt(cd[0, 0] ** 2 + cd[1, 0] ** 2) * 3600  # deg→arcsec
    scale_y = math.sqrt(cd[0, 1] ** 2 + cd[1, 1] ** 2) * 3600
    print(f"像素尺度: X={scale_x:.4f}\"/px, Y={scale_y:.4f}\"/px (从CD提取)")
    print(f"  与s0偏差: X={abs(scale_x-s0)/s0*100:.2f}%, Y={abs(scale_y-s0)/s0*100:.2f}%")

    # 从CD矩阵提取旋转角
    rot_x = math.degrees(math.atan2(cd[1, 0], cd[0, 0]))
    rot_y = math.degrees(math.atan2(-cd[0, 1], cd[1, 1]))
    print(f"旋转角: θ_x={rot_x:.2f}°, θ_y={rot_y:.2f}° (从CD提取)")

    # 查询Gaia亮星
    logger.info("查询Gaia星表...")
    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    fov_deg = math.sqrt(w * w + h * h) * s0 / 3600.0
    _, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        gaia, crval[0], crval[1], max(0.8, fov_deg * 1.2), args.n_gaia + 200)
    gaia.close()

    mag_a = np.array(cat_mag, np.float64)
    idx = np.argsort(mag_a)[:args.n_gaia]
    ra_src = np.array(cat_ra, np.float64)[idx]
    dec_src = np.array(cat_dec, np.float64)[idx]
    mag_src = mag_a[idx]
    logger.info(f"Gaia查询: {M}颗, 取前{args.n_gaia}亮星")

    # WCS-SIP逆投影
    logger.info("执行WCS-SIP逆投影...")
    x_gaia, y_gaia = wcs_sip_inverse_project(ra_src, dec_src, wcs)

    # 帧内统计
    in_frame = np.isfinite(x_gaia) & (x_gaia > 0) & (x_gaia < w) & \
               (y_gaia > 0) & (y_gaia < h)
    print(f"\n投影结果: {in_frame.sum()}/{len(ra_src)}颗在帧内")

    # 检测星点用于残差计算
    logger.info("检测星点用于残差计算...")
    from star_detector import StarDetector, SDetParamsPy
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    x_det = np.array(det.x, np.float64)
    y_det = np.array(det.y, np.float64)
    logger.info(f"检测到{len(x_det)}颗星")

    # 计算匹配残差
    stats = compute_residual_stats(x_gaia, y_gaia, x_det, y_det, s0)
    print(f"\n{'='*60}")
    print("匹配残差统计 (Gaia投影→最近检测星)")
    print(f"{'='*60}")
    if stats:
        print(f"匹配对数: {stats['n_matched']}")
        print(f"均值残差: {stats['mean_px']:.3f} px ({stats['mean_arcsec']:.3f}\")")
        print(f"中位残差: {stats['median_px']:.3f} px")
        print(f"RMS残差:  {stats['rms_px']:.3f} px ({stats['rms_arcsec']:.3f}\")")
        print(f"P90残差:  {stats['p90_px']:.3f} px")
        print(f"最大残差: {stats['max_px']:.3f} px")

        # 评价
        if stats['rms_px'] < 0.5:
            print("\n[优秀] RMS < 0.5px, WCS-SIP拟合精度极高")
        elif stats['rms_px'] < 1.0:
            print("\n[良好] RMS < 1.0px, WCS-SIP拟合精度可接受")
        elif stats['rms_px'] < 2.0:
            print("\n[一般] RMS < 2.0px, 可能存在系统误差")
        else:
            print("\n[较差] RMS > 2.0px, WCS-SIP拟合可能有问题")
    else:
        print("匹配对不足，无法计算残差统计")

    # 线性投影对比（不用SIP）
    if sip_order > 0:
        wcs_linear = dict(wcs)
        wcs_linear['SIP_ORDER'] = 0
        x_lin, y_lin = wcs_sip_inverse_project(ra_src, dec_src, wcs_linear)
        in_frame_lin = np.isfinite(x_lin) & (x_lin > 0) & (x_lin < w) & \
                       (y_lin > 0) & (y_lin < h)
        stats_lin = compute_residual_stats(x_lin, y_lin, x_det, y_det, s0)
        if stats_lin:
            print(f"\n{'='*60}")
            print("线性投影残差 (无SIP)")
            print(f"{'='*60}")
            print(f"匹配对数: {stats_lin['n_matched']}")
            print(f"RMS残差:  {stats_lin['rms_px']:.3f} px ({stats_lin['rms_arcsec']:.3f}\")")
            if stats:
                improvement = (stats_lin['rms_px'] - stats['rms_px']) / stats_lin['rms_px'] * 100
                print(f"SIP改善: {improvement:.1f}%")

    # 渲染验证图
    if not args.no_render:
        out_path = args.out
        if out_path is None:
            base = os.path.splitext(os.path.basename(fits_path))[0]
            out_path = os.path.join(PROJECT_ROOT, f"verify_v35_{base}.png")

        title = f"V3.5 Verify | SIP order={sip_order}"
        if stats:
            title += f" | RMS={stats['rms_px']:.2f}px"
        render_verification(img.data, w, h, x_gaia, y_gaia, out_path, title)

    print(f"\n{'='*60}")
    print("验证完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
