"""
V3.5 批量覆盖图生成脚本
随机抽取50帧 → 全流程solve(含SIP) → WCS-SIP投影Gaia前1000亮星 → 红色十字覆盖图
输出: overlay_output/{目标}_{滤镜}_{日期}.png 全尺寸无边框
"""
import sys, os, math, json, time, random, glob, re, logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger("batch_overlay")

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "overlay_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

try:
    from astro_image_io import ImageReader
    from vector_match_v3_5_cpp import VectorMatchV35Cpp as VM35
    from vector_match_v2 import GaiaClientPy, bisection_mag_limit
    from star_detector import StarDetector, SDetParamsPy
    from astropy.io import fits as afits
    from astropy.coordinates import SkyCoord
    import astropy.units as u
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_all_fits():
    """收集testdata/lights和testdata/lights1下所有.fts文件"""
    files = []
    for root_dir in ["testdata/lights", "testdata/lights1"]:
        full_dir = os.path.join(PROJECT_ROOT, root_dir)
        if not os.path.isdir(full_dir):
            continue
        for dirpath, _, filenames in os.walk(full_dir):
            for fn in filenames:
                if fn.endswith('.fts'):
                    files.append(os.path.relpath(os.path.join(dirpath, fn), PROJECT_ROOT))
    return files


def parse_fits_header(fits_path):
    """解析FITS头的RA/DEC/EXPTIME/FL/PS，不依赖ImageReader的WCS解析"""
    with afits.open(fits_path) as hdul:
        hdr = hdul[0].header
    ra_str = hdr.get('RA', hdr.get('OBJCTRA', None))
    dec_str = hdr.get('DEC', hdr.get('OBJCTDEC', None))
    exptime = float(hdr.get('EXPTIME', hdr.get('EXPOSURE', 1.0)))
    fl = float(hdr.get('FOCALLEN', 0))
    ps = float(hdr.get('XPIXSZ', hdr.get('PIXSIZE', 0)))
    bin_x = int(hdr.get('XBINNING', 1))
    if ra_str is None or dec_str is None:
        return None
    try:
        sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    except Exception:
        sc = SkyCoord(ra_str, dec_str, unit=(u.deg, u.deg))
    cra, cdec = sc.ra.deg, sc.dec.deg
    if fl <= 0 or ps <= 0:
        return None
    ps = ps * bin_x
    return {'cra': cra, 'cdec': cdec, 'exptime': exptime, 'fl': fl, 'ps': ps}


def wcs_sip_project(ra_src, dec_src, cd, crval, crpix, sip_A, sip_B, sip_order, w, h):
    """WCS-SIP逆投影: sky(α,δ) → pixel(x,y), 参考verify_wcs_v35.py"""
    cdet = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
    if abs(cdet) < 1e-30:
        return np.full(len(ra_src), np.nan), np.full(len(ra_src), np.nan)
    cd_inv = np.array([[cd[1,1], -cd[0,1]], [-cd[1,0], cd[0,0]]]) / cdet

    dra = ra_src - crval[0]
    ddec = dec_src - crval[1]
    xi_prime = cd_inv[0,0]*dra + cd_inv[0,1]*ddec
    eta_prime = cd_inv[1,0]*dra + cd_inv[1,1]*ddec

    x_lin = xi_prime + crpix[0]
    y_lin = eta_prime + crpix[1]
    margin = 500
    prelim = (x_lin > -margin) & (x_lin < w+margin) & (y_lin > -margin) & (y_lin < h+margin)

    xi = xi_prime[prelim].copy()
    eta = eta_prime[prelim].copy()
    xi_ps = xi_prime[prelim].copy()
    eta_ps = eta_prime[prelim].copy()

    max_order = min(sip_order, 6) if sip_order > 0 else 0
    sip_terms = []
    if max_order >= 2:
        for p in range(max_order + 1):
            for q in range(max_order + 1):
                if p + q < 2 or p + q > max_order:
                    continue
                if p >= 6 or q >= 6:
                    continue
                a_c = sip_A[p, q] if p < 6 and q < 6 else 0.0
                b_c = sip_B[p, q] if p < 6 and q < 6 else 0.0
                if abs(a_c) > 1e-30 or abs(b_c) > 1e-30:
                    sip_terms.append((p, q, a_c, b_c))

    for _ in range(20):
        sip_dx = np.zeros_like(xi)
        sip_dy = np.zeros_like(eta)
        for p, q, a_c, b_c in sip_terms:
            xi_c = np.clip(xi, -1e4, 1e4)
            eta_c = np.clip(eta, -1e4, 1e4)
            term = xi_c**p * eta_c**q
            term = np.where(np.isfinite(term), term, 0.0)
            sip_dx += a_c * term
            sip_dy += b_c * term
        xi_new = xi_ps - sip_dx
        eta_new = eta_ps - sip_dy
        if np.max(np.abs(xi_new - xi)) < 1e-6 and np.max(np.abs(eta_new - eta)) < 1e-6:
            break
        xi, eta = xi_new, eta_new

    x_pix = np.full(len(ra_src), np.nan)
    y_pix = np.full(len(ra_src), np.nan)
    x_pix[prelim] = xi + crpix[0]
    y_pix[prelim] = eta + crpix[1]
    return x_pix, y_pix


def process_one(rel_path, vm35, gaia_dir, reader, detector, idx, total):
    """处理单帧: solve → 渲染覆盖图"""
    fits_path = os.path.join(PROJECT_ROOT, rel_path)
    basename = os.path.splitext(os.path.basename(rel_path))[0]

    # 解析filter - 从文件名中的曝光时间后面提取
    match = re.search(r'-(\d+S)-(Red|Green|Blue|Lum|H-alpha|Oiii)$', basename, re.IGNORECASE)
    flt = match.group(2) if match else 'Unknown'

    # 解析目标
    parts = basename.split('_')
    target = parts[0] if parts else 'Unknown'

    # 解析日期时间 (唯一标识)
    dt_match = re.search(r'(\d{8})@(\d{6})', basename)
    if dt_match:
        date_str = dt_match.group(1)
        time_str = dt_match.group(2)
        label = f"{target}_{flt}_{date_str}_{time_str}"
    else:
        date_str = 'unknown'
        label = f"{target}_{flt}_{os.path.basename(rel_path).replace('.fts','')}"[-80:]
    out_path = os.path.join(OUTPUT_DIR, f"{label}.png")
    log_path = os.path.join(OUTPUT_DIR, f"{label}.json")

    if os.path.exists(out_path):
        logger.info(f"[{idx}/{total}] SKIP exists: {label}")
        return {"label": label, "status": "SKIP"}

    logger.info(f"[{idx}/{total}] {label}")

    t0 = time.time()
    try:
        hinfo = parse_fits_header(fits_path)
        if hinfo is None:
            logger.warning(f"  no header info")
            return {"label": label, "status": "NO_HEADER"}

        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = hinfo['fl']
        ps = hinfo['ps']
        s0 = 206.265 * ps / fl

        det = detector.detect_ex(img.data)
        img_x = np.array(det.x, np.float64)
        img_y = np.array(det.y, np.float64)
        img_flux = np.array(det.flux, np.float64)
        img_sat = np.array(det.saturated, np.int32)

        wcs_json = os.path.join(OUTPUT_DIR, f"_tmp_{label}.json")
        vm = VM35(gaia_dir, db_type=1)
        result = vm.solve(img_x, img_y, img_flux, img_sat,
                          hinfo['cra'], hinfo['cdec'], fl, ps, w, h,
                          wcs_out=wcs_json, skip_sip=False, exptime=hinfo['exptime'])
        vm.close()

        if not result:
            logger.warning(f"  solve failed")
            return {"label": label, "status": "SOLVE_FAIL"}

        # 尝试读取WCS JSON；如果不存在则从result对象直接构建
        sip_A = np.zeros((6, 6), dtype=np.float64)
        sip_B = np.zeros((6, 6), dtype=np.float64)
        sip_order = 0
        if os.path.exists(wcs_json):
            with open(wcs_json, 'r', encoding='utf-8') as f:
                wcs = json.load(f)
            try: os.remove(wcs_json)
            except OSError: pass
            cd = np.array(wcs['CD'], dtype=np.float64)
            crval = np.array(wcs['CRVAL'], dtype=np.float64)
            crpix = np.array(wcs['CRPIX'], dtype=np.float64)
            sip_A = np.array(wcs.get('SIP_A', [0]*36), dtype=np.float64).reshape(6, 6)
            sip_B = np.array(wcs.get('SIP_B', [0]*36), dtype=np.float64).reshape(6, 6)
            sip_order = wcs.get('SIP_ORDER', 0)
            sip_rms = float(wcs.get('RMS_PX', 0))
        else:
            try: os.remove(wcs_json)
            except OSError: pass
            cos_d = math.cos(hinfo['cdec'] * math.pi / 180.0)
            if cos_d < 1e-10: cos_d = 1e-10
            s_s = result.solve_s
            theta_r = math.radians(result.rotation_deg)
            ct, st = math.cos(theta_r), math.sin(theta_r)
            fx = (result.flip_mode == 1 or result.flip_mode == 3)
            fy = (result.flip_mode == 2 or result.flip_mode == 3)
            sx = -1.0 if fx else 1.0
            sy = -1.0 if fy else 1.0
            s0_3600_s = s0 / (s_s * 3600.0)
            cd = np.array([
                [sx * s0_3600_s * ct / cos_d, -sx * s0_3600_s * st / cos_d],
                [-sy * s0_3600_s * st, -sy * s0_3600_s * ct]
            ], dtype=np.float64)
            crval = np.array([
                hinfo['cra'] - result.solve_tx / (cos_d * 3600.0),
                hinfo['cdec'] - result.solve_ty / 3600.0
            ], dtype=np.float64)
            crpix = np.array([w / 2.0, h / 2.0], dtype=np.float64)
            sip_rms = 0.0
            logger.info(f"  WCS from result (no JSON): CD={cd.tolist()} CRVAL={crval}")

        gaia = GaiaClientPy(gaia_dir, 1)
        fov_deg = math.sqrt(w*w+h*h)*s0/3600.0
        _, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
            gaia, crval[0], crval[1], max(0.8, fov_deg*1.2), 1200)
        gaia.close()

        mag_a = np.array(cat_mag, np.float64)
        top_idx = np.argsort(mag_a)[:1000]
        ra_src = np.array(cat_ra, np.float64)[top_idx]
        dec_src = np.array(cat_dec, np.float64)[top_idx]

        x_gaia, y_gaia = wcs_sip_project(ra_src, dec_src, cd, crval, crpix,
                                          sip_A, sip_B, sip_order, w, h)

        in_frame = np.isfinite(x_gaia) & (x_gaia > 0) & (x_gaia < w) & (y_gaia > 0) & (y_gaia < h)
        n_gaia = in_frame.sum()

        data_f = img.data.astype(np.float32)
        dd = data_f[data_f > 0]
        lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) > 1 else (0, 1)
        img_s = np.clip((data_f - lo) / max(hi - lo, 1), 0, 1)

        DPI = 100
        fig = plt.figure(figsize=(w / DPI, h / DPI), dpi=DPI, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")

        bt = np.isfinite(x_gaia) & (x_gaia > 0) & (x_gaia < w) & (y_gaia > 0) & (y_gaia < h)
        if bt.sum() > 0:
            ax.scatter(x_gaia[bt], y_gaia[bt], marker="+", color="#FF0000",
                       s=60, linewidths=1.5, alpha=0.85)

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.axis("off")

        fig.savefig(out_path, dpi=DPI, pad_inches=0)
        plt.close(fig)

        elapsed = time.time() - t0
        info = {
            "label": label, "status": "OK", "elapsed": round(elapsed, 1),
            "w": w, "h": h, "fl": fl, "ps": ps, "s0": round(s0, 4),
            "s": round(result.solve_s, 6), "theta": round(result.rotation_deg, 3),
            "flip": result.flip_mode, "n_inliers": result.matched_count,
            "sip_order": sip_order, "sip_rms": round(sip_rms, 4),
            "n_gaia_in_frame": int(n_gaia),
            "nsat": int(hinfo.get('nsat', 0)),
        }
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        logger.info(f"  OK s={result.solve_s:.4f} θ={result.rotation_deg:.1f}° "
                     f"n={result.matched_count} sip_rms={sip_rms:.3f}px "
                     f"gaia={n_gaia}/1000 {elapsed:.1f}s")
        return info

    except Exception as e:
        logger.error(f"  exception: {e}")
        return {"label": label, "status": "EXCEPTION", "error": str(e)}


def main():
    all_files = collect_all_fits()
    logger.info(f"发现{len(all_files)}个.fts文件")

    random.seed(42)
    selected = random.sample(all_files, min(50, len(all_files)))

    logger.info(f"随机选取{len(selected)}帧")

    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    reader = ImageReader()
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))

    results = []
    for idx, rel_path in enumerate(selected, 1):
        r = process_one(rel_path, VM35, gaia_dir, reader, detector, idx, len(selected))
        results.append(r)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    fail_count = len(selected) - ok_count
    logger.info(f"\n完成: OK={ok_count} FAIL={fail_count} 共{len(selected)}")
    logger.info(f"输出目录: {OUTPUT_DIR}")

    summary_path = os.path.join(OUTPUT_DIR, "_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"汇总: {summary_path}")


if __name__ == "__main__":
    main()
