"""V4.0 抽样投票向量法 单帧端到端测试

功能:
    读取 FITS 图像 → 星点检测 → V4.0 求解（密度查询 + PROSAC + k-vector + 贝叶斯/三角形 + 分层拟合）
    输出求解结果和 V4.0 调试信息（lnK、三角形通过率、k-vector 耗时等）

用途:
    验证 V4.0 完整流程在真实帧上的功能正确性，对比 V3.5 基线
"""
import sys, os, json, time
# Windows UTF-8 输出
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v4_cpp import VectorMatchV4Cpp
from star_detector import StarDetector, SDetParamsPy


def _parse_ra_dec_from_fits(fits_path: str):
    """从 FITS 头解析初始指向 (RA, DEC in degrees)。

    优先级: CRVAL1/CRVAL2 > OBJCTRA/OBJCTDEC > RA/DEC。
    支持 "HH MM SS.SS" / "DD MM SS.S" 格式与十进制格式。
    """
    from astropy.io import fits
    hdr = fits.getheader(fits_path)
    # 1) 标准 WCS CRVAL
    if "CRVAL1" in hdr and "CRVAL2" in hdr:
        return float(hdr["CRVAL1"]), float(hdr["CRVAL2"])
    # 2) OBJCTRA/OBJCTDEC
    for ra_key, dec_key in [("OBJCTRA", "OBJCTDEC"), ("RA", "DEC")]:
        if ra_key in hdr and dec_key in hdr:
            ra_str = str(hdr[ra_key]).strip()
            dec_str = str(hdr[dec_key]).strip()
            try:
                ra = _parse_sexagesimal_ra(ra_str)
                dec = _parse_sexagesimal_dec(dec_str)
                return ra, dec
            except Exception as e:
                print(f"  [warn] 解析 {ra_key}/{dec_key} 失败: {e}")
    raise ValueError(f"FITS 头中未找到 CRVAL1/2、OBJCTRA/OBJCTDEC 或 RA/DEC: {fits_path}")


def _parse_sexagesimal_ra(s: str) -> float:
    """解析 RA 字符串 → 十进制度。支持 'HH MM SS.SS' 或十进制。"""
    s = s.strip()
    if " " in s:
        parts = s.split()
        if len(parts) == 3:
            h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
            return (h + m / 60.0 + sec / 3600.0) * 15.0
        if len(parts) == 2:
            h, m = float(parts[0]), float(parts[1])
            return (h + m / 60.0) * 15.0
    return float(s)  # 已是十进制度


def _parse_sexagesimal_dec(s: str) -> float:
    """解析 DEC 字符串 → 十进制度。支持 'DD MM SS.S' 或十进制。"""
    s = s.strip()
    sign = 1.0
    if s.startswith("-"):
        sign = -1.0
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    if " " in s:
        parts = s.split()
        if len(parts) == 3:
            d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
            return sign * (d + m / 60.0 + sec / 3600.0)
        if len(parts) == 2:
            d, m = float(parts[0]), float(parts[1])
            return sign * (d + m / 60.0)
    return sign * float(s)


def solve_one_frame(fits_path: str, gaia_dir: str, wcs_out: str = None, log_path: str = None):
    """对单帧运行 V4.0 求解

    Returns:
        (result, elapsed_sec) 或 (None, elapsed_sec)
    """
    print(f"\n=== V4.0 单帧测试 ===")
    print(f"FITS: {os.path.basename(fits_path)}")

    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    # 优先使用 WCS CRVAL，无 WCS 时回退到 OBJCTRA/OBJCTDEC（宽视场 mosaic 帧无 WCS）
    if img.metadata.wcs is not None:
        cra0 = img.metadata.wcs.crval1
        cdec0 = img.metadata.wcs.crval2
    else:
        cra0, cdec0 = _parse_ra_dec_from_fits(fits_path)
        print(f"  [info] 无 WCS 元数据，从 FITS 头解析初始指向: ({cra0:.4f}, {cdec0:.4f})")
    exptime = getattr(img.metadata.observation, 'exposure', 1.0) or 1.0
    print(f"图像: {w}x{h} fl={fl}mm ps={ps}um 中心=({cra0:.4f}, {cdec0:.4f}) exptime={exptime}s")

    # 星点检测
    t0 = time.time()
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    t_detect = time.time() - t0
    print(f"星点检测: {len(det.x)} 颗 ({t_detect:.2f}s)")

    # V4.0 求解
    if wcs_out is None:
        wcs_out = os.path.join(PROJECT_ROOT, "vm4_wcs_output.json")

    vm = VectorMatchV4Cpp(gaia_dir, db_type=1)
    try:
        t0 = time.time()
        result = vm.solve(
            np.array(det.x, np.float64), np.array(det.y, np.float64),
            np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
            cra0, cdec0, fl, ps, w, h,
            wcs_out=wcs_out, exptime=exptime,
            log_file_path=log_path,
        )
        t_solve = time.time() - t0
    finally:
        vm.close()

    if result is None:
        print(f"❌ 求解失败 (耗时 {t_solve:.2f}s)")
        return None, t_solve

    # 输出结果
    print(f"\n✅ 求解成功 (耗时 {t_solve:.2f}s)")
    print(f"  best_mode={result.flip_mode}  s={result.solve_s:.6f}  θ={result.rotation_deg:.4f}°")
    print(f"  matched={result.matched_count}  rms_px={result.rms_px:.4f}  sip_rms={result.sip_rms_px:.4f}px")
    print(f"  sip_order={result.sip_order}  n_sip_total={result.n_sip_total}")
    print(f"\n--- V4.0 调试信息 ---")
    print(f"  Phase 0: m_lim={result.m_lim_final:.2f}  n_gaia={result.n_gaia_final}  "
          f"ρ_img={result.rho_img:.3f}  ρ_target={result.rho_target:.3f}  iter={result.m_lim_iterations}")
    print(f"  Phase A: θ_SNR={result.theta_snr:.1f}x  θ_peak={result.theta_peak_deg:.2f}°  "
          f"prosac_pool={result.prosac_pool_final}  quality_med={result.prosac_quality_median:.3f}")
    print(f"  Phase B: n_pairs={result.n_phaseb_pairs}  n_corr={result.n_phaseb_corr}  "
          f"n_records={result.n_phasea_records}")
    print(f"  Phase C: n_expanded={result.n_phasec_expanded}  "
          f"kv_build={result.kvector_build_ms:.1f}ms  kv_queries={result.kvector_queries}  "
          f"avg_cand={result.kvector_avg_candidates:.2f}")
    print(f"  Phase D: n_clean={result.n_phased_clean}  iters={result.n_phased_iterations}  "
          f"MAD_RMS={result.mad_rms_arcsec:.3f}\"")
    print(f"  Phase D': lnK={result.bayes_lnK:.2f}  bayes_n={result.bayes_n_match}  "
          f"decision={result.bayes_decision}  tri_total={result.triangle_total}  "
          f"tri_ratio={result.triangle_pass_ratio:.3f}")

    # WCS 验证
    if os.path.exists(wcs_out):
        with open(wcs_out, "r", encoding="utf-8") as f:
            d = json.load(f)
        print(f"\n--- WCS JSON ---")
        print(f"  CD: {d['CD']}")
        print(f"  CRVAL: {d['CRVAL']}")
        print(f"  CRPIX: {d['CRPIX']}")
        print(f"  SIP_ORDER: {d.get('SIP_ORDER', 'N/A')}")
        print(f"  RMS_PX: {d['RMS_PX']:.4f}")
        a_nonzero = sum(1 for x in d["SIP_A"] if abs(x) > 1e-30)
        b_nonzero = sum(1 for x in d["SIP_B"] if abs(x) > 1e-30)
        print(f"  SIP_A nonzero: {a_nonzero}, SIP_B nonzero: {b_nonzero}")

    return result, t_solve


if __name__ == "__main__":
    # 默认测试帧: M20_T2 Red
    fits_path = os.path.join(
        PROJECT_ROOT, "testdata", "lights",
        "M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"
    )
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    wcs_out = os.path.join(PROJECT_ROOT, "vm4_wcs_output.json")
    log_path = os.path.join(
        PROJECT_ROOT, "lib", "plate_solve", "logs", "v4",
        f"v4_test_{time.strftime('%Y%m%d_%H%M%S')}.log"
    )

    if len(sys.argv) > 1:
        fits_path = sys.argv[1]
    if len(sys.argv) > 2:
        gaia_dir = sys.argv[2]

    result, elapsed = solve_one_frame(fits_path, gaia_dir, wcs_out, log_path)
    if result:
        print(f"\n日志文件: {log_path}")
        print(f"总耗时: {elapsed:.2f}s")
        sys.exit(0)
    else:
        sys.exit(1)
