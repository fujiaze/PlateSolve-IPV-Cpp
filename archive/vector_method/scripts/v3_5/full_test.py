# 全局强制UTF-8编码
import sys, os, json, time
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

"""
V3.5 全量测试脚本
每个目标/滤镜取1帧，统计成功率、RMS、SIP阶数等
"""
import numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy

# 每个目标/滤镜取1帧（代表性帧）
TEST_FRAMES = [
    # M20 T2
    ("M20_T2", "Red",     "testdata/lights/M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"),
    ("M20_T2", "Blue",    "testdata/lights/M20_T2_flying_dutchman-20250719@005635-300S-Blue.fts"),
    # LDN43
    ("LDN43", "Red",      "testdata/lights/LDN43_LRGBH_flying_dutchman-20250716@235722-1200S-Red.fts"),
    ("LDN43", "Green",    "testdata/lights/LDN43_LRGBH_flying_dutchman-20250717@001813-1200S-Green.fts"),
    ("LDN43", "Blue",     "testdata/lights/LDN43_LRGBH_flying_dutchman-20250620@052246-1200S-Blue.fts"),
    ("LDN43", "Lum",      "testdata/lights/LDN43_LRGBH_flying_dutchman-20250716@233436-600S-Lum.fts"),
    ("LDN43", "H-alpha",  "testdata/lights/LDN43_LRGBH_flying_dutchman-20250717@004958-1200S-H-alpha.fts"),
    # Galaxy Center Panel1
    ("GC_P1", "Red",      "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts"),
    ("GC_P1", "Green",    "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@012151-180S-Green.fts"),
    ("GC_P1", "Blue",     "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@062557-180S-Blue.fts"),
    ("GC_P1", "H-alpha",  "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@001306-300S-H-alpha.fts"),
    ("GC_P1", "Oiii",     "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts"),
    # Galaxy Center Panel2
    ("GC_P2", "Red",      "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@032620-180S-Red.fts"),
    ("GC_P2", "Green",    "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@002955-180S-Green.fts"),
    ("GC_P2", "Blue",     "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@005249-180S-Blue.fts"),
    ("GC_P2", "H-alpha",  "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@030735-300S-H-alpha.fts"),
    ("GC_P2", "Oiii",     "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts"),
    # Galaxy Center Panel3
    ("GC_P3", "Red",      "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@005108-180S-Red.fts"),
    ("GC_P3", "Green",    "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@012315-180S-Green.fts"),
    ("GC_P3", "Blue",     "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@021257-180S-Blue.fts"),
    ("GC_P3", "H-alpha",  "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@022237-300S-H-alpha.fts"),
    ("GC_P3", "Oiii",     "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@025838-600S-Oiii.fts"),
]

def extract_filter(filename):
    """从文件名提取滤镜名"""
    parts = filename.split('-')
    for p in parts:
        if p in ('Red', 'Green', 'Blue', 'Lum', 'H-alpha', 'Oiii'):
            return p
    return 'Unknown'

def run_test(target, filt, rel_path):
    """对单帧运行V3.5 solve"""
    fits_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(fits_path):
        return {"target": target, "filter": filt, "status": "FILE_NOT_FOUND"}

    t0 = time.time()
    try:
        reader = ImageReader()
        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = img.metadata.observation.focallen
        ps = img.metadata.observation.xpixsz
        s0 = 206.265 * ps / fl
        
        # GC帧可能没有WCS头信息，从FITS头解析RA/DEC和EXPTIME
        if img.metadata.wcs is None or img.metadata.wcs.crval1 is None:
            from astropy.io import fits as afits
            hdul = afits.open(fits_path)
            hdr = hdul[0].header
            hdul.close()
            ra_str = hdr.get('RA', hdr.get('OBJCTRA', None))
            dec_str = hdr.get('DEC', hdr.get('OBJCTDEC', None))
            exptime = float(hdr.get('EXPTIME', hdr.get('EXPOSURE', 1.0)))
            if ra_str is None or dec_str is None:
                return {"target": target, "filter": filt, "status": "NO_RA_DEC", "elapsed": time.time()-t0}
            # 解析 "18 11 14.00" 格式
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            try:
                sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
                cra0 = sc.ra.deg
                cdec0 = sc.dec.deg
            except:
                return {"target": target, "filter": filt, "status": f"PARSE_RA_DEC_FAIL: {ra_str} {dec_str}", "elapsed": time.time()-t0}
        else:
            cra0 = img.metadata.wcs.crval1
            cdec0 = img.metadata.wcs.crval2
            # 从FITS头获取EXPTIME
            from astropy.io import fits as afits
            hdul = afits.open(fits_path)
            exptime = float(hdul[0].header.get('EXPTIME', hdul[0].header.get('EXPOSURE', 1.0)))
            hdul.close()

        detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        det = detector.detect_ex(img.data)
        img_x = np.array(det.x, np.float64)
        img_y = np.array(det.y, np.float64)
        img_flux = np.array(det.flux, np.float64)
        img_sat = np.array(det.saturated, np.int32)
        nsat = int(img_sat.sum())

        gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
        vm = VectorMatchV35Cpp(gaia_dir)
        wcs_json = os.path.join(PROJECT_ROOT, f"vm35_test_{target}_{filt}.json")
        result = vm.solve(img_x, img_y, img_flux, img_sat, cra0, cdec0, fl, ps, w, h, wcs_out=wcs_json, skip_sip=True, exptime=exptime)
        vm.close()

        elapsed = time.time() - t0

        # solve可能返回None（所有模式都失败）
        if result is None:
            return {
                "target": target, "filter": filt,
                "status": "ALL_MODES_FAIL", "elapsed": elapsed,
            }

        # 验证RMS
        verify_rms = None
        try:
            with open(wcs_json, "r", encoding="utf-8") as f:
                d = json.load(f)
            CD = np.array(d["CD"]).reshape(2, 2)
            CRVAL = np.array(d["CRVAL"])
            CRPIX = np.array(d["CRPIX"])
            A = np.array(d["SIP_A"]).reshape(6, 6)
            B = np.array(d["SIP_B"]).reshape(6, 6)
            sip_order = d["SIP_ORDER"]

            cdet = CD[0,0]*CD[1,1] - CD[0,1]*CD[1,0]
            CD_inv = np.array([[CD[1,1], -CD[0,1]], [-CD[1,0], CD[0,0]]]) / cdet

            gaia = GaiaClientPy(gaia_dir, 1)
            fov = np.sqrt(w*w + h*h) * s0 / 3600
            _, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
                gaia, CRVAL[0], CRVAL[1], max(0.8, fov*1.2), 1000)
            gaia.close()

            ra_src = np.array(cat_ra, np.float64)[:500]
            dec_src = np.array(cat_dec, np.float64)[:500]
            dra = ra_src - CRVAL[0]
            ddec = dec_src - CRVAL[1]
            xi_p = CD_inv[0,0]*dra + CD_inv[0,1]*ddec
            eta_p = CD_inv[1,0]*dra + CD_inv[1,1]*ddec

            # SIP
            xi = xi_p.copy(); eta = eta_p.copy()
            for it in range(30):
                sdx = np.zeros_like(xi); sdy = np.zeros_like(eta)
                for p in range(6):
                    for q in range(6):
                        if p+q<2 or p+q>sip_order: continue
                        xc = np.clip(xi,-1e4,1e4); yc = np.clip(eta,-1e4,1e4)
                        t = xc**p * yc**q
                        sdx += A[p,q]*t; sdy += B[p,q]*t
                xn = xi_p - sdx; yn = eta_p - sdy
                if np.max(np.abs(xn-xi))<1e-6 and np.max(np.abs(yn-eta))<1e-6: break
                xi = xn; eta = yn

            x_g = xi + CRPIX[0]; y_g = eta + CRPIX[1]
            in_f = (x_g>0)&(x_g<w)&(y_g>0)&(y_g<h)
            from scipy.spatial import cKDTree
            tree = cKDTree(np.column_stack([img_x, img_y]))
            dists, _ = tree.query(np.column_stack([x_g[in_f], y_g[in_f]]), k=1)
            matched = dists < 5.0
            if matched.sum() > 0:
                verify_rms = np.sqrt(np.mean(dists[matched]**2))
        except Exception as e:
            verify_rms = None

        # skip_sip模式: 直接用result字段
        n_inliers = result.matched_count if result.matched_count > 0 else result.n_phased_clean
        sip_rms = result.sip_rms_px if result.sip_rms_px > 0 else 0.0
        sip_order = result.sip_order if result.sip_order > 0 else 0

        return {
            "target": target, "filter": filt,
            "status": "OK",
            "s": result.solve_s,
            "theta": result.rotation_deg,
            "flip_mode": result.flip_mode,
            "n_inliers": n_inliers,
            "sip_rms": sip_rms,
            "sip_order": sip_order,
            "nsat": nsat,
            "n_stars": len(img_x),
            "w": w, "h": h,
            "s0": s0,
            "elapsed": elapsed,
            "verify_rms": verify_rms,
        }
    except Exception as e:
        return {
            "target": target, "filter": filt,
            "status": f"FAIL: {str(e)[:80]}",
            "elapsed": time.time() - t0,
        }

# 运行全量测试
results = []
print(f"{'目标':<8} {'滤镜':<8} {'状态':<6} {'s':>8} {'θ':>8} {'flip':>4} {'n':>5} {'SIP阶':>5} {'SIP_RMS':>8} {'验证RMS':>8} {'nsat':>5} {'耗时':>6}")
print("-" * 100)

for target, filt, rel_path in TEST_FRAMES:
    r = run_test(target, filt, rel_path)
    results.append(r)
    if r["status"] == "OK":
        vr = f"{r['verify_rms']:>8.3f}" if r.get('verify_rms') else f"{'N/A':>8}"
        print(f"{r['target']:<8} {r['filter']:<8} {'OK':<6} {r['s']:>8.4f} {r['theta']:>8.3f} {r['flip_mode']:>4} "
              f"{r['n_inliers']:>5} {r['sip_order']:>5} {r['sip_rms']:>8.3f} "
              f"{vr} "
              f"{r['nsat']:>5} {r['elapsed']:>5.1f}s")
    else:
        print(f"{r['target']:<8} {r['filter']:<8} {r['status']:<20} {r.get('elapsed',0):>5.1f}s")

# 汇总统计
print("\n" + "=" * 100)
print("汇总统计")
print("=" * 100)

ok_results = [r for r in results if r["status"] == "OK"]
fail_results = [r for r in results if r["status"] != "OK"]

print(f"总帧数: {len(results)}, 成功: {len(ok_results)}, 失败: {len(fail_results)}")
if ok_results:
    s_vals = [r["s"] for r in ok_results]
    rms_vals = [r["sip_rms"] for r in ok_results]
    verify_vals = [r["verify_rms"] for r in ok_results if r.get("verify_rms")]
    print(f"s范围: [{min(s_vals):.4f}, {max(s_vals):.4f}]")
    print(f"SIP RMS范围: [{min(rms_vals):.3f}, {max(rms_vals):.3f}]px, 均值={np.mean(rms_vals):.3f}px")
    if verify_vals:
        print(f"验证RMS范围: [{min(verify_vals):.3f}, {max(verify_vals):.3f}]px, 均值={np.mean(verify_vals):.3f}px")

# 按滤镜统计
print(f"\n按滤镜统计:")
for filt in ["Red", "Green", "Blue", "Lum", "H-alpha", "Oiii"]:
    fr = [r for r in ok_results if r["filter"] == filt]
    if fr:
        success_rate = len(fr) / len([r for r in results if r["filter"] == filt]) * 100
        rms_mean = np.mean([r["sip_rms"] for r in fr])
        verify_mean = np.mean([r["verify_rms"] for r in fr if r.get("verify_rms")]) if any(r.get("verify_rms") for r in fr) else float('nan')
        print(f"  {filt:<8}: {len(fr)}/{len([r for r in results if r['filter']==filt])} 成功({success_rate:.0f}%), "
              f"SIP RMS={rms_mean:.3f}px, 验证RMS={verify_mean:.3f}px")

# 保存结果
out_path = os.path.join(PROJECT_ROOT, "vm35_full_test_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)
print(f"\n结果已保存: {out_path}")
