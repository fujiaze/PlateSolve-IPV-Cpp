# 全局强制UTF-8编码
import sys, os, json, time
if sys.platform == 'win32':
    import locale; locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

"""V3.5 快速验证 - 只跑之前失败的帧"""
import numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy

# 之前失败的帧
TEST_FRAMES = [
    ("M20_T2", "Blue",   "testdata/lights/M20_T2_flying_dutchman-20250719@005635-300S-Blue.fts"),
    ("GC_P1", "Green",   "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@012151-180S-Green.fts"),
    ("GC_P1", "Blue",    "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@062557-180S-Blue.fts"),
    ("GC_P1", "Oiii",    "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts"),
    ("GC_P2", "Red",     "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@032620-180S-Red.fts"),
    ("GC_P2", "Blue",    "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@005249-180S-Blue.fts"),
    ("GC_P3", "Blue",    "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@021257-180S-Blue.fts"),
    ("GC_P3", "H-alpha", "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@022237-300S-H-alpha.fts"),
    ("GC_P3", "Oiii",    "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@025838-600S-Oiii.fts"),
]

results = []
print(f"{'目标':<8} {'滤镜':<8} {'状态':<8} {'s':>8} {'θ':>8} {'flip':>4} {'n':>5} {'SIP阶':>5} {'SIP_RMS':>8} {'验证RMS':>8}")
print("-" * 80)

for target, filt, rel_path in TEST_FRAMES:
    fits_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(fits_path):
        print(f"{target:<8} {filt:<8} FILE_NOT_FOUND")
        continue

    t0 = time.time()
    try:
        reader = ImageReader()
        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = img.metadata.observation.focallen
        ps = img.metadata.observation.xpixsz
        s0 = 206.265 * ps / fl

        if img.metadata.wcs is None or img.metadata.wcs.crval1 is None:
            from astropy.io import fits as afits
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            hdul = afits.open(fits_path)
            hdr = hdul[0].header
            hdul.close()
            ra_str = hdr.get('RA', hdr.get('OBJCTRA', None))
            dec_str = hdr.get('DEC', hdr.get('OBJCTDEC', None))
            sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
            cra0, cdec0 = sc.ra.deg, sc.dec.deg
        else:
            cra0 = img.metadata.wcs.crval1
            cdec0 = img.metadata.wcs.crval2

        detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        det = detector.detect_ex(img.data)
        img_x = np.array(det.x, np.float64)
        img_y = np.array(det.y, np.float64)
        img_flux = np.array(det.flux, np.float64)
        img_sat = np.array(det.saturated, np.int32)

        gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
        vm = VectorMatchV35Cpp(gaia_dir)
        wcs_json = os.path.join(PROJECT_ROOT, f"vm35_test_{target}_{filt}.json")
        result = vm.solve(img_x, img_y, img_flux, img_sat, cra0, cdec0, fl, ps, w, h, wcs_out=wcs_json)
        vm.close()
        elapsed = time.time() - t0

        if result is None:
            print(f"{target:<8} {filt:<8} {'ALL_FAIL':<8} {'':>8} {'':>8} {'':>4} {'':>5} {'':>5} {'':>8} {'':>8} {elapsed:.1f}s")
            results.append({"target": target, "filter": filt, "status": "ALL_FAIL"})
            continue

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
            fov = np.sqrt(w*w+h*h)*s0/3600
            _, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(gaia, CRVAL[0], CRVAL[1], max(0.8, fov*1.2), 1000)
            gaia.close()
            ra_s = np.array(cat_ra, np.float64)[:500]
            dec_s = np.array(cat_dec, np.float64)[:500]
            dra = ra_s - CRVAL[0]; ddec = dec_s - CRVAL[1]
            xi_p = CD_inv[0,0]*dra + CD_inv[0,1]*ddec
            eta_p = CD_inv[1,0]*dra + CD_inv[1,1]*ddec
            xi = xi_p.copy(); eta = eta_p.copy()
            for it in range(30):
                sdx = np.zeros_like(xi); sdy = np.zeros_like(eta)
                for p in range(6):
                    for q in range(6):
                        if p+q<2 or p+q>sip_order: continue
                        xc = np.clip(xi,-1e4,1e4); yc = np.clip(eta,-1e4,1e4)
                        t = xc**p * yc**q; sdx += A[p,q]*t; sdy += B[p,q]*t
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
        except:
            pass

        vr = f"{verify_rms:.3f}" if verify_rms else "N/A"
        n = result.n_phased_clean
        sip_rms = result.sip_rms_px
        ok = "OK" if (n >= 10 and sip_rms < 10) else "WEAK"
        print(f"{target:<8} {filt:<8} {ok:<8} {result.solve_s:>8.4f} {result.rotation_deg:>8.3f} {result.flip_mode:>4} "
              f"{n:>5} {result.sip_order:>5} {sip_rms:>8.3f} {vr:>8} {elapsed:.1f}s")
        results.append({"target": target, "filter": filt, "status": ok, "n": n, "sip_rms": sip_rms,
                        "verify_rms": verify_rms, "s": result.solve_s})
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{target:<8} {filt:<8} ERROR: {str(e)[:60]}")
        results.append({"target": target, "filter": filt, "status": f"ERROR: {str(e)[:60]}"})

print(f"\n=== 改进效果 ===")
ok_count = sum(1 for r in results if r["status"] in ("OK", "WEAK"))
weak_count = sum(1 for r in results if r["status"] == "WEAK")
fail_count = sum(1 for r in results if r["status"] not in ("OK", "WEAK"))
print(f"之前失败的9帧: OK={ok_count-weak_count}, WEAK={weak_count}, FAIL={fail_count}")
