"""快速生成单帧覆盖图 - 检验Umeyama SVD修复"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy, bisection_mag_limit
from star_detector import StarDetector, SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")

# 测试帧列表: 长焦+短焦各几个
test_frames = [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts", "NGC7293_Ha"),
    ("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts", "GC_P2_Oiii"),
    ("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts", "GC_P1_Red"),
]

reader = ImageReader()
detector = StarDetector(params=SDetParamsPy(fitRadius=0))
gaia_dir = 'GaiaDR3'

for fits_path, label in test_frames:
    print(f"\n=== {label} ===")
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    hdul = afits.open(fits_path)
    hdr = hdul[0].header
    exptime = float(hdr.get('EXPTIME', 1.0))
    ra_str = hdr.get('RA', hdr.get('OBJCTRA', None))
    dec_str = hdr.get('DEC', hdr.get('OBJCTDEC', None))
    sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    cra0, cdec0 = sc.ra.deg, sc.dec.deg
    hdul.close()

    det = detector.detect_ex(img.data)
    vm = VectorMatchV35Cpp(gaia_dir)
    wcs_out = f'overlay_output/_fix_{label}.json'
    result = vm.solve(
        np.array(det.x, np.float64), np.array(det.y, np.float64),
        np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
        cra0, cdec0, fl, ps, w, h,
        wcs_out=wcs_out, skip_sip=False, exptime=exptime)
    vm.close()

    if not result:
        print(f"  FAILED")
        continue

    with open(wcs_out) as f:
        wcs = json.load(f)

    cd = np.array(wcs['CD'], dtype=np.float64)
    crval = np.array(wcs['CRVAL'], dtype=np.float64)
    crpix = np.array(wcs['CRPIX'], dtype=np.float64)
    sip_A = np.array(wcs.get('SIP_A', [0]*36), dtype=np.float64).reshape(6,6)
    sip_B = np.array(wcs.get('SIP_B', [0]*36), dtype=np.float64).reshape(6,6)
    sip_order = wcs.get('SIP_ORDER', 0)

    gaia = GaiaClientPy(gaia_dir, 1)
    s0 = 206.265 * ps / fl
    fov_deg = math.sqrt(w*w+h*h)*s0/3600.0
    _, M, cat_ra, cat_dec, cat_mag = bisection_mag_limit(
        gaia, crval[0], crval[1], max(0.8, fov_deg*1.2), 1200)
    gaia.close()

    mag_a = np.array(cat_mag, np.float64)
    top_idx = np.argsort(mag_a)[:1000]
    ra_src = np.array(cat_ra, np.float64)[top_idx]
    dec_src = np.array(cat_dec, np.float64)[top_idx]

    # WCS-SIP投影
    cdet = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
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
        for p in range(max_order+1):
            for q in range(max_order+1):
                if p+q < 2 or p+q > max_order: continue
                if p>=6 or q>=6: continue
                a_c = sip_A[p,q]; b_c = sip_B[p,q]
                if abs(a_c)>1e-30 or abs(b_c)>1e-30:
                    sip_terms.append((p,q,a_c,b_c))

    for _ in range(20):
        sip_dx = np.zeros_like(xi); sip_dy = np.zeros_like(eta)
        for p,q,a_c,b_c in sip_terms:
            xi_c = np.clip(xi, -1e4, 1e4); eta_c = np.clip(eta, -1e4, 1e4)
            term = xi_c**p * eta_c**q
            term = np.where(np.isfinite(term), term, 0.0)
            sip_dx += a_c*term; sip_dy += b_c*term
        xi_new = xi_ps - sip_dx; eta_new = eta_ps - sip_dy
        if np.max(np.abs(xi_new-xi))<1e-6 and np.max(np.abs(eta_new-eta))<1e-6: break
        xi, eta = xi_new, eta_new

    x_pix = np.full(len(ra_src), np.nan)
    y_pix = np.full(len(ra_src), np.nan)
    x_pix[prelim] = xi + crpix[0]
    y_pix[prelim] = eta + crpix[1]

    bt = np.isfinite(x_pix) & (x_pix>0) & (x_pix<w) & (y_pix>0) & (y_pix<h)
    n_gaia = bt.sum()
    print(f"  s={result.solve_s:.4f} θ={result.rotation_deg:.1f}° n={result.matched_count} "
          f"SIP_RMS={wcs['RMS_PX']:.3f}px gaia_in_frame={n_gaia}")

    # 渲染
    data_f = img.data.astype(np.float32)
    dd = data_f[data_f>0]
    lo, hi = np.percentile(dd, (1,99.5)) if len(dd)>1 else (0,1)
    img_s = np.clip((data_f-lo)/max(hi-lo,1), 0, 1)
    DPI=100
    fig=plt.figure(figsize=(w/DPI,h/DPI), dpi=DPI, frameon=False)
    ax=fig.add_axes([0,0,1,1])
    ax.imshow(img_s, cmap="gray", origin="lower", interpolation="nearest")
    if bt.sum()>0:
        ax.scatter(x_pix[bt], y_pix[bt], marker="+", color="#FF0000", s=60, linewidths=1.5, alpha=0.85)
    ax.set_xlim(0,w); ax.set_ylim(0,h); ax.axis("off")
    out_png = f'overlay_output/_fix_{label}.png'
    fig.savefig(out_png, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print(f"  Saved: {out_png}")
