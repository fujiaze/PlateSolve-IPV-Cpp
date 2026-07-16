"""
标准控制点对可视化调试示例 - V3.5最终版
红色十字=Gaia前1000亮星经WCS+SIP投影  蓝色圈=Phase B匹配星  黄箭头=残差(Gaia→检测)
"""
import sys,os,numpy as np,math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy
from star_detector import StarDetector,SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")
OUT = "overlay_output"

for fits_path,label in [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293"),
    ("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2"),
    ("testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts","GC_P1"),
]:
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl

    hdul = afits.open(fits_path)
    hdr = hdul[0].header
    hdul.close()
    sc = SkyCoord(hdr.get('RA',''), hdr.get('DEC',''), unit=(u.hourangle, u.deg))
    cra, cdec = sc.ra.deg, sc.dec.deg

    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)

    vm = VectorMatchV35Cpp('GaiaDR3')
    js = os.path.join(OUT, f'_cp_{label}.json')
    vm.solve(det.x, det.y, np.array(det.flux), np.array(det.saturated),
             cra, cdec, fl, ps, w, h, wcs_out=js)
    vm.close()

    with open(js) as f:
        wc = json.load(f)
    cd = np.array(wc['CD'])
    crv = np.array(wc['CRVAL'])
    crp = np.array(wc['CRPIX'])
    sipA = np.array(wc['SIP_A']).reshape(6, 6)
    sipB = np.array(wc['SIP_B']).reshape(6, 6)
    so = wc['SIP_ORDER']

    cdet = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
    cdi = np.array([[cd[1,1], -cd[0,1]], [-cd[1,0], cd[0,0]]]) / cdet

    fov = math.sqrt(w*w + h*h) * s0 / 3600.0

    def sky_to_pixel(ra_arr, dec_arr):
        """WCS标准逆投影: 天球 → 中间坐标 → SIP迭代 → 像素"""
        dra = np.asarray(ra_arr) - crv[0]
        ddec = np.asarray(dec_arr) - crv[1]
        xi = cdi[0,0]*dra + cdi[0,1]*ddec
        eta = cdi[1,0]*dra + cdi[1,1]*ddec
        if so >= 2:
            xio, eto = xi.copy(), eta.copy()
            for _ in range(15):
                sdx = np.zeros_like(xi)
                sdy = np.zeros_like(eta)
                for p in range(so+1):
                    for q in range(so+1):
                        if p+q < 2 or p+q > so:
                            continue
                        if abs(sipA[p,q]) < 1e-30 and abs(sipB[p,q]) < 1e-30:
                            continue
                        xc = np.clip(xi, -5e3, 5e3)
                        yc = np.clip(eta, -5e3, 5e3)
                        term = xc**p * yc**q
                        term = np.where(np.isfinite(term), term, 0.0)
                        sdx += sipA[p,q] * term
                        sdy += sipB[p,q] * term
                xn = xio - sdx
                yn = eto - sdy
                if np.max(np.abs(xn-xi)) < 1e-6 and np.max(np.abs(yn-eta)) < 1e-6:
                    xi, eta = xn, yn
                    break
                xi, eta = xn, yn
        px = xi + crp[0]
        py = eta + crp[1]
        return px, py

    gaia = GaiaClientPy('GaiaDR3', 1)
    qr = max(fov * 0.65, 3.5)
    ra_t, dec_t, mag_t = gaia.cone_search(crv[0], crv[1], qr, 22.0)
    gaia.close()
    ra_all = np.array(ra_t)
    dec_all = np.array(dec_t)
    mag_all = np.array(mag_t)
    gx_all, gy_all = sky_to_pixel(ra_all, dec_all)
    in_frame = (gx_all > 0) & (gx_all < w) & (gy_all > 0) & (gy_all < h)
    gx_if = gx_all[in_frame]
    gy_if = gy_all[in_frame]
    mag_if = mag_all[in_frame]

    grid = 32
    gx_b, gy_b = [], []
    idx_all = np.arange(len(gx_if))
    for gi in range(grid):
        x0 = gi * w / grid
        x1 = (gi + 1) * w / grid
        for gj in range(grid):
            y0 = gj * h / grid
            y1 = (gj + 1) * h / grid
            mask = (gx_if > x0) & (gx_if < x1) & (gy_if > y0) & (gy_if < y1)
            if not np.any(mask):
                continue
            indices = idx_all[mask]
            best = indices[np.argmin(mag_if[indices])]
            gx_b.append(float(gx_if[best]))
            gy_b.append(float(gy_if[best]))
            if len(gx_b) >= 1000:
                break
        if len(gx_b) >= 1000:
            break
    gx_b = np.array(gx_b)
    gy_b = np.array(gy_b)
    n_b = len(gx_b)
    if n_b > 0:
        print(f"  [DEBUG] 查询{len(ra_t)}颗 帧内{np.sum(in_frame)}颗 {grid}×{grid}网格 → {n_b}个 "
              f"范围 x=[{gx_b.min():.0f},{gx_b.max():.0f}] y=[{gy_b.min():.0f},{gy_b.max():.0f}]")

    match_pairs = wc.get('MATCH_PAIRS', [])

    data = img.data.astype(np.float32)
    dd = data[data > 0]
    lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) > 1 else (0, 1)
    ims = np.clip((data - lo) / max(hi-lo, 1), 0, 1)

    DPI = 100
    fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(ims, cmap="gray", origin="lower", interpolation="nearest")

    csz = 3
    ax.plot(gx_b, gy_b, '+', color='red', markersize=csz, mew=0.6, alpha=0.6)

    for mp in match_pairs:
        dx_px, dy_px, gaia_ra, gaia_dec = mp
        gx, gy = sky_to_pixel([gaia_ra], [gaia_dec])
        gx, gy = float(gx[0]), float(gy[0])

        ax.arrow(gx, gy, dx_px-gx, dy_px-gy,
                 head_width=6, head_length=5, fc='yellow', ec='yellow',
                 alpha=0.8, lw=0.6, length_includes_head=True)
        circ = plt.Circle((dx_px, dy_px), 2.5, fc='none', ec='cyan', lw=0.9, alpha=0.95)
        ax.add_patch(circ)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    out_png = os.path.join(OUT, f"_cp_{label}.png")
    fig.savefig(out_png, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print(f"{label}: 红色十字{n_b}个 + PhaseB匹配{len(match_pairs)}对 → {out_png}")
