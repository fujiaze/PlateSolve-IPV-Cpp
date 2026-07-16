"""CD投影残差快速诊断"""
import sys,os
import numpy as np
import math,json
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import GaiaClientPy
from star_detector import StarDetector,SDetParamsPy
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")

for fits,lab in [
    ("testdata/lights/NGC7293_T2_HO_flying_dutchman-20250607@085204-1200S-H-alpha.fts","NGC7293"),
    ("testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts","GC_P2"),
]:
    reader = ImageReader()
    img = reader.read(fits)
    w = img.width
    h = img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl

    hdul = afits.open(fits)
    hdr = hdul[0].header
    hdul.close()
    sc = SkyCoord(hdr.get('RA',''), hdr.get('DEC',''), unit=(u.hourangle, u.deg))
    cra = sc.ra.deg
    cdec = sc.dec.deg

    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)

    vm = VectorMatchV35Cpp('GaiaDR3')
    json_path = f'overlay_output/_q_{lab}.json'
    r = vm.solve(
        np.array(det.x), np.array(det.y),
        np.array(det.flux), np.array(det.saturated),
        cra, cdec, fl, ps, w, h,
        wcs_out=json_path
    )
    vm.close()

    if not r:
        print(f'{lab}: FAIL')
        continue

    with open(json_path) as f:
        wc = json.load(f)

    cd = np.array(wc['CD'])
    crv = np.array(wc['CRVAL'])
    crp = np.array(wc['CRPIX'])
    sipA = np.array(wc['SIP_A']).reshape(6,6)
    sipB = np.array(wc['SIP_B']).reshape(6,6)
    so = wc['SIP_ORDER']

    print(f'\n{lab}: s={r.solve_s:.4f} th={r.rotation_deg:.1f} flip={r.flip_mode} s0={s0:.4f}')
    print(f'CD=[{cd[0,0]:.6e},{cd[0,1]:.6e};{cd[1,0]:.6e},{cd[1,1]:.6e}]')
    print(f'CRVAL=[{crv[0]:.10f},{crv[1]:.10f}]')
    print(f'SIP_order={so} RMS_px={wc["RMS_PX"]:.4f}')

    fd = math.sqrt(w*w+h*h) * s0 / 3600.
    gaia = GaiaClientPy('GaiaDR3', 1)
    ra_a, dec_a, _ = gaia.cone_search(cra, cdec, fd*0.55, 22.0)
    gaia.close()
    ra_a = np.array(ra_a)
    dec_a = np.array(dec_a)

    cdet = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
    cdi = np.array([[cd[1,1], -cd[0,1]], [-cd[1,0], cd[0,0]]]) / cdet
    xi_p = cdi[0,0]*(ra_a-crv[0]) + cdi[0,1]*(dec_a-crv[1])
    et_p = cdi[1,0]*(ra_a-crv[0]) + cdi[1,1]*(dec_a-crv[1])
    xB = xi_p + crp[0]
    yB = et_p + crp[1]

    mrg = 500
    ifv = (xB > -mrg) & (xB < w+mrg) & (yB > -mrg) & (yB < h+mrg)
    xi = xi_p[ifv].copy()
    et = et_p[ifv].copy()
    xio = xi_p[ifv].copy()
    eto = et_p[ifv].copy()

    mo = min(so, 6) if so > 0 else 0
    sts = []
    if mo >= 2:
        for p in range(mo+1):
            for q in range(mo+1):
                if p+q < 2 or p+q > mo:
                    continue
                ac = sipA[p,q]
                bc = sipB[p,q]
                if abs(ac) > 1e-30 or abs(bc) > 1e-30:
                    sts.append((p, q, ac, bc))

    for _ in range(20):
        sdx = np.zeros_like(xi)
        sdy = np.zeros_like(et)
        for p, q, ac, bc in sts:
            xc = np.clip(xi, -1e4, 1e4)
            yc = np.clip(et, -1e4, 1e4)
            term = xc**p * yc**q
            term = np.where(np.isfinite(term), term, 0)
            sdx += ac * term
            sdy += bc * term
        xn = xio - sdx
        yn = eto - sdy
        if np.max(np.abs(xn-xi)) < 1e-6 and np.max(np.abs(yn-et)) < 1e-6:
            break
        xi = xn
        et = yn

    xS = np.full(len(ra_a), np.nan)
    yS = np.full(len(ra_a), np.nan)
    xS[ifv] = xi + crp[0]
    yS[ifv] = et + crp[1]

    iS = np.isfinite(xS) & (xS > 0) & (xS < w) & (yS > 0) & (yS < h)
    if iS.sum() < 3:
        print(f'  帧内Gaia星不足')
        continue

    td = cKDTree(np.column_stack([det.x, det.y]))
    gp = np.column_stack([xS[iS], yS[iS]])
    gil = np.where(iS)[0]
    ds, ids = td.query(gp, k=1)
    used = np.zeros(len(det.x), dtype=bool)
    pairs = []
    for kk in np.argsort(ds):
        if ds[kk] > 3:
            break
        ii = ids[kk]
        if used[ii]:
            continue
        used[ii] = True
        pairs.append((ii, gil[kk]))

    npairs = len(pairs)
    if npairs < 3:
        print(f'  匹配对不足({npairs})')
        continue

    rx_vals = []
    ry_vals = []
    for di, gi in pairs:
        rx_vals.append(det.x[di] - xS[gi])
        ry_vals.append(det.y[di] - yS[gi])
    rxd = np.array(rx_vals)
    ryd = np.array(ry_vals)
    rdist = np.sqrt(rxd**2 + ryd**2)

    print(f'CD+SIP 1对1匹配({npairs}对, <3px):')
    print(f'  med=[{np.median(rxd):+.3f},{np.median(ryd):+.3f}]px  '
          f'mean=[{np.mean(rxd):+.3f},{np.mean(ryd):+.3f}]px  '
          f'RMS={np.sqrt(np.mean(rxd**2+ryd**2)):.3f}px')

    bins = [0, 0.5, 1, 2, 3, 5, 10, 50]
    print('  残差分布:', end='')
    for i in range(len(bins)-1):
        c = ((rdist >= bins[i]) & (rdist < bins[i+1])).sum()
        if c > 0:
            print(f' [{bins[i]:.1f}-{bins[i+1]:.1f}):{c}', end='')
    print()
