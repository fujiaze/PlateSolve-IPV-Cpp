"""
诊断U和W的共同星比例
检查图像侧最亮的100颗星和Gaia侧最亮的150颗星有多少是同一颗星
"""
import sys, os, math, numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v2 import (
    GaiaClientPy, _DEGTORAD,
    _build_image_vectors, _build_catalog_vectors, _apply_flip,
    gnomonic_forward,
)
from star_detector import StarDetector, SDetParamsPy
from scipy.spatial import cKDTree
from astropy.io import fits as afits
from astropy.coordinates import SkyCoord
import astropy.units as u

def diagnose_overlap(rel_path, target, filt):
    fits_path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(fits_path):
        print(f"  文件不存在: {rel_path}")
        return

    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl

    hdul = afits.open(fits_path)
    hdr = hdul[0].header
    hdul.close()
    ra_str = hdr.get('RA', hdr.get('OBJCTRA', None))
    dec_str = hdr.get('DEC', hdr.get('OBJCTDEC', None))
    exptime = float(hdr.get('EXPTIME', hdr.get('EXPOSURE', 1.0)))
    sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    cra0, cdec0 = sc.ra.deg, sc.dec.deg

    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    img_x = np.array(det.x, np.float64)
    img_y = np.array(det.y, np.float64)
    img_flux = np.array(det.flux, np.float64)
    img_sat = np.array(det.saturated, np.int32)
    nsat = int(img_sat.sum())

    U, N, nsat_v, _ = _build_image_vectors(img_x, img_y, img_flux, img_sat, s0, w, h)

    # 查询Gaia - 用mag=22获取所有星
    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    fov_diag = math.sqrt(w*w+h*h)*s0/3600.0
    query_radius = fov_diag * 0.5
    ra_all, dec_all, mag_all = gaia.cone_search(cra0, cdec0, query_radius, 22.0)
    gaia.close()

    # gnomonic投影所有Gaia星
    xi_all, eta_all, valid_all = gnomonic_forward(ra_all, dec_all, cra0, cdec0)
    fov_half_w = w/2 * s0
    fov_half_h = h/2 * s0
    in_fov = valid_all & (np.abs(xi_all) < fov_half_w) & (np.abs(eta_all) < fov_half_h)
    fov_idx = np.where(in_fov)[0]

    print(f"\n{'='*80}")
    print(f"诊断: {target} {filt}")
    print(f"  焦距={fl}mm s0={s0:.3f}\"/px N={N} nsat={nsat}")
    print(f"  FOV内Gaia星(22等): {len(fov_idx)}")

    # 构建全部FOV内Gaia星的向量
    W_all = _build_catalog_vectors(ra_all[fov_idx], dec_all[fov_idx], cra0, cdec0)

    # 对每个flip mode，检查U中的星在全部Gaia星中的匹配情况
    for mode in [1, 2]:  # 只看最常见的两个mode
        Wf = _apply_flip(W_all, mode)

        # 用全部Gaia星建KDTree
        tree_all = cKDTree(Wf)

        # 检查U中每颗星在全部Gaia星中的最近邻距离
        dists_all, idxs_all = tree_all.query(U, k=1)

        # 不同阈值下的匹配数
        for thresh_px in [1, 2, 3, 5, 10]:
            thresh_asec = thresh_px * s0
            n_match = np.sum(dists_all < thresh_asec)
            print(f"  mode{mode} 全部Gaia星中 距离<{thresh_px}px({thresh_asec:.1f}\"): {n_match}/{N} ({n_match/N*100:.1f}%)")

        # 检查U中星在Gaia中的星等分布
        if mode == 1:
            matched_1px = dists_all < 1.0 * s0
            if matched_1px.sum() > 0:
                matched_gaia_mag = mag_all[fov_idx][idxs_all[matched_1px]]
                print(f"  mode{mode} 1px匹配的Gaia星星等: min={matched_gaia_mag.min():.1f} P50={np.median(matched_gaia_mag):.1f} max={matched_gaia_mag.max():.1f}")

    # 现在检查：用最亮150颗Gaia星时，U中能匹配多少？
    Ngaia = 150
    fov_mag = mag_all[fov_idx]
    sorted_order = np.argsort(fov_mag)
    sel_order = sorted_order[:Ngaia]
    sel_idx = fov_idx[sel_order]
    W_150 = _build_catalog_vectors(ra_all[sel_idx], dec_all[sel_idx], cra0, cdec0)

    for mode in [1, 2]:
        Wf_150 = _apply_flip(W_150, mode)
        tree_150 = cKDTree(Wf_150)
        dists_150, _ = tree_150.query(U, k=1)
        for thresh_px in [1, 2, 3, 5]:
            thresh_asec = thresh_px * s0
            n_match = np.sum(dists_150 < thresh_asec)
            print(f"  mode{mode} 最亮{Ngaia}颗Gaia 距离<{thresh_px}px({thresh_asec:.1f}\"): {n_match}/{N} ({n_match/N*100:.1f}%)")

# 测试帧
test_frames = [
    ("GC_P2", "Oiii", "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts"),
    ("GC_P1", "Red",  "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts"),
    ("M20_T2", "Red", "testdata/lights/M20_T2_flying_dutchman-20250701@073331-300S-Red.fts"),
]

for target, filt, rel_path in test_frames:
    diagnose_overlap(rel_path, target, filt)
