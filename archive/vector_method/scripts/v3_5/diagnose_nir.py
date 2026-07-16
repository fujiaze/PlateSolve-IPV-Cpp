"""
诊断短焦距帧1点抽样n_in_range低的原因
检查：正确变换下，图像星和Gaia星的位置匹配情况
"""
import sys, os, math, numpy as np
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v2 import (
    GaiaClientPy, bisection_mag_limit, _DEGTORAD,
    _build_image_vectors, _build_catalog_vectors, _apply_flip, _apply_similarity,
    gnomonic_forward,
)
from star_detector import StarDetector, SDetParamsPy
from scipy.spatial import cKDTree

def diagnose_frame(rel_path, target, filt):
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

    # 获取RA/DEC
    from astropy.io import fits as afits
    hdul = afits.open(fits_path)
    hdr = hdul[0].header
    hdul.close()
    ra_str = hdr.get('RA', hdr.get('OBJCTRA', None))
    dec_str = hdr.get('DEC', hdr.get('OBJCTDEC', None))
    exptime = float(hdr.get('EXPTIME', hdr.get('EXPOSURE', 1.0)))
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg))
    cra0, cdec0 = sc.ra.deg, sc.dec.deg

    # 星点检测
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    img_x = np.array(det.x, np.float64)
    img_y = np.array(det.y, np.float64)
    img_flux = np.array(det.flux, np.float64)
    img_sat = np.array(det.saturated, np.int32)
    nsat = int(img_sat.sum())

    # 构建图像向量
    U, N, nsat_v, _ = _build_image_vectors(img_x, img_y, img_flux, img_sat, s0, w, h)

    # 构建星表向量（用极限星等公式）
    Ngaia = math.ceil(1.5 * nsat) if nsat >= 50 else 150
    m_cut = 6.0 + 1.5 * math.log10(max(fl, 1.0)) + 2.0 * math.log10(max(exptime, 0.1))
    fov_diag = math.sqrt(w*w+h*h)*s0/3600.0
    query_radius_deg = fov_diag * 0.5

    gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
    # 查询足够多的星
    mag_query = m_cut
    for attempt in range(10):
        ra_t, dec_t, mag_t = gaia.cone_search(cra0, cdec0, query_radius_deg, mag_query)
        if len(ra_t) >= 500000:
            break
        if len(ra_t) >= 5000:
            break
        mag_query += 0.5

    # gnomonic投影
    xi_all, eta_all, valid_all = gnomonic_forward(ra_t, dec_t, cra0, cdec0)
    fov_half_w = w/2 * s0
    fov_half_h = h/2 * s0
    in_fov = valid_all & (np.abs(xi_all) < fov_half_w) & (np.abs(eta_all) < fov_half_h)
    fov_idx = np.where(in_fov)[0]

    # 取最亮的Ngaia颗
    fov_mag = mag_t[fov_idx]
    sorted_order = np.argsort(fov_mag)
    sel_order = sorted_order[:min(Ngaia, len(sorted_order))]
    sel_idx = fov_idx[sel_order]

    cat_ra = ra_t[sel_idx]
    cat_dec = dec_t[sel_idx]
    cat_mag = mag_t[sel_idx]
    M = len(cat_ra)

    W = _build_catalog_vectors(cat_ra, cat_dec, cra0, cdec0)

    print(f"\n{'='*80}")
    print(f"诊断: {target} {filt}")
    print(f"  焦距={fl}mm s0={s0:.3f}\"/px FOV={w*s0/3600:.1f}°×{h*s0/3600:.1f}°")
    print(f"  N(图像星)={N} M(Gaia星)={M} nsat={nsat}")
    print(f"  m_cut={m_cut:.1f} mag_query={mag_query:.1f}")
    print(f"  FOV内Gaia星: {len(fov_idx)}")

    # 分析U和W的模长分布
    norm_U = np.sqrt(U[:,0]**2 + U[:,1]**2)
    norm_W = np.sqrt(W[:,0]**2 + W[:,1]**2)
    print(f"\n  U模长: min={norm_U.min():.0f}\" P50={np.median(norm_U):.0f}\" max={norm_U.max():.0f}\"")
    print(f"  W模长: min={norm_W.min():.0f}\" P50={np.median(norm_W):.0f}\" max={norm_W.max():.0f}\"")
    print(f"  FOV半径: {fov_diag*3600/2:.0f}\"")

    # 对每个flip mode，用正确变换(θ≈0或180°, s≈1)检查匹配
    for mode in range(4):
        Wf = _apply_flip(W, mode)
        norm_Wf = np.sqrt(Wf[:,0]**2 + Wf[:,1]**2)

        # 用所有N颗图像星和M颗Gaia星做1点抽样
        # 模拟正确变换: s=1.0, θ=0 (mode 0/1) 或 θ=π (mode 2/3)
        # 实际上1点抽样中θ由angle(U_i)-angle(W_j)决定

        # 更好的方法：直接用identity变换（s=1, θ=0, tx=0, ty=0）看Wf和U的重叠
        Wt_identity = Wf.copy()  # s=1, θ=0, tx=0, ty=0

        tree_Wt = cKDTree(Wt_identity)
        dists, idxs = tree_Wt.query(U, k=1)

        # 不同距离阈值下的匹配数
        for thresh in [1.0*s0, 2.0*s0, 3.0*s0, 5.0*s0, 10.0*s0, 20.0*s0]:
            n_match = np.sum(dists < thresh)
            print(f"  mode{mode} identity变换 距离<{thresh:.1f}\": {n_match}/{N} ({n_match/N*100:.1f}%)")

        # 模拟1点抽样: 随机选一对(i,j)，计算s和θ
        # 然后看变换后的匹配数
        best_n = 0
        best_info = None
        n_trials = 200
        for trial in range(n_trials):
            i = np.random.randint(N)
            j = np.random.randint(M)
            s = norm_U[i] / norm_Wf[j]
            if s < 0.9 or s > 1.1:
                continue
            theta = math.atan2(U[i,1], U[i,0]) - math.atan2(Wf[j,1], Wf[j,0])
            ct, st = math.cos(theta), math.sin(theta)
            tx = U[i,0] - s*(ct*Wf[j,0] - st*Wf[j,1])
            ty = U[i,1] - s*(st*Wf[j,0] + ct*Wf[j,1])
            max_t = fov_diag*3600*0.6
            if abs(tx) > max_t or abs(ty) > max_t:
                continue
            Wt = _apply_similarity(Wf, s, theta, tx, ty)
            tree_Wt2 = cKDTree(Wt)
            d2, idx2 = tree_Wt2.query(U, k=1)
            n_5s0 = np.sum(d2 < 5.0*s0)
            if n_5s0 > best_n:
                best_n = n_5s0
                best_info = (i, j, s, math.degrees(theta), tx, ty, n_5s0)

        if best_info:
            i, j, s, theta_deg, tx, ty, n = best_info
            print(f"  mode{mode} 1点抽样最佳: n={n}/{N} s={s:.4f} θ={theta_deg:.2f}° tx={tx:.1f}\" ty={ty:.1f}\"")

            # 对最佳变换做详细分析
            Wt_best = _apply_similarity(Wf, s, math.radians(theta_deg), tx, ty)
            tree_best = cKDTree(Wt_best)
            d_best, idx_best = tree_best.query(U, k=1)

            # 距离分布
            matched = d_best < 5.0*s0
            if matched.sum() > 0:
                d_matched = d_best[matched]
                print(f"    匹配距离: min={d_matched.min():.2f}\" P50={np.median(d_matched):.2f}\" max={d_matched.max():.2f}\"")

            # scale检查
            norm_Wf_arr = np.sqrt(Wf[:,0]**2 + Wf[:,1]**2)
            scale_ok = 0
            for k in range(N):
                if d_best[k] < 5.0*s0:
                    sr = norm_U[k] / norm_Wf_arr[idx_best[k]]
                    if 0.9 <= sr <= 1.1:
                        scale_ok += 1
            print(f"    距离OK: {matched.sum()}/{N}  scaleOK: {scale_ok}/{N}  两者OK: {scale_ok}/{N}")

            # 关键诊断: 变换后Wt和U的中心偏移
            print(f"    U中心: ({U[:,0].mean():.1f}\", {U[:,1].mean():.1f}\")")
            print(f"    Wt中心: ({Wt_best[:,0].mean():.1f}\", {Wt_best[:,1].mean():.1f}\")")
            print(f"    U范围: x=[{U[:,0].min():.0f}\", {U[:,0].max():.0f}\"] y=[{U[:,1].min():.0f}\", {U[:,1].max():.0f}\"]")
            print(f"    Wt范围: x=[{Wt_best[:,0].min():.0f}\", {Wt_best[:,0].max():.0f}\"] y=[{Wt_best[:,1].min():.0f}\", {Wt_best[:,1].max():.0f}\"]")

            # 检查: U和Wt中是否有共同的星？
            # 用2*s0阈值做双向匹配
            tree_u = cKDTree(U)
            d_uw, idx_uw = tree_u.query(Wt_best, k=1)
            d_wu, idx_wu = tree_Wt2.query(U, k=1)
            mutual = 0
            for k in range(N):
                if d_wu[k] < 2.0*s0 and idx_uw[idx_wu[k]] == k:
                    mutual += 1
            print(f"    双向互匹配(2*s0): {mutual}/{N}")

    gaia.close()

# 测试帧
test_frames = [
    ("GC_P2", "H-alpha", "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@030735-300S-H-alpha.fts"),
    ("GC_P2", "Oiii",    "testdata/lights1/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@022723-600S-Oiii.fts"),
    ("GC_P3", "Red",     "testdata/lights1/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@005108-180S-Red.fts"),
    ("GC_P1", "Red",     "testdata/lights1/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@011752-180S-Red.fts"),
]

for target, filt, rel_path in test_frames:
    diagnose_frame(rel_path, target, filt)
