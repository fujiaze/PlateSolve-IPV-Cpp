"""
V4.0 标准控制点对可视化调试 — 复用V3.5可视化脚本结构 (数据来自V4 WCS JSON)
红色十字=Gaia前1000亮星经WCS+SIP投影  蓝色圈=确认匹配对  黄箭头=残差(Gaia→检测)
"""
import sys,os,numpy as np,math,json
from scipy.spatial import cKDTree
sys.path.insert(0,'lib/plate_solve/python')
sys.path.insert(0,'lib/plate_solve/scripts/v4_0')
sys.path.insert(0,'lib/astro_image_io/python')
sys.path.insert(0,'lib/star_detector/python')
from astro_image_io import ImageReader
from vector_match_v2 import GaiaClientPy
from star_detector import StarDetector,SDetParamsPy
from test_v40_prototype import V40PrototypeSolver
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(r"F:\Astro dev\Astro CS Normalization Database")
OUT = "overlay_output"

for fits_path,label in [
    ("testdata/lights/M20_T2_flying_dutchman-20250701@073331-300S-Red.fts","V4_M20T2"),
]:
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    cra0 = img.metadata.wcs.crval1
    cdec0 = img.metadata.wcs.crval2
    s0 = 206.265 * ps / fl

    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)

    solver = V40PrototypeSolver('GaiaDR3', db_type=1)
    result = solver.solve(
        np.array(det.x, np.float64), np.array(det.y, np.float64),
        np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
        cra0, cdec0, fl, ps, w, h, exptime=300.0,
        K_max=3000, tau_scm=0.40, tau_nmin=3, verbose=True,
    )
    solver.close()

    if not result['success']:
        print(f"{label}: 求解失败")
        continue

    wcs = result['wcs_data']
    if wcs is None:
        print(f"{label}: 无WCS数据")
        continue

    js_path = os.path.join(OUT, f"_cp_{label}.json")
    with open(js_path, 'w', encoding='utf-8') as f:
        json.dump(wcs, f, indent=2)

    cd = np.array(wcs['CD'])
    crv = np.array(wcs['CRVAL'])
    crp = np.array(wcs['CRPIX'])
    sipA = np.array(wcs['SIP_A']).reshape(6, 6)
    sipB = np.array(wcs['SIP_B']).reshape(6, 6)
    so = wcs['SIP_ORDER']

    cdet = cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]
    cdi = np.array([[cd[1,1], -cd[0,1]], [-cd[1,0], cd[0,0]]]) / cdet

    fov = math.sqrt(w*w + h*h) * s0 / 3600.0

    cos_d = math.cos(crv[1] * math.pi / 180.0)

    def sky_to_pixel(ra_arr, dec_arr):
        """WCS标准逆投影: 天球 → 中间坐标 → SIP迭代 → 像素"""
        dra_cosd = (np.asarray(ra_arr) - crv[0]) * cos_d
        ddec = np.asarray(dec_arr) - crv[1]
        xi = cdi[0,0]*dra_cosd + cdi[0,1]*ddec
        eta = cdi[1,0]*dra_cosd + cdi[1,1]*ddec
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
    idx_all2 = np.arange(len(gx_if))
    for gi in range(grid):
        x0 = gi * w / grid
        x1 = (gi+1) * w / grid
        for gj in range(grid):
            y0 = gj * h / grid
            y1 = (gj+1) * h / grid
            mask = (gx_if > x0) & (gx_if < x1) & (gy_if > y0) & (gy_if < y1)
            if not np.any(mask):
                continue
            indices = idx_all2[mask]
            best = indices[np.argmin(mag_if[indices])]
            gx_b.append(float(gx_if[best]))
            gy_b.append(float(gy_if[best]))
            if len(gx_b) >= 1000:
                break
        if len(gx_b) >= 1000:
            break
    gx_b = np.array(gx_b)
    gy_b = np.array(gy_b)
    print(f"  [{label}] Gaia投影十字分布: x=[{gx_b.min():.0f},{gx_b.max():.0f}] "
          f"y=[{gy_b.min():.0f},{gy_b.max():.0f}] 共{len(gx_b)}个")

    # 诊断: CD逆投影质量 — 前1000亮Gaia星投影后附近有无检测星
    det_x_arr = np.array(det.x, np.float64)
    det_y_arr = np.array(det.y, np.float64)
    det_tree = cKDTree(np.column_stack([det_x_arr, det_y_arr]))
    # 只用在V4投影中在图像内的Gaia星
    in_frame_mask = (gx_if >= 0) & (gx_if < w) & (gy_if >= 0) & (gy_if < h)
    in_frame_idx = np.where(in_frame_mask)[0]
    bright_in_frame = in_frame_idx[mag_if[in_frame_idx].argsort()[:1000]]
    test_gx = gx_if[bright_in_frame]
    test_gy = gy_if[bright_in_frame]
    dists_to_det, _ = det_tree.query(np.column_stack([test_gx, test_gy]))
    n_within_3px = np.sum(dists_to_det < 3)
    n_within_5px = np.sum(dists_to_det < 5)
    n_within_10px = np.sum(dists_to_det < 10)
    n_test = len(bright_in_frame)
    print(f"  [{label}] CD逆投影质量 (前{n_test}亮帧内Gaia星): "
          f"3px内={n_within_3px} 5px内={n_within_5px} 10px内={n_within_10px}/{n_test}")

    # 对比: 用原始WCS做逆投影
    try:
        orig_cd = np.array([[img.metadata.wcs.cd1_1, img.metadata.wcs.cd1_2],
                             [img.metadata.wcs.cd2_1, img.metadata.wcs.cd2_2]])
        orig_crv = [img.metadata.wcs.crval1, img.metadata.wcs.crval2]
        orig_crp = [img.metadata.wcs.crpix1, img.metadata.wcs.crpix2]
        print(f"  [{label}] 原始WCS: CD={orig_cd.tolist()} CRVAL={orig_crv} CRPIX={orig_crp}")
        if abs(orig_cd[0,0]) > 1e-10:
            orig_cdet = orig_cd[0,0]*orig_cd[1,1] - orig_cd[0,1]*orig_cd[1,0]
            orig_cdi = np.array([[orig_cd[1,1], -orig_cd[0,1]], [-orig_cd[1,0], orig_cd[0,0]]]) / orig_cdet
            orig_cos_d = math.cos(orig_crv[1] * math.pi / 180.0)
            orig_gx_list, orig_gy_list = [], []
            # 用全部Gaia星投影, 筛选在图像内的
            for k in range(len(ra_all)):
                dra_cosd = (ra_all[k] - orig_crv[0]) * orig_cos_d
                ddec = dec_all[k] - orig_crv[1]
                ox = orig_crp[0] + orig_cdi[0,0]*dra_cosd + orig_cdi[0,1]*ddec
                oy = orig_crp[1] + orig_cdi[1,0]*dra_cosd + orig_cdi[1,1]*ddec
                if 0 <= ox < w and 0 <= oy < h:
                    orig_gx_list.append(ox)
                    orig_gy_list.append(oy)
            if len(orig_gx_list) > 10:
                orig_dists, _ = det_tree.query(np.column_stack([orig_gx_list, orig_gy_list]))
                print(f"  [{label}] 原始WCS逆投影: 3px内={np.sum(orig_dists<3)} "
                      f"5px内={np.sum(orig_dists<5)} 10px内={np.sum(orig_dists<10)}/{len(orig_gx_list)}")
                # CD差异分析
                cd_diff = cd - orig_cd
                print(f"  [{label}] CD差异: {cd_diff.tolist()}")
                print(f"  [{label}] CD差异(对角): cd00={cd_diff[0,0]:.2e} cd11={cd_diff[1,1]:.2e}")
                print(f"  [{label}] CD差异(非对角): cd01={cd_diff[0,1]:.2e} cd10={cd_diff[1,0]:.2e}")
                # 角度差异
                ang_v4 = math.degrees(math.atan2(cd[1,0], cd[0,0]))
                ang_orig = math.degrees(math.atan2(orig_cd[1,0], orig_cd[0,0]))
                print(f"  [{label}] 角度: V4={ang_v4:.4f}° 原始={ang_orig:.4f}° 差={ang_v4-ang_orig:.4f}°")
                # CRVAL差异
                dra_crv = (crv[0] - orig_crv[0]) * 3600 * math.cos(orig_crv[1] * math.pi / 180)
                ddec_crv = (crv[1] - orig_crv[1]) * 3600
                print(f"  [{label}] CRVAL差异: ΔRA·cosδ={dra_crv:.2f}\" ΔDec={ddec_crv:.2f}\" "
                      f"({dra_crv/s0:.1f}px, {ddec_crv/s0:.1f}px)")
                print(f"  [{label}] CRVAL: V4={crv} 原始={orig_crv}")
    except Exception as e:
        print(f"  [{label}] 原始WCS对比失败: {e}")

    match_pairs = wcs.get('MATCH_PAIRS', [])
    # 诊断: 匹配对空间分布
    if len(match_pairs) > 0:
        mp_arr = np.array(match_pairs)
        print(f"  [{label}] 匹配对分布: x=[{mp_arr[:,0].min():.0f},{mp_arr[:,0].max():.0f}] "
              f"y=[{mp_arr[:,1].min():.0f},{mp_arr[:,1].max():.0f}] "
              f"图像尺寸={w}x{h}")
        # 网格密度统计
        gn = 8
        for gi in range(gn):
            row_parts = []
            for gj in range(gn):
                x0, x1 = gi * w / gn, (gi+1) * w / gn
                y0, y1 = gj * h / gn, (gj+1) * h / gn
                cnt = np.sum((mp_arr[:,0] >= x0) & (mp_arr[:,0] < x1) &
                             (mp_arr[:,1] >= y0) & (mp_arr[:,1] < y1))
                row_parts.append(f"{cnt:3d}")
            print(f"    行{gi}: {' '.join(row_parts)}")
        # 残差空间分布: 检查CD投影是否有系统性偏差
        res_x_list, res_y_list = [], []
        for mp in match_pairs:
            dx_px, dy_px, g_ra, g_dec = mp
            dra_cosd = (g_ra - crv[0]) * cos_d
            ddec = g_dec - crv[1]
            px_cd = crp[0] + cdi[0,0] * dra_cosd + cdi[0,1] * ddec
            py_cd = crp[1] + cdi[1,0] * dra_cosd + cdi[1,1] * ddec
            res_x_list.append(dx_px - px_cd)
            res_y_list.append(dy_px - py_cd)
        res_x_arr = np.array(res_x_list)
        res_y_arr = np.array(res_y_list)
        print(f"  [{label}] 残差空间分布: dx_med={np.median(res_x_arr):.2f} dx_std={np.std(res_x_arr):.2f} "
              f"dy_med={np.median(res_y_arr):.2f} dy_std={np.std(res_y_arr):.2f} px")
        # 按象限统计
    cx_q, cy_q = w/2, h/2
    for qi, (xlo, xhi, ylo, yhi) in enumerate([
        (0, cx_q, 0, cy_q), (cx_q, w, 0, cy_q), (0, cx_q, cy_q, h), (cx_q, w, cy_q, h)]):
            mask = (mp_arr[:,0] >= xlo) & (mp_arr[:,0] < xhi) & (mp_arr[:,1] >= ylo) & (mp_arr[:,1] < yhi)
            if np.any(mask):
                print(f"    象限{qi}: n={np.sum(mask)} dx={np.median(res_x_arr[mask]):.2f} dy={np.median(res_y_arr[mask]):.2f}")
    print(f"  [{label}] CD={cd.tolist()} CRVAL={crv.tolist()} MATCH_PAIRS={len(match_pairs)}对")

    # ── 向量诊断 ──
    print(f"\n  ╔══ 向量诊断 [{label}] ══╗")
    print(f"  ║ flip_mode = {result['best_mode']}   (0=原 1=flipX 2=flipY 3=flipXY)")
    print(f"  ║ s = {result['s_final']:.6f}   θ = {math.degrees(result['theta_final']):.4f}°")
    print(f"  ║ tx = {result['tx']:.2f}\"  ty = {result['ty']:.2f}\"")
    print(f"  ║ CD det = {cd[0,0]*cd[1,1] - cd[0,1]*cd[1,0]:.10f}")
    print(f"  ║ Image center: {w/2:.0f}, {h/2:.0f}")
    cx_i, cy_i = w/2, h/2
    if len(match_pairs) > 0:
        print(f"  ║ 前10对匹配向量:")
        for i in range(min(10, len(match_pairs))):
            dx_px, dy_px, g_ra, g_dec = match_pairs[i]
            # 用CD投影: sky→pixel
            dra = (g_ra - crv[0]) * math.cos(crv[1] * math.pi / 180.0)
            ddec = g_dec - crv[1]
            px_cd = crp[0] + cdi[0,0] * dra + cdi[0,1] * ddec
            py_cd = crp[1] + cdi[1,0] * dra + cdi[1,1] * ddec
            det_vec_x = dx_px - cx_i
            det_vec_y = dy_px - cy_i
            gaia_vec_x = px_cd - cx_i
            gaia_vec_y = py_cd - cy_i
            res_px = math.hypot(dx_px - px_cd, dy_px - py_cd)
            print(f"  ║   #{i}: 检测({dx_px:7.1f},{dy_px:7.1f})  Gaia投影({px_cd:7.1f},{py_cd:7.1f})  "
                  f"残差{res_px:5.1f}px  RA/Dec=({g_ra:.6f},{g_dec:.6f})")
            print(f"  ║       检测向量=({det_vec_x:7.1f},{det_vec_y:7.1f})  "
                  f"Gaia向量=({gaia_vec_x:7.1f},{gaia_vec_y:7.1f})")
        # 检测向量和Gaia向量的统计
        det_vecs = np.array([[mp[0] - cx_i, mp[1] - cy_i] for mp in match_pairs[:20]])
        gaia_vecs = []
        for mp in match_pairs[:20]:
            dx_px, dy_px, g_ra, g_dec = mp
            dra = (g_ra - crv[0]) * math.cos(crv[1] * math.pi / 180.0)
            ddec = g_dec - crv[1]
            px_cd = crp[0] + cdi[0,0] * dra + cdi[0,1] * ddec
            py_cd = crp[1] + cdi[1,0] * dra + cdi[1,1] * ddec
            gaia_vecs.append([px_cd - cx_i, py_cd - cy_i])
        gaia_vecs = np.array(gaia_vecs)
        det_angles = np.degrees(np.arctan2(det_vecs[:,1], det_vecs[:,0]))
        gaia_angles = np.degrees(np.arctan2(gaia_vecs[:,1], gaia_vecs[:,0]))
        angle_diffs = det_angles - gaia_angles
        angle_diffs = np.where(angle_diffs > 180, angle_diffs - 360, angle_diffs)
        angle_diffs = np.where(angle_diffs < -180, angle_diffs + 360, angle_diffs)
        print(f"  ║  前20对角度差(检测-Gaia): med={np.median(angle_diffs):.2f}° "
              f"mean={np.mean(angle_diffs):.2f}° std={np.std(angle_diffs):.2f}°")
        det_mags = np.linalg.norm(det_vecs, axis=1)
        gaia_mags = np.linalg.norm(gaia_vecs, axis=1)
        mag_ratios = det_mags / (gaia_mags + 1e-10)
        print(f"  ║  前20对模长比(检测/Gaia): med={np.median(mag_ratios):.4f} "
              f"mean={np.mean(mag_ratios):.4f}")
    print(f"  ╚{'═'*40}╝\n")

    # 诊断: 检查sky_to_pixel投影是否正确
    if len(match_pairs) > 0:
        mp0 = match_pairs[0]
        gx_test, gy_test = sky_to_pixel([mp0[2]], [mp0[3]])
        dra_test = (mp0[2] - crv[0]) * cos_d
        ddec_test = mp0[3] - crv[1]
        px_lin = crp[0] + cdi[0,0]*dra_test + cdi[0,1]*ddec_test
        py_lin = crp[1] + cdi[1,0]*dra_test + cdi[1,1]*ddec_test
        print(f"  [诊断] pair[0]: 检测=({mp0[0]:.1f},{mp0[1]:.1f})")
        print(f"         sky_to_pixel=({float(gx_test[0]):.1f},{float(gy_test[0]):.1f})")
        print(f"         CD逆线性投影=({px_lin:.1f},{py_lin:.1f})")
        print(f"         SIP_ORDER={so}, max|SIP_A|={np.max(np.abs(sipA)):.6f}, max|SIP_B|={np.max(np.abs(sipB)):.6f}")

    data = img.data.astype(np.float32)
    dd = data[data > 0]
    lo, hi = np.percentile(dd, (1, 99.5)) if len(dd) > 1 else (0, 1)
    ims = np.clip((data - lo) / max(hi-lo, 1), 0, 1)

    DPI = 100
    fig = plt.figure(figsize=(w/DPI, h/DPI), dpi=DPI, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(ims, cmap="gray", origin="lower", interpolation="nearest")

    csz = 8
    ax.plot(gx_b, gy_b, '+', color='red', markersize=csz, mew=1.5, alpha=0.9)

    for mp in match_pairs:
        dx_px, dy_px, gaia_ra, gaia_dec = mp
        gx2, gy2 = sky_to_pixel([gaia_ra], [gaia_dec])
        gx2, gy2 = float(gx2[0]), float(gy2[0])
        ax.arrow(gx2, gy2, dx_px-gx2, dy_px-gy2,
                 head_width=10, head_length=8, fc='yellow', ec='yellow',
                 alpha=0.9, lw=1.2, length_includes_head=True)
        circ = plt.Circle((dx_px, dy_px), 5, fc='none', ec='cyan', lw=1.5, alpha=1.0)
        ax.add_patch(circ)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")

    out_png = os.path.join(OUT, f"_cp_{label}.png")
    fig.savefig(out_png, dpi=DPI, pad_inches=0)
    plt.close(fig)
    print(f"{label}: 红色十字{len(gx_b)}个 + 匹配{len(match_pairs)}对 → {out_png}")
