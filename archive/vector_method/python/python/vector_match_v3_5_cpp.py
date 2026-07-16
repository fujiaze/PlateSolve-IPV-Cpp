"""Vector Match V3.5 Python ctypes wrapper - Phase A+B SNR验证 → 直送分层拟合"""
import ctypes, math, os, sys, json, logging, numpy as np
from typing import Optional
from vector_match_v2 import (
    GaiaClientPy, bisection_mag_limit, VectorMatchResult, _DEGTORAD,
    _build_image_vectors, _build_catalog_vectors, _apply_flip, _apply_similarity,
    gnomonic_forward,
)
logger = logging.getLogger("vector_match_v3_5_cpp")
_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"

class VM35SolveParamsC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("s0", ctypes.c_double), ("s_min", ctypes.c_double), ("s_max", ctypes.c_double),
        ("n_modes", ctypes.c_int), ("seed", ctypes.c_int),
        ("K_total", ctypes.c_int), ("batch_size", ctypes.c_int), ("min_samples", ctypes.c_int),
        ("K_top", ctypes.c_int), ("min_inliers", ctypes.c_int),
        ("fov_diag_asec", ctypes.c_double), ("img_width", ctypes.c_double), ("img_height", ctypes.c_double),
        ("center_ra", ctypes.c_double), ("center_dec", ctypes.c_double),
        ("wcs_out_path", ctypes.c_char_p),
        # V3.5新增参数（snr_stop已移除，回退5N/10N停止）
        ("skip_sip", ctypes.c_int),
        ("expand_n_gaia", ctypes.c_int),
        ("expand_n_img", ctypes.c_int),
        ("radial_n_bins", ctypes.c_int),
        ("radial_fit_order", ctypes.c_int),
        ("radial_n_iters", ctypes.c_int),
    ]

class VM35DebugInfoC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("theta_snr", ctypes.c_double), ("theta_peak_deg", ctypes.c_double),
        ("best_n_range", ctypes.c_int), ("median_noise", ctypes.c_double),
        ("n_phaseb_pairs", ctypes.c_int), ("n_phaseb_corr", ctypes.c_int),
        ("n_phasea_records", ctypes.c_int), ("n_phasec_expanded", ctypes.c_int),
        ("n_phased_clean", ctypes.c_int), ("n_phased_iterations", ctypes.c_int),
        ("mad_rms_arcsec", ctypes.c_double),
        # V3.5新增：扩展匹配与SIP拟合调试信息
        ("n_expand_mutual", ctypes.c_int),
        ("n_expand_after_filter", ctypes.c_int),
        ("n_sip_total", ctypes.c_int),
        ("sip_order", ctypes.c_int),
    ]

class VM35SolveResultC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("s", ctypes.c_double), ("theta", ctypes.c_double), ("tx", ctypes.c_double), ("ty", ctypes.c_double),
        ("n_inliers", ctypes.c_int), ("rms", ctypes.c_double),
        ("best_mode", ctypes.c_int), ("norm_score", ctypes.c_double),
        ("inlier_mask", ctypes.POINTER(ctypes.c_int)),
        ("success", ctypes.c_int), ("peak_snr", ctypes.c_double), ("n_samples", ctypes.c_int),
        ("debug", VM35DebugInfoC),
        ("sip_A", ctypes.c_double * 36), ("sip_B", ctypes.c_double * 36),
        ("cd", ctypes.c_double * 4), ("crval", ctypes.c_double * 2), ("crpix", ctypes.c_double * 2),
    ]

def _find_dll():
    p = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v3_5", "vector_match_v3_5.dll")
    if os.path.exists(p): return p
    raise FileNotFoundError(p)

def _read_wcs_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return {
        'CD': np.array(d['CD'], dtype=np.float64),
        'CRVAL': np.array(d['CRVAL'], dtype=np.float64),
        'CRPIX': np.array(d['CRPIX'], dtype=np.float64),
        'SIP_A': np.array(d['SIP_A'], dtype=np.float64),
        'SIP_B': np.array(d['SIP_B'], dtype=np.float64),
        'RMS_PX': float(d['RMS_PX']),
    }

class VectorMatchV35Cpp:
    def __init__(self, gaia_data_dir, db_type=1):
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._dll = ctypes.CDLL(_find_dll())
        self._dll.vm35_solve.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(VM35SolveParamsC), ctypes.POINTER(VM35SolveResultC)]
        self._dll.vm35_solve.restype = ctypes.c_int
        self._closed = False

    def solve(self, img_x, img_y, img_flux, img_saturated, cra, cdec, fl, ps, w, h,
              wcs_out=None, skip_sip=False, exptime=1.0):
        """Phase A+B信噪比验证匹配对 → 直送分层拟合(CD/SIP)

        skip_sip已废弃，V3.5最终版始终启用SIP拟合"""
        s0 = 206.265 * ps / fl
        fov_diag = math.sqrt(w*w+h*h)*s0/3600.0

        # V3.5: N=250 — 取全部饱和星 + flux最高的非饱和星补足250颗
        img_x_arr = np.asarray(img_x, np.float64)
        img_y_arr = np.asarray(img_y, np.float64)
        img_flux_arr = np.asarray(img_flux, np.float64)
        img_sat_arr = np.asarray(img_saturated, np.bool_)
        sat_idx = np.where(img_sat_arr)[0]
        nsat = len(sat_idx)
        non_sat_idx = np.where(~img_sat_arr)[0]
        if len(non_sat_idx) > 0:
            non_sat_sorted = non_sat_idx[np.argsort(-img_flux_arr[non_sat_idx])]
        else:
            non_sat_sorted = np.array([], dtype=np.int64)
        n_needed = max(0, 250 - nsat)
        top_non_sat = non_sat_sorted[:n_needed]
        sel_idx = np.concatenate([sat_idx, top_non_sat])
        N = len(sel_idx)
        if N < 2: raise ValueError(f"N={N}")
        cx, cy = w/2.0, h/2.0
        ux = (img_x_arr[sel_idx] - cx) * s0
        uy = -(img_y_arr[sel_idx] - cy) * s0
        U = np.column_stack([ux, uy])

        # Gaia星数: V3.3策略
        Ngaia = math.ceil(1.5 * nsat) if nsat >= 50 else 150

        # V3.5修复: 极限星等公式确定查询星等 + gnomonic投影 + 取FOV内最亮的Ngaia颗
        # 极限星等: m_cut ≈ 6 + 1.5*log10(f_mm) + 2*log10(t_s)
        m_cut = 6.0 + 1.5 * math.log10(max(fl, 1.0)) + 2.0 * math.log10(max(exptime, 0.1))
        # 按星点密度计算所需查询星等: 需要圆形查询区域内任一FOV大小的区域有≥Ngaia颗星
        # 图像星点密度 (颗/deg²)
        fov_area_deg2 = (w * s0 / 3600.0) * (h * s0 / 3600.0)
        # 所需Gaia星密度: 至少Ngaia颗/FOV面积，加1.5x安全系数确保边缘区域也够
        required_density = max(Ngaia, 50) / max(fov_area_deg2, 1e-6) * 1.5
        # 查询圆面积 (0.5×FOV对角线半径，确保覆盖旋转范围)
        query_radius_deg = fov_diag * 0.5
        query_area_deg2 = math.pi * query_radius_deg ** 2
        # 需要的总星数
        n_required_total = int(required_density * query_area_deg2)

        # 从m_cut开始迭代，0.5等步长，找到够用的星等限制
        mag_query = m_cut
        cat_ra_all = cat_dec_all = cat_mag_all = None
        for attempt in range(10):
            ra_t, dec_t, mag_t = self._gaia.cone_search(cra, cdec, query_radius_deg, mag_query)
            n_stars = len(ra_t)
            sys.stderr.write(f"[vm35] Gaia迭代#{attempt}: mag={mag_query:.1f} 返回{n_stars}颗 (需要≥{n_required_total})\n")
            if n_stars >= n_required_total or n_stars >= 500000:
                cat_ra_all, cat_dec_all, cat_mag_all = ra_t, dec_t, mag_t
                break
            mag_query += 0.5
        else:
            # 最终用mag=22兜底
            cat_ra_all, cat_dec_all, cat_mag_all = self._gaia.cone_search(cra, cdec, query_radius_deg, 22.0)
            sys.stderr.write(f"[vm35] Gaia迭代未满足，兜底mag=22 返回{len(cat_ra_all)}颗\n")

        if cat_ra_all is None or len(cat_ra_all) < 2:
            raise ValueError(f"Gaia查询返回{len(cat_ra_all) if cat_ra_all is not None else 0}颗星")

        sys.stderr.write(f"[vm35] Gaia查询: m_cut={m_cut:.1f} mag_query={mag_query:.1f} 返回{len(cat_ra_all)}颗 (需要≥{n_required_total})\n")

        # gnomonic投影
        xi_all, eta_all, valid_all = gnomonic_forward(cat_ra_all, cat_dec_all, cra, cdec)
        fov_half_w = w/2 * s0
        fov_half_h = h/2 * s0

        # 筛选FOV内的星
        in_fov = valid_all & (np.abs(xi_all) < fov_half_w) & (np.abs(eta_all) < fov_half_h)
        fov_idx = np.where(in_fov)[0]

        if len(fov_idx) < 2:
            # FOV内星太少，放宽到1.5x FOV
            in_fov = valid_all & (np.abs(xi_all) < fov_half_w*1.5) & (np.abs(eta_all) < fov_half_h*1.5)
            fov_idx = np.where(in_fov)[0]

        if len(fov_idx) < 2:
            raise ValueError(f"FOV内只有{len(fov_idx)}颗Gaia星")

        sys.stderr.write(f"[vm35] FOV内Gaia星: {len(fov_idx)}/{len(cat_ra_all)} (需要≥{Ngaia})\n")

        # 按星等排序，取最亮的Ngaia颗
        fov_mag = cat_mag_all[fov_idx]
        sorted_order = np.argsort(fov_mag)
        sel_order = sorted_order[:min(Ngaia, len(sorted_order))]
        sel_idx = fov_idx[sel_order]

        cat_ra = cat_ra_all[sel_idx]
        cat_dec = cat_dec_all[sel_idx]
        M = len(cat_ra)

        W = _build_catalog_vectors(cat_ra, cat_dec, cra, cdec)

        wcs_json_path = wcs_out
        if wcs_json_path is None:
            import tempfile
            fd, wcs_json_path = tempfile.mkstemp(suffix='.json', prefix='vm35_wcs_')
            os.close(fd)

        # V3.5最终版: Phase B匹配对直送拟合，无NN扩充/扩增
        params = VM35SolveParamsC()
        params.s0=s0; params.s_min=0.9; params.s_max=1.1; params.n_modes=4; params.seed=42
        params.K_total=10000; params.batch_size=1000; params.min_samples=2000
        params.K_top=50; params.min_inliers=max(5,int(N*0.1))
        params.fov_diag_asec=fov_diag*3600.0; params.img_width=float(w); params.img_height=float(h)
        params.center_ra=float(cra); params.center_dec=float(cdec)
        # V3.5新增参数（snr_stop已移除，回退5N/10N停止）
        params.skip_sip=1 if skip_sip else 0
        params.expand_n_gaia=1500
        params.expand_n_img=1000
        params.radial_n_bins=20
        params.radial_fit_order=3
        params.radial_n_iters=3
        params.wcs_out_path = wcs_json_path.encode('utf-8') if isinstance(wcs_json_path, str) else wcs_json_path

        imask = np.zeros(N, dtype=np.int32)
        result = VM35SolveResultC()
        result.inlier_mask = imask.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        ret = self._dll.vm35_solve(
            np.ascontiguousarray(U,np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), N,
            np.ascontiguousarray(W,np.float64).ctypes.data_as(ctypes.POINTER(ctypes.c_double)), M,
            ctypes.byref(params), ctypes.byref(result))

        if ret != 0 or result.success == 0:
            logger.warning("fail: ret=%d succ=%d", ret, result.success); return None
        if result.s < 0.9 or result.s > 1.1:
            logger.warning("s=%.4f out of range", result.s); return None

        s, theta, tx, ty = result.s, result.theta, result.tx, result.ty
        best_mode = result.best_mode
        cos_d = math.cos(cdec*_DEGTORAD)
        dra = -tx/(max(cos_d,1e-10)*3600.0); ddec = -ty/3600.0
        cur_ra, cur_dec = cra+dra, cdec+ddec

        Wf = _apply_flip(W, best_mode)
        inl_mask = imask.astype(bool)
        s_final = s0 * s
        rot_deg = math.degrees(theta)

        rms_px, rms_asec = 0.0, 0.0
        if np.any(inl_mask):
            Wt_arr = _apply_similarity(Wf, s, theta, tx, ty)
            from scipy.spatial import cKDTree
            tree = cKDTree(Wt_arr); dists, idxs = tree.query(U, k=1)
            diffs = U[inl_mask] - Wt_arr[idxs[inl_mask]]
            rms_asec = float(np.sqrt(np.mean(np.sum(diffs**2,axis=1))))
            rms_px = rms_asec / s0

        ct, st = math.cos(theta), math.sin(theta)
        affine = (tx, s*ct, -s*st, ty, s*st, s*ct)
        r = VectorMatchResult(center_ra=cur_ra, center_dec=cur_dec,
            original_ra=cra, original_dec=cdec, rotation_deg=rot_deg,
            scale_arcsec_px=s_final, flip_mode=best_mode, matched_count=int(np.sum(inl_mask)),
            rms_px=rms_px, rms_arcsec=rms_asec, affine=affine)
        r.solve_tx=tx; r.solve_ty=ty; r.solve_s=s; r.s0=s0
        r.theta_snr=float(result.debug.theta_snr)
        r.theta_peak_deg=float(result.debug.theta_peak_deg)
        r.best_n_range=int(result.debug.best_n_range)
        r.median_noise=float(result.debug.median_noise)
        r.n_phaseb_pairs=int(result.debug.n_phaseb_pairs)
        r.n_phaseb_corr=int(result.debug.n_phaseb_corr)
        r.n_phasea_records=int(result.debug.n_phasea_records)
        r.n_phasec_expanded=int(result.debug.n_phasec_expanded)
        r.n_phased_clean=int(result.debug.n_phased_clean)
        r.n_phased_iterations=int(result.debug.n_phased_iterations)
        r.mad_rms_arcsec=float(result.debug.mad_rms_arcsec)
        # V3.5新增调试信息
        r.n_expand_mutual=int(result.debug.n_expand_mutual)
        r.n_expand_after_filter=int(result.debug.n_expand_after_filter)
        r.n_sip_total=int(result.debug.n_sip_total)
        r.sip_order=int(result.debug.sip_order)

        # 从JSON文件读取WCS参数
        if os.path.exists(wcs_json_path):
            try:
                wcs_data = _read_wcs_json(wcs_json_path)
                r.cd = wcs_data['CD']
                r.crval = wcs_data['CRVAL']
                r.crpix = wcs_data['CRPIX']
                r.sip_A = wcs_data['SIP_A'].reshape(6,6)
                r.sip_B = wcs_data['SIP_B'].reshape(6,6)
                r.sip_rms_px = wcs_data['RMS_PX']
                logger.info("WCS JSON loaded: CD=%s CRVAL=%s SIP_RMS=%.3fpx",
                            r.cd.tolist(), r.crval.tolist(), r.sip_rms_px)
            except Exception as e:
                logger.warning("WCS JSON read failed: %s, fallback to ctypes", e)
                r.sip_rms_px = float(result.rms)
                r.cd = np.array([[float(result.cd[0]),float(result.cd[1])],
                                 [float(result.cd[2]),float(result.cd[3])]], dtype=np.float64)
                r.crval = np.array([float(result.crval[0]),float(result.crval[1])])
                r.crpix = np.array([float(result.crpix[0]),float(result.crpix[1])])
                r.sip_A = np.array([float(result.sip_A[i]) for i in range(36)], dtype=np.float64).reshape(6,6)
                r.sip_B = np.array([float(result.sip_B[i]) for i in range(36)], dtype=np.float64).reshape(6,6)
            if wcs_out is None:
                try: os.remove(wcs_json_path)
                except: pass
        else:
            logger.warning("WCS JSON not found: %s", wcs_json_path)
            r.sip_rms_px = float(result.rms)
            r.cd = np.array([[float(result.cd[0]),float(result.cd[1])],
                             [float(result.cd[2]),float(result.cd[3])]], dtype=np.float64)
            r.crval = np.array([float(result.crval[0]),float(result.crval[1])])
            r.crpix = np.array([float(result.crpix[0]),float(result.crpix[1])])
            r.sip_A = np.zeros((6,6), dtype=np.float64)
            r.sip_B = np.zeros((6,6), dtype=np.float64)

        return r

    def close(self):
        if not self._closed and self._gaia: self._gaia.close(); self._gaia=None; self._closed=True
    def __del__(self): self.close()
    def __enter__(self): return self
    def __exit__(self,*a): self.close()
