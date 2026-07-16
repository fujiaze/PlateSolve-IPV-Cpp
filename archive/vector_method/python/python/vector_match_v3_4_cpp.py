"""Vector Match V3.4 Python ctypes wrapper - _pack_=8 alignment + JSON WCS"""
import ctypes, math, os, json, logging, numpy as np
from typing import Optional
from vector_match_v2 import (
    GaiaClientPy, bisection_mag_limit, VectorMatchResult, _DEGTORAD,
    _build_image_vectors, _build_catalog_vectors, _apply_flip, _apply_similarity,
)
logger = logging.getLogger("vector_match_v3_4_cpp")
_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"

class VM34SolveParamsC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("s0", ctypes.c_double), ("s_min", ctypes.c_double), ("s_max", ctypes.c_double),
        ("n_modes", ctypes.c_int), ("seed", ctypes.c_int),
        ("K_total", ctypes.c_int), ("batch_size", ctypes.c_int), ("min_samples", ctypes.c_int),
        ("K_top", ctypes.c_int), ("min_inliers", ctypes.c_int),
        ("fov_diag_asec", ctypes.c_double), ("img_width", ctypes.c_double), ("img_height", ctypes.c_double),
        ("center_ra", ctypes.c_double), ("center_dec", ctypes.c_double),
        ("wcs_out_path", ctypes.c_char_p),
    ]

class VM34DebugInfoC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("theta_snr", ctypes.c_double), ("theta_peak_deg", ctypes.c_double),
        ("best_n_range", ctypes.c_int), ("median_noise", ctypes.c_double),
        ("n_phaseb_pairs", ctypes.c_int), ("n_phaseb_corr", ctypes.c_int),
        ("n_phasea_records", ctypes.c_int), ("n_phasec_expanded", ctypes.c_int),
        ("n_phased_clean", ctypes.c_int), ("n_phased_iterations", ctypes.c_int),
        ("mad_rms_arcsec", ctypes.c_double),
    ]

class VM34SolveResultC(ctypes.Structure):
    _pack_ = 8
    _fields_ = [
        ("s", ctypes.c_double), ("theta", ctypes.c_double), ("tx", ctypes.c_double), ("ty", ctypes.c_double),
        ("n_inliers", ctypes.c_int), ("rms", ctypes.c_double),
        ("best_mode", ctypes.c_int), ("norm_score", ctypes.c_double),
        ("inlier_mask", ctypes.POINTER(ctypes.c_int)),
        ("success", ctypes.c_int), ("peak_snr", ctypes.c_double), ("n_samples", ctypes.c_int),
        ("debug", VM34DebugInfoC),
        ("sip_A", ctypes.c_double * 36), ("sip_B", ctypes.c_double * 36),
        ("cd", ctypes.c_double * 4), ("crval", ctypes.c_double * 2), ("crpix", ctypes.c_double * 2),
    ]

def _find_dll():
    p = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v3_4", "vector_match_v3_4.dll")
    if os.path.exists(p): return p
    raise FileNotFoundError(p)

def _read_wcs_json(path):
    """从C++输出的JSON文件读取WCS参数"""
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

class VectorMatchV34Cpp:
    def __init__(self, gaia_data_dir, db_type=1):
        self._gaia = GaiaClientPy(gaia_data_dir, db_type)
        self._dll = ctypes.CDLL(_find_dll())
        self._dll.vm34_solve.argtypes = [
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(VM34SolveParamsC), ctypes.POINTER(VM34SolveResultC)]
        self._dll.vm34_solve.restype = ctypes.c_int
        self._closed = False

    def solve(self, img_x, img_y, img_flux, img_saturated, cra, cdec, fl, ps, w, h,
              wcs_out=None):
        s0 = 206.265 * ps / fl
        fov_diag = math.sqrt(w*w+h*h)*s0/3600.0
        U, N, nsat, _ = _build_image_vectors(
            np.asarray(img_x,np.float64), np.asarray(img_y,np.float64),
            np.asarray(img_flux,np.float64), np.asarray(img_saturated,np.int32), s0, w, h)
        if N < 2: raise ValueError(f"N={N}")

        Ngaia = math.ceil(1.5*nsat) if nsat>=50 else 200
        maglim, M, cat_ra, cat_dec, _ = bisection_mag_limit(
            self._gaia, cra, cdec, fov_diag*1.2/2.0, Ngaia)
        if M < 2: raise ValueError(f"M={M}")

        W = _build_catalog_vectors(cat_ra, cat_dec, cra, cdec)

        # WCS JSON输出路径
        wcs_json_path = wcs_out
        if wcs_json_path is None:
            import tempfile
            fd, wcs_json_path = tempfile.mkstemp(suffix='.json', prefix='vm34_wcs_')
            os.close(fd)

        params = VM34SolveParamsC()
        params.s0=s0; params.s_min=0.9; params.s_max=1.1; params.n_modes=4; params.seed=42
        params.K_total=10000; params.batch_size=1000; params.min_samples=2000
        params.K_top=50; params.min_inliers=max(5,int(N*0.1))
        params.fov_diag_asec=fov_diag*3600.0; params.img_width=float(w); params.img_height=float(h)
        params.center_ra=float(cra); params.center_dec=float(cdec)
        params.wcs_out_path = wcs_json_path.encode('utf-8') if isinstance(wcs_json_path, str) else wcs_json_path

        imask = np.zeros(N, dtype=np.int32)
        result = VM34SolveResultC()
        result.inlier_mask = imask.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        ret = self._dll.vm34_solve(
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

        # 从JSON文件读取WCS参数（绕开ctypes对齐问题）
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
            # 清理临时文件
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
