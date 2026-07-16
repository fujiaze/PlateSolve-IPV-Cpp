"""
V42Pipeline 模块 - V4.2 管线编排器

功能:
    串联 5 个独立模块: StarSelector → VectorMatcher → PairExpander → PairVerifier → WcsFitter
    每阶段输出 JSON 落盘到 logs/v4_2/<frame>/phase_*.json
    每阶段输出日志到 logs/v4_2/<frame>/phase_*.log
    断点续跑: 检查 phase_*.json 是否存在, 存在则跳过该阶段(可强制重跑)
    返回与 V4.1 兼容的 SolveResult

用途:
    V4.2 模块化管线一键求解接口, 输入 FITS 图像路径 + 中心指向 + 焦距/像元尺寸,
    输出标准 WCS 参数 (CD/CRVAL/CRPIX/SIP)。

依赖:
    - star_selector.dll, vector_matcher.dll, pair_expander.dll,
      pair_verifier.dll, wcs_fitter.dll
    - gaia_client.dll (经 GaiaClientPy 封装)
    - star_detector.dll (经 StarDetector 封装)
    - numpy, ctypes, logging
"""

import os
import sys
import json
import math
import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

# 项目根与路径
_PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
_PLATE_SOLVE_PY = os.path.join(_PROJECT_ROOT, "lib", "plate_solve", "python")
if _PLATE_SOLVE_PY not in sys.path:
    sys.path.insert(0, _PLATE_SOLVE_PY)
_STAR_DET_PY = os.path.join(_PROJECT_ROOT, "lib", "star_detector", "python")
if _STAR_DET_PY not in sys.path:
    sys.path.insert(0, _STAR_DET_PY)

from vector_match_v2 import GaiaClientPy  # noqa: E402

from .star_selector import StarSelector
from .vector_matcher import VectorMatcher
from .pair_expander import PairExpander
from .pair_verifier import PairVerifier
from .wcs_fitter import WcsFitter

logger = logging.getLogger("v4_2_pipeline")

# 阶段名常量
PHASE_0 = "phase_0_star_selector"
PHASE_AB = "phase_ab_vector_matcher"
PHASE_C = "phase_c_pair_expander"
PHASE_D = "phase_d_pair_verifier"
PHASE_E = "phase_e_wcs_fitter"
_ALL_PHASES = [PHASE_0, PHASE_AB, PHASE_C, PHASE_D, PHASE_E]


class V42Pipeline:
    """V4.2 管线编排器: 串联 StarSelector → VectorMatcher → PairExpander → PairVerifier → WcsFitter"""

    def __init__(self,
                 dll_dir: Optional[str] = None,
                 gaia_client: Optional[GaiaClientPy] = None,
                 star_detector: Optional[Any] = None):
        """初始化 5 个模块

        Args:
            dll_dir: 5 个模块 DLL 所在的根目录 (lib/plate_solve/cpp/v4_2/)。
                     None 时各模块使用各自默认路径。
                     若指定, 则从 <dll_dir>/<module_name>/<module_name>.dll 加载。
            gaia_client: 已实例化的 GaiaClientPy。None 时内部创建 (默认 GaiaDR3)。
            star_detector: 已实例化的 StarDetector。None 时内部创建。
        """
        self._dll_dir = dll_dir

        # --- Gaia 客户端: 注入优先, 否则内部创建 ---
        self._gaia_external = gaia_client is not None
        if gaia_client is not None:
            self._gaia_client: Optional[GaiaClientPy] = gaia_client
        else:
            gaia_data_dir = os.path.join(_PROJECT_ROOT, "GaiaDR3")
            self._gaia_client = GaiaClientPy(gaia_data_dir, db_type=0)

        # --- StarDetector: 注入优先, 否则内部创建 ---
        self._star_detector_external = star_detector is not None
        if star_detector is not None:
            self._star_detector = star_detector
        else:
            try:
                from star_detector import StarDetector, SDetParamsPy
                self._star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
            except Exception as e:
                logger.warning("StarDetector 内部创建失败: %s (延迟到 select 时再试)", e)
                self._star_detector = None

        # --- 构造各模块 DLL 路径 ---
        if dll_dir:
            ss_dll = os.path.join(dll_dir, "star_selector", "star_selector.dll")
            vm_dll = os.path.join(dll_dir, "vector_matcher", "vector_matcher.dll")
            pe_dll = os.path.join(dll_dir, "pair_expander", "pair_expander.dll")
            pv_dll = os.path.join(dll_dir, "pair_verifier", "pair_verifier.dll")
            wf_dll = os.path.join(dll_dir, "wcs_fitter", "wcs_fitter.dll")
        else:
            ss_dll = vm_dll = pe_dll = pv_dll = wf_dll = None

        # --- 实例化 5 个模块 ---
        self._star_selector = StarSelector(
            dll_path=ss_dll,
            gaia_client=self._gaia_client,
            star_detector=self._star_detector,
        )
        self._vector_matcher = VectorMatcher(dll_path=vm_dll)
        self._pair_expander = PairExpander(dll_path=pe_dll)
        self._pair_verifier = PairVerifier(dll_path=pv_dll)
        self._wcs_fitter = WcsFitter(dll_path=wf_dll)

        self._closed = False

    # ========================================================================
    # 公共求解接口
    # ========================================================================

    def solve(self,
              image_path: str,
              ra: float,
              dec: float,
              focal_length_mm: float,
              pixel_size_um: float,
              log_dir: Optional[str] = None,
              resume: bool = True,
              force_phase: Optional[str] = None) -> Dict[str, Any]:
        """一键求解: StarSelector → VectorMatcher → PairExpander → PairVerifier → WcsFitter

        Args:
            image_path: FITS 图像路径
            ra, dec: 图像中心赤经赤纬(度)
            focal_length_mm: 焦距(mm)
            pixel_size_um: 像元尺寸(um)
            log_dir: 日志目录(可选, 默认 logs/v4_2/<frame_basename>/)
            resume: 是否启用断点续跑(默认True, 检查 phase_*.json 存在则跳过)
            force_phase: 强制从指定阶段重跑(忽略该阶段及之后阶段的 checkpoint)
                         取值: PHASE_0/PHASE_AB/PHASE_C/PHASE_D/PHASE_E

        Returns:
            dict: 与 V4.1 兼容的 SolveResult, 含:
                success, cd, crval, crpix, sip_A, sip_B, sip_order, rms_px,
                matched_count, scale_arcsec_px, rotation_deg, flip_mode,
                center_ra, center_dec, s0, s, theta, tx, ty,
                theta_snr, bayes_lnK, triangle_pass_ratio, ...
        """
        frame_base = os.path.splitext(os.path.basename(image_path))[0]
        if log_dir is None:
            log_dir = os.path.join(
                _PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_2", frame_base)
        os.makedirs(log_dir, exist_ok=True)

        # 像素尺度与 FOV
        s0 = 206.265 * pixel_size_um / focal_length_mm

        # 确定 force_phase 起始位置 (该阶段及之后全部重跑)
        force_start_idx = _ALL_PHASES.index(force_phase) if force_phase else len(_ALL_PHASES)

        # 阶段执行结果
        ctx: Dict[str, Any] = {
            "image_path": image_path,
            "ra": ra, "dec": dec,
            "focal_length_mm": focal_length_mm,
            "pixel_size_um": pixel_size_um,
            "s0": s0,
            "frame_base": frame_base,
            "log_dir": log_dir,
        }
        phase_status: Dict[str, str] = {}  # phase -> "skipped"/"ok"/"fail"

        # ====================================================================
        # Phase 0: StarSelector
        # ====================================================================
        if resume and force_start_idx > 0 and self._has_checkpoint(log_dir, PHASE_0):
            logger.info("[%s] Phase 0 checkpoint found, skipping", frame_base)
            phase_status[PHASE_0] = "skipped"
            ckpt = self._load_checkpoint(log_dir, PHASE_0)
            ctx["U"] = np.array(ckpt["U"], dtype=np.float64)
            ctx["W"] = np.array(ckpt["W"], dtype=np.float64)
            ctx["meta_p0"] = ckpt["meta"]
        else:
            try:
                result = self._star_selector.select(
                    image_path=image_path,
                    ra=ra, dec=dec,
                    focal_length_mm=focal_length_mm,
                    pixel_size_um=pixel_size_um,
                    log_dir=log_dir,
                )
                ctx["U"] = result["U"]
                ctx["W"] = result["W"]
                ctx["meta_p0"] = result["meta"]
                self._save_checkpoint(log_dir, PHASE_0, {
                    "U": result["U"].tolist(),
                    "W": result["W"].tolist(),
                    "meta": _json_safe(result["meta"]),
                })
                phase_status[PHASE_0] = "ok"
                logger.info("[%s] Phase 0 OK: U=%d×2 W=%d×2 m_lim=%.3f",
                            frame_base, len(result["U"]), len(result["W"]),
                            result["meta"].get("m_lim_final", 0))
            except Exception as e:
                logger.error("[%s] Phase 0 FAIL: %s", frame_base, e)
                phase_status[PHASE_0] = "fail"
                return self._build_failure(ctx, phase_status, str(e))

        img_width = ctx["meta_p0"].get("img_width", 0)
        img_height = ctx["meta_p0"].get("img_height", 0)
        fov_diag_deg = ctx["meta_p0"].get("fov_diag_deg", 0.0)

        # ====================================================================
        # Phase A+B: VectorMatcher
        # ====================================================================
        if resume and force_start_idx > 1 and self._has_checkpoint(log_dir, PHASE_AB):
            logger.info("[%s] Phase A+B checkpoint found, skipping", frame_base)
            phase_status[PHASE_AB] = "skipped"
            ckpt = self._load_checkpoint(log_dir, PHASE_AB)
            ctx["vm_result"] = ckpt
        else:
            try:
                U = ctx["U"]
                W = ctx["W"]
                vm_log = os.path.join(log_dir, "phase_ab_vector_matcher.log")
                vm_result = self._vector_matcher.match(
                    U=U, W=W,
                    s0=s0,
                    s_min=0.9, s_max=1.1,
                    log_file_path=vm_log,
                )
                if not vm_result.get("success", False):
                    raise RuntimeError(
                        f"VectorMatcher failed: n_pairs={vm_result.get('n_pairs', 0)}")
                ctx["vm_result"] = vm_result
                self._save_checkpoint(log_dir, PHASE_AB, _json_safe(vm_result))
                phase_status[PHASE_AB] = "ok"
                logger.info("[%s] Phase A+B OK: n_pairs=%d s=%.4f θ=%.4f° SNR=%.2f",
                            frame_base, vm_result["n_pairs"], vm_result["s"],
                            math.degrees(vm_result["theta"]), vm_result["theta_snr"])
            except Exception as e:
                logger.error("[%s] Phase A+B FAIL: %s", frame_base, e)
                phase_status[PHASE_AB] = "fail"
                return self._build_failure(ctx, phase_status, str(e))

        vm = ctx["vm_result"]

        # ====================================================================
        # 应用 flip: VectorMatcher 内部对 W 应用 flip(best_mode) 后拟合变换,
        # 下游模块 (PairExpander/PairVerifier/WcsFitter) 设计上不处理 flip,
        # 因此需根据 best_mode 对 W 应用 flip 生成 W_eff, 后续阶段统一使用 W_eff。
        # ====================================================================
        best_mode = int(vm.get("best_mode", 0))
        W_orig = ctx["W"]
        W_eff = _apply_flip(W_orig, best_mode)
        ctx["W_eff"] = W_eff
        if best_mode != 0:
            logger.info("[%s] 应用 flip: best_mode=%d (W → W_eff)", frame_base, best_mode)

        # ====================================================================
        # Phase C: PairExpander
        # ====================================================================
        if resume and force_start_idx > 2 and self._has_checkpoint(log_dir, PHASE_C):
            logger.info("[%s] Phase C checkpoint found, skipping", frame_base)
            phase_status[PHASE_C] = "skipped"
            ctx["pe_result"] = self._load_checkpoint(log_dir, PHASE_C)
        else:
            try:
                T = {
                    "s": vm["s"], "theta": vm["theta"],
                    "tx": vm["tx"], "ty": vm["ty"],
                }
                pe_result = self._pair_expander.expand(
                    U=ctx["U"], W=W_eff,
                    T=T,
                    init_cu=vm.get("cu", []),
                    init_cw=vm.get("cw", []),
                    s0=s0,
                    img_width=img_width, img_height=img_height,
                    log_dir=log_dir,
                )
                if not pe_result.get("success", False):
                    raise RuntimeError(
                        f"PairExpander failed: {pe_result.get('meta', {}).get('error', '')}")
                ctx["pe_result"] = pe_result
                self._save_checkpoint(log_dir, PHASE_C, _json_safe(pe_result))
                phase_status[PHASE_C] = "ok"
                logger.info("[%s] Phase C OK: n_pairs=%d (expanded=%d)",
                            frame_base, pe_result["n_pairs"], pe_result["n_expanded"])
            except Exception as e:
                logger.error("[%s] Phase C FAIL: %s", frame_base, e)
                phase_status[PHASE_C] = "fail"
                return self._build_failure(ctx, phase_status, str(e))

        pe = ctx["pe_result"]

        # ====================================================================
        # Phase D+D': PairVerifier
        # ====================================================================
        if resume and force_start_idx > 3 and self._has_checkpoint(log_dir, PHASE_D):
            logger.info("[%s] Phase D checkpoint found, skipping", frame_base)
            phase_status[PHASE_D] = "skipped"
            ctx["pv_result"] = self._load_checkpoint(log_dir, PHASE_D)
        else:
            try:
                # 构造 pairs [[u, w], ...]
                pairs = list(zip(pe["expand_u"], pe["expand_w"]))
                pairs = [[int(u), int(w)] for u, w in pairs]
                pv_result = self._pair_verifier.verify(
                    U=ctx["U"], W=W_eff,
                    pairs=pairs,
                    s0=s0,
                    fov_diag_deg=fov_diag_deg,
                    log_dir=log_dir,
                )
                if not pv_result.get("success", False):
                    raise RuntimeError(
                        f"PairVerifier failed: {pv_result.get('meta', {}).get('error', '')}")
                ctx["pv_result"] = pv_result
                self._save_checkpoint(log_dir, PHASE_D, _json_safe(pv_result))
                phase_status[PHASE_D] = "ok"
                logger.info("[%s] Phase D OK: n_clean=%d validated=%s lnK=%.2f tri=%.3f",
                            frame_base, pv_result["n_clean"], pv_result["validated"],
                            pv_result["bayes"]["lnK"], pv_result["triangle"]["pass_ratio"])
            except Exception as e:
                logger.error("[%s] Phase D FAIL: %s", frame_base, e)
                phase_status[PHASE_D] = "fail"
                return self._build_failure(ctx, phase_status, str(e))

        pv = ctx["pv_result"]

        # ====================================================================
        # Phase E: WcsFitter
        # ====================================================================
        if resume and force_start_idx > 4 and self._has_checkpoint(log_dir, PHASE_E):
            logger.info("[%s] Phase E checkpoint found, skipping", frame_base)
            phase_status[PHASE_E] = "skipped"
            ctx["wf_result"] = self._load_checkpoint(log_dir, PHASE_E)
        else:
            try:
                wf_result = self._wcs_fitter.fit(
                    U=ctx["U"], W=W_eff,
                    pairs=pv["pairs"],
                    ra=ra, dec=dec,
                    focal_length_mm=focal_length_mm,
                    pixel_size_um=pixel_size_um,
                    img_width=img_width, img_height=img_height,
                    log_dir=log_dir,
                )
                if not wf_result.get("success", False):
                    raise RuntimeError(
                        f"WcsFitter failed: {wf_result.get('error', '')}")
                ctx["wf_result"] = wf_result
                self._save_checkpoint(log_dir, PHASE_E, _json_safe(wf_result))
                phase_status[PHASE_E] = "ok"
                logger.info("[%s] Phase E OK: rms_px=%.4f sip_order=%d n_pairs=%d",
                            frame_base, wf_result["rms_px"], wf_result["sip_order"],
                            wf_result["n_pairs"])
            except Exception as e:
                logger.error("[%s] Phase E FAIL: %s", frame_base, e)
                phase_status[PHASE_E] = "fail"
                return self._build_failure(ctx, phase_status, str(e))

        wf = ctx["wf_result"]

        # ====================================================================
        # 组装 V4.1 兼容的 SolveResult
        # ====================================================================
        result = self._build_solve_result(
            ctx=ctx, vm=vm, pe=pe, pv=pv, wf=wf,
            ra=ra, dec=dec, s0=s0,
            focal_length_mm=focal_length_mm, pixel_size_um=pixel_size_um,
            phase_status=phase_status,
        )
        # 最终 WCS JSON 落盘
        wcs_json_path = os.path.join(log_dir, "wcs_final.json")
        try:
            with open(wcs_json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "CD": wf["cd"],
                    "CRVAL": wf["crval"],
                    "CRPIX": wf["crpix"],
                    "SIP_A": wf["sip_A"],
                    "SIP_B": wf["sip_B"],
                    "SIP_ORDER": wf["sip_order"],
                    "RMS_PX": wf["rms_px"],
                    "N_PAIRS": wf["n_pairs"],
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("wcs_final.json 写入失败: %s", e)

        logger.info("[%s] Pipeline 完成: success=%s RMS=%.4fpx matched=%d",
                    frame_base, result["success"], result["rms_px"],
                    result["matched_count"])
        return result

    # ========================================================================
    # 内部工具方法
    # ========================================================================

    @staticmethod
    def _has_checkpoint(log_dir: str, phase: str) -> bool:
        return os.path.exists(os.path.join(log_dir, f"{phase}.json"))

    @staticmethod
    def _load_checkpoint(log_dir: str, phase: str) -> Dict[str, Any]:
        path = os.path.join(log_dir, f"{phase}.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _save_checkpoint(log_dir: str, phase: str, data: Dict[str, Any]) -> None:
        path = os.path.join(log_dir, f"{phase}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _build_failure(ctx: Dict[str, Any],
                       phase_status: Dict[str, str],
                       error: str) -> Dict[str, Any]:
        """构建失败结果 (与 V4.1 兼容的 SolveResult 字段)"""
        s0 = ctx.get("s0", 0.0)
        img_w = ctx.get("meta_p0", {}).get("img_width", 0)
        img_h = ctx.get("meta_p0", {}).get("img_height", 0)
        return {
            "success": False,
            "error": error,
            "phase_status": phase_status,
            "frame_base": ctx.get("frame_base", ""),
            "log_dir": ctx.get("log_dir", ""),
            # V4.1 兼容字段
            "cd": [0.0, 0.0, 0.0, 0.0],
            "crval": [ctx.get("ra", 0.0), ctx.get("dec", 0.0)],
            "crpix": [img_w / 2.0, img_h / 2.0],
            "sip_A": [0.0] * 36,
            "sip_B": [0.0] * 36,
            "sip_order": 0,
            "sip_rms_px": 0.0,
            "rms_px": 0.0,
            "rms_arcsec": 0.0,
            "matched_count": 0,
            "scale_arcsec_px": s0,
            "rotation_deg": 0.0,
            "flip_mode": -1,
            "center_ra": ctx.get("ra", 0.0),
            "center_dec": ctx.get("dec", 0.0),
            "original_ra": ctx.get("ra", 0.0),
            "original_dec": ctx.get("dec", 0.0),
            "s0": s0,
            "s": 0.0, "theta": 0.0, "tx": 0.0, "ty": 0.0,
            "theta_snr": 0.0, "theta_peak_deg": 0.0,
            "bayes_lnK": 0.0, "triangle_pass_ratio": 0.0,
        }

    @staticmethod
    def _build_solve_result(ctx: Dict[str, Any],
                            vm: Dict[str, Any],
                            pe: Dict[str, Any],
                            pv: Dict[str, Any],
                            wf: Dict[str, Any],
                            ra: float, dec: float, s0: float,
                            focal_length_mm: float, pixel_size_um: float,
                            phase_status: Dict[str, str]) -> Dict[str, Any]:
        """构建 V4.1 兼容的 SolveResult"""
        s = vm.get("s", 1.0)
        theta = vm.get("theta", 0.0)
        tx = vm.get("tx", 0.0)
        ty = vm.get("ty", 0.0)
        best_mode = vm.get("best_mode", 0)
        rms_px = wf.get("rms_px", 0.0)
        matched_count = wf.get("n_pairs", pv.get("n_clean", 0))
        scale_arcsec_px = s0 * s
        rotation_deg = math.degrees(theta)
        rms_arcsec = rms_px * s0

        # 中心修正 (与 V4.1 一致: 用 tx/ty 反向推算中心偏移)
        cos_d = math.cos(dec * math.pi / 180.0)
        dra = -tx / (max(cos_d, 1e-10) * 3600.0)
        ddec = -ty / 3600.0
        center_ra = ra + dra
        center_dec = dec + ddec

        return {
            "success": True,
            "phase_status": phase_status,
            "frame_base": ctx.get("frame_base", ""),
            "log_dir": ctx.get("log_dir", ""),
            # WCS 参数
            "cd": wf["cd"],
            "crval": wf["crval"],
            "crpix": wf["crpix"],
            "sip_A": wf["sip_A"],
            "sip_B": wf["sip_B"],
            "sip_order": wf["sip_order"],
            "sip_rms_px": wf["rms_px"],
            "rms_px": rms_px,
            "rms_arcsec": rms_arcsec,
            "matched_count": matched_count,
            # 变换参数
            "scale_arcsec_px": scale_arcsec_px,
            "rotation_deg": rotation_deg,
            "flip_mode": best_mode,
            "center_ra": center_ra,
            "center_dec": center_dec,
            "original_ra": ra,
            "original_dec": dec,
            "s0": s0,
            "s": s, "theta": theta, "tx": tx, "ty": ty,
            # 调试信息
            "theta_snr": vm.get("theta_snr", 0.0),
            "theta_peak_deg": vm.get("theta_peak_deg", 0.0),
            "best_n_range": vm.get("best_n_range", 0),
            "n_phasea_records": vm.get("n_phasea_records", 0),
            "prosac_quality_median": vm.get("prosac_quality_median", 0.0),
            "prosac_pool_final": vm.get("prosac_pool_final", 0),
            # Phase 0 元数据
            "m_lim_final": ctx.get("meta_p0", {}).get("m_lim_final", 0.0),
            "n_gaia_final": ctx.get("meta_p0", {}).get("n_gaia_final", 0),
            "m_lim_iterations": ctx.get("meta_p0", {}).get("m_lim_iterations", 0),
            "rho_img": ctx.get("meta_p0", {}).get("rho_img", 0.0),
            "rho_target": ctx.get("meta_p0", {}).get("rho_target", 0.0),
            "fov_diag_deg": ctx.get("meta_p0", {}).get("fov_diag_deg", 0.0),
            # Phase C 元数据
            "n_phasec_expanded": pe.get("n_expanded", 0),
            "n_phasec_total": pe.get("n_pairs", 0),
            # Phase D 元数据
            "n_phased_clean": pv.get("n_clean", 0),
            "mad_rms_arcsec": pv.get("mad", {}).get("rms_arcsec", 0.0),
            "mad_iterations": pv.get("mad", {}).get("iterations", 0),
            "bayes_lnK": pv.get("bayes", {}).get("lnK", 0.0),
            "bayes_n_match": pv.get("bayes", {}).get("n_match", 0),
            "bayes_decision": pv.get("bayes", {}).get("decision", 0),
            "triangle_total": pv.get("triangle", {}).get("total", 0),
            "triangle_pass_ratio": pv.get("triangle", {}).get("pass_ratio", 0.0),
            "validated": pv.get("validated", False),
        }

    # ========================================================================
    # 资源管理
    # ========================================================================

    def close(self):
        if self._closed:
            return
        # 关闭内部创建的资源 (外部注入的不负责关闭)
        try:
            if self._star_selector is not None:
                self._star_selector.close()
        except Exception:
            pass
        if not self._gaia_external and self._gaia_client is not None:
            try:
                self._gaia_client.close()
            except Exception:
                pass
            self._gaia_client = None
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ============================================================================
# 工具函数
# ============================================================================

def _apply_flip(W: np.ndarray, mode: int) -> np.ndarray:
    """对 W 应用翻转模式 (与 C++ vector_matcher apply_flip 一致)

    mode: 0=无翻转, 1=X翻转, 2=Y翻转, 3=XY翻转
    W: (M, 2) 数组, 列 0=X, 列 1=Y

    返回 flip 后的 W 副本 (不修改原数组)
    """
    if mode == 0:
        return W
    W_eff = np.array(W, dtype=np.float64, copy=True)
    if mode == 1 or mode == 3:
        W_eff[:, 0] = -W_eff[:, 0]
    if mode == 2 or mode == 3:
        W_eff[:, 1] = -W_eff[:, 1]
    return W_eff


def _json_safe(obj: Any) -> Any:
    """递归将 numpy 对象转为 JSON 可序列化的 Python 原生类型"""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
