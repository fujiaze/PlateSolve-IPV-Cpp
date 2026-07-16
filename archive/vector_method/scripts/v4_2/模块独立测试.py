"""V4.2 模块独立测试 - 验证每个模块可单独调用

功能:
    1. 对 M20_T2 Red 帧先用 V42Pipeline 完整运行一次（生成所有 phase_*.json）
    2. 然后逐个模块独立加载 JSON 中间结果, 单独调用每个模块
    3. 验证每个模块的输出与完整管线运行时一致
    4. 输出测试报告

验证项:
    1. StarSelector 可单独调用 select()
    2. VectorMatcher 可单独调用 match()（用 StarSelector 输出）
    3. PairExpander 可单独调用 expand()（用 VectorMatcher 输出）
    4. PairVerifier 可单独调用 verify()（用 PairExpander 输出）
    5. WcsFitter 可单独调用 fit()（用 PairVerifier 输出）

用 M20_T2 Red 帧作为测试数据, 中间结果从 logs/v4_2/<frame>/phase_*.json 加载

用法:
    python 模块独立测试.py
"""
import os
import sys
import json
import time

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# MSYS2 MinGW DLL 路径（vector_matcher.dll 依赖 libwinpthread-1.dll）
os.environ["PATH"] = r"C:\msys64\mingw64\bin;" + os.environ.get("PATH", "")

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import numpy as np

import logging
logging.basicConfig(level=logging.WARNING, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger("V4.2模块独立测试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_2.pipeline import V42Pipeline, _apply_flip, PHASE_0, PHASE_AB, PHASE_C, PHASE_D, PHASE_E
from v4_2.star_selector import StarSelector
from v4_2.vector_matcher import VectorMatcher
from v4_2.pair_expander import PairExpander
from v4_2.pair_verifier import PairVerifier
from v4_2.wcs_fitter import WcsFitter


# ============================================================================
# 测试帧与路径
# ============================================================================

_TEST_FRAME = os.path.join(
    PROJECT_ROOT, "testdata", "lights",
    "M20_T2_flying_dutchman-20250719@004357-300S-Red.fts")
_FRAME_BASE = os.path.splitext(os.path.basename(_TEST_FRAME))[0]
_LOG_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_2", _FRAME_BASE)
_REPORT_PATH = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_2",
    "module_independent_test_report.json")


# ============================================================================
# 工具函数
# ============================================================================

def _parse_ra_hms(s):
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m / 60.0 + sec / 3600.0) * 15.0
    return float(s)


def _parse_dec_dms(s):
    s = str(s).strip()
    sign = 1.0
    if s.startswith("-"):
        sign = -1.0
        s = s[1:]
    elif s.startswith("+"):
        s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m / 60.0 + sec / 3600.0)
    return float(s)


def _load_checkpoint(phase):
    """加载指定阶段的 checkpoint JSON"""
    path = os.path.join(_LOG_DIR, f"{phase}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _arrays_close(a, b, tol=1e-6):
    """比较两个浮点数组是否接近"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return False
    return np.allclose(a, b, atol=tol, rtol=1e-5)


def _float_close(a, b, tol=1e-6):
    """比较两个浮点数是否接近"""
    return abs(float(a) - float(b)) <= tol


def _read_fits_header(fits_path):
    """读取 FITS 头, 返回 (ra, dec, focal_length, pixel_size, width, height, exptime)"""
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    exptime = getattr(img.metadata.observation, "exptime", 1.0)
    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    cra0 = _parse_ra_hms(kw_dict.get("OBJCTRA") or kw_dict.get("RA"))
    cdec0 = _parse_dec_dms(kw_dict.get("OBJCTDEC") or kw_dict.get("DEC"))
    return cra0, cdec0, fl, ps, w, h, exptime


# ============================================================================
# 测试报告
# ============================================================================

class TestReport:
    def __init__(self):
        self.tests = []

    def add(self, name, passed, detail=""):
        self.tests.append({
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  [{status}] {name}")
        if detail:
            for line in detail.strip().split("\n"):
                print(f"           {line}")

    def summary(self):
        n_pass = sum(1 for t in self.tests if t["passed"])
        n_total = len(self.tests)
        return n_pass, n_total

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        n_pass, n_total = self.summary()
        data = {
            "frame": _FRAME_BASE,
            "n_pass": n_pass,
            "n_total": n_total,
            "all_passed": n_pass == n_total,
            "tests": self.tests,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================================
# 模块独立测试
# ============================================================================

def test_star_selector(report, ra, dec, fl, ps, exptime):
    """测试 1: StarSelector 可单独调用 select()"""
    print(f"\n{'=' * 70}")
    print(f"  测试 1: StarSelector.select() 独立调用")
    print(f"{'=' * 70}")

    try:
        # 加载 checkpoint 作为对比基准
        ckpt = _load_checkpoint(PHASE_0)
        ref_U = np.array(ckpt["U"], dtype=np.float64)
        ref_W = np.array(ckpt["W"], dtype=np.float64)
        ref_meta = ckpt["meta"]

        # 独立调用 StarSelector
        gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
        gaia_client = GaiaClientPy(gaia_dir, db_type=0)
        star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")

        selector = StarSelector(
            dll_path=os.path.join(dll_dir, "star_selector", "star_selector.dll"),
            gaia_client=gaia_client,
            star_detector=star_detector,
        )

        # 独立日志目录（避免覆盖管线日志）
        indep_log_dir = os.path.join(_LOG_DIR, "independent", "phase_0")

        result = selector.select(
            image_path=_TEST_FRAME,
            ra=ra, dec=dec,
            focal_length_mm=fl, pixel_size_um=ps,
            log_dir=indep_log_dir,
        )

        selector.close()
        gaia_client.close()
        star_detector.close()

        U = result["U"]
        W = result["W"]
        meta = result["meta"]

        # 验证 U 形状
        u_shape_ok = U.shape == ref_U.shape
        report.add("StarSelector: U 形状一致", u_shape_ok,
                   f"独立={U.shape} vs 参考={ref_U.shape}")

        # 验证 W 形状
        w_shape_ok = W.shape == ref_W.shape
        report.add("StarSelector: W 形状一致", w_shape_ok,
                   f"独立={W.shape} vs 参考={ref_W.shape}")

        # 验证 U 数值一致（选星是确定性的, 应完全一致）
        if u_shape_ok:
            u_close = _arrays_close(U, ref_U, tol=1e-6)
            max_diff = float(np.max(np.abs(U - ref_U))) if u_shape_ok else 0
            report.add("StarSelector: U 数值一致", u_close,
                       f"max_diff={max_diff:.2e}")
        else:
            report.add("StarSelector: U 数值一致", False, "形状不匹配, 跳过")

        # 验证 W 数值一致（Gaia 查询顺序可能因缓存失效而不同, 用排序后比较）
        if w_shape_ok:
            # 按 (xi, eta) 字典序排序后再比较, 消除查询顺序非确定性影响
            def _sort_xy(arr):
                idx = np.lexsort((arr[:, 1], arr[:, 0]))
                return arr[idx]
            w_sorted = _sort_xy(np.asarray(W, dtype=np.float64))
            ref_w_sorted = _sort_xy(np.asarray(ref_W, dtype=np.float64))
            w_close = _arrays_close(w_sorted, ref_w_sorted, tol=1e-6)
            max_diff = float(np.max(np.abs(w_sorted - ref_w_sorted)))
            report.add("StarSelector: W 数值一致 (排序后比较)", w_close,
                       f"max_diff={max_diff:.2e}")
        else:
            report.add("StarSelector: W 数值一致 (排序后比较)", False, "形状不匹配, 跳过")

        # 验证 meta 关键字段
        meta_keys = ["s0", "fov_diag_deg", "m_lim_final", "n_gaia_final",
                     "n_img_selected", "n_gaia_selected", "converged"]
        for k in meta_keys:
            if k in ref_meta and k in meta:
                v_ref = ref_meta[k]
                v_new = meta[k]
                if isinstance(v_ref, float):
                    ok = _float_close(v_new, v_ref, tol=1e-6)
                    report.add(f"StarSelector: meta.{k} 一致", ok,
                               f"独立={v_new} vs 参考={v_ref}")
                else:
                    ok = v_new == v_ref
                    report.add(f"StarSelector: meta.{k} 一致", ok,
                               f"独立={v_new} vs 参考={v_ref}")

        return U, W, meta

    except Exception as e:
        report.add("StarSelector: 独立调用", False, f"异常: {e}")
        return None, None, None


def test_vector_matcher(report, U, W, s0):
    """测试 2: VectorMatcher 可单独调用 match()"""
    print(f"\n{'=' * 70}")
    print(f"  测试 2: VectorMatcher.match() 独立调用")
    print(f"{'=' * 70}")

    try:
        # 加载 checkpoint 作为对比基准
        ckpt = _load_checkpoint(PHASE_AB)
        ref_s = ckpt["s"]
        ref_theta = ckpt["theta"]
        ref_n_pairs = ckpt["n_pairs"]
        ref_best_mode = ckpt["best_mode"]
        ref_success = ckpt["success"]

        # 独立调用 VectorMatcher
        dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")
        vm = VectorMatcher(dll_path=os.path.join(dll_dir, "vector_matcher", "vector_matcher.dll"))

        indep_log_dir = os.path.join(_LOG_DIR, "independent", "phase_ab")
        os.makedirs(indep_log_dir, exist_ok=True)
        vm_log = os.path.join(indep_log_dir, "phase_ab_vector_matcher.log")

        result = vm.match(
            U=U, W=W,
            s0=s0,
            s_min=0.9, s_max=1.1,
            log_file_path=vm_log,
        )

        vm.close()

        # 验证 success 标志
        success_ok = result["success"] == ref_success
        report.add("VectorMatcher: success 一致", success_ok,
                   f"独立={result['success']} vs 参考={ref_success}")

        if not result["success"]:
            report.add("VectorMatcher: 独立调用失败, 跳过后续比较", False,
                       f"error={result.get('error', '')}")
            return result

        # 验证 s（PROSAC 有随机性, 用容差比较）
        s_ok = _float_close(result["s"], ref_s, tol=0.05)
        report.add("VectorMatcher: s 一致 (容差0.05)", s_ok,
                   f"独立={result['s']:.6f} vs 参考={ref_s:.6f}")

        # 验证 theta（弧度, 容差 0.01 rad ≈ 0.57°）
        theta_ok = _float_close(result["theta"], ref_theta, tol=0.01)
        report.add("VectorMatcher: theta 一致 (容差0.01rad)", theta_ok,
                   f"独立={result['theta']:.6f} vs 参考={ref_theta:.6f}")

        # 验证 best_mode
        mode_ok = result["best_mode"] == ref_best_mode
        report.add("VectorMatcher: best_mode 一致", mode_ok,
                   f"独立={result['best_mode']} vs 参考={ref_best_mode}")

        # 验证 n_pairs（PROSAC 随机性可能导致对数略有不同, 容差 ±10）
        n_pairs_ok = abs(result["n_pairs"] - ref_n_pairs) <= 10
        report.add("VectorMatcher: n_pairs 接近 (容差±10)", n_pairs_ok,
                   f"独立={result['n_pairs']} vs 参考={ref_n_pairs}")

        return result

    except Exception as e:
        report.add("VectorMatcher: 独立调用", False, f"异常: {e}")
        return None


def test_pair_expander(report, U, W, vm_result, s0, img_width, img_height):
    """测试 3: PairExpander 可单独调用 expand()"""
    print(f"\n{'=' * 70}")
    print(f"  测试 3: PairExpander.expand() 独立调用")
    print(f"{'=' * 70}")

    try:
        # 加载 checkpoint 作为对比基准
        ckpt = _load_checkpoint(PHASE_C)
        ref_expand_u = ckpt["expand_u"]
        ref_expand_w = ckpt["expand_w"]
        ref_n_pairs = ckpt["n_pairs"]
        ref_n_expanded = ckpt["n_expanded"]
        ref_success = ckpt["success"]

        # 应用 flip 生成 W_eff（与 pipeline 一致）
        best_mode = int(vm_result["best_mode"])
        W_eff = _apply_flip(W, best_mode)

        # 构造 T 变换
        T = {
            "s": vm_result["s"],
            "theta": vm_result["theta"],
            "tx": vm_result["tx"],
            "ty": vm_result["ty"],
        }

        # 独立调用 PairExpander
        dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")
        pe = PairExpander(dll_path=os.path.join(dll_dir, "pair_expander", "pair_expander.dll"))

        indep_log_dir = os.path.join(_LOG_DIR, "independent", "phase_c")

        result = pe.expand(
            U=U, W=W_eff,
            T=T,
            init_cu=vm_result["cu"],
            init_cw=vm_result["cw"],
            s0=s0,
            img_width=img_width, img_height=img_height,
            log_dir=indep_log_dir,
        )

        # PairExpander 无 close() 方法, DLL 句柄在析构时自动释放
        if hasattr(pe, "close"):
            pe.close()

        # 验证 success
        success_ok = result["success"] == ref_success
        report.add("PairExpander: success 一致", success_ok,
                   f"独立={result['success']} vs 参考={ref_success}")

        if not result["success"]:
            report.add("PairExpander: 独立调用失败, 跳过后续比较", False,
                       f"error={result.get('meta', {}).get('error', '')}")
            return result

        # 验证 n_pairs（输入 T 略有不同可能导致对数不同, 容差 ±20）
        n_pairs_ok = abs(result["n_pairs"] - ref_n_pairs) <= 20
        report.add("PairExpander: n_pairs 接近 (容差±20)", n_pairs_ok,
                   f"独立={result['n_pairs']} vs 参考={ref_n_pairs}")

        # 验证 n_expanded
        n_expanded_ok = abs(result["n_expanded"] - ref_n_expanded) <= 20
        report.add("PairExpander: n_expanded 接近 (容差±20)", n_expanded_ok,
                   f"独立={result['n_expanded']} vs 参考={ref_n_expanded}")

        # 验证 expand_u/expand_w 列表（如果输入 T 相同, 应完全一致）
        # 由于 VectorMatcher 的 s/theta 可能有微小差异, 这里用集合交集比较
        ref_pairs_set = set(zip(ref_expand_u, ref_expand_w))
        new_pairs_set = set(zip(result["expand_u"], result["expand_w"]))
        if ref_pairs_set and new_pairs_set:
            overlap = len(ref_pairs_set & new_pairs_set)
            overlap_ratio = overlap / max(len(ref_pairs_set), 1)
            overlap_ok = overlap_ratio >= 0.8
            report.add("PairExpander: 匹配对重叠率 ≥ 80%", overlap_ok,
                       f"重叠={overlap}/{len(ref_pairs_set)} ({overlap_ratio*100:.1f}%)")
        else:
            report.add("PairExpander: 匹配对重叠率", False,
                       f"独立对数={len(new_pairs_set)} 参考对数={len(ref_pairs_set)}")

        return result

    except Exception as e:
        report.add("PairExpander: 独立调用", False, f"异常: {e}")
        return None


def test_pair_verifier(report, U, W, vm_result, pe_result, s0, fov_diag_deg):
    """测试 4: PairVerifier 可单独调用 verify()"""
    print(f"\n{'=' * 70}")
    print(f"  测试 4: PairVerifier.verify() 独立调用")
    print(f"{'=' * 70}")

    try:
        # 加载 checkpoint 作为对比基准
        ckpt = _load_checkpoint(PHASE_D)
        ref_n_clean = ckpt["n_clean"]
        ref_validated = ckpt["validated"]
        ref_bayes_lnK = ckpt["bayes"]["lnK"]
        ref_tri_ratio = ckpt["triangle"]["pass_ratio"]
        ref_success = ckpt["success"]

        # 应用 flip 生成 W_eff
        best_mode = int(vm_result["best_mode"])
        W_eff = _apply_flip(W, best_mode)

        # 构造 pairs [[u, w], ...]
        pairs = list(zip(pe_result["expand_u"], pe_result["expand_w"]))
        pairs = [[int(u), int(w)] for u, w in pairs]

        # 独立调用 PairVerifier
        dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")
        pv = PairVerifier(dll_path=os.path.join(dll_dir, "pair_verifier", "pair_verifier.dll"))

        indep_log_dir = os.path.join(_LOG_DIR, "independent", "phase_d")

        result = pv.verify(
            U=U, W=W_eff,
            pairs=pairs,
            s0=s0,
            fov_diag_deg=fov_diag_deg,
            log_dir=indep_log_dir,
        )

        # PairVerifier 无 close() 方法, DLL 句柄在析构时自动释放
        if hasattr(pv, "close"):
            pv.close()

        # 验证 success
        success_ok = result["success"] == ref_success
        report.add("PairVerifier: success 一致", success_ok,
                   f"独立={result['success']} vs 参考={ref_success}")

        if not result["success"]:
            report.add("PairVerifier: 独立调用失败, 跳过后续比较", False,
                       f"error={result.get('meta', {}).get('error', '')}")
            return result

        # 验证 validated
        validated_ok = result["validated"] == ref_validated
        report.add("PairVerifier: validated 一致", validated_ok,
                   f"独立={result['validated']} vs 参考={ref_validated}")

        # 验证 n_clean（输入对数可能不同, 容差 ±20）
        n_clean_ok = abs(result["n_clean"] - ref_n_clean) <= 20
        report.add("PairVerifier: n_clean 接近 (容差±20)", n_clean_ok,
                   f"独立={result['n_clean']} vs 参考={ref_n_clean}")

        # 验证 bayes lnK（容差 ±50, 因为输入对数可能不同）
        lnK_ok = abs(result["bayes"]["lnK"] - ref_bayes_lnK) <= 50
        report.add("PairVerifier: bayes lnK 接近 (容差±50)", lnK_ok,
                   f"独立={result['bayes']['lnK']:.2f} vs 参考={ref_bayes_lnK:.2f}")

        # 验证 triangle pass_ratio（容差 ±0.1）
        tri_ok = abs(result["triangle"]["pass_ratio"] - ref_tri_ratio) <= 0.1
        report.add("PairVerifier: triangle pass_ratio 接近 (容差±0.1)", tri_ok,
                   f"独立={result['triangle']['pass_ratio']:.3f} vs 参考={ref_tri_ratio:.3f}")

        return result

    except Exception as e:
        report.add("PairVerifier: 独立调用", False, f"异常: {e}")
        return None


def test_wcs_fitter(report, U, W, vm_result, pv_result, ra, dec, fl, ps, img_width, img_height):
    """测试 5: WcsFitter 可单独调用 fit()"""
    print(f"\n{'=' * 70}")
    print(f"  测试 5: WcsFitter.fit() 独立调用")
    print(f"{'=' * 70}")

    try:
        # 加载 checkpoint 作为对比基准
        ckpt = _load_checkpoint(PHASE_E)
        ref_rms_px = ckpt["rms_px"]
        ref_n_pairs = ckpt["n_pairs"]
        ref_sip_order = ckpt["sip_order"]
        ref_success = ckpt["success"]

        # 应用 flip 生成 W_eff
        best_mode = int(vm_result["best_mode"])
        W_eff = _apply_flip(W, best_mode)

        # 使用 PairVerifier 清洗后的 pairs
        pairs = pv_result["pairs"]

        # 独立调用 WcsFitter
        dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")
        wf = WcsFitter(dll_path=os.path.join(dll_dir, "wcs_fitter", "wcs_fitter.dll"))

        indep_log_dir = os.path.join(_LOG_DIR, "independent", "phase_e")

        result = wf.fit(
            U=U, W=W_eff,
            pairs=pairs,
            ra=ra, dec=dec,
            focal_length_mm=fl, pixel_size_um=ps,
            img_width=img_width, img_height=img_height,
            log_dir=indep_log_dir,
        )

        wf.close()

        # 验证 success
        success_ok = result["success"] == ref_success
        report.add("WcsFitter: success 一致", success_ok,
                   f"独立={result['success']} vs 参考={ref_success}")

        if not result["success"]:
            report.add("WcsFitter: 独立调用失败, 跳过后续比较", False,
                       f"error={result.get('error', '')}")
            return result

        # 验证 n_pairs（应与 PairVerifier 输出一致）
        n_pairs_ok = result["n_pairs"] == ref_n_pairs
        report.add("WcsFitter: n_pairs 一致", n_pairs_ok,
                   f"独立={result['n_pairs']} vs 参考={ref_n_pairs}")

        # 验证 rms_px（输入对数可能不同, 容差 ±1.0px）
        rms_ok = abs(result["rms_px"] - ref_rms_px) <= 1.0
        report.add("WcsFitter: rms_px 接近 (容差±1.0px)", rms_ok,
                   f"独立={result['rms_px']:.4f} vs 参考={ref_rms_px:.4f}")

        # 验证 sip_order
        sip_ok = result["sip_order"] == ref_sip_order
        report.add("WcsFitter: sip_order 一致", sip_ok,
                   f"独立={result['sip_order']} vs 参考={ref_sip_order}")

        return result

    except Exception as e:
        report.add("WcsFitter: 独立调用", False, f"异常: {e}")
        return None


# ============================================================================
# 主流程
# ============================================================================

def main():
    print(f"=== V4.2 模块独立测试 ===")
    print(f"测试帧: {_FRAME_BASE}")
    print(f"日志目录: {_LOG_DIR}")

    report = TestReport()

    # Step 0: 确保完整管线的 phase_*.json 已存在
    print(f"\n--- Step 0: 确保完整管线 checkpoint 存在 ---")
    needed_phases = [PHASE_0, PHASE_AB, PHASE_C, PHASE_D, PHASE_E]
    missing = [p for p in needed_phases if not os.path.exists(os.path.join(_LOG_DIR, f"{p}.json"))]

    if missing:
        print(f"  缺失 checkpoint: {missing}")
        print(f"  运行完整管线生成 checkpoint...")
        cra0, cdec0, fl, ps, w, h, exptime = _read_fits_header(_TEST_FRAME)
        gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
        gaia_client = GaiaClientPy(gaia_dir, db_type=0)
        star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        dll_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_2")
        pipeline = V42Pipeline(
            dll_dir=dll_dir,
            gaia_client=gaia_client,
            star_detector=star_detector,
        )
        result = pipeline.solve(
            image_path=_TEST_FRAME,
            ra=cra0, dec=cdec0,
            focal_length_mm=fl, pixel_size_um=ps,
            log_dir=_LOG_DIR,
            resume=True,
        )
        pipeline.close()
        gaia_client.close()
        star_detector.close()
        if not result.get("success"):
            print(f"  完整管线运行失败: {result.get('error')}")
            report.add("完整管线运行", False, f"error={result.get('error')}")
            report.save(_REPORT_PATH)
            return 1
        print(f"  完整管线运行成功: RMS={result.get('rms_px'):.4f}px")
    else:
        print(f"  所有 checkpoint 已存在, 跳过完整管线运行")

    # 读取 FITS 头
    cra0, cdec0, fl, ps, w, h, exptime = _read_fits_header(_TEST_FRAME)
    s0 = 206.265 * ps / fl
    print(f"\n  帧参数: RA={cra0:.6f}° Dec={cdec0:.6f}° fl={fl}mm ps={ps}um s0={s0:.4f}\"/px")

    # 读取 Phase 0 checkpoint 获取 fov_diag_deg
    ckpt_p0 = _load_checkpoint(PHASE_0)
    fov_diag_deg = ckpt_p0["meta"].get("fov_diag_deg", 2.0)
    img_width = ckpt_p0["meta"].get("img_width", w)
    img_height = ckpt_p0["meta"].get("img_height", h)

    # 测试 1: StarSelector
    U, W, meta = test_star_selector(report, cra0, cdec0, fl, ps, exptime)

    # 如果 StarSelector 失败, 用 checkpoint 中的 U/W 继续
    if U is None or W is None:
        print(f"\n  StarSelector 独立调用失败, 使用 checkpoint 中的 U/W 继续后续测试")
        U = np.array(ckpt_p0["U"], dtype=np.float64)
        W = np.array(ckpt_p0["W"], dtype=np.float64)

    # 测试 2: VectorMatcher
    vm_result = test_vector_matcher(report, U, W, s0)

    # 如果 VectorMatcher 失败, 用 checkpoint 中的结果继续
    if vm_result is None or not vm_result.get("success"):
        print(f"\n  VectorMatcher 独立调用失败, 使用 checkpoint 中的结果继续")
        ckpt_ab = _load_checkpoint(PHASE_AB)
        vm_result = ckpt_ab

    # 测试 3: PairExpander
    pe_result = test_pair_expander(report, U, W, vm_result, s0, img_width, img_height)

    # 如果 PairExpander 失败, 用 checkpoint 中的结果继续
    if pe_result is None or not pe_result.get("success"):
        print(f"\n  PairExpander 独立调用失败, 使用 checkpoint 中的结果继续")
        ckpt_c = _load_checkpoint(PHASE_C)
        pe_result = {
            "success": ckpt_c["success"],
            "expand_u": ckpt_c["expand_u"],
            "expand_w": ckpt_c["expand_w"],
            "n_pairs": ckpt_c["n_pairs"],
            "n_expanded": ckpt_c["n_expanded"],
        }

    # 测试 4: PairVerifier
    pv_result = test_pair_verifier(report, U, W, vm_result, pe_result, s0, fov_diag_deg)

    # 如果 PairVerifier 失败, 用 checkpoint 中的结果继续
    if pv_result is None or not pv_result.get("success"):
        print(f"\n  PairVerifier 独立调用失败, 使用 checkpoint 中的结果继续")
        ckpt_d = _load_checkpoint(PHASE_D)
        pv_result = {
            "success": ckpt_d["success"],
            "validated": ckpt_d["validated"],
            "pairs": ckpt_d["pairs"],
            "n_clean": ckpt_d["n_clean"],
            "bayes": ckpt_d["bayes"],
            "triangle": ckpt_d["triangle"],
        }

    # 测试 5: WcsFitter
    test_wcs_fitter(report, U, W, vm_result, pv_result,
                    cra0, cdec0, fl, ps, img_width, img_height)

    # 汇总
    n_pass, n_total = report.summary()
    print(f"\n{'=' * 70}")
    print(f"  模块独立测试汇总")
    print(f"{'=' * 70}")
    print(f"  通过: {n_pass}/{n_total}")
    print(f"  结果: {'✓ 全部通过' if n_pass == n_total else '✗ 有失败项'}")

    report.save(_REPORT_PATH)
    print(f"\n  报告保存: {_REPORT_PATH}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
