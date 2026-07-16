"""相对向量法 V4.4 实际数据验证

对比 V4.3 (单θ Phase A) vs V4.4 (相对向量法 Phase A) 在 4 帧上的表现:
  1. Galaxy_Center_mosaic2 - Type3失败帧 (t≠0, 单θ SNR<5)
  2. NGC7293 - Bug4修复帧
  3. LDN43 - Bug4修复帧 (U=271 候选爆炸)
  4. Galaxy_Center_mosaic1 - 成功帧对比

验证目标:
  - V4.4 相对向量法 SNR vs V4.3 单θ SNR
  - V4.4 RMS / matched_count / success 与 V4.3 对比
  - V4.4 在 t≠0 场景能否成功 (V4.3 失败的帧)
"""
import os
import sys
import time
import json
import functools

print = functools.partial(print, flush=True)  # 强制无缓冲

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_4", "validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_MINGW_BIN)
    except Exception:
        pass

sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2 import GaiaClientPy
from star_detector import StarDetector, SDetParamsPy
from v4_3.vector_match_v4_3_cpp import V43Solver
from v4_4.vector_match_v4_4_cpp import V44Solver

TESTDATA = os.path.join(PROJECT_ROOT, "testdata")


def find_fits_path(filename):
    """在 testdata 目录递归查找 FITS 文件"""
    for dirpath, _, filenames in os.walk(TESTDATA):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def test_frame(fits_path, solver_v43, solver_v44):
    """对单帧对比 V4.3 和 V4.4 求解结果

    Returns:
        dict: 包含 v43 和 v44 的求解结果 + 耗时
    """
    base = os.path.basename(fits_path)
    print(f"\n{'='*70}")
    print(f"=== {base} ===")
    print(f"{'='*70}")

    # --- V4.3 求解 ---
    log_dir_v43 = os.path.join(OUTPUT_DIR, "v43_logs", os.path.splitext(base)[0])
    t0 = time.time()
    try:
        result_v43 = solver_v43.solve(
            image_path=fits_path,
            ra=solver_v43_solver_ra(fits_path),
            dec=solver_v43_solver_dec(fits_path),
            focal_length_mm=solver_v43_focal(fits_path),
            pixel_size_um=solver_v43_pixel(fits_path),
            log_dir=log_dir_v43,
        )
    except Exception as e:
        print(f"V4.3 异常: {e}")
        result_v43 = {"success": False, "error": str(e)}
    t_v43 = time.time() - t0

    # --- V4.4 求解 ---
    log_dir_v44 = os.path.join(OUTPUT_DIR, "v44_logs", os.path.splitext(base)[0])
    t0 = time.time()
    try:
        result_v44 = solver_v44.solve(
            image_path=fits_path,
            ra=solver_v43_solver_ra(fits_path),
            dec=solver_v43_solver_dec(fits_path),
            focal_length_mm=solver_v43_focal(fits_path),
            pixel_size_um=solver_v43_pixel(fits_path),
            log_dir=log_dir_v44,
        )
    except Exception as e:
        print(f"V4.4 异常: {e}")
        result_v44 = {"success": False, "error": str(e)}
    t_v44 = time.time() - t0

    # --- 打印对比 ---
    print(f"\n--- V4.3 (单θ Phase A) ---")
    _print_result(result_v43, t_v43)

    print(f"\n--- V4.4 (相对向量法 Phase A) ---")
    _print_result(result_v44, t_v44)

    # --- 对比摘要 ---
    v43_snr = result_v43.get("theta_snr", 0)
    v44_snr = result_v44.get("theta_snr", 0)
    v43_ok = result_v43.get("success", False)
    v44_ok = result_v44.get("success", False)
    snr_ratio = v44_snr / max(v43_snr, 0.01)
    print(f"\n--- 对比 ---")
    print(f"  success:    V4.3={v43_ok}  V4.4={v44_ok}")
    print(f"  θ SNR:      V4.3={v43_snr:.2f}x  V4.4={v44_snr:.2f}x  (比值={snr_ratio:.2f})")
    print(f"  RMS(px):    V4.3={result_v43.get('rms_px', 0):.4f}  V4.4={result_v44.get('rms_px', 0):.4f}")
    print(f"  matched:    V4.3={result_v43.get('matched_count', 0)}  V4.4={result_v44.get('matched_count', 0)}")
    print(f"  耗时(s):    V4.3={t_v43:.2f}  V4.4={t_v44:.2f}")

    return {
        "v43": _extract_summary(result_v43, t_v43),
        "v44": _extract_summary(result_v44, t_v44),
    }


def _print_result(r, elapsed):
    if not r.get("success", False):
        print(f"  失败: {r.get('error', '未知')}")
        return
    print(f"  θ_peak={r.get('theta_peak_deg', 0):.2f}°  SNR={r.get('theta_snr', 0):.2f}x")
    print(f"  mode={r.get('flip_mode', -1)}  s={r.get('s', 0):.4f}  θ={r.get('rotation_deg', 0):.2f}°")
    print(f"  tx={r.get('tx', 0):.1f}\"  ty={r.get('ty', 0):.1f}\"")
    print(f"  RMS={r.get('rms_px', 0):.4f}px  S_robust={r.get('s_robust', 0):.4f}\"")
    print(f"  matched={r.get('matched_count', 0)}  n_inliers={r.get('n_inliers', 0)}")
    print(f"  IRM: iter={r.get('n_iters', 0)}  converged={r.get('irm_converged', False)}")
    print(f"  lnK={r.get('bayes_lnK', 0):.1f}  triangle={r.get('triangle_pass_ratio', 0):.2f}")
    print(f"  耗时: {elapsed:.2f}s")


def _extract_summary(r, elapsed):
    return {
        "success": r.get("success", False),
        "theta_snr": r.get("theta_snr", 0),
        "theta_peak_deg": r.get("theta_peak_deg", 0),
        "rms_px": r.get("rms_px", 0),
        "s_robust": r.get("s_robust", 0),
        "matched_count": r.get("matched_count", 0),
        "n_inliers": r.get("n_inliers", 0),
        "n_iters": r.get("n_iters", 0),
        "irm_converged": r.get("irm_converged", False),
        "bayes_lnK": r.get("bayes_lnK", 0),
        "triangle_pass_ratio": r.get("triangle_pass_ratio", 0),
        "flip_mode": r.get("flip_mode", -1),
        "rotation_deg": r.get("rotation_deg", 0),
        "s": r.get("s", 0),
        "tx": r.get("tx", 0),
        "ty": r.get("ty", 0),
        "elapsed_sec": elapsed,
        "error": r.get("error", ""),
    }


# ============================================================================
# 从 FITS 头读取参数 (与 V4.3 实测脚本一致)
# ============================================================================
from astro_image_io import ImageReader


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


# 缓存 FITS 元数据 (避免重复读取)
_FITS_CACHE = {}


def _get_fits_meta(fits_path):
    if fits_path not in _FITS_CACHE:
        reader = ImageReader()
        img = reader.read(fits_path)
        kws = img.keywords
        kw_dict = {k.name.upper(): k.value for k in kws}
        obj_ra_str = kw_dict.get("OBJCTRA") or kw_dict.get("RA")
        obj_dec_str = kw_dict.get("OBJCTDEC") or kw_dict.get("DEC")
        _FITS_CACHE[fits_path] = {
            "ra": _parse_ra_hms(obj_ra_str),
            "dec": _parse_dec_dms(obj_dec_str),
            "focal": img.metadata.observation.focallen,
            "pixel": img.metadata.observation.xpixsz,
        }
    return _FITS_CACHE[fits_path]


def solver_v43_solver_ra(fits_path):
    return _get_fits_meta(fits_path)["ra"]


def solver_v43_solver_dec(fits_path):
    return _get_fits_meta(fits_path)["dec"]


def solver_v43_focal(fits_path):
    return _get_fits_meta(fits_path)["focal"]


def solver_v43_pixel(fits_path):
    return _get_fits_meta(fits_path)["pixel"]


# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 70)
    print("相对向量法 V4.4 实际数据验证")
    print("对比 V4.3 (单θ Phase A) vs V4.4 (相对向量法 Phase A)")
    print("=" * 70)

    # 初始化共享客户端 (V4.3 和 V4.4 共用同一 GaiaClient 和 StarDetector)
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=0)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))

    solver_v43 = V43Solver(gaia_client=gaia_client, star_detector=star_detector)
    solver_v44 = V44Solver(gaia_client=gaia_client, star_detector=star_detector)

    # 测试帧 (与 V4.3 实测脚本一致)
    test_frames = [
        # Type3 失败帧 (t≠0, 单θ SNR<5)
        "Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",
        "NGC7293_T2_HO_flying_dutchman-20250706@081055-1200S-H-alpha.fts",
        "LDN43_LRGBH_flying_dutchman-20250520@022502-600S-Lum.fts",
        # 成功帧对比
        "Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
    ]

    all_results = {}
    for filename in test_frames:
        fits_path = find_fits_path(filename)
        if not fits_path:
            print(f"\n!! 未找到: {filename}")
            continue
        try:
            result = test_frame(fits_path, solver_v43, solver_v44)
            all_results[filename] = result
        except Exception as e:
            import traceback
            print(f"\n!! 测试失败 {filename}: {e}")
            print(traceback.format_exc())
            all_results[filename] = {"error": str(e)}

    # --- 总结表 ---
    print("\n\n" + "=" * 110)
    print("总结: V4.3 (单θ) vs V4.4 (相对向量法)")
    print("=" * 110)
    header = f"{'帧':<50} {'V4.3SNR':<10} {'V4.4SNR':<10} {'V4.3OK':<7} {'V4.4OK':<7} {'V4.3RMS':<10} {'V4.4RMS':<10} {'V4.3t':<7} {'V4.4t':<7}"
    print(header)
    print("-" * 110)
    for fn, r in all_results.items():
        if "error" in r:
            print(f"{os.path.basename(fn)[:49]:<50} ERROR: {r['error'][:50]}")
            continue
        v43 = r["v43"]
        v44 = r["v44"]
        print(f"{os.path.basename(fn)[:49]:<50} "
              f"{v43['theta_snr']:<10.2f} {v44['theta_snr']:<10.2f} "
              f"{str(v43['success']):<7} {str(v44['success']):<7} "
              f"{v43['rms_px']:<10.4f} {v44['rms_px']:<10.4f} "
              f"{v43['elapsed_sec']:<7.2f} {v44['elapsed_sec']:<7.2f}")
    print("=" * 110)

    # --- 改善分析 ---
    print("\n--- 改善分析 ---")
    n_v43_ok = sum(1 for r in all_results.values() if "v43" in r and r["v43"]["success"])
    n_v44_ok = sum(1 for r in all_results.values() if "v44" in r and r["v44"]["success"])
    print(f"成功率: V4.3={n_v43_ok}/{len(all_results)}  V4.4={n_v44_ok}/{len(all_results)}")

    snr_improvements = []
    for fn, r in all_results.items():
        if "v43" in r and "v44" in r and r["v43"]["success"] and r["v44"]["success"]:
            v43_snr = r["v43"]["theta_snr"]
            v44_snr = r["v44"]["theta_snr"]
            if v43_snr > 0.1:
                ratio = v44_snr / v43_snr
                snr_improvements.append((fn, v43_snr, v44_snr, ratio))
                print(f"  {os.path.basename(fn)[:40]}: SNR {v43_snr:.2f}→{v44_snr:.2f} (×{ratio:.2f})")

    # --- 保存结果 ---
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {summary_path}")

    # 关闭
    solver_v43.close()
    solver_v44.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n脚本异常: {e}")
        print(traceback.format_exc())
