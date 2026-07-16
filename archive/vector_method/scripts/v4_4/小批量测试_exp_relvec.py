"""V4.4 exp_relvec 独立实验 - 小批量真实数据测试

测试新的 exp_relvec.dll (3D 密度场 + θ 过滤 + RANSAC 一步求解)
复用 V4.4 的 GaiaClientPy + StarDetector 基础设施注入句柄

输出:
    lib/plate_solve/logs/v4_4/exp_relvec_test/batch_summary.csv
    lib/plate_solve/logs/v4_4/exp_relvec_test/<frame>/result_summary.csv

用法:
    py 小批量测试_exp_relvec.py
"""
import os
import sys
import csv
import time
import re
import ctypes
import functools

print = functools.partial(print, flush=True)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# MinGW 运行时 DLL 路径
_MINGW_BIN = r"C:\msys64\mingw64\bin"
os.environ["PATH"] = _MINGW_BIN + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(_MINGW_BIN)
    except (OSError, FileNotFoundError):
        pass

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import logging
logging.basicConfig(level=logging.WARNING)

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy


# ============================================================================
# exp_relvec.dll 接口
# ============================================================================

class ExpResultSummary(ctypes.Structure):
    """匹配 C 端 exp_dll.cpp 的 ExpResultSummary 结构体"""
    _fields_ = [
        ("success", ctypes.c_int),
        ("ransac_theta_deg", ctypes.c_double),
        ("ransac_tx", ctypes.c_double),
        ("ransac_ty", ctypes.c_double),
        ("ransac_s", ctypes.c_double),
        ("n_inliers", ctypes.c_int),
        ("n_ransac_iters", ctypes.c_int),
        ("ransac_rms", ctypes.c_double),
        ("snr_final", ctypes.c_double),
        ("n_samples_actual", ctypes.c_int),
        ("n_passed", ctypes.c_int),
        ("n_focused", ctypes.c_int),
        ("err_theta_deg", ctypes.c_double),
        ("err_tx", ctypes.c_double),
        ("err_ty", ctypes.c_double),
        ("err_s", ctypes.c_double),
        ("error_msg", ctypes.c_char * 256),
    ]


class ExpRelvecSolver:
    """exp_relvec.dll 的 Python 封装"""

    def __init__(self, dll_path, gaia_client, star_detector):
        if not os.path.exists(dll_path):
            raise FileNotFoundError(f"exp_relvec.dll 未找到: {dll_path}")

        self._dll = ctypes.CDLL(dll_path)
        self._setup_argtypes()

        # 注入句柄
        gaia_handle = getattr(gaia_client, "_handle", None)
        if not gaia_handle:
            raise RuntimeError("GaiaClientPy._handle 为空")
        gaia_handle_int = gaia_handle if isinstance(gaia_handle, int) else gaia_handle
        self._dll.exp_set_gaia_client(ctypes.c_void_p(gaia_handle_int))
        print(f"  注入 GaiaClient handle=0x{gaia_handle_int:x}")

        sdet_handle = getattr(star_detector, "_handle", None)
        if not sdet_handle:
            raise RuntimeError("StarDetector._handle 为空")
        sdet_handle_int = sdet_handle if isinstance(sdet_handle, int) else sdet_handle
        self._dll.exp_set_star_detector(ctypes.c_void_p(sdet_handle_int))
        print(f"  注入 StarDetector handle=0x{sdet_handle_int:x}")

    def _setup_argtypes(self):
        self._dll.exp_set_gaia_client.argtypes = [ctypes.c_void_p]
        self._dll.exp_set_gaia_client.restype = None
        self._dll.exp_set_star_detector.argtypes = [ctypes.c_void_p]
        self._dll.exp_set_star_detector.restype = None

        self._dll.exp_solve.argtypes = [
            ctypes.c_char_p,   # image_path
            ctypes.c_double,   # ra
            ctypes.c_double,   # dec
            ctypes.c_double,   # focal_length_mm
            ctypes.c_double,   # pixel_size_um
            ctypes.c_char_p,   # output_dir
            ctypes.POINTER(ExpResultSummary),
        ]
        self._dll.exp_solve.restype = ctypes.c_int

        self._dll.exp_solve_synthetic.argtypes = [
            ctypes.c_int,      # seed
            ctypes.c_int,      # n_stars
            ctypes.c_double,   # theta_true_deg
            ctypes.c_char_p,   # output_dir
            ctypes.POINTER(ExpResultSummary),
        ]
        self._dll.exp_solve_synthetic.restype = ctypes.c_int

    def solve(self, image_path, ra, dec, focal_mm, pixel_um, output_dir):
        summary = ExpResultSummary()
        rc = self._dll.exp_solve(
            image_path.encode("utf-8"),
            ctypes.c_double(ra), ctypes.c_double(dec),
            ctypes.c_double(focal_mm), ctypes.c_double(pixel_um),
            output_dir.encode("utf-8"),
            ctypes.byref(summary),
        )
        return rc, summary

    def solve_synthetic(self, seed, n_stars, theta_true_deg, output_dir):
        summary = ExpResultSummary()
        rc = self._dll.exp_solve_synthetic(
            ctypes.c_int(seed), ctypes.c_int(n_stars),
            ctypes.c_double(theta_true_deg),
            output_dir.encode("utf-8"),
            ctypes.byref(summary),
        )
        return rc, summary


# ============================================================================
# 工具函数 (复用 V4.4 批量测试脚本)
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


def get_fits_meta(fits_path):
    """读取 FITS 元数据 (RA/DEC/焦距/像元)"""
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    kw_dict = {k.name.upper(): k.value for k in img.keywords}
    obj_ra_str = kw_dict.get("OBJCTRA") or kw_dict.get("RA")
    obj_dec_str = kw_dict.get("OBJCTDEC") or kw_dict.get("DEC")
    if not obj_ra_str or not obj_dec_str:
        return None
    cra = _parse_ra_hms(obj_ra_str)
    cdec = _parse_dec_dms(obj_dec_str)
    return {
        "width": w, "height": h,
        "focallen": fl, "pixel_size": ps,
        "ra": cra, "dec": cdec,
        "s0": 206.265 * ps / fl,
    }


# ============================================================================
# 小批量测试帧 (10 帧: 代表性目标)
# ============================================================================

TEST_FRAMES = [
    # Galaxy_Center mosaic1/2 (正常, 密集星场)
    r"testdata\lights1\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
    r"testdata\lights1\panel2\Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@002647-180S-Red.fts",
    # LDN43 (曾异常: θ≈±90° 镜像歧义)
    r"testdata\lights\LDN43_LRGBH_flying_dutchman-20250503@031525-600S-Lum.fts",
    r"testdata\lights\LDN43_LRGBH_flying_dutchman-20250520@022502-600S-Lum.fts",
    # M20 (曾异常: 多帧 RMS 异常)
    r"testdata\lights\M20_T2_flying_dutchman-20250701@073331-300S-Red.fts",
    r"testdata\lights\M20_T2_flying_dutchman-20250719@004357-300S-Red.fts",
    r"testdata\lights\M20_T2_flying_dutchman-20250813@021541-300S-Red.fts",
    # NGC247 (窄场)
    r"testdata\lights\NGC247_T2_flying_dutchman-20250816@033428-600S-Lum.fts",
    # NGC55 (南天窄场)
    r"testdata\lights\NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts",
    # LDN43 H-alpha (窄带)
    r"testdata\lights\LDN43_LRGBH_flying_dutchman-20250503@042947-1200S-H-alpha.fts",
]


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("V4.4 exp_relvec 独立实验 - 小批量真实数据测试 (10 帧)")
    print("架构: 3D 密度场(定位θ) → θ过滤提取点对 → RANSAC + Umeyama SVD")
    print("=" * 70)

    # 初始化 GaiaClient + StarDetector
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    print(f"初始化 GaiaClient (GaiaDR3: {gaia_dir})...")
    gaia_client = GaiaClientPy(gaia_dir, db_type=0)
    print("初始化 StarDetector (fitRadius=0 自动)...")
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))

    # 加载 exp_relvec.dll
    dll_path = os.path.join(
        PROJECT_ROOT, "lib", "plate_solve", "cpp", "v4_4",
        "experiment", "exp_relvec.dll"
    )
    print(f"加载 exp_relvec.dll: {dll_path}")
    solver = ExpRelvecSolver(dll_path, gaia_client, star_detector)
    print("DLL 加载完成, 句柄已注入")

    # 输出目录
    output_root = os.path.join(
        PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_4", "exp_relvec_test"
    )
    os.makedirs(output_root, exist_ok=True)

    # 求解
    results = []
    t_total_start = time.time()
    for i, rel_path in enumerate(TEST_FRAMES):
        fits_path = os.path.join(PROJECT_ROOT, rel_path)
        base = os.path.basename(fits_path)
        frame_name = os.path.splitext(base)[0]
        print(f"\n[{i+1}/{len(TEST_FRAMES)}] {base}")

        if not os.path.exists(fits_path):
            print(f"  SKIP  文件不存在")
            results.append({"frame": base, "status": "skip_no_file"})
            continue

        t_start = time.time()
        try:
            meta = get_fits_meta(fits_path)
            if meta is None:
                print(f"  FAIL  无 OBJCTRA/OBJCTDEC")
                results.append({"frame": base, "status": "fail_no_objctra"})
                continue

            frame_out_dir = os.path.join(output_root, frame_name)
            os.makedirs(frame_out_dir, exist_ok=True)

            rc, summary = solver.solve(
                image_path=fits_path,
                ra=meta["ra"], dec=meta["dec"],
                focal_mm=meta["focallen"], pixel_um=meta["pixel_size"],
                output_dir=frame_out_dir,
            )
            solve_time = time.time() - t_start

            if rc != 0:
                err = summary.error_msg.decode("utf-8", errors="replace")
                print(f"  FAIL  rc={rc}  err={err[:80]}")
                results.append({
                    "frame": base, "status": f"fail_rc{rc}",
                    "error": err[:100], "solve_time_s": round(solve_time, 3),
                })
                continue

            # RMS 转换为像素 (s0 = 角秒/像素)
            rms_px = summary.ransac_rms / meta["s0"] if summary.ransac_rms > 0 else 0
            status = "success" if summary.success else "fail_low_inliers"
            print(f"  {status.upper():4s}  θ={summary.ransac_theta_deg:.3f}°  "
                  f"tx={summary.ransac_tx:.1f}\"  ty={summary.ransac_ty:.1f}\"  "
                  f"s={summary.ransac_s:.4f}  inliers={summary.n_inliers}  "
                  f"RMS={summary.ransac_rms:.3f}\"/{rms_px:.3f}px  "
                  f"SNR={summary.snr_final:.1f}  t={solve_time:.2f}s")

            results.append({
                "frame": base,
                "status": status,
                "solve_time_s": round(solve_time, 3),
                "ra": meta["ra"], "dec": meta["dec"],
                "focallen": meta["focallen"], "pixel_size": meta["pixel_size"],
                "s0": round(meta["s0"], 4),
                "ransac_theta_deg": round(summary.ransac_theta_deg, 4),
                "ransac_tx": round(summary.ransac_tx, 4),
                "ransac_ty": round(summary.ransac_ty, 4),
                "ransac_s": round(summary.ransac_s, 6),
                "n_inliers": summary.n_inliers,
                "n_ransac_iters": summary.n_ransac_iters,
                "ransac_rms": round(summary.ransac_rms, 4),
                "rms_px": round(rms_px, 4),
                "snr_final": round(summary.snr_final, 2),
                "n_samples_actual": summary.n_samples_actual,
                "n_passed": summary.n_passed,
                "n_focused": summary.n_focused,
            })
        except Exception as e:
            solve_time = time.time() - t_start
            print(f"  ERROR  {str(e)[:100]}")
            results.append({
                "frame": base, "status": f"error: {str(e)[:80]}",
                "solve_time_s": round(solve_time, 3),
            })

    t_total = time.time() - t_total_start

    # 汇总统计
    n_total = len(results)
    n_success = sum(1 for r in results if r.get("status") == "success")
    n_fail = n_total - n_success
    rms_list = [r["rms_px"] for r in results if r.get("status") == "success" and "rms_px" in r]
    snr_list = [r["snr_final"] for r in results if r.get("status") == "success"]
    t_list = [r["solve_time_s"] for r in results if r.get("status") == "success"]

    print("\n" + "=" * 70)
    print(f"小批量测试完成: {n_success}/{n_total} 成功  耗时 {t_total:.1f}s")
    if rms_list:
        print(f"  RMS: min={min(rms_list):.3f}  med={sorted(rms_list)[len(rms_list)//2]:.3f}  "
              f"max={max(rms_list):.3f} px")
    if snr_list:
        print(f"  SNR: min={min(snr_list):.1f}  med={sorted(snr_list)[len(snr_list)//2]:.1f}  "
              f"max={max(snr_list):.1f}")
    if t_list:
        print(f"  时间: min={min(t_list):.2f}  med={sorted(t_list)[len(t_list)//2]:.2f}  "
              f"max={max(t_list):.2f} s")

    # 写汇总 CSV
    summary_csv = os.path.join(output_root, "batch_summary.csv")
    if results:
        fields = list(results[0].keys())
        with open(summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\n汇总 CSV: {summary_csv}")

    # 详细结果表
    print("\n=== 详细结果 ===")
    print(f"{'帧':<55} {'状态':<10} {'θ°':>8} {'inliers':>8} {'RMS_px':>8} {'SNR':>6} {'t(s)':>6}")
    for r in results:
        status = r.get("status", "unknown")[:10]
        frame = r.get("frame", "")[:55]
        if r.get("status") == "success":
            print(f"{frame:<55} {status:<10} {r.get('ransac_theta_deg',0):>8.2f} "
                  f"{r.get('n_inliers',0):>8d} {r.get('rms_px',0):>8.3f} "
                  f"{r.get('snr_final',0):>6.1f} {r.get('solve_time_s',0):>6.2f}")
        else:
            print(f"{frame:<55} {status:<10} {'-':>8} {'-':>8} {'-':>8} {'-':>6} {r.get('solve_time_s',0):>6.2f}")


if __name__ == "__main__":
    main()
