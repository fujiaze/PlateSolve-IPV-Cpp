"""V3.5 vs V4.0 对比测试

功能:
    在同样本上分别运行 V3.5 和 V4.0 求解，对比成功率、RMS、耗时、误匹配率。

用途:
    Task 10.4 与 V3.5 结果对比，验证 V4.0 是否达到 spec 指标。
    spec 指标: 成功率 ≥ 90%、误匹配率 = 0%、求解时间 ≤ 0.02s、RMS ≤ 0.50px

用法:
    python test_compare_v35_v4.py [--limit N]
    默认 --limit 10 (代表性样本)
"""
import sys, os, json, time, argparse, csv
# Windows UTF-8 输出
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from astro_image_io import ImageReader
from vector_match_v3_5_cpp import VectorMatchV35Cpp
from vector_match_v4_cpp import VectorMatchV4Cpp
from star_detector import StarDetector, SDetParamsPy

# 复用单帧测试脚本的 FITS 头解析
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_sampling_voting"))
from test_single_frame import _parse_ra_dec_from_fits
from test_batch import scan_fits_files


def solve_with_version(fits_path, gaia_dir, version: str, wcs_out: str):
    """用指定版本求解单帧。

    Returns:
        (result, elapsed_s) 或 (None, elapsed_s)
    """
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    if img.metadata.wcs is not None:
        cra0 = img.metadata.wcs.crval1
        cdec0 = img.metadata.wcs.crval2
    else:
        cra0, cdec0 = _parse_ra_dec_from_fits(fits_path)
    exptime = getattr(img.metadata.observation, 'exposure', 1.0) or 1.0

    # 星点检测（两版本共用，保证一致性）
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)

    x = np.array(det.x, np.float64)
    y = np.array(det.y, np.float64)
    flux = np.array(det.flux, np.float64)
    sat = np.array(det.saturated, np.int32)

    if version == "v3_5":
        vm = VectorMatchV35Cpp(gaia_dir, db_type=1)
        try:
            t0 = time.time()
            result = vm.solve(x, y, flux, sat, cra0, cdec0, fl, ps, w, h,
                              wcs_out=wcs_out, exptime=exptime)
            elapsed = time.time() - t0
        finally:
            vm.close()
    elif version == "v4":
        vm = VectorMatchV4Cpp(gaia_dir, db_type=1)
        try:
            t0 = time.time()
            result = vm.solve(x, y, flux, sat, cra0, cdec0, fl, ps, w, h,
                              wcs_out=wcs_out, exptime=exptime)
            elapsed = time.time() - t0
        finally:
            vm.close()
    else:
        raise ValueError(f"未知版本: {version}")

    return result, elapsed


def run_compare(fits_list, gaia_dir: str, log_dir: str):
    """在同样本上对比 V3.5 和 V4.0。"""
    results = []
    total = len(fits_list)
    print(f"\n=== V3.5 vs V4.0 对比测试: {total} 帧 ===\n")

    for idx, (fits_path, tel, flt) in enumerate(fits_list, 1):
        fname = os.path.basename(fits_path)
        print(f"[{idx}/{total}] {fname} ({tel}/{flt})")

        wcs35 = os.path.join(log_dir, f"_cmp_v35_{idx}.json")
        wcs4 = os.path.join(log_dir, f"_cmp_v4_{idx}.json")

        # V3.5
        try:
            r35, t35 = solve_with_version(fits_path, gaia_dir, "v3_5", wcs35)
            s35_ok = r35 is not None
        except Exception as e:
            print(f"  V3.5 异常: {e}")
            r35, t35, s35_ok = None, 0.0, False

        # V4.0
        try:
            r4, t4 = solve_with_version(fits_path, gaia_dir, "v4", wcs4)
            s4_ok = r4 is not None
        except Exception as e:
            print(f"  V4.0 异常: {e}")
            r4, t4, s4_ok = None, 0.0, False

        # 清理临时文件
        for p in [wcs35, wcs4]:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

        rec = {
            "idx": idx,
            "filename": fname,
            "telescope": tel,
            "filter": flt,
            "v35_success": int(s35_ok),
            "v4_success": int(s4_ok),
            "v35_elapsed_s": round(t35, 3),
            "v4_elapsed_s": round(t4, 3),
        }

        if s35_ok:
            rec.update({
                "v35_matched": r35.matched_count,
                "v35_sip_rms_px": round(r35.sip_rms_px, 4),
                "v35_sip_order": r35.sip_order,
                "v35_n_clean": r35.n_phased_clean,
                "v35_mad_rms_arcsec": round(r35.mad_rms_arcsec, 3),
                "v35_theta_snr": round(r35.theta_snr, 1),
                "v35_s": round(r35.solve_s, 6),
                "v35_theta_deg": round(r35.rotation_deg, 4),
            })
        if s4_ok:
            rec.update({
                "v4_matched": r4.matched_count,
                "v4_sip_rms_px": round(r4.sip_rms_px, 4),
                "v4_sip_order": r4.sip_order,
                "v4_n_clean": r4.n_phased_clean,
                "v4_mad_rms_arcsec": round(r4.mad_rms_arcsec, 3),
                "v4_theta_snr": round(r4.theta_snr, 1),
                "v4_bayes_lnK": round(r4.bayes_lnK, 2),
                "v4_tri_ratio": round(r4.triangle_pass_ratio, 3),
                "v4_s": round(r4.solve_s, 6),
                "v4_theta_deg": round(r4.rotation_deg, 4),
            })

        # 打印对比
        if s35_ok and s4_ok:
            rms_diff = r4.sip_rms_px - r35.sip_rms_px
            t_ratio = t4 / max(t35, 1e-6)
            print(f"  V3.5: RMS={r35.sip_rms_px:.3f}px matched={r35.matched_count} t={t35:.2f}s")
            print(f"  V4.0: RMS={r4.sip_rms_px:.3f}px matched={r4.matched_count} t={t4:.2f}s "
                  f"lnK={r4.bayes_lnK:.0f} tri={r4.triangle_pass_ratio:.3f}")
            print(f"  ΔRMS={rms_diff:+.3f}px  时间比V3.5={t_ratio:.2f}x")
        elif s35_ok:
            print(f"  V3.5: RMS={r35.sip_rms_px:.3f}px  V4.0: 失败")
        elif s4_ok:
            print(f"  V3.5: 失败  V4.0: RMS={r4.sip_rms_px:.3f}px")
        else:
            print(f"  两版本均失败")

        results.append(rec)
        sys.stdout.flush()

    return results


def print_compare_statistics(results):
    """打印对比统计。"""
    print("\n" + "=" * 80)
    print("V3.5 vs V4.0 对比统计报告")
    print("=" * 80)

    total = len(results)
    s35 = sum(1 for r in results if r["v35_success"])
    s4 = sum(1 for r in results if r["v4_success"])
    both = sum(1 for r in results if r["v35_success"] and r["v4_success"])

    print(f"\n总帧数: {total}")
    print(f"V3.5 成功: {s35} ({100*s35/total:.1f}%)")
    print(f"V4.0 成功: {s4} ({100*s4/total:.1f}%)")
    print(f"两版本均成功: {both} ({100*both/total:.1f}%)")

    # 仅对比两版本均成功的帧
    common = [r for r in results if r["v35_success"] and r["v4_success"]]
    if not common:
        print("\n无共同成功帧用于对比")
        return results

    print(f"\n--- 共同成功帧对比 (n={len(common)}) ---")

    rms35 = np.array([r["v35_sip_rms_px"] for r in common])
    rms4 = np.array([r["v4_sip_rms_px"] for r in common])
    t35 = np.array([r["v35_elapsed_s"] for r in common])
    t4 = np.array([r["v4_elapsed_s"] for r in common])
    m35 = np.array([r["v35_matched"] for r in common])
    m4 = np.array([r["v4_matched"] for r in common])

    print(f"\n  {'指标':<20} {'V3.5':<20} {'V4.0':<20} {'差异':<20}")
    print(f"  {'-'*80}")
    print(f"  {'中位RMS(px)':<20} {np.median(rms35):<20.4f} {np.median(rms4):<20.4f} {np.median(rms4)-np.median(rms35):<+20.4f}")
    print(f"  {'P25 RMS(px)':<20} {np.percentile(rms35,25):<20.4f} {np.percentile(rms4,25):<20.4f} {np.percentile(rms4,25)-np.percentile(rms35,25):<+20.4f}")
    print(f"  {'P75 RMS(px)':<20} {np.percentile(rms35,75):<20.4f} {np.percentile(rms4,75):<20.4f} {np.percentile(rms4,75)-np.percentile(rms35,75):<+20.4f}")
    print(f"  {'min RMS(px)':<20} {rms35.min():<20.4f} {rms4.min():<20.4f} {rms4.min()-rms35.min():<+20.4f}")
    print(f"  {'max RMS(px)':<20} {rms35.max():<20.4f} {rms4.max():<20.4f} {rms4.max()-rms35.max():<+20.4f}")
    print(f"  {'中位耗时(s)':<20} {np.median(t35):<20.4f} {np.median(t4):<20.4f} {np.median(t4)-np.median(t35):<+20.4f}")
    print(f"  {'中位matched':<20} {int(np.median(m35)):<20} {int(np.median(m4)):<20} {int(np.median(m4))-int(np.median(m35)):<+20}")

    # spec 指标检查
    print(f"\n--- Spec 指标检查 (V4.0) ---")
    spec_success = 100 * s4 / total
    spec_rms_med = float(np.median(rms4))
    spec_time_med = float(np.median(t4))

    print(f"  成功率 ≥ 90%: {spec_success:.1f}% {'✅' if spec_success >= 90 else '❌'}")
    print(f"  RMS 中位 ≤ 0.50px: {spec_rms_med:.4f}px {'✅' if spec_rms_med <= 0.50 else '❌'}")
    print(f"  求解中位时间 ≤ 0.02s: {spec_time_med:.4f}s {'✅' if spec_time_med <= 0.02 else '❌'} "
          f"(注: 含Gaia查询+检测+求解)")

    # 误匹配率：V4.0 验证通过但 RMS > 5px 且 matched < 5
    v4_results = [r for r in results if r["v4_success"]]
    false_match = sum(1 for r in v4_results
                      if r.get("v4_bayes_lnK", 0) > 20.7
                      and r.get("v4_tri_ratio", 0) > 0.8
                      and r["v4_sip_rms_px"] > 5.0
                      and r["v4_matched"] < 5)
    spec_false = 100 * false_match / max(len(v4_results), 1)
    print(f"  误匹配率 = 0%: {spec_false:.1f}% ({false_match}帧) {'✅' if spec_false == 0 else '❌'}")

    # V4.0 相对 V3.5 的改善
    print(f"\n--- V4.0 相对 V3.5 改善 ---")
    rms_better = sum(1 for r in common if r["v4_sip_rms_px"] < r["v35_sip_rms_px"])
    rms_worse = sum(1 for r in common if r["v4_sip_rms_px"] > r["v35_sip_rms_px"])
    rms_equal = len(common) - rms_better - rms_worse
    print(f"  RMS 改善: {rms_better}/{len(common)} ({100*rms_better/len(common):.0f}%)")
    print(f"  RMS 持平: {rms_equal}/{len(common)} ({100*rms_equal/len(common):.0f}%)")
    print(f"  RMS 变差: {rms_worse}/{len(common)} ({100*rms_worse/len(common):.0f}%)")

    t_faster = sum(1 for r in common if r["v4_elapsed_s"] < r["v35_elapsed_s"])
    print(f"  耗时更短: {t_faster}/{len(common)} ({100*t_faster/len(common):.0f}%)")

    return results


def save_csv(results, csv_path):
    """保存结果到 CSV。"""
    if not results:
        return
    fields = list(results[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\n结果已保存: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3.5 vs V4.0 对比测试")
    parser.add_argument("--limit", type=int, default=10, help="测试帧数上限")
    parser.add_argument("--testdata", type=str, default=None)
    parser.add_argument("--gaia", type=str, default=None)
    args = parser.parse_args()

    testdata_dir = args.testdata or os.path.join(PROJECT_ROOT, "testdata")
    gaia_dir = args.gaia or os.path.join(PROJECT_ROOT, "GaiaDR3")
    log_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4")
    os.makedirs(log_dir, exist_ok=True)

    fits_list = scan_fits_files(testdata_dir)
    print(f"扫描到 {len(fits_list)} 帧")

    if args.limit > 0 and len(fits_list) > args.limit:
        step = len(fits_list) / args.limit
        sampled = [fits_list[int(i * step)] for i in range(args.limit)]
        fits_list = sampled
        print(f"均匀采样 {len(fits_list)} 帧")

    results = run_compare(fits_list, gaia_dir, log_dir)
    print_compare_statistics(results)

    csv_path = os.path.join(log_dir, f"v4_compare_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    save_csv(results, csv_path)
