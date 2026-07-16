"""V4.0 抽样投票向量法 批量端到端测试

功能:
    扫描 testdata 目录下所有 FITS 文件，对每帧运行 V4.0 求解，
    统计成功率、误匹配率、中位时间、RMS 中位等指标。

用途:
    Task 10.3 批量测试，验证 V4.0 在多望远镜多滤镜场景下的鲁棒性。

用法:
    python test_batch.py [--limit N] [--telescope T2|T3|T4] [--filter Red|Green|Blue|H-alpha|Oiii|Lum]
    默认 --limit 30 (代表性样本)
    --limit 0 表示全量测试 (562帧, 耗时约 30 分钟)
"""
import sys, os, json, time, argparse, csv
from collections import defaultdict
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

# 复用单帧测试脚本的求解逻辑
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_sampling_voting"))
from test_single_frame import solve_one_frame, _parse_ra_dec_from_fits


def scan_fits_files(testdata_dir: str, telescope: str = None, filt: str = None):
    """扫描 testdata 目录下所有 FITS 文件，按望远镜/滤镜筛选。

    Returns:
        list of (fits_path, telescope_tag, filter_tag)
    """
    results = []
    for root, dirs, files in os.walk(testdata_dir):
        for f in files:
            if not f.lower().endswith(".fts"):
                continue
            path = os.path.join(root, f)
            # 解析望远镜标识 (T2/T3/T4) 和滤镜
            # 文件名格式: <Target>_<Tel>_flying_dutchman-<date>@<time>-<exp>S-<Filter>.fts
            tel_tag = ""
            if "_T2_" in f:
                tel_tag = "T2"
            elif "_T3_" in f:
                tel_tag = "T3"
            elif "_T4_" in f:
                tel_tag = "T4"
            elif "LDN43" in f:
                tel_tag = "T1"  # LDN43 命名无 T1，按惯例归类

            # 滤镜
            filt_tag = ""
            for flt in ["Red", "Green", "Blue", "H-alpha", "H-alpha", "Oiii", "OIII", "Lum"]:
                if f"-{flt}.fts" in f or f"-{flt}." in f:
                    filt_tag = flt.replace("OIII", "Oiii").replace("H-alpha", "H-alpha")
                    break

            if telescope and tel_tag != telescope:
                continue
            if filt:
                # 模糊匹配滤镜
                if filt.lower() not in filt_tag.lower():
                    continue
            results.append((path, tel_tag, filt_tag))
    return results


def run_batch(fits_list, gaia_dir: str, limit: int = 30, log_dir: str = None):
    """批量运行 V4.0 求解。

    Returns:
        list of dict (每帧结果摘要)
    """
    if limit > 0:
        # 均匀采样以覆盖不同望远镜/滤镜
        fits_list = fits_list[:limit] if len(fits_list) > limit else fits_list

    results = []
    log_dir = log_dir or os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4")
    os.makedirs(log_dir, exist_ok=True)

    total = len(fits_list)
    print(f"\n=== V4.0 批量测试: {total} 帧 ===\n")

    for idx, (fits_path, tel, flt) in enumerate(fits_list, 1):
        fname = os.path.basename(fits_path)
        wcs_out = os.path.join(log_dir, f"_batch_wcs_{idx}.json")
        log_path = os.path.join(log_dir, f"v4_batch_{time.strftime('%Y%m%d_%H%M%S')}_{idx}.log")

        print(f"[{idx}/{total}] {fname} ({tel}/{flt})")

        t0 = time.time()
        try:
            result, t_solve = solve_one_frame(fits_path, gaia_dir, wcs_out=wcs_out, log_path=log_path)
            success = result is not None
        except Exception as e:
            print(f"  ❌ 异常: {e}")
            result = None
            success = False
            t_solve = time.time() - t0

        # 清理临时 WCS JSON
        try:
            if os.path.exists(wcs_out):
                os.remove(wcs_out)
        except Exception:
            pass

        rec = {
            "idx": idx,
            "filename": fname,
            "telescope": tel,
            "filter": flt,
            "success": success,
            "elapsed_s": round(t_solve, 3),
        }
        if success:
            rec.update({
                "best_mode": result.flip_mode,
                "s": round(result.solve_s, 6),
                "theta_deg": round(result.rotation_deg, 4),
                "matched": result.matched_count,
                "rms_px": round(result.rms_px, 4),
                "sip_rms_px": round(result.sip_rms_px, 4),
                "sip_order": result.sip_order,
                "m_lim": round(result.m_lim_final, 2),
                "n_gaia": result.n_gaia_final,
                "rho_img": round(result.rho_img, 3),
                "rho_target": round(result.rho_target, 3),
                "theta_snr": round(result.theta_snr, 1),
                "n_clean": result.n_phased_clean,
                "mad_rms_arcsec": round(result.mad_rms_arcsec, 3),
                "bayes_lnK": round(result.bayes_lnK, 2),
                "bayes_decision": result.bayes_decision,
                "tri_ratio": round(result.triangle_pass_ratio, 3),
                "validated": 1 if result.triangle_pass_ratio > 0.8 and result.bayes_lnK > 20.7 else 0,
                "kv_build_ms": round(result.kvector_build_ms, 2),
            })
            status = "✅" if rec["validated"] else "⚠️"
            print(f"  {status} s={rec['s']:.4f} θ={rec['theta_deg']}° matched={rec['matched']} "
                  f"sip_rms={rec['sip_rms_px']:.2f}px lnK={rec['bayes_lnK']} tri={rec['tri_ratio']} "
                  f"t={rec['elapsed_s']:.2f}s")
        else:
            print(f"  ❌ 失败 t={rec['elapsed_s']:.2f}s")

        results.append(rec)
        sys.stdout.flush()

    return results


def print_statistics(results):
    """打印批量测试统计报告。"""
    print("\n" + "=" * 70)
    print("V4.0 批量测试统计报告")
    print("=" * 70)

    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    validated_count = sum(1 for r in results if r.get("validated", 0))
    fail_count = total - success_count

    print(f"\n总帧数: {total}")
    print(f"求解成功: {success_count} ({100*success_count/max(total,1):.1f}%)")
    print(f"验证通过 (lnK>20.7 & tri>0.8): {validated_count} ({100*validated_count/max(total,1):.1f}%)")
    print(f"求解失败: {fail_count} ({100*fail_count/max(total,1):.1f}%)")

    # 按望远镜分组
    by_tel = defaultdict(list)
    for r in results:
        by_tel[r["telescope"]].append(r)

    print(f"\n--- 按望远镜分组 ---")
    for tel in sorted(by_tel.keys()):
        group = by_tel[tel]
        n = len(group)
        succ = sum(1 for r in group if r["success"])
        vali = sum(1 for r in group if r.get("validated", 0))
        rms_list = [r["sip_rms_px"] for r in group if r["success"]]
        t_list = [r["elapsed_s"] for r in group]
        med_rms = float(np.median(rms_list)) if rms_list else 0.0
        med_t = float(np.median(t_list)) if t_list else 0.0
        print(f"  {tel}: n={n} 成功={succ}({100*succ/n:.0f}%) 验证={vali}({100*vali/n:.0f}%) "
              f"中位RMS={med_rms:.2f}px 中位耗时={med_t:.2f}s")

    # 按滤镜分组
    by_filt = defaultdict(list)
    for r in results:
        by_filt[r["filter"]].append(r)

    print(f"\n--- 按滤镜分组 ---")
    for flt in sorted(by_filt.keys()):
        group = by_filt[flt]
        n = len(group)
        succ = sum(1 for r in group if r["success"])
        vali = sum(1 for r in group if r.get("validated", 0))
        rms_list = [r["sip_rms_px"] for r in group if r["success"]]
        med_rms = float(np.median(rms_list)) if rms_list else 0.0
        print(f"  {flt}: n={n} 成功={succ}({100*succ/n:.0f}%) 验证={vali}({100*vali/n:.0f}%) "
              f"中位RMS={med_rms:.2f}px")

    # 全局统计
    success_results = [r for r in results if r["success"]]
    if success_results:
        rms_arr = np.array([r["sip_rms_px"] for r in success_results])
        t_arr = np.array([r["elapsed_s"] for r in success_results])
        snr_arr = np.array([r["theta_snr"] for r in success_results])
        lnK_arr = np.array([r["bayes_lnK"] for r in success_results])
        tri_arr = np.array([r["tri_ratio"] for r in success_results])

        print(f"\n--- 成功帧统计 (n={len(success_results)}) ---")
        print(f"  SIP RMS: 中位={np.median(rms_arr):.3f}px P25={np.percentile(rms_arr,25):.3f}px "
              f"P75={np.percentile(rms_arr,75):.3f}px min={rms_arr.min():.3f}px max={rms_arr.max():.3f}px")
        print(f"  耗时: 中位={np.median(t_arr):.3f}s P25={np.percentile(t_arr,25):.3f}s "
              f"P75={np.percentile(t_arr,75):.3f}s min={t_arr.min():.3f}s max={t_arr.max():.3f}s")
        print(f"  θ SNR: 中位={np.median(snr_arr):.1f}x min={snr_arr.min():.1f}x max={snr_arr.max():.1f}x")
        print(f"  lnK: 中位={np.median(lnK_arr):.2f} min={lnK_arr.min():.2f} max={lnK_arr.max():.2f}")
        print(f"  三角形通过率: 中位={np.median(tri_arr):.4f} min={tri_arr.min():.4f}")

    # 误匹配检测：validated=True 但 RMS > 5px（可疑）
    suspicious = [r for r in success_results if r.get("validated", 0) and r["sip_rms_px"] > 5.0]
    if suspicious:
        print(f"\n--- 可疑帧 (validated=True 但 RMS>5px): {len(suspicious)} ---")
        for r in suspicious[:10]:
            print(f"  {r['filename']}: RMS={r['sip_rms_px']:.2f}px lnK={r['bayes_lnK']} matched={r['matched']}")

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
    parser = argparse.ArgumentParser(description="V4.0 批量测试")
    parser.add_argument("--limit", type=int, default=30, help="测试帧数上限 (0=全部)")
    parser.add_argument("--telescope", type=str, default=None, help="按望远镜筛选 (T2/T3/T4)")
    parser.add_argument("--filter", type=str, default=None, help="按滤镜筛选")
    parser.add_argument("--testdata", type=str, default=None, help="testdata 目录")
    parser.add_argument("--gaia", type=str, default=None, help="Gaia 数据目录")
    args = parser.parse_args()

    testdata_dir = args.testdata or os.path.join(PROJECT_ROOT, "testdata")
    gaia_dir = args.gaia or os.path.join(PROJECT_ROOT, "GaiaDR3")

    fits_list = scan_fits_files(testdata_dir, args.telescope, args.filter)
    print(f"扫描到 {len(fits_list)} 帧 (telescope={args.telescope}, filter={args.filter})")

    if args.limit > 0 and len(fits_list) > args.limit:
        # 均匀采样覆盖不同望远镜/滤镜
        step = len(fits_list) / args.limit
        sampled = []
        for i in range(args.limit):
            idx = int(i * step)
            sampled.append(fits_list[idx])
        fits_list = sampled
        print(f"均匀采样 {len(fits_list)} 帧")

    results = run_batch(fits_list, gaia_dir, limit=0)  # limit=0 因为已采样
    print_statistics(results)

    csv_path = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4",
                            f"v4_batch_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    save_csv(results, csv_path)
