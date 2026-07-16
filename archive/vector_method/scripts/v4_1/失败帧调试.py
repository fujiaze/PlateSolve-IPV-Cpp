"""V4.1 失败帧调试分析 — 详细诊断3个V4.1批量测试中失败的FITS帧

功能:
    对3个失败帧执行完整的V4.0调试流程:
      步骤1: 读取FITS头信息(OBJCTRA/OBJCTDEC/焦距/像元/曝光/滤镜/宽高)
             计算 s0=206.265*ps/fl, FOV对角线=sqrt(w²+h²)*s0/3600
      步骤2: 星点检测(fitRadius=0), 统计 n_detected/n_saturated/n_non_sat
             flux分布(min/P25/P50/P75/P90/max), 前10亮星
             与同目标同滤镜成功帧对比
      步骤3: V4.0求解(verbose=True), 捕获C++完整stderr日志
      步骤4: 分析各Phase(0/A/B/C/D/D'/E)日志, 定位失败位置
      步骤5: 输出失败根因总结

用途:
    定位算法层面失败原因(非数据本身问题), 指导V4.1后续优化

用法:
    python 失败帧调试.py
输出:
    lib/plate_solve/logs/v4/debug/ 下:
      - wcs_debug_<filename>.json (每帧WCS JSON)
      - cpp_log_<filename>.log (C++详细日志)
      - debug_report_<filename>.txt (调试报告)
      - 调试总报告.md (3个帧汇总分析)
"""
import os, sys, math, json, time, traceback, io
from contextlib import redirect_stderr
import numpy as np

# ============================================================================
# UTF-8 编码初始化（Windows GBK 兼容）
# ============================================================================
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    # 不替换 sys.stderr，避免与 redirect_stderr 上下文冲突

# ============================================================================
# 路径初始化
# ============================================================================
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("失败帧调试")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_cpp import VectorMatchV4Cpp

# ============================================================================
# 失败帧 + 同目标成功对比帧
# ============================================================================
FRAMES = [
    {
        "name": "NGC55_T3_Oiii",
        "fail_path": r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@042902-1200S-Oiii.fts",
        "succ_path": r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\NGC55_T3_flying_dutchman-20250915@025221-1200S-Oiii.fts",
        "succ_note": "同目标同滤镜同夜, 早2小时拍摄",
    },
    {
        "name": "Victory_mosaic2_Lum",
        "fail_path": r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic2_flying_dutchman-20250205@062533-180S-Lum.fts",
        "succ_path": r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic2_flying_dutchman-20250205@062145-180S-Lum.fts",
        "succ_note": "同目标同滤镜同夜, 早4分钟拍摄",
    },
    {
        "name": "Victory_mosaic1_Green",
        "fail_path": r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic1_flying_dutchman-20250206@054309-180S-Green.fts",
        "succ_path": r"f:\Astro dev\Astro CS Normalization Database\testdata\lights\Victory_Nebula_mosaic1_flying_dutchman-20250206@053908-180S-Green.fts",
        "succ_note": "同目标同滤镜同夜, 早4分钟拍摄",
    },
]

# 输出目录
DEBUG_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)


# ============================================================================
# 工具函数
# ============================================================================
def _parse_ra_hms(s):
    """RA: '13 05 40.00' → 度数"""
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m / 60.0 + sec / 3600.0) * 15.0
    return float(s)


def _parse_dec_dms(s):
    """Dec: '-49 35 39.0' → 度数"""
    s = str(s).strip()
    sign = 1.0
    if s.startswith('-'):
        sign = -1.0
        s = s[1:]
    elif s.startswith('+'):
        s = s[1:]
    parts = s.split()
    if len(parts) == 3:
        d, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return sign * (d + m / 60.0 + sec / 3600.0)
    return float(s)


def _flux_stats(flux):
    """计算flux分布统计"""
    if len(flux) == 0:
        return {"count": 0}
    arr = np.asarray(flux, dtype=np.float64)
    # 排除饱和星flux=-1
    arr_valid = arr[arr >= 0]
    if len(arr_valid) == 0:
        return {"count": len(arr), "all_saturated": True}
    return {
        "count": len(arr),
        "valid_count": len(arr_valid),
        "min": float(arr_valid.min()),
        "P25": float(np.percentile(arr_valid, 25)),
        "P50": float(np.percentile(arr_valid, 50)),
        "P75": float(np.percentile(arr_valid, 75)),
        "P90": float(np.percentile(arr_valid, 90)),
        "max": float(arr_valid.max()),
    }


def _top10_brightest(det):
    """返回前10颗最亮星(flux降序), 区分饱和/非饱和"""
    flux_arr = np.asarray(det.flux, dtype=np.float64)
    x_arr = np.asarray(det.x)
    y_arr = np.asarray(det.y)
    sat_arr = np.asarray(det.saturated, dtype=bool)

    # 饱和星优先(按r排序假设已排), 非饱和按flux降序
    sat_idx = np.where(sat_arr)[0]
    non_sat_idx = np.where(~sat_arr)[0]
    if len(non_sat_idx) > 0:
        non_sat_sorted = non_sat_idx[np.argsort(-flux_arr[non_sat_idx])]
    else:
        non_sat_sorted = np.array([], dtype=np.int64)

    # 前10: 饱和优先, 然后非饱和补足
    top_idx = np.concatenate([sat_idx, non_sat_sorted])[:10]
    result = []
    for i in top_idx:
        result.append({
            "x": float(x_arr[i]),
            "y": float(y_arr[i]),
            "flux": float(flux_arr[i]),
            "saturated": bool(sat_arr[i]),
        })
    return result


# ============================================================================
# 步骤1: 读取FITS头信息
# ============================================================================
def step1_read_header(fits_path, label):
    """读取FITS头信息并计算s0/FOV

    Returns:
        dict: 包含img对象、w/h/fl/ps/s0/fov/cra0/cdec0/exptime/filter等
    """
    print(f"\n{'='*70}")
    print(f"  步骤1: 读取FITS头信息 — {label}")
    print(f"  路径: {fits_path}")
    print(f"{'='*70}")

    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl
    fov_diag = math.sqrt(w * w + h * h) * s0 / 3600.0

    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    obj_ra_str = kw_dict.get('OBJCTRA') or kw_dict.get('RA')
    obj_dec_str = kw_dict.get('OBJCTDEC') or kw_dict.get('DEC')
    exptime = getattr(img.metadata.calibration, 'exptime', 1.0) or 1.0
    filter_name = getattr(img.metadata.calibration, 'filter_name', 'Unknown')

    cra0 = _parse_ra_hms(obj_ra_str) if obj_ra_str else 0.0
    cdec0 = _parse_dec_dms(obj_dec_str) if obj_dec_str else 0.0

    info = {
        "img": img,
        "w": w, "h": h, "fl": fl, "ps": ps, "s0": s0,
        "fov_diag_deg": fov_diag,
        "cra0": cra0, "cdec0": cdec0,
        "exptime": exptime,
        "filter": filter_name,
        "OBJCTRA": obj_ra_str,
        "OBJCTDEC": obj_dec_str,
    }
    print(f"  宽高: {w} × {h}")
    print(f"  焦距: {fl} mm")
    print(f"  像元: {ps} µm")
    print(f"  s0 = 206.265 × {ps} / {fl} = {s0:.4f} 角秒/像素")
    print(f"  FOV对角线 = sqrt({w}²+{h}²) × {s0:.4f} / 3600 = {fov_diag:.4f}°")
    print(f"  OBJCTRA: {obj_ra_str} → {cra0:.6f}°")
    print(f"  OBJCTDEC: {obj_dec_str} → {cdec0:.6f}°")
    print(f"  曝光: {exptime}s")
    print(f"  滤镜: {filter_name}")
    return info


# ============================================================================
# 步骤2: 星点检测分析
# ============================================================================
def step2_detect_stars(img, label):
    """星点检测分析, 返回det对象和统计信息"""
    print(f"\n{'='*70}")
    print(f"  步骤2: 星点检测 — {label}")
    print(f"{'='*70}")

    t0 = time.time()
    detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    det = detector.detect_ex(img.data)
    elapsed = time.time() - t0

    n_detected = len(det.x)
    n_saturated = int(np.sum(det.saturated))
    n_non_sat = n_detected - n_saturated
    flux_stat = _flux_stats(det.flux)
    top10 = _top10_brightest(det)

    print(f"  检测耗时: {elapsed:.2f}s")
    print(f"  n_detected = {n_detected}")
    print(f"  n_saturated = {n_saturated}")
    print(f"  n_non_sat = {n_non_sat}")
    print(f"  flux分布(非饱和): "
          f"min={flux_stat.get('min', 0):.1f} "
          f"P25={flux_stat.get('P25', 0):.1f} "
          f"P50={flux_stat.get('P50', 0):.1f} "
          f"P75={flux_stat.get('P75', 0):.1f} "
          f"P90={flux_stat.get('P90', 0):.1f} "
          f"max={flux_stat.get('max', 0):.1f}")
    print(f"  前10颗最亮星:")
    for i, s in enumerate(top10):
        sat_mark = "[饱和]" if s["saturated"] else ""
        print(f"    #{i+1}: x={s['x']:.1f} y={s['y']:.1f} flux={s['flux']:.1f} {sat_mark}")

    return {
        "det": det,
        "n_detected": n_detected,
        "n_saturated": n_saturated,
        "n_non_sat": n_non_sat,
        "flux_stat": flux_stat,
        "top10": top10,
        "detect_time_s": elapsed,
    }


# ============================================================================
# 步骤3: V4.0求解(verbose=True)
# ============================================================================
def step3_solve_verbose(fits_path, info, det, label, n_total=250):
    """V4.0求解, 捕获Python端sys.stderr日志, C++日志通过log_file_path获取

    使用 redirect_stderr 捕获 vector_match_v4_cpp.py 中 sys.stderr.write() 输出的
    Phase 0 密度匹配日志; C++ DLL 完整日志通过 log_file_path 参数写入文件。
    verbose=False 让 vector_match_v4_cpp.py 内部将 C++ stderr 重定向到 /devnull,
    避免与 fd 2 重定向冲突导致崩溃。
    """
    print(f"\n{'='*70}")
    print(f"  步骤3: V4.0求解 (N_total={n_total}) — {label}")
    print(f"{'='*70}")

    base = os.path.basename(fits_path)
    wcs_json = os.path.join(DEBUG_DIR, f"wcs_debug_{base}.json")
    cpp_log = os.path.join(DEBUG_DIR, f"cpp_log_{base}.log")
    os.makedirs(os.path.dirname(wcs_json), exist_ok=True)

    solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)

    # 使用 redirect_stderr 捕获 Python 端 sys.stderr.write() 输出 (Phase 0 日志等)
    # verbose=False: C++ DLL 的 stderr 被 vector_match_v4_cpp.py 内部重定向到 /devnull
    # C++ 完整日志通过 log_file_path 参数写入 cpp_log 文件
    stderr_capture = io.StringIO()

    t0 = time.time()
    result = None
    err_msg = None
    try:
        with redirect_stderr(stderr_capture):
            result = solver.solve(
                np.array(det.x, np.float64), np.array(det.y, np.float64),
                np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
                info["cra0"], info["cdec0"], info["fl"], info["ps"], info["w"], info["h"],
                wcs_out=wcs_json,
                exptime=info["exptime"],
                n_img_total=n_total,
                log_file_path=cpp_log,
                verbose=False,
            )
    except Exception as e:
        err_msg = str(e)
        traceback.print_exc(file=sys.stdout)

    elapsed = time.time() - t0
    stderr_text = stderr_capture.getvalue()

    # 保存stderr到文件
    stderr_log_path = os.path.join(DEBUG_DIR, f"stderr_{base}.log")
    with open(stderr_log_path, 'w', encoding='utf-8') as f:
        f.write(stderr_text)

    # 打印 Python 端 stderr 内容 (Phase 0 密度匹配日志)
    if stderr_text:
        print(f"  --- Python端日志 (Phase 0 等) ---")
        for line in stderr_text.strip().split('\n'):
            print(f"    {line}")

    print(f"  求解耗时: {elapsed:.2f}s")
    if err_msg:
        print(f"  异常: {err_msg}")
    if result:
        print(f"  求解成功!")
        print(f"    flip_mode = {result.flip_mode}")
        print(f"    matched_count = {result.matched_count}")
        print(f"    rms_px = {result.rms_px:.4f}px")
        print(f"    sip_rms_px = {result.sip_rms_px:.4f}px")
        print(f"    scale = {result.scale_arcsec_px:.4f}\"/px")
        print(f"    rotation = {result.rotation_deg:.4f}°")
        print(f"    s = {result.solve_s:.6f}")
        print(f"    θ_SNR = {result.theta_snr:.2f}")
        print(f"    θ_peak = {result.theta_peak_deg:.4f}°")
        print(f"    PhaseB对数 = {result.n_phaseb_pairs}")
        print(f"    PhaseC扩展 = {result.n_phasec_expanded}")
        print(f"    PhaseD清洗 = {result.n_phased_clean}")
        print(f"    MAD-RMS = {result.mad_rms_arcsec:.4f}\"")
        print(f"    bayes_lnK = {result.bayes_lnK:.2f}")
        print(f"    triangle_pass_ratio = {result.triangle_pass_ratio:.3f}")
        print(f"    m_lim = {result.m_lim_final:.2f}")
        print(f"    n_gaia = {result.n_gaia_final}")
        print(f"    prosac_pool = {result.prosac_pool_final}")
    else:
        print(f"  求解失败 (None)")
        if os.path.exists(cpp_log):
            print(f"  C++日志文件: {cpp_log}")

    solver.close()

    return {
        "result": result,
        "solve_time_s": elapsed,
        "wcs_json": wcs_json if os.path.exists(wcs_json) else None,
        "cpp_log": cpp_log if os.path.exists(cpp_log) else None,
        "stderr_log": stderr_log_path,
        "stderr_text": stderr_text,
        "err_msg": err_msg,
    }


# ============================================================================
# 步骤4: 分析C++日志, 定位失败Phase
# ============================================================================
def step4_analyze_log(solve_out, label):
    """分析C++日志, 提取各Phase关键指标"""
    print(f"\n{'='*70}")
    print(f"  步骤4: 分析C++日志 — {label}")
    print(f"{'='*70}")

    cpp_log_path = solve_out.get("cpp_log")
    stderr_text = solve_out.get("stderr_text", "")

    cpp_log_text = ""
    if cpp_log_path and os.path.exists(cpp_log_path):
        with open(cpp_log_path, 'r', encoding='utf-8') as f:
            cpp_log_text = f.read()

    # 合并分析(stderr含Python端密度匹配日志, cpp_log含C++各Phase日志)
    analysis = {
        "phase0": {},
        "phaseA": {},
        "phaseB": {},
        "phaseC": {},
        "phaseD": {},
        "phaseD_prime": {},
        "phaseE": {},
        "failure_phase": "unknown",
        "root_cause": "",
    }

    # 从stderr提取Phase 0信息
    for line in stderr_text.split('\n'):
        if 'Phase 0 密度匹配' in line:
            # n_img=200 rho_img=... n_target=... query_r=... m_cut=...
            parts = line.split()
            for p in parts:
                if '=' in p:
                    k, v = p.split('=', 1)
                    try:
                        analysis["phase0"][k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        analysis["phase0"][k] = v
        elif 'Phase 0 iter#' in line:
            analysis["phase0"].setdefault('iterations', []).append(line.strip())
        elif 'Phase 0 收敛' in line or 'Phase 0 未收敛' in line or 'Phase 0 兜底' in line:
            analysis["phase0"]['final_status'] = line.strip()
        elif 'Phase 0 完成' in line:
            analysis["phase0"]['final_summary'] = line.strip()
        elif 'FOV 内 Gaia 星' in line:
            analysis["phase0"]['fov_gaia_count'] = line.strip()

    # 从cpp_log提取Phase A/B/C/D/D'/E信息
    log_text = cpp_log_text
    for line in log_text.split('\n'):
        ls = line.strip()
        if 'Phase A' in ls or 'PROSAC' in ls:
            if 'peak_snr' in ls.lower() or 'best_n' in ls.lower():
                analysis["phaseA"]['records'] = analysis["phaseA"].get('records', []) + [ls]
        elif 'Phase B' in ls or 'SVD' in ls:
            analysis["phaseB"]['records'] = analysis["phaseB"].get('records', []) + [ls]
        elif 'Phase C' in ls or 'k-vector' in ls.lower() or 'kvector' in ls.lower():
            analysis["phaseC"]['records'] = analysis["phaseC"].get('records', []) + [ls]
        elif 'Phase D' in ls and 'Phase D\'' not in ls:
            analysis["phaseD"]['records'] = analysis["phaseD"].get('records', []) + [ls]
        elif 'Bayes' in ls or '贝叶斯' in ls or 'triangle' in ls.lower() or '三角形' in ls:
            analysis["phaseD_prime"]['records'] = analysis["phaseD_prime"].get('records', []) + [ls]
        elif 'Phase E' in ls or 'SIP' in ls:
            analysis["phaseE"]['records'] = analysis["phaseE"].get('records', []) + [ls]

    # 输出关键摘要
    print(f"\n  --- Phase 0 (密度匹配) ---")
    p0 = analysis["phase0"]
    print(f"    n_img_bright = {p0.get('n_img', '?')}")
    print(f"    rho_img = {p0.get('rho_img', '?')}")
    print(f"    n_target = {p0.get('n_target', '?')}")
    print(f"    query_r = {p0.get('query_r', '?')}")
    print(f"    m_cut = {p0.get('m_cut', '?')}")
    if 'final_summary' in p0:
        print(f"    最终: {p0['final_summary']}")
    if 'fov_gaia_count' in p0:
        print(f"    {p0['fov_gaia_count']}")

    print(f"\n  --- Phase A (PROSAC抽样) ---")
    pa = analysis["phaseA"]
    if 'records' in pa:
        for r in pa['records'][-5:]:
            print(f"    {r}")
    else:
        print(f"    (无明确日志)")

    print(f"\n  --- Phase B (SVD) ---")
    pb = analysis["phaseB"]
    if 'records' in pb:
        for r in pb['records'][-5:]:
            print(f"    {r}")
    else:
        print(f"    (无明确日志)")

    print(f"\n  --- Phase C (k-vector) ---")
    pc = analysis["phaseC"]
    if 'records' in pc:
        for r in pc['records'][-5:]:
            print(f"    {r}")
    else:
        print(f"    (无明确日志)")

    print(f"\n  --- Phase D (MAD清洗) ---")
    pd = analysis["phaseD"]
    if 'records' in pd:
        for r in pd['records'][-5:]:
            print(f"    {r}")
    else:
        print(f"    (无明确日志)")

    print(f"\n  --- Phase D' (贝叶斯+三角形) ---")
    pdp = analysis["phaseD_prime"]
    if 'records' in pdp:
        for r in pdp['records'][-5:]:
            print(f"    {r}")
    else:
        print(f"    (无明确日志)")

    print(f"\n  --- Phase E (SIP拟合) ---")
    pe = analysis["phaseE"]
    if 'records' in pe:
        for r in pe['records'][-5:]:
            print(f"    {r}")
    else:
        print(f"    (无明确日志)")

    # 定位失败Phase (基于求解结果)
    result = solve_out.get("result")
    if result is None:
        # 完全失败, 需要从日志判断
        if not p0:
            analysis["failure_phase"] = "Phase 0"
            analysis["root_cause"] = "密度匹配失败或异常"
        elif 'final_status' not in p0 and 'final_summary' not in p0:
            analysis["failure_phase"] = "Phase 0"
            analysis["root_cause"] = "密度匹配未收敛"
        elif not pa:
            analysis["failure_phase"] = "Phase A"
            analysis["root_cause"] = "PROSAC抽样无有效记录, 可能N/M星点太少"
        elif not pb:
            analysis["failure_phase"] = "Phase B"
            analysis["root_cause"] = "SVD匹配对太少"
        elif not pc:
            analysis["failure_phase"] = "Phase C"
            analysis["root_cause"] = "k-vector扩展失败"
        elif not pdp:
            analysis["failure_phase"] = "Phase D'"
            analysis["root_cause"] = "贝叶斯/三角形验证未通过"
        else:
            analysis["failure_phase"] = "Phase E"
            analysis["root_cause"] = "SIP拟合失败"
    else:
        # 求解成功但有可疑迹象
        if result.matched_count < 5:
            analysis["failure_phase"] = "Phase D'/E"
            analysis["root_cause"] = f"matched_count过少({result.matched_count})"
        elif result.bayes_lnK < 20.7:
            analysis["failure_phase"] = "Phase D'"
            analysis["root_cause"] = f"贝叶斯lnK过低({result.bayes_lnK:.2f})"
        elif result.triangle_pass_ratio < 0.8:
            analysis["failure_phase"] = "Phase D'"
            analysis["root_cause"] = f"三角形通过率过低({result.triangle_pass_ratio:.3f})"
        else:
            analysis["failure_phase"] = "成功"
            analysis["root_cause"] = ""

    print(f"\n  → 失败Phase: {analysis['failure_phase']}")
    print(f"  → 根因: {analysis['root_cause']}")

    analysis["cpp_log_text"] = cpp_log_text
    analysis["stderr_text"] = stderr_text
    return analysis


# ============================================================================
# 步骤5: 单帧调试报告
# ============================================================================
def step5_write_report(frame_info, fail_header, fail_detect, fail_solve, fail_analysis,
                        succ_header=None, succ_detect=None, succ_solve=None, succ_analysis=None):
    """写单帧调试报告"""
    name = frame_info["name"]
    report_path = os.path.join(DEBUG_DIR, f"debug_report_{name}.txt")

    lines = []
    lines.append(f"=" * 70)
    lines.append(f"  V4.1 失败帧调试报告 — {name}")
    lines.append(f"=" * 70)
    lines.append(f"")
    lines.append(f"失败帧: {frame_info['fail_path']}")
    if succ_header:
        lines.append(f"对比帧: {frame_info['succ_path']} ({frame_info['succ_note']})")
    lines.append(f"")

    # === 头信息 ===
    lines.append(f"--- 1. FITS头信息 ---")
    lines.append(f"{'指标':<20} {'失败帧':<25} {'对比帧':<25}")
    lines.append(f"{'-'*70}")
    rows = [
        ("宽×高", f"{fail_header['w']}×{fail_header['h']}",
         f"{succ_header['w']}×{succ_header['h']}" if succ_header else "-"),
        ("焦距(mm)", f"{fail_header['fl']}",
         f"{succ_header['fl']}" if succ_header else "-"),
        ("像元(µm)", f"{fail_header['ps']}",
         f"{succ_header['ps']}" if succ_header else "-"),
        ("s0(\"/px)", f"{fail_header['s0']:.4f}",
         f"{succ_header['s0']:.4f}" if succ_header else "-"),
        ("FOV对角(°)", f"{fail_header['fov_diag_deg']:.4f}",
         f"{succ_header['fov_diag_deg']:.4f}" if succ_header else "-"),
        ("OBJCTRA", f"{fail_header['OBJCTRA']}",
         f"{succ_header['OBJCTRA']}" if succ_header else "-"),
        ("OBJCTDEC", f"{fail_header['OBJCTDEC']}",
         f"{succ_header['OBJCTDEC']}" if succ_header else "-"),
        ("曝光(s)", f"{fail_header['exptime']}",
         f"{succ_header['exptime']}" if succ_header else "-"),
        ("滤镜", f"{fail_header['filter']}",
         f"{succ_header['filter']}" if succ_header else "-"),
    ]
    for k, v1, v2 in rows:
        lines.append(f"{k:<20} {v1:<25} {v2:<25}")

    # === 星点检测 ===
    lines.append(f"")
    lines.append(f"--- 2. 星点检测 ---")
    lines.append(f"{'指标':<20} {'失败帧':<25} {'对比帧':<25}")
    lines.append(f"{'-'*70}")
    rows2 = [
        ("n_detected", str(fail_detect["n_detected"]),
         str(succ_detect["n_detected"]) if succ_detect else "-"),
        ("n_saturated", str(fail_detect["n_saturated"]),
         str(succ_detect["n_saturated"]) if succ_detect else "-"),
        ("n_non_sat", str(fail_detect["n_non_sat"]),
         str(succ_detect["n_non_sat"]) if succ_detect else "-"),
    ]
    for k, v1, v2 in rows2:
        lines.append(f"{k:<20} {v1:<25} {v2:<25}")

    lines.append(f"")
    lines.append(f"失败帧 flux分布(非饱和): {fail_detect['flux_stat']}")
    if succ_detect:
        lines.append(f"对比帧 flux分布(非饱和): {succ_detect['flux_stat']}")

    lines.append(f"")
    lines.append(f"失败帧 前10亮星:")
    for i, s in enumerate(fail_detect["top10"]):
        sat = "[饱和]" if s["saturated"] else ""
        lines.append(f"  #{i+1}: x={s['x']:.1f} y={s['y']:.1f} flux={s['flux']:.1f} {sat}")
    if succ_detect:
        lines.append(f"对比帧 前10亮星:")
        for i, s in enumerate(succ_detect["top10"]):
            sat = "[饱和]" if s["saturated"] else ""
            lines.append(f"  #{i+1}: x={s['x']:.1f} y={s['y']:.1f} flux={s['flux']:.1f} {sat}")

    # === 求解结果 ===
    lines.append(f"")
    lines.append(f"--- 3. V4.0求解结果 ---")
    if fail_solve["result"]:
        r = fail_solve["result"]
        lines.append(f"失败帧: 求解成功(但质量可疑) — matched={r.matched_count} rms={r.rms_px:.4f}px "
                     f"lnK={r.bayes_lnK:.2f} tri_ratio={r.triangle_pass_ratio:.3f} "
                     f"m_lim={r.m_lim_final:.2f} n_gaia={r.n_gaia_final}")
    else:
        lines.append(f"失败帧: 求解失败 (返回None)")
        if fail_solve["err_msg"]:
            lines.append(f"  异常: {fail_solve['err_msg']}")

    if succ_solve and succ_solve["result"]:
        r = succ_solve["result"]
        lines.append(f"对比帧: 求解成功 — matched={r.matched_count} rms={r.rms_px:.4f}px "
                     f"lnK={r.bayes_lnK:.2f} tri_ratio={r.triangle_pass_ratio:.3f} "
                     f"m_lim={r.m_lim_final:.2f} n_gaia={r.n_gaia_final}")
    elif succ_solve:
        lines.append(f"对比帧: 求解失败 (返回None)")

    # === Phase分析 ===
    lines.append(f"")
    lines.append(f"--- 4. Phase分析 ---")
    lines.append(f"失败帧:")
    lines.append(f"  失败Phase: {fail_analysis['failure_phase']}")
    lines.append(f"  根因: {fail_analysis['root_cause']}")
    lines.append(f"  Phase 0: {fail_analysis['phase0']}")
    if succ_analysis:
        lines.append(f"对比帧:")
        lines.append(f"  失败Phase: {succ_analysis['failure_phase']}")
        lines.append(f"  Phase 0: {succ_analysis['phase0']}")

    # === 关键差异 ===
    lines.append(f"")
    lines.append(f"--- 5. 关键差异分析 ---")
    if succ_detect:
        det_diff = fail_detect["n_detected"] - succ_detect["n_detected"]
        sat_diff = fail_detect["n_saturated"] - succ_detect["n_saturated"]
        lines.append(f"  星点检测差异: n_detected {fail_detect['n_detected']} vs {succ_detect['n_detected']} (差{det_diff})")
        lines.append(f"  饱和星差异: n_saturated {fail_detect['n_saturated']} vs {succ_detect['n_saturated']} (差{sat_diff})")
        if fail_detect["n_saturated"] == 0 and succ_detect["n_saturated"] > 0:
            lines.append(f"  ⚠ 失败帧无饱和星 — PROSAC优先采样无地标, 抽样质量降低")
        if fail_detect["n_detected"] < 100 and succ_detect["n_detected"] >= 100:
            lines.append(f"  ⚠ 失败帧星点过少 — 可能密度匹配目标星数过少")

    if fail_solve["result"] and succ_solve and succ_solve["result"]:
        fr, sr = fail_solve["result"], succ_solve["result"]
        if abs(fr.n_gaia_final - sr.n_gaia_final) > 100:
            lines.append(f"  ⚠ Gaia查询星数差异大: {fr.n_gaia_final} vs {sr.n_gaia_final}")
        if abs(fr.m_lim_final - sr.m_lim_final) > 0.5:
            lines.append(f"  ⚠ 极限星等差异: {fr.m_lim_final:.2f} vs {sr.m_lim_final:.2f}")

    report_text = "\n".join(lines)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  报告已保存: {report_path}")
    return report_text, report_path


# ============================================================================
# 主流程
# ============================================================================
def main():
    print(f"=== V4.1 失败帧调试分析 ===")
    print(f"调试目录: {DEBUG_DIR}")
    print(f"待分析帧数: {len(FRAMES)} 个失败帧 + 对应成功对比帧")

    all_reports = []

    for fi, frame in enumerate(FRAMES, 1):
        print(f"\n\n{'#'*70}")
        print(f"# [{fi}/{len(FRAMES)}] {frame['name']}")
        print(f"{'#'*70}")

        # === 失败帧 ===
        fail_header = step1_read_header(frame["fail_path"], f"{frame['name']} (失败)")
        fail_detect = step2_detect_stars(fail_header["img"], f"{frame['name']} (失败)")
        fail_solve = step3_solve_verbose(frame["fail_path"], fail_header, fail_detect["det"],
                                          f"{frame['name']} (失败)")
        fail_analysis = step4_analyze_log(fail_solve, f"{frame['name']} (失败)")

        # === 对比成功帧 ===
        succ_header = succ_detect = succ_solve = succ_analysis = None
        if frame.get("succ_path") and os.path.exists(frame["succ_path"]):
            succ_header = step1_read_header(frame["succ_path"], f"{frame['name']} (对比)")
            succ_detect = step2_detect_stars(succ_header["img"], f"{frame['name']} (对比)")
            succ_solve = step3_solve_verbose(frame["succ_path"], succ_header, succ_detect["det"],
                                              f"{frame['name']} (对比)")
            succ_analysis = step4_analyze_log(succ_solve, f"{frame['name']} (对比)")
        else:
            print(f"\n  对比帧不存在或未指定: {frame.get('succ_path')}")

        # === 单帧报告 ===
        report_text, report_path = step5_write_report(
            frame, fail_header, fail_detect, fail_solve, fail_analysis,
            succ_header, succ_detect, succ_solve, succ_analysis
        )
        all_reports.append({
            "name": frame["name"],
            "fail_path": frame["fail_path"],
            "succ_path": frame.get("succ_path"),
            "fail_header": fail_header,
            "fail_detect": fail_detect,
            "fail_solve": {
                "result": fail_solve["result"],
                "err_msg": fail_solve["err_msg"],
                "solve_time_s": fail_solve["solve_time_s"],
            },
            "fail_analysis": {k: v for k, v in fail_analysis.items()
                              if k not in ('cpp_log_text', 'stderr_text')},
            "succ_header": succ_header,
            "succ_detect": succ_detect,
            "succ_solve": {
                "result": succ_solve["result"] if succ_solve else None,
                "err_msg": succ_solve["err_msg"] if succ_solve else None,
                "solve_time_s": succ_solve["solve_time_s"] if succ_solve else None,
            } if succ_solve else None,
            "succ_analysis": {k: v for k, v in succ_analysis.items()
                              if k not in ('cpp_log_text', 'stderr_text')} if succ_analysis else None,
            "report_path": report_path,
        })

    # === 输出汇总报告 ===
    print(f"\n\n{'#'*70}")
    print(f"# 汇总分析")
    print(f"{'#'*70}")

    summary_path = os.path.join(DEBUG_DIR, "调试总报告.md")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("# V4.1 失败帧调试总报告\n\n")
        f.write(f"调试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 1. 失败帧概览\n\n")
        f.write(f"| 帧名 | 失败Phase | 根因 | n_detected | n_saturated | matched | lnK |\n")
        f.write(f"|------|-----------|------|------------|------------|---------|-----|\n")
        for r in all_reports:
            fd = r["fail_detect"]
            fs = r["fail_solve"]
            fa = r["fail_analysis"]
            matched = fs["result"].matched_count if fs["result"] else "None"
            lnK = f"{fs['result'].bayes_lnK:.2f}" if fs["result"] else "None"
            f.write(f"| {r['name']} | {fa['failure_phase']} | {fa['root_cause'][:40]} | "
                    f"{fd['n_detected']} | {fd['n_saturated']} | {matched} | {lnK} |\n")

        f.write(f"\n## 2. 各帧详细分析\n\n")
        for r in all_reports:
            f.write(f"### 2.{all_reports.index(r)+1} {r['name']}\n\n")
            f.write(f"**失败帧**: `{r['fail_path']}`\n\n")
            if r["succ_path"]:
                f.write(f"**对比帧**: `{r['succ_path']}`\n\n")

            fh = r["fail_header"]
            sh = r["succ_header"]
            f.write(f"#### FITS头对比\n\n")
            f.write(f"| 指标 | 失败帧 | 对比帧 |\n")
            f.write(f"|------|--------|--------|\n")
            sh_w_h = f"{sh['w']}×{sh['h']}" if sh else "-"
            sh_fl = f"{sh['fl']}" if sh else "-"
            sh_s0 = f"{sh['s0']:.4f}" if sh else "-"
            sh_fov = f"{sh['fov_diag_deg']:.4f}" if sh else "-"
            sh_ra = f"{sh['OBJCTRA']}" if sh else "-"
            sh_dec = f"{sh['OBJCTDEC']}" if sh else "-"
            sh_exp = f"{sh['exptime']}" if sh else "-"
            f.write(f"| 宽×高 | {fh['w']}×{fh['h']} | {sh_w_h} |\n")
            f.write(f"| 焦距(mm) | {fh['fl']} | {sh_fl} |\n")
            f.write(f"| s0(\\\"/px) | {fh['s0']:.4f} | {sh_s0} |\n")
            f.write(f"| FOV对角(°) | {fh['fov_diag_deg']:.4f} | {sh_fov} |\n")
            f.write(f"| OBJCTRA | {fh['OBJCTRA']} | {sh_ra} |\n")
            f.write(f"| OBJCTDEC | {fh['OBJCTDEC']} | {sh_dec} |\n")
            f.write(f"| 曝光(s) | {fh['exptime']} | {sh_exp} |\n")
            f.write(f"\n")

            fd = r["fail_detect"]
            sd = r["succ_detect"]
            f.write(f"#### 星点检测对比\n\n")
            f.write(f"| 指标 | 失败帧 | 对比帧 |\n")
            f.write(f"|------|--------|--------|\n")
            f.write(f"| n_detected | {fd['n_detected']} | {sd['n_detected'] if sd else '-'} |\n")
            f.write(f"| n_saturated | {fd['n_saturated']} | {sd['n_saturated'] if sd else '-'} |\n")
            f.write(f"| n_non_sat | {fd['n_non_sat']} | {sd['n_non_sat'] if sd else '-'} |\n")
            if fd["flux_stat"] and sd and sd["flux_stat"]:
                f.write(f"| flux P50 | {fd['flux_stat'].get('P50', 0):.1f} | {sd['flux_stat'].get('P50', 0):.1f} |\n")
                f.write(f"| flux max | {fd['flux_stat'].get('max', 0):.1f} | {sd['flux_stat'].get('max', 0):.1f} |\n")
            f.write(f"\n")

            fs = r["fail_solve"]
            ss = r["succ_solve"]
            f.write(f"#### 求解结果对比\n\n")
            f.write(f"| 指标 | 失败帧 | 对比帧 |\n")
            f.write(f"|------|--------|--------|\n")
            if fs["result"]:
                ss_matched = f"{ss['result'].matched_count}" if ss and ss['result'] else "-"
                ss_rms = f"{ss['result'].rms_px:.4f}" if ss and ss['result'] else "-"
                ss_lnK = f"{ss['result'].bayes_lnK:.2f}" if ss and ss['result'] else "-"
                ss_mlim = f"{ss['result'].m_lim_final:.2f}" if ss and ss['result'] else "-"
                ss_ngaia = f"{ss['result'].n_gaia_final}" if ss and ss['result'] else "-"
                f.write(f"| matched | {fs['result'].matched_count} | {ss_matched} |\n")
                f.write(f"| rms_px | {fs['result'].rms_px:.4f} | {ss_rms} |\n")
                f.write(f"| bayes_lnK | {fs['result'].bayes_lnK:.2f} | {ss_lnK} |\n")
                f.write(f"| m_lim | {fs['result'].m_lim_final:.2f} | {ss_mlim} |\n")
                f.write(f"| n_gaia | {fs['result'].n_gaia_final} | {ss_ngaia} |\n")
            else:
                f.write(f"| 求解状态 | 失败(None) | {'成功' if ss and ss['result'] else '失败'} |\n")
            f.write(f"\n")

            fa = r["fail_analysis"]
            f.write(f"#### 失败Phase与根因\n\n")
            f.write(f"- **失败Phase**: {fa['failure_phase']}\n")
            f.write(f"- **根因**: {fa['root_cause']}\n")
            f.write(f"- **Phase 0**: {fa['phase0']}\n\n")

        f.write(f"\n## 3. 共性问题与特殊问题\n\n")
        f.write(f"### 共性问题\n")
        # 检测共性问题
        sat_zero_count = sum(1 for r in all_reports if r["fail_detect"]["n_saturated"] == 0)
        if sat_zero_count >= 2:
            f.write(f"- **无饱和星**: {sat_zero_count}/3 帧无饱和星, PROSAC优先采样失效\n")
        low_det = sum(1 for r in all_reports if r["fail_detect"]["n_detected"] < 200)
        if low_det >= 2:
            f.write(f"- **星点偏少**: {low_det}/3 帧检测星点<200\n")
        f.write(f"\n### 各帧特殊问题\n")
        for r in all_reports:
            f.write(f"- **{r['name']}**: {r['fail_analysis']['root_cause']}\n")

    print(f"\n汇总报告: {summary_path}")
    print(f"\n=== 调试完成 ===")
    print(f"所有日志和报告保存在: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
