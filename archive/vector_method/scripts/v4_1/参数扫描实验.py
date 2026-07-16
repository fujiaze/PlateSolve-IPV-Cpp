"""V4.1 参数扫描实验 — 星点数量 N_total 最优值确定

功能: 对 N_total ∈ {100,150,200,250,300,400,500,800} 8 个值,
      在 testdata/lights 全部帧(633帧)上运行 V4.0 求解,
      记录成功率/RMS/耗时/10px命中率, 找出最优 N_total。

用途: V4.1 版本参数整定的实验数据来源

用法: python 参数扫描实验.py
      支持断点续跑(自动跳过已完成的 N×帧 组合)
      python 参数扫描实验.py --smoke  # 冒烟测试: 1帧×2个N值
"""
import os, sys, math, json, time, traceback
import numpy as np

# ============================================================================
# UTF-8 编码初始化（Windows GBK 兼容）
# ============================================================================
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ============================================================================
# 路径初始化
# ============================================================================
PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "scripts", "v4_0"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

import logging
logging.basicConfig(level=logging.WARNING, format='%(name)s %(levelname)s: %(message)s')
logger = logging.getLogger("参数扫描实验")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_cpp import VectorMatchV4Cpp
from vector_match_v2 import GaiaClientPy
from WCS重投影校核V4 import sky_to_pixel_wcs

# ============================================================================
# 扫描参数
# ============================================================================
N_TOTAL_LIST = [100, 150, 200, 250, 300, 400, 500, 800]
SMOKE_N_LIST = [100, 250]
N_BRIGHT_EVAL = 1000  # 投影评估用的 Gaia 亮星数

# 输出目录
SWEEP_DIR = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "sweep")
WCS_DIR = os.path.join(SWEEP_DIR, "wcs")
SWEEP_JSON = os.path.join(SWEEP_DIR, "sweep_all.json")
SMOKE_JSON = os.path.join(SWEEP_DIR, "sweep_smoke.json")


# ============================================================================
# 收集 FITS 文件（复用 v4_0 脚本逻辑）
# ============================================================================
def collect_fits_files(root_dir, max_depth=3):
    """递归收集所有 FITS 文件(.fts/.fit/.fits)"""
    fits_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel_depth = os.path.relpath(dirpath, root_dir).count(os.sep)
        if rel_depth >= max_depth:
            dirnames.clear()
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ('.fts', '.fit', '.fits'):
                fits_files.append(os.path.join(dirpath, fn))
    return fits_files


def parse_filename(path):
    """从文件名提取目标、滤镜、曝光等信息

    示例: NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts
    → target=NGC55_T3, filter=Red, exposure=600S
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    info = {'filename': base}

    import re
    m = re.search(r'-(\d+)S-', name)
    if m:
        info['exposure_s'] = int(m.group(1))
    else:
        info['exposure_s'] = 0

    m = re.search(r'-([A-Za-z][A-Za-z0-9_-]*)$', name)
    if m:
        info['filter'] = m.group(1)
    else:
        info['filter'] = 'unknown'

    if '_' in name:
        info['target'] = name.split('_')[0]
    elif '-' in name:
        info['target'] = name.split('-')[0]
    else:
        info['target'] = 'unknown'

    return info


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


def _compute_n_selected(n_total, n_saturated, n_detected):
    """计算实际选入求解的星点数

    solve() 内部逻辑: 饱和全选 + 非饱和按 flux 降序补足到 n_total
    → N = min(max(n_total, n_saturated), n_detected)
    """
    return int(min(max(n_total, n_saturated), n_detected))


# ============================================================================
# 断点续跑: 加载/保存扫描结果
# ============================================================================
def load_existing_results(json_path):
    """加载已有的扫描结果, 返回 dict[(filename, n_total)] = result_info"""
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        done = {}
        for r in data:
            key = (r.get('filename'), r.get('n_total'))
            done[key] = r
        return done
    except Exception as e:
        logger.warning(f"加载已有结果失败: {e}, 重新开始")
        return {}


def save_results(done_dict, json_path):
    """增量保存扫描结果到 JSON 文件"""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    results_list = list(done_dict.values())
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, ensure_ascii=False, indent=2)


# ============================================================================
# 单帧单 N 值求解 + 投影评估
# ============================================================================
def solve_frame_with_n(fits_path, solver, gaia_eval, n_total,
                       det, img_info, t_frame_start):
    """对单帧用指定 n_total 求解, 返回 result_info dict

    Args:
        fits_path: FITS 文件路径
        solver: VectorMatchV4Cpp 实例(复用)
        gaia_eval: GaiaClientPy 实例(投影评估用, 复用缓存)
        n_total: N_total 参数值
        det: 已检测的星点数据 (detect_ex 返回对象)
        img_info: 图像信息 dict
        t_frame_start: 帧处理开始时间(用于计算总耗时)

    Returns:
        dict: 结果信息
    """
    base = os.path.basename(fits_path)
    w = img_info['w']
    h = img_info['h']
    fl = img_info['fl']
    ps = img_info['ps']
    n_detected = len(det.x)
    n_saturated = int(np.sum(det.saturated))

    result_info = parse_filename(fits_path)
    result_info['n_total'] = n_total
    result_info['width'] = w
    result_info['height'] = h
    result_info['focallen'] = fl
    result_info['pixel_size'] = ps
    result_info['n_detected'] = n_detected
    result_info['n_saturated'] = n_saturated
    result_info['n_selected'] = _compute_n_selected(n_total, n_saturated, n_detected)

    t_solve_start = time.time()

    try:
        # ── 1. V4.0 求解 WCS ──
        wcs_json = os.path.join(WCS_DIR, f"wcs_N{n_total}_{base}.json")
        os.makedirs(WCS_DIR, exist_ok=True)

        result = solver.solve(
            np.array(det.x, np.float64), np.array(det.y, np.float64),
            np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
            img_info['cra0'], img_info['cdec0'], fl, ps, w, h,
            wcs_out=wcs_json,
            exptime=img_info['exptime'],
            n_img_total=n_total,
        )
        solve_time = time.time() - t_solve_start

        if not result:
            result_info['status'] = 'fail_solve'
            result_info['solve_time_s'] = round(solve_time, 2)
            result_info['total_time_s'] = round(time.time() - t_frame_start, 2)
            return result_info

        result_info['status'] = 'success'
        result_info['solve_time_s'] = round(solve_time, 2)
        result_info['flip_mode'] = result.flip_mode
        result_info['matched_count'] = result.matched_count
        result_info['phaseb_rms_px'] = round(result.rms_px, 4)
        result_info['scale_arcsec_px'] = round(result.scale_arcsec_px, 4)
        result_info['rotation_deg'] = round(result.rotation_deg, 4)

        # ── 2. 读取 WCS JSON (获取 SIP 拟合后的最终 RMS) ──
        if os.path.exists(wcs_json):
            with open(wcs_json, 'r', encoding='utf-8') as f:
                wcs = json.load(f)
        else:
            # 兜底: solver.solve()返回成功但WCS JSON未写入, 使用result对象数据
            logger.warning(f"  WCS JSON未写入, 使用result兜底: {wcs_json}")
            wcs = {
                'RMS_PX': getattr(result, 'sip_rms_px', result.rms_px),
                'CD': getattr(result, 'cd_matrix', [[1.0, 0.0], [0.0, 1.0]]),
                'CRVAL': [0.0, 0.0],  # 无法获取, 跳过投影评估
                'CRPIX': [w / 2.0, h / 2.0],
                'SIP_A': [[0.0]*6 for _ in range(6)],
                'SIP_B': [[0.0]*6 for _ in range(6)],
                'SIP_ORDER': 0,
            }
            result_info['rms_px'] = round(float(wcs['RMS_PX']), 4)
            result_info['total_time_s'] = round(time.time() - t_frame_start, 2)
            result_info['wcs_json_missing'] = True
            return result_info

        result_info['rms_px'] = round(
            float(wcs.get('RMS_PX', getattr(result, 'sip_rms_px', 0.0))), 4)
        cd = np.array(wcs['CD'], dtype=np.float64).reshape(2, 2)
        crval = np.array(wcs['CRVAL'], dtype=np.float64)
        crpix = np.array(wcs['CRPIX'], dtype=np.float64)
        sip_A = np.array(wcs['SIP_A'], dtype=np.float64).reshape(6, 6)
        sip_B = np.array(wcs['SIP_B'], dtype=np.float64).reshape(6, 6)
        sip_order = int(wcs.get('SIP_ORDER', 0))
        result_info['sip_order'] = sip_order

        # ── 3. 查询 Gaia 亮星做投影评估 (复用全局 gaia_eval 实例利用缓存) ──
        fov_deg = math.sqrt(w * w + h * h) * img_info['s0'] / 3600.0
        query_radius = max(fov_deg * 0.7, 1.0)
        ra_t, dec_t, mag_t = gaia_eval.cone_search(
            crval[0], crval[1], query_radius, 22.0)

        ra_all = np.array(ra_t, dtype=np.float64)
        dec_all = np.array(dec_t, dtype=np.float64)
        mag_all = np.array(mag_t, dtype=np.float64)

        if len(ra_all) > N_BRIGHT_EVAL:
            idx_bright = np.argsort(mag_all)[:N_BRIGHT_EVAL]
            ra_src = ra_all[idx_bright]
            dec_src = dec_all[idx_bright]
        else:
            ra_src = ra_all
            dec_src = dec_all

        # ── 4. 标准 WCS-SIP 逆投影 ──
        x_pix, y_pix = sky_to_pixel_wcs(
            ra_src, dec_src, cd, crval, crpix, sip_A, sip_B, sip_order)

        in_frame = np.isfinite(x_pix) & (x_pix > 0) & (x_pix < w) & \
                   (y_pix > 0) & (y_pix < h)
        x_in = x_pix[in_frame]
        y_in = y_pix[in_frame]

        # ── 5. 投影质量诊断 ──
        from scipy.spatial import cKDTree
        det_tree = cKDTree(np.column_stack([det.x, det.y]))
        n_test = len(x_in)
        if n_test > 0:
            dists, _ = det_tree.query(np.column_stack([x_in, y_in]))
            n_5px = int(np.sum(dists < 5))
            n_10px = int(np.sum(dists < 10))
            result_info['pct_5px'] = round(100 * n_5px / n_test, 1)
            result_info['pct_10px'] = round(100 * n_10px / n_test, 1)
        else:
            result_info['pct_5px'] = 0.0
            result_info['pct_10px'] = 0.0

        result_info['total_time_s'] = round(time.time() - t_frame_start, 2)

    except Exception as e:
        result_info['status'] = f'error: {str(e)[:100]}'
        result_info['solve_time_s'] = round(time.time() - t_solve_start, 2)
        result_info['total_time_s'] = round(time.time() - t_frame_start, 2)
        logger.error(f"  异常: {base} N={n_total}: {e}")
        traceback.print_exc()

    return result_info


# ============================================================================
# 单帧处理: 读取 + 检测(一次) + 对每个 N 值求解
# ============================================================================
def process_single_frame(fits_path, solver, gaia_eval, n_list,
                         done_dict, json_path, frame_idx, n_total_frames):
    """处理单帧: 读取+检测(一次)+对每个N值求解+投影评估

    Args:
        fits_path: FITS 文件路径
        solver: VectorMatchV4Cpp 实例(复用)
        gaia_eval: GaiaClientPy 实例(投影评估用)
        n_list: N_total 值列表
        done_dict: 已完成结果字典(会更新)
        json_path: 结果保存路径
        frame_idx: 当前帧索引(1-based, 用于进度显示)
        n_total_frames: 总帧数
    """
    base = os.path.basename(fits_path)
    t_frame_start = time.time()

    # 检查这帧的所有 N 值是否都已完成
    pending_n = [n for n in n_list if (base, n) not in done_dict]
    if not pending_n:
        return

    # ── 1. 读取 FITS ──
    try:
        reader = ImageReader()
        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = img.metadata.observation.focallen
        ps = img.metadata.observation.xpixsz
        s0 = 206.265 * ps / fl

        kws = img.keywords
        kw_dict = {k.name.upper(): k.value for k in kws}
        obj_ra_str = kw_dict.get('OBJCTRA') or kw_dict.get('RA')
        obj_dec_str = kw_dict.get('OBJCTDEC') or kw_dict.get('DEC')

        img_info = {
            'w': w, 'h': h, 'fl': fl, 'ps': ps, 's0': s0,
            'exptime': getattr(img.metadata.observation, 'exptime', 1.0),
        }

        if not obj_ra_str or not obj_dec_str:
            # 所有 N 值标记为失败
            for n_total in pending_n:
                result_info = parse_filename(fits_path)
                result_info['n_total'] = n_total
                result_info['width'] = w
                result_info['height'] = h
                result_info['focallen'] = fl
                result_info['pixel_size'] = ps
                result_info['status'] = 'fail_no_objctra'
                result_info['solve_time_s'] = 0.0
                result_info['total_time_s'] = round(time.time() - t_frame_start, 2)
                done_dict[(base, n_total)] = result_info
                print(f"  [N={n_total}][{frame_idx}/{n_total_frames}] {base} → fail_no_objctra")
            save_results(done_dict, json_path)
            return

        img_info['cra0'] = _parse_ra_hms(obj_ra_str)
        img_info['cdec0'] = _parse_dec_dms(obj_dec_str)

    except Exception as e:
        # 读取异常, 所有 N 值标记为错误
        for n_total in pending_n:
            result_info = parse_filename(fits_path)
            result_info['n_total'] = n_total
            result_info['status'] = f'error: {str(e)[:100]}'
            result_info['solve_time_s'] = 0.0
            result_info['total_time_s'] = round(time.time() - t_frame_start, 2)
            done_dict[(base, n_total)] = result_info
            print(f"  [N={n_total}][{frame_idx}/{n_total_frames}] {base} → error: read failed")
        save_results(done_dict, json_path)
        logger.error(f"  读取异常: {fits_path}: {e}")
        return

    # ── 2. 星点检测 (只做一次, 同帧不同 N 值共享) ──
    try:
        detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        det = detector.detect_ex(img.data)
        n_detected = len(det.x)
        n_saturated = int(np.sum(det.saturated))
        detector = None  # 释放
    except Exception as e:
        for n_total in pending_n:
            result_info = parse_filename(fits_path)
            result_info['n_total'] = n_total
            result_info['width'] = w
            result_info['height'] = h
            result_info['focallen'] = fl
            result_info['pixel_size'] = ps
            result_info['status'] = f'error: detect {str(e)[:80]}'
            result_info['solve_time_s'] = 0.0
            result_info['total_time_s'] = round(time.time() - t_frame_start, 2)
            done_dict[(base, n_total)] = result_info
            print(f"  [N={n_total}][{frame_idx}/{n_total_frames}] {base} → error: detect failed")
        save_results(done_dict, json_path)
        logger.error(f"  检测异常: {fits_path}: {e}")
        return

    if n_detected < 5:
        for n_total in pending_n:
            result_info = parse_filename(fits_path)
            result_info['n_total'] = n_total
            result_info['width'] = w
            result_info['height'] = h
            result_info['focallen'] = fl
            result_info['pixel_size'] = ps
            result_info['n_detected'] = n_detected
            result_info['n_saturated'] = n_saturated
            result_info['n_selected'] = 0
            result_info['status'] = 'fail_too_few_stars'
            result_info['solve_time_s'] = 0.0
            result_info['total_time_s'] = round(time.time() - t_frame_start, 2)
            done_dict[(base, n_total)] = result_info
            print(f"  [N={n_total}][{frame_idx}/{n_total_frames}] {base} → fail_too_few_stars (n={n_detected})")
        save_results(done_dict, json_path)
        return

    # ── 3. 对每个 N 值求解 + 投影评估 ──
    for n_total in pending_n:
        result_info = solve_frame_with_n(
            fits_path, solver, gaia_eval, n_total, det, img_info, t_frame_start
        )
        done_dict[(base, n_total)] = result_info

        # 进度显示
        status = result_info['status']
        if status == 'success':
            rms = result_info.get('rms_px', 0.0)
            pct10 = result_info.get('pct_10px', 0.0)
            solve_t = result_info.get('solve_time_s', 0.0)
            print(f"  [N={n_total}][{frame_idx}/{n_total_frames}] {base} → "
                  f"success RMS={rms:.3f}px 10px={pct10}% 耗时={solve_t}s")
        else:
            print(f"  [N={n_total}][{frame_idx}/{n_total_frames}] {base} → {status}")

        # 增量保存(每帧每 N 值完成后)
        save_results(done_dict, json_path)


# ============================================================================
# 汇总统计
# ============================================================================
def print_summary(done_dict, n_list):
    """打印汇总统计"""
    print(f"\n=== 汇总统计 ===")
    print(f"{'N_total':>8} {'总数':>6} {'成功':>6} {'成功率':>8} "
          f"{'中位RMS':>10} {'中位10px':>10} {'中位耗时':>10}")

    for n_total in n_list:
        results_n = [r for (fn, n), r in done_dict.items() if n == n_total]
        n_total_count = len(results_n)
        if n_total_count == 0:
            continue
        n_success = sum(1 for r in results_n if r.get('status') == 'success')
        succ = [r for r in results_n if r.get('status') == 'success']

        success_rate = 100 * n_success / n_total_count
        if succ:
            rms_list = [r['rms_px'] for r in succ if 'rms_px' in r]
            pct10_list = [r.get('pct_10px', 0) for r in succ]
            time_list = [r['solve_time_s'] for r in succ if 'solve_time_s' in r]
            med_rms = f"{np.median(rms_list):.3f}px" if rms_list else "-"
            med_pct10 = f"{np.median(pct10_list):.1f}%" if pct10_list else "-"
            med_time = f"{np.median(time_list):.2f}s" if time_list else "-"
        else:
            med_rms = "-"
            med_pct10 = "-"
            med_time = "-"

        print(f"{n_total:>8} {n_total_count:>6} {n_success:>6} {success_rate:>7.1f}% "
              f"{med_rms:>10} {med_pct10:>10} {med_time:>10}")

    # 失败统计
    all_results = list(done_dict.values())
    n_fail = sum(1 for r in all_results if r.get('status') != 'success')
    if n_fail > 0:
        print(f"\n失败组合: {n_fail} 个")
        fail_types = {}
        for r in all_results:
            if r.get('status') != 'success':
                st = r.get('status', 'unknown')
                fail_types[st] = fail_types.get(st, 0) + 1
        for st, cnt in sorted(fail_types.items(), key=lambda x: -x[1]):
            print(f"  {st}: {cnt}")


# ============================================================================
# 主流程
# ============================================================================
def main():
    smoke_mode = '--smoke' in sys.argv

    if smoke_mode:
        n_list = SMOKE_N_LIST
        max_frames = 1
        json_path = SMOKE_JSON
        print(f"=== V4.1 参数扫描实验 — 冒烟测试 ===")
    else:
        n_list = N_TOTAL_LIST
        max_frames = None
        json_path = SWEEP_JSON
        print(f"=== V4.1 参数扫描实验 — 全量扫描 ===")

    # 收集 FITS 文件
    lights_dir = os.path.join(PROJECT_ROOT, "testdata", "lights")
    all_fits = collect_fits_files(lights_dir, max_depth=3)
    n_total_frames = len(all_fits)
    print(f"N_total 列表: {n_list}")
    print(f"找到 FITS 文件: {n_total_frames} 个")

    if max_frames is not None and n_total_frames > max_frames:
        all_fits = all_fits[:max_frames]
        n_total_frames = len(all_fits)
        print(f"冒烟测试模式: 取前 {n_total_frames} 帧")

    # 创建输出目录
    os.makedirs(WCS_DIR, exist_ok=True)

    # 加载已有结果 (断点续跑, 冒烟测试不加载)
    if smoke_mode:
        done_dict = {}
    else:
        done_dict = load_existing_results(json_path)
        n_done = len(done_dict)
        if n_done > 0:
            print(f"断点续跑: 已完成 {n_done} 个 (帧, N) 组合, 将跳过")

    # 复用 solver 实例 (整个扫描过程只创建一个)
    print(f"初始化 V4.0 求解器...")
    solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)
    # 复用 Gaia 客户端实例 (投影评估用, 利用 60s TTL 缓存)
    gaia_eval = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)

    t_sweep_start = time.time()
    n_processed = 0

    try:
        for i, fits_path in enumerate(all_fits, 1):
            base = os.path.basename(fits_path)

            # 检查是否所有 N 值都已完成
            pending_n = [n for n in n_list if (base, n) not in done_dict]
            if not pending_n:
                continue

            print(f"\n[{i}/{n_total_frames}] {base} (待处理 N 值: {pending_n})")

            process_single_frame(
                fits_path, solver, gaia_eval, n_list,
                done_dict, json_path, i, n_total_frames
            )
            n_processed += 1

    finally:
        solver.close()
        gaia_eval.close()

    elapsed = time.time() - t_sweep_start
    print(f"\n本次扫描处理 {n_processed} 帧, 总耗时 {elapsed:.1f}s")

    # 汇总统计
    print_summary(done_dict, n_list)

    print(f"\n结果已保存: {json_path}")
    print(f"WCS 文件目录: {WCS_DIR}")


if __name__ == "__main__":
    main()
