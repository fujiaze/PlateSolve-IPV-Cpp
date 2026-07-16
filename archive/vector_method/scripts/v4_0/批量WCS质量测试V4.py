"""V4.0 批量WCS质量测试 — 随机抽取testdata中的帧进行WCS求解+重投影质量评估

功能:
    随机抽取testdata/lights下的FITS帧(默认50帧) → V4.0求解WCS
    → 标准WCS-SIP重投影Gaia亮星 → 统计投影命中率
    → 结果保存JSON+CSV供分析

用途:
    评估V4.0在不同目标/滤镜/曝光下的WCS求解成功率和精度

用法:
    python 批量WCS质量测试V4.py [帧数] [随机种子]
    默认: 50帧, 种子=42(可复现)

V4.1 参数整定:
    - n_img_total=250 (参数扫描实验最优值, 363帧×8个N值验证)
    - 算法成功率94.4% (排除路径错误bug和全失败帧后98.7%)
"""
import os, sys, math, json, time, random, csv, traceback
import numpy as np

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
logger = logging.getLogger("批量WCS质量测试V4")

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v4_cpp import VectorMatchV4Cpp
from vector_match_v2 import GaiaClientPy
from WCS重投影校核V4 import sky_to_pixel_wcs


# ============================================================================
# 收集FITS文件
# ============================================================================
def collect_fits_files(root_dir, max_depth=3):
    """递归收集所有FITS文件(.fts/.fit/.fits)"""
    fits_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 限制深度
        rel_depth = os.path.relpath(dirpath, root_dir).count(os.sep)
        if rel_depth >= max_depth:
            dirnames.clear()
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in ('.fts', '.fit', '.fits'):
                fits_files.append(os.path.join(dirpath, fn))
    return fits_files


# ============================================================================
# 解析文件名提取目标/滤镜/曝光信息
# ============================================================================
def parse_filename(path):
    """从文件名提取目标、滤镜、曝光等信息

    示例: NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts
    → target=NGC55_T3, filter=Red, exposure=600S
    """
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    info = {'filename': base, 'path': path}

    # 提取曝光时间 (数字+S)
    import re
    m = re.search(r'-(\d+)S-', name)
    if m:
        info['exposure_s'] = int(m.group(1))
    else:
        info['exposure_s'] = 0

    # 提取滤镜 (最后一段 -XXX)
    m = re.search(r'-([A-Za-z][A-Za-z0-9_-]*)$', name)
    if m:
        info['filter'] = m.group(1)
    else:
        info['filter'] = 'unknown'

    # 提取目标名 (第一个 _ 之前或第一个 - 之前)
    # 常见格式: TARGET_T..._flying_dutchman-... 或 TARGET-...
    if '_' in name:
        parts = name.split('_')
        info['target'] = parts[0]
    elif '-' in name:
        info['target'] = name.split('-')[0]
    else:
        info['target'] = 'unknown'

    return info


# ============================================================================
# 解析FITS头中的RA/DEC字符串(格式: "HH MM SS.SS" / "DD MM SS.S")
# ============================================================================
def _parse_ra_hms(s):
    """RA: '13 05 40.00' → 度数"""
    parts = str(s).strip().split()
    if len(parts) == 3:
        h, m, sec = float(parts[0]), float(parts[1]), float(parts[2])
        return (h + m/60.0 + sec/3600.0) * 15.0
    return float(s)  # 已是数值


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
        return sign * (d + m/60.0 + sec/3600.0)
    return float(s)  # 已是数值


# ============================================================================
# 单帧WCS求解+质量评估
# ============================================================================
def solve_single_frame(fits_path, solver, output_dir, n_bright=1000):
    """对单帧FITS进行V4.0 WCS求解+重投影质量评估

    Args:
        fits_path: FITS文件路径
        solver: VectorMatchV4Cpp 实例(复用)
        output_dir: WCS JSON输出目录
        n_bright: 投影的Gaia亮星数

    Returns:
        dict: 包含WCS参数和质量指标的结果字典
    """
    result_info = parse_filename(fits_path)
    t_start = time.time()

    try:
        # ── 1. 读取FITS ──
        reader = ImageReader()
        img = reader.read(fits_path)
        w, h = img.width, img.height
        fl = img.metadata.observation.focallen
        ps = img.metadata.observation.xpixsz
        s0 = 206.265 * ps / fl

        # WCS求解器: 只用望远镜指向(OBJCTRA/DEC)作为初始值, 不用已有CRVAL
        # 从FITS头关键字中提取OBJCTRA/OBJCTDEC (或RA/DEC), 格式如 "13 05 40.00"
        kws = img.keywords
        kw_dict = {k.name.upper(): k.value for k in kws}
        obj_ra_str = kw_dict.get('OBJCTRA') or kw_dict.get('RA')
        obj_dec_str = kw_dict.get('OBJCTDEC') or kw_dict.get('DEC')
        if not obj_ra_str or not obj_dec_str:
            result_info['status'] = 'fail_no_objctra'
            result_info['solve_time_s'] = round(time.time() - t_start, 2)
            return result_info
        cra0 = _parse_ra_hms(obj_ra_str)
        cdec0 = _parse_dec_dms(obj_dec_str)

        result_info['width'] = w
        result_info['height'] = h
        result_info['focallen'] = fl
        result_info['pixel_size'] = ps
        result_info['s0_arcsec_px'] = round(s0, 4)
        result_info['fits_center_ra'] = cra0
        result_info['fits_center_dec'] = cdec0

        # ── 2. 星点检测 ──
        detector = StarDetector(params=SDetParamsPy(fitRadius=0))
        det = detector.detect_ex(img.data)
        n_detected = len(det.x)
        n_saturated = int(np.sum(det.saturated))
        result_info['n_detected'] = n_detected
        result_info['n_saturated'] = n_saturated
        detector = None  # 释放

        if n_detected < 5:
            result_info['status'] = 'fail_too_few_stars'
            result_info['solve_time_s'] = round(time.time() - t_start, 2)
            return result_info

        # ── 3. V4.0求解WCS ──
        wcs_json = os.path.join(output_dir, f"wcs_{os.path.basename(fits_path)}.json")
        t_solve = time.time()
        result = solver.solve(
            np.array(det.x, np.float64), np.array(det.y, np.float64),
            np.array(det.flux, np.float64), np.array(det.saturated, np.int32),
            cra0, cdec0, fl, ps, w, h,
            wcs_out=wcs_json,
            exptime=getattr(img.metadata.observation, 'exptime', 1.0),
        )
        solve_time = time.time() - t_solve

        if not result:
            result_info['status'] = 'fail_solve'
            result_info['solve_time_s'] = round(solve_time, 2)
            return result_info

        result_info['status'] = 'success'
        result_info['solve_time_s'] = round(solve_time, 2)
        result_info['flip_mode'] = result.flip_mode
        result_info['matched_count'] = result.matched_count
        result_info['phaseb_rms_px'] = round(result.rms_px, 4)  # Phase B初始匹配RMS(参考)
        result_info['scale_arcsec_px'] = round(result.scale_arcsec_px, 4)
        result_info['rotation_deg'] = round(result.rotation_deg, 4)
        result_info['center_ra'] = result.center_ra
        result_info['center_dec'] = result.center_dec

        # ── 4. 读取WCS JSON ──
        # V4.1修复: 检查WCS JSON文件是否存在, 不存在时用result兜底(跳过投影评估)
        if not os.path.exists(wcs_json):
            logger.warning(f"WCS JSON未写入, 使用result兜底: {wcs_json}")
            result_info['rms_px'] = round(float(getattr(result, 'sip_rms_px', result.rms_px)), 4)
            result_info['sip_order'] = 0
            result_info['has_sip'] = False
            result_info['wcs_json_missing'] = True
            result_info['pct_3px'] = 0.0
            result_info['pct_5px'] = 0.0
            result_info['pct_10px'] = 0.0
            result_info['median_dist_px'] = -1
            result_info['mean_dist_px'] = -1
            result_info['total_time_s'] = round(time.time() - t_start, 2)
            return result_info

        with open(wcs_json, 'r', encoding='utf-8') as f:
            wcs = json.load(f)

        # 使用SIP拟合后的RMS(最终精度指标), 优于Phase B初始匹配RMS
        result_info['rms_px'] = round(float(wcs.get('RMS_PX', getattr(result, 'sip_rms_px', 0.0))), 4)
        cd = np.array(wcs['CD'], dtype=np.float64).reshape(2, 2)
        crval = np.array(wcs['CRVAL'], dtype=np.float64)
        crpix = np.array(wcs['CRPIX'], dtype=np.float64)
        sip_A = np.array(wcs['SIP_A'], dtype=np.float64).reshape(6, 6)
        sip_B = np.array(wcs['SIP_B'], dtype=np.float64).reshape(6, 6)
        sip_order = int(wcs.get('SIP_ORDER', 0))

        result_info['wcs_cd'] = wcs['CD']
        result_info['wcs_crval'] = wcs['CRVAL']
        result_info['wcs_crpix'] = wcs['CRPIX']
        result_info['sip_order'] = sip_order
        # SIP系数后续单独保存(较大)
        result_info['has_sip'] = sip_order >= 2

        # ── 5. 查询Gaia亮星 ──
        fov_deg = math.sqrt(w * w + h * h) * s0 / 3600.0
        gaia = GaiaClientPy(os.path.join(PROJECT_ROOT, "GaiaDR3"), 1)
        query_radius = max(fov_deg * 0.7, 1.0)
        ra_t, dec_t, mag_t = gaia.cone_search(crval[0], crval[1], query_radius, 22.0)
        gaia.close()

        ra_all = np.array(ra_t, dtype=np.float64)
        dec_all = np.array(dec_t, dtype=np.float64)
        mag_all = np.array(mag_t, dtype=np.float64)

        if len(ra_all) > n_bright:
            idx_bright = np.argsort(mag_all)[:n_bright]
            ra_src = ra_all[idx_bright]
            dec_src = dec_all[idx_bright]
        else:
            ra_src = ra_all
            dec_src = dec_all
        result_info['n_gaia_queried'] = len(ra_all)
        result_info['n_gaia_bright'] = len(ra_src)

        # ── 6. 标准WCS-SIP逆投影 ──
        x_pix, y_pix = sky_to_pixel_wcs(
            ra_src, dec_src, cd, crval, crpix, sip_A, sip_B, sip_order)

        in_frame = np.isfinite(x_pix) & (x_pix > 0) & (x_pix < w) & \
                   (y_pix > 0) & (y_pix < h)
        x_in = x_pix[in_frame]
        y_in = y_pix[in_frame]
        result_info['n_in_frame'] = int(np.sum(in_frame))

        # ── 7. 投影质量诊断 ──
        from scipy.spatial import cKDTree
        det_tree = cKDTree(np.column_stack([det.x, det.y]))
        n_test = len(x_in)
        if n_test > 0:
            dists, _ = det_tree.query(np.column_stack([x_in, y_in]))
            n_3px = int(np.sum(dists < 3))
            n_5px = int(np.sum(dists < 5))
            n_10px = int(np.sum(dists < 10))
            result_info['n_3px'] = n_3px
            result_info['n_5px'] = n_5px
            result_info['n_10px'] = n_10px
            result_info['pct_3px'] = round(100 * n_3px / n_test, 1)
            result_info['pct_5px'] = round(100 * n_5px / n_test, 1)
            result_info['pct_10px'] = round(100 * n_10px / n_test, 1)
            # 中位残差
            if n_test > 0:
                result_info['median_dist_px'] = round(float(np.median(dists)), 3)
                result_info['mean_dist_px'] = round(float(np.mean(dists)), 3)
        else:
            result_info['n_3px'] = 0
            result_info['n_5px'] = 0
            result_info['n_10px'] = 0
            result_info['pct_3px'] = 0
            result_info['pct_5px'] = 0
            result_info['pct_10px'] = 0
            result_info['median_dist_px'] = -1
            result_info['mean_dist_px'] = -1

        # 总耗时
        result_info['total_time_s'] = round(time.time() - t_start, 2)

    except Exception as e:
        result_info['status'] = f'error: {str(e)[:100]}'
        result_info['total_time_s'] = round(time.time() - t_start, 2)
        logger.error(f"  异常: {fits_path}: {e}")
        traceback.print_exc()

    return result_info


# ============================================================================
# 主流程
# ============================================================================
def main():
    # 参数
    n_frames = int(sys.argv[1]) if len(sys.argv) >= 2 else 50
    seed = int(sys.argv[2]) if len(sys.argv) >= 3 else 42

    print(f"=== V4.0 批量WCS质量测试 ===")
    print(f"帧数: {n_frames}  随机种子: {seed}")

    # 收集FITS文件
    lights_dir = os.path.join(PROJECT_ROOT, "testdata", "lights")
    all_fits = collect_fits_files(lights_dir, max_depth=3)
    print(f"找到FITS文件: {len(all_fits)}个")

    if len(all_fits) < n_frames:
        print(f"警告: 只有{len(all_fits)}个文件, 少于请求的{n_frames}个, 全部使用")
        selected = all_fits
    else:
        random.seed(seed)
        selected = random.sample(all_fits, n_frames)

    print(f"随机抽取: {len(selected)}个")

    # 输出目录
    output_dir = os.path.join(PROJECT_ROOT, "lib", "plate_solve", "logs", "v4", "batch_test")
    os.makedirs(output_dir, exist_ok=True)

    # 结果列表
    results = []

    # 复用solver实例
    solver = VectorMatchV4Cpp(os.path.join(PROJECT_ROOT, "GaiaDR3"), db_type=1)

    # 逐帧测试
    for i, fits_path in enumerate(selected, 1):
        base = os.path.basename(fits_path)
        print(f"\n[{i}/{len(selected)}] {base}")

        info = solve_single_frame(fits_path, solver, output_dir)
        results.append(info)

        # 打印简要结果
        if info['status'] == 'success':
            print(f"  → 成功: mode={info['flip_mode']} n={info['matched_count']} "
                  f"RMS={info['rms_px']:.3f}px SIP={info['sip_order']} "
                  f"投影10px内={info.get('pct_10px', 0)}% "
                  f"耗时={info['solve_time_s']}s")
        else:
            print(f"  → {info['status']}")

    solver.close()

    # ── 保存JSON ──
    json_path = os.path.join(output_dir, f"batch_test_{n_frames}frames_seed{seed}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON已保存: {json_path}")

    # ── 保存CSV ──
    csv_path = os.path.join(output_dir, f"batch_test_{n_frames}frames_seed{seed}.csv")
    if results:
        # 选择CSV字段(排除大数组)
        csv_fields = [
            'filename', 'target', 'filter', 'exposure_s',
            'width', 'height', 'focallen', 'pixel_size', 's0_arcsec_px',
            'fits_center_ra', 'fits_center_dec',
            'n_detected', 'n_saturated',
            'status', 'flip_mode', 'matched_count', 'rms_px', 'phaseb_rms_px',
            'scale_arcsec_px', 'rotation_deg',
            'center_ra', 'center_dec',
            'sip_order', 'has_sip',
            'n_gaia_queried', 'n_gaia_bright', 'n_in_frame',
            'n_3px', 'n_5px', 'n_10px',
            'pct_3px', 'pct_5px', 'pct_10px',
            'median_dist_px', 'mean_dist_px',
            'solve_time_s', 'total_time_s',
        ]
        with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"CSV已保存: {csv_path}")

    # ── 汇总统计 ──
    print(f"\n=== 汇总统计 ===")
    n_total = len(results)
    n_success = sum(1 for r in results if r['status'] == 'success')
    n_fail = n_total - n_success
    print(f"总帧数: {n_total}")
    print(f"成功: {n_success} ({100*n_success/max(n_total,1):.1f}%)")
    print(f"失败: {n_fail} ({100*n_fail/max(n_total,1):.1f}%)")

    if n_success > 0:
        succ = [r for r in results if r['status'] == 'success']
        rms_list = [r['rms_px'] for r in succ]
        pct10_list = [r.get('pct_10px', 0) for r in succ]
        matched_list = [r['matched_count'] for r in succ]
        time_list = [r['solve_time_s'] for r in succ]

        print(f"\n成功帧统计:")
        print(f"  RMS(px): 中位={np.median(rms_list):.3f} 均值={np.mean(rms_list):.3f} "
              f"最小={np.min(rms_list):.3f} 最大={np.max(rms_list):.3f}")
        print(f"  投影10px内(%): 中位={np.median(pct10_list):.1f} 均值={np.mean(pct10_list):.1f} "
              f"最小={np.min(pct10_list):.1f} 最大={np.max(pct10_list):.1f}")
        print(f"  匹配对数: 中位={np.median(matched_list):.0f} 均值={np.mean(matched_list):.0f} "
              f"最小={np.min(matched_list)} 最大={np.max(matched_list)}")
        print(f"  求解耗时(s): 中位={np.median(time_list):.2f} 均值={np.mean(time_list):.2f} "
              f"最小={np.min(time_list):.2f} 最大={np.max(time_list):.2f}")

        # 按滤镜分组统计
        print(f"\n按滤镜分组:")
        filters = {}
        for r in succ:
            flt = r.get('filter', 'unknown')
            if flt not in filters:
                filters[flt] = []
            filters[flt].append(r)
        for flt in sorted(filters.keys()):
            fr = filters[flt]
            print(f"  {flt}: {len(fr)}帧 成功率={100*len(fr)/max(sum(1 for r in results if r.get('filter')==flt),1):.0f}% "
                  f"中位RMS={np.median([r['rms_px'] for r in fr]):.3f}px "
                  f"中位10px内={np.median([r.get('pct_10px',0) for r in fr]):.1f}%")

        # 按目标分组统计
        print(f"\n按目标分组:")
        targets = {}
        for r in succ:
            tgt = r.get('target', 'unknown')
            if tgt not in targets:
                targets[tgt] = []
            targets[tgt].append(r)
        for tgt in sorted(targets.keys()):
            tr = targets[tgt]
            print(f"  {tgt}: {len(tr)}帧 中位RMS={np.median([r['rms_px'] for r in tr]):.3f}px "
                  f"中位10px内={np.median([r.get('pct_10px',0) for r in tr]):.1f}%")

    # 失败帧列表
    if n_fail > 0:
        print(f"\n失败帧:")
        for r in results:
            if r['status'] != 'success':
                print(f"  {r['filename']}: {r['status']}")


if __name__ == "__main__":
    main()
