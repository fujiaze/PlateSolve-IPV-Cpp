"""向量匹配调试图生成器 - 在原图上标注用于匹配的 U 向量组

功能:
    对低质量帧 (Type2/Type3) 生成调试图, 在原图上标注:
    - 绿色圆: 所有检测到的星点
    - 红色箭头: 从图像中心到 U 向量组饱和星的向量
    - 蓝色箭头: 从图像中心到 U 向量组非饱和补充星的向量
    - 黄色十字: 最终匹配对中的图像侧星点
    - 图像标题: 文件名 + 关键指标 (RMS, lnK, matched, mode)

    输出 PNG 缩略图 (最大 2048px 边长)

用法:
    python 向量匹配调试图.py                          # 生成所有低质量帧调试图
    python 向量匹配调试图.py --type type3              # 仅 Type3 (lnK=0+RMS<1+matched≥5)
    python 向量匹配调试图.py --type type2              # 仅 Type2 (RMS>50px)
    python 向量匹配调试图.py --frames "xxx.fts yyy.fts" # 指定帧
"""
import os
import sys
import json
import argparse
import re
import math

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

import numpy as np

# Pillow 用于图像绘制
from PIL import Image, ImageDraw, ImageFont

from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy
from vector_match_v2 import GaiaClientPy
from v4_3.vector_match_v4_3_cpp import V43Solver


_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "debug_images")

_FULL_TEST_JSON = os.path.join(
    PROJECT_ROOT, "lib", "plate_solve", "logs", "v4_3", "full_test", "full_test_all.json")

TESTDATA = os.path.join(PROJECT_ROOT, "testdata")


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


def find_fits_path(filename):
    """在 testdata 中递归查找 FITS 文件"""
    for dirpath, dirnames, filenames in os.walk(TESTDATA):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def classify_frame(r):
    """分类帧类型"""
    rms = r.get("rms_px", 0)
    lnK = r.get("bayes_lnK", 0)
    matched = r.get("matched_count", 0)
    if matched <= 2 and lnK == 0 and rms == 0:
        return "type1_sparse"
    if rms > 50:
        return "type2_wrong"
    if lnK == 0 and rms < 1 and matched >= 5:
        return "type3_verify"
    if 0 < lnK < 10:
        return "low_lnK"
    if rms > 5:
        return "high_rms"
    return "other"


def generate_debug_image(fits_path, solver, output_path):
    """生成单帧调试图"""
    base = os.path.basename(fits_path)

    # 读取图像
    reader = ImageReader()
    img = reader.read(fits_path)
    w, h = img.width, img.height
    pixels = img.to_numpy()  # numpy array (float32 or uint16)

    # 获取 FITS header 信息
    fl = img.metadata.observation.focallen
    ps = img.metadata.observation.xpixsz
    s0 = 206.265 * ps / fl

    kws = img.keywords
    kw_dict = {k.name.upper(): k.value for k in kws}
    obj_ra_str = kw_dict.get("OBJCTRA") or kw_dict.get("RA")
    obj_dec_str = kw_dict.get("OBJCTDEC") or kw_dict.get("DEC")
    cra0 = _parse_ra_hms(obj_ra_str)
    cdec0 = _parse_dec_dms(obj_dec_str)

    # 星点检测 (与 solver 内部使用相同的 detector)
    star_detector = solver._star_detector
    # detect_ex 接受 uint16 或 float32 (内部会 clip 转 uint16)
    pixels_u16 = np.clip(pixels, 0, 65535).astype(np.uint16) if pixels.dtype != np.uint16 else pixels
    det_result = star_detector.detect_ex(pixels_u16)

    all_x = det_result.x
    all_y = det_result.y
    all_flux = det_result.flux
    all_sat = det_result.saturated
    n_total = len(all_x)

    # 分类: 饱和星 vs 正常星
    sat_indices = [i for i in range(n_total) if all_sat[i]]
    normal_indices = [i for i in range(n_total) if not all_sat[i]]

    # U 向量组选取逻辑 (与 C++ vm43_select 一致)
    # 饱和星全选 (如果 > img_n_target=50)
    # 否则取所有饱和星 + 补充亮正常星到 50
    img_n_target = 50
    if len(sat_indices) >= img_n_target:
        u_indices = list(sat_indices[:])  # 全选饱和星
        u_type = "sat_only"
    else:
        # 取所有饱和星 + 正常星补到 img_n_target
        n_need = img_n_target - len(sat_indices)
        u_indices = list(sat_indices)
        u_indices.extend(normal_indices[:n_need])
        u_type = "sat+normal"

    u_set = set(u_indices)

    # 求解 (获取匹配对 cu)
    frame_base = os.path.splitext(base)[0]
    log_dir = os.path.join(_OUTPUT_DIR, "logs", frame_base)
    result = solver.solve(
        image_path=fits_path,
        ra=cra0, dec=cdec0,
        focal_length_mm=fl, pixel_size_um=ps,
        log_dir=log_dir,
    )

    # 匹配对中的图像侧索引
    cu = result.get("cu", [])
    cu_set = set(cu)

    # ---- 绘制图像 ----
    # 缩放到最大 2048
    max_dim = 2048
    scale = min(max_dim / w, max_dim / h, 1.0)
    sw, sh = int(w * scale), int(h * scale)

    # 转为 8bit 灰度 (float32 → 百分位拉伸)
    pf = pixels.astype(np.float32)
    p2 = np.percentile(pf, 2)
    p98 = np.percentile(pf, 98)
    if p98 <= p2:
        p98 = p2 + 1
    stretched = np.clip((pf - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

    # RGB 图像
    rgb = np.stack([stretched, stretched, stretched], axis=-1)
    pil_img = Image.fromarray(rgb)
    pil_img = pil_img.resize((sw, sh), Image.LANCZOS)
    draw = ImageDraw.Draw(pil_img)

    # 绘制参数
    r_all = max(1, int(2 * scale))       # 所有星点半径
    r_u = max(2, int(4 * scale))          # U 向量组星点半径
    r_match = max(3, int(6 * scale))      # 匹配星点半径

    # 图像中心 (缩放后坐标)
    center_x = sw / 2.0
    center_y = sh / 2.0

    def _draw_arrow(draw, x0, y0, x1, y1, color, width=1, head_len=6, head_angle=25):
        """绘制带箭头的向量线"""
        draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
        # 箭头
        dx = x1 - x0
        dy = y1 - y0
        length = math.sqrt(dx * dx + dy * dy)
        if length < head_len:
            return
        # 单位方向向量
        ux, uy = dx / length, dy / length
        # 法线
        nx, ny = -uy, ux
        angle_rad = math.radians(head_angle)
        # 箭头两翼
        hl = head_len
        ax = x1 - hl * (ux * math.cos(angle_rad) + nx * math.sin(angle_rad))
        ay = y1 - hl * (uy * math.cos(angle_rad) + ny * math.sin(angle_rad))
        bx = x1 - hl * (ux * math.cos(angle_rad) - nx * math.sin(angle_rad))
        by = y1 - hl * (uy * math.cos(angle_rad) - ny * math.sin(angle_rad))
        draw.polygon([(x1, y1), (ax, ay), (bx, by)], fill=color)

    # 1. 先画所有星点 (绿色小圆)
    for i in range(n_total):
        cx = int(all_x[i] * scale)
        cy = int(all_y[i] * scale)
        if 0 <= cx < sw and 0 <= cy < sh:
            draw.ellipse([cx - r_all, cy - r_all, cx + r_all, cy + r_all],
                         outline=(0, 180, 0), width=1)

    # 2. 画 U 向量组: 从图像中心出发的向量箭头
    #    饱和星=红色箭头, 非饱和补充星=蓝色箭头
    arrow_w = max(1, int(1 * scale))
    head_l = max(4, int(6 * scale))
    for idx in u_indices:
        cx = int(all_x[idx] * scale)
        cy = int(all_y[idx] * scale)
        if 0 <= cx < sw and 0 <= cy < sh:
            if all_sat[idx]:
                color = (255, 60, 60)   # 红色: 饱和星
            else:
                color = (60, 120, 255)  # 蓝色: 非饱和补充星
            _draw_arrow(draw, center_x, center_y, cx, cy,
                        color=color, width=arrow_w, head_len=head_l)
            # 星点位置画小圆
            draw.ellipse([cx - r_u, cy - r_u, cx + r_u, cy + r_u],
                         outline=color, width=2)

    # 3. 画匹配对中的图像侧星点 (黄色十字)
    for idx in cu:
        if 0 <= idx < n_total:
            cx = int(all_x[idx] * scale)
            cy = int(all_y[idx] * scale)
            if 0 <= cx < sw and 0 <= cy < sh:
                arm = r_match
                draw.line([cx - arm, cy, cx + arm, cy], fill=(255, 255, 0), width=2)
                draw.line([cx, cy - arm, cx, cy + arm], fill=(255, 255, 0), width=2)

    # 4. 画图像中心标记 (白色十字 + 圆)
    cr = max(4, int(8 * scale))
    draw.line([center_x - cr, center_y, center_x + cr, center_y], fill=(255, 255, 255), width=2)
    draw.line([center_x, center_y - cr, center_x, center_y + cr], fill=(255, 255, 255), width=2)
    draw.ellipse([center_x - cr//2, center_y - cr//2, center_x + cr//2, center_y + cr//2],
                 outline=(255, 255, 255), width=2)

    # 4. 图像标题
    rms_px = result.get("rms_px", 0)
    lnK = result.get("bayes_lnK", 0)
    matched = result.get("matched_count", 0)
    mode = result.get("flip_mode", -1)
    theta = result.get("rotation_deg", 0)
    s_robust = result.get("s_robust", 0)
    irm_conv = result.get("irm_converged", False)
    n_iters = result.get("n_iters", 0)

    title = (f"{base}\n"
             f"RMS={rms_px:.3f}px lnK={lnK:.1f} matched={matched} "
             f"mode={mode} θ={theta:.1f}°\n"
             f"S_robust={s_robust:.2f}\" iters={n_iters} conv={irm_conv} | "
             f"U: {len(sat_indices)}sat+{len(u_indices)-len(sat_indices)}norm "
             f"(total {n_total} stars)")

    # 标题位置
    draw.text((10, 10), title, fill=(255, 255, 0))

    # 5. 图例
    legend_y = sh - 100
    # 白色中心标记
    draw.line([10, legend_y + 6, 18, legend_y + 6], fill=(255, 255, 255), width=2)
    draw.line([14, legend_y + 2, 14, legend_y + 10], fill=(255, 255, 255), width=2)
    draw.text((24, legend_y), "Image center", fill=(255, 255, 255))
    # 绿色小圆 = 所有星
    draw.ellipse([10, legend_y + 18, 18, legend_y + 26], outline=(0, 180, 0), width=1)
    draw.text((24, legend_y + 16), "All stars", fill=(0, 180, 0))
    # 红色箭头 = U饱和星向量
    _draw_arrow(draw, 10, legend_y + 38, 18, legend_y + 38, (255, 60, 60), width=arrow_w, head_len=4)
    draw.text((24, legend_y + 32), "U: sat vectors", fill=(255, 60, 60))
    # 蓝色箭头 = U非饱和向量
    _draw_arrow(draw, 10, legend_y + 56, 18, legend_y + 56, (60, 120, 255), width=arrow_w, head_len=4)
    draw.text((24, legend_y + 50), "U: normal vectors", fill=(60, 120, 255))
    # 黄色十字 = 匹配对
    arm = r_match
    lx = 14
    ly = legend_y + 72
    draw.line([lx - arm, ly, lx + arm, ly], fill=(255, 255, 0), width=2)
    draw.line([lx, ly - arm, lx, ly + arm], fill=(255, 255, 0), width=2)
    draw.text((24, legend_y + 68), "Matched pair", fill=(255, 255, 0))

    pil_img.save(output_path, "PNG")
    return {
        "filename": base,
        "rms_px": rms_px,
        "lnK": lnK,
        "matched": matched,
        "n_total_stars": n_total,
        "n_sat": len(sat_indices),
        "n_u": len(u_indices),
        "n_matched_img": len(cu),
        "u_type": u_type,
        "mode": mode,
        "theta": theta,
    }


def main():
    parser = argparse.ArgumentParser(description="向量匹配调试图生成器")
    parser.add_argument("--type", choices=["type1", "type2", "type3", "low_lnK", "all"],
                        default="all", help="帧类型过滤")
    parser.add_argument("--frames", nargs="*", default=None,
                        help="指定帧文件名 (不含路径)")
    parser.add_argument("--limit", type=int, default=50,
                        help="最大生成帧数 (默认 50)")
    args = parser.parse_args()

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    print("=== 向量匹配调试图生成器 ===\n")

    # 读取全量测试结果
    if not os.path.exists(_FULL_TEST_JSON):
        print(f"错误: 未找到全量测试结果 {_FULL_TEST_JSON}")
        return

    with open(_FULL_TEST_JSON, "r", encoding="utf-8") as f:
        all_results = json.load(f)

    # 筛选低质量帧
    type_map = {
        "type1": "type1_sparse",
        "type2": "type2_wrong",
        "type3": "type3_verify",
        "low_lnK": "low_lnK",
    }

    low_quality = []
    for r in all_results:
        if r.get("status") != "success":
            continue
        ft = classify_frame(r)
        if ft == "other":
            continue
        r["_type"] = ft
        if args.type == "all" or type_map.get(args.type) == ft:
            low_quality.append(r)

    # 按指定帧名过滤
    if args.frames:
        name_set = set(args.frames)
        low_quality = [r for r in low_quality if r.get("filename", "") in name_set]

    # 限制数量
    low_quality = low_quality[:args.limit]

    print(f"待生成帧数: {len(low_quality)}")
    if not low_quality:
        print("无符合条件的帧")
        return

    # 按类型统计
    type_count = {}
    for r in low_quality:
        t = r["_type"]
        type_count[t] = type_count.get(t, 0) + 1
    for t, c in sorted(type_count.items()):
        print(f"  {t}: {c} 帧")

    # 创建求解器
    print("\n初始化求解器...")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    gaia_client = GaiaClientPy(gaia_dir, db_type=1)
    star_detector = StarDetector(params=SDetParamsPy(fitRadius=0))
    solver = V43Solver(gaia_client=gaia_client, star_detector=star_detector)

    results = []
    for i, r in enumerate(low_quality, 1):
        fn = r.get("filename", "")
        fits_path = find_fits_path(fn)
        if not fits_path:
            print(f"[{i}/{len(low_quality)}] 跳过 (文件未找到): {fn}")
            continue

        print(f"[{i}/{len(low_quality)}] {fn} ({r['_type']})")

        output_path = os.path.join(_OUTPUT_DIR, os.path.splitext(fn)[0] + ".png")
        try:
            info = generate_debug_image(fits_path, solver, output_path)
            info["type"] = r["_type"]
            results.append(info)
            print(f"  → U={info['n_u']}({info['u_type']}) matched={info['n_matched_img']} "
                  f"RMS={info['rms_px']:.3f}px lnK={info['lnK']:.1f} mode={info['mode']}")
        except Exception as e:
            print(f"  → 错误: {e}")

    solver.close()
    gaia_client.close()
    star_detector.close()

    # 汇总
    print(f"\n=== 汇总 ===")
    print(f"生成调试图: {len(results)} 帧")
    print(f"输出目录: {_OUTPUT_DIR}")

    # 按类型汇总
    for t in ["type2_wrong", "type3_verify", "low_lnK", "type1_sparse"]:
        subset = [r for r in results if r.get("type") == t]
        if subset:
            print(f"\n{t} ({len(subset)} 帧):")
            for r in subset:
                print(f"  {r['filename']}: U={r['n_u']} matched={r['n_matched_img']} "
                      f"RMS={r['rms_px']:.3f}px lnK={r['lnK']:.1f} mode={r['mode']}")

    # 保存元数据
    meta_path = os.path.join(_OUTPUT_DIR, "debug_images_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n元数据已保存: {meta_path}")


if __name__ == "__main__":
    main()
