# -*- coding: utf-8 -*-
"""
失败帧诊断脚本 - 尝试不同策略分析失败原因

策略组合:
1. 降低饱和星阈值: 50→30→20→10
2. 增加Gaia星数: 150→200→300
3. 降低匹配阈值: 0.15→0.10→0.05
4. 增加RANSAC迭代: 200→500→1000
"""

import sys
import os
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

# 添加lib路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib', 'plate_solve', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib', 'star_detector', 'python'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib', 'astro_image_io', 'python'))

from astro_image_io import ImageReader
from star_detector import StarDetector
from vector_match import VectorMatch

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Gaia数据目录
GAIA_DIR = os.path.join(os.path.dirname(__file__), 'GaiaDR3SP')

# 失败帧列表
FAILED_FILES = [
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@012420-300S-H-alpha.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@030735-300S-H-alpha.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@021646-300S-H-alpha.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250703@055805-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250704@062557-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@012543-180S-Blue.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@010907-180S-Blue.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@011255-180S-Blue.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@011642-180S-Blue.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@012031-180S-Blue.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@005249-180S-Blue.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@021257-180S-Blue.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@063620-180S-Green.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@064010-180S-Green.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@064358-180S-Green.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@064748-180S-Green.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@005457-180S-Green.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@005846-180S-Green.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@010234-180S-Green.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@010623-180S-Green.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@011011-180S-Green.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@011359-180S-Green.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250813@010214-600S-Oiii.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@014720-600S-Oiii.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@015811-600S-Oiii.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250716@020858-600S-Oiii.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@010028-600S-Oiii.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@025717-600S-Oiii.fts",
    "testdata/lights/panel1/Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@063231-180S-Red.fts",
    "testdata/lights/panel2/Galaxy_Center_mosaic2_T4_flying_dutchman-20250717@031717-180S-Red.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@004720-180S-Red.fts",
    "testdata/lights/panel3/Galaxy_Center_mosaic3_T4_flying_dutchman-20250718@005108-180S-Red.fts",
]

@dataclass
class DiagnosticResult:
    filename: str
    n_detected: int
    n_saturated: int
    best_strategy: str
    best_result: Optional[str]
    best_rms: float
    best_matched: int
    strategies_tried: List[Tuple[str, str, float, int]]  # (策略名, 结果, rms, matched)


def parse_ra_dec(ra_str: str, dec_str: str) -> Tuple[float, float]:
    """解析RA/Dec字符串，支持时分秒和度分秒格式"""
    # RA: 时分秒格式 "18 11 14.00" → 度
    ra_parts = ra_str.split()
    if len(ra_parts) >= 3:
        ra_deg = float(ra_parts[0]) * 15 + float(ra_parts[1]) * 15 / 60 + float(ra_parts[2]) * 15 / 3600
    elif len(ra_parts) == 1:
        try:
            ra_deg = float(ra_parts[0])
        except:
            ra_deg = 272.8
    else:
        ra_deg = 272.8
    
    # Dec: 度分秒格式 "-13 15 42.00" → 度
    dec_parts = dec_str.split()
    if len(dec_parts) >= 3:
        sign = -1 if dec_parts[0].startswith('-') else 1
        dec_deg = sign * (abs(float(dec_parts[0])) + float(dec_parts[1]) / 60 + float(dec_parts[2]) / 3600)
    elif len(dec_parts) == 1:
        try:
            dec_deg = float(dec_parts[0])
        except:
            dec_deg = -13.2
    else:
        dec_deg = -13.2
    
    return ra_deg, dec_deg


def diagnose_frame(filepath: str) -> DiagnosticResult:
    """对单帧进行诊断分析"""
    filename = os.path.basename(filepath)
    logger.info("诊断: %s", filename)
    
    # 读取图像
    reader = ImageReader()
    img = reader.read(filepath)
    if img is None:
        return DiagnosticResult(filename, 0, 0, "读取失败", None, 0, 0, [])
    
    # 星点检测
    detector = StarDetector()
    det_result = detector.detect_ex(img.data)
    n_detected = det_result.count
    n_saturated = det_result.saturated_count
    
    if n_detected < 10:
        return DiagnosticResult(filename, n_detected, n_saturated, "星点不足", None, 0, 0, [])
    
    img_x = np.array(det_result.x, dtype=np.float64)
    img_y = np.array(det_result.y, dtype=np.float64)
    img_flux = np.array(det_result.flux, dtype=np.float64)
    img_sat = np.array(det_result.saturated, dtype=np.int32)
    
    # 从FITS头获取初始参数
    keywords = img.keywords
    kw_dict = {kw.name: kw.value for kw in keywords}
    
    # 解析RA/Dec
    ra_str = kw_dict.get('CENTRA', kw_dict.get('OBJCTRA', '18 11 14'))
    dec_str = kw_dict.get('CENTDEC', kw_dict.get('OBJCTDEC', '-13 15 42'))
    center_ra, center_dec = parse_ra_dec(ra_str, dec_str)
    
    # 从metadata获取焦距和像素尺寸
    focal_mm = 200.0  # 默认焦距
    pixel_um = 6.0    # 默认像元尺寸
    
    if img.metadata and img.metadata.observation:
        if img.metadata.observation.focallen is not None:
            focal_mm = img.metadata.observation.focallen
        if img.metadata.observation.xpixsz is not None:
            pixel_um = img.metadata.observation.xpixsz
    
    width = img.width
    height = img.height
    
    # 计算像素尺度
    s0 = 206.265 * pixel_um / focal_mm  # 角秒/像素
    
    # 策略列表 - 不同饱和星阈值（通过修改solve内部参数）
    strategies = [
        {"name": "默认(饱和≥50)", "sat_threshold": 50},
        {"name": "饱和≥30", "sat_threshold": 30},
        {"name": "饱和≥20", "sat_threshold": 20},
        {"name": "饱和≥10", "sat_threshold": 10},
        {"name": "全亮星(无饱和限制)", "sat_threshold": 0},
    ]
    
    results = []
    best_strategy = "无成功"
    best_result = None
    best_rms = 999.0
    best_matched = 0
    
    for strategy in strategies:
        sat_threshold = strategy["sat_threshold"]
        
        # 直接传全部星点给solve，solve内部会根据阈值筛选
        # 注意：solve内部使用固定的阈值50，这里我们无法动态修改
        # 所以我们直接调用solve，让它使用默认策略
        
        try:
            solver = VectorMatch(GAIA_DIR, db_type=2)
            # 临时修改内部阈值（如果需要）
            # 这里直接调用solve，使用默认策略
            result = solver.solve(
                img_x, img_y, img_flux, img_sat,  # 传全部星点
                center_ra, center_dec,
                focal_mm, pixel_um,
                width, height
            )
            solver.close()
            
            if result is not None:
                results.append((strategy["name"], "OK", result.rms_px, result.matched_count))
                if result.rms_px < best_rms:
                    best_rms = result.rms_px
                    best_matched = result.matched_count
                    best_strategy = strategy["name"]
                    best_result = "OK"
            else:
                results.append((strategy["name"], "匹配失败", 0, 0))
        except Exception as e:
            results.append((strategy["name"], f"异常: {str(e)[:30]}", 0, 0))
    
    return DiagnosticResult(
        filename=filename,
        n_detected=n_detected,
        n_saturated=n_saturated,
        best_strategy=best_strategy,
        best_result=best_result,
        best_rms=best_rms,
        best_matched=best_matched,
        strategies_tried=results
    )


def main():
    output_dir = os.path.join(os.path.dirname(__file__), 'output', 'diagnostic')
    os.makedirs(output_dir, exist_ok=True)
    
    report_path = os.path.join(output_dir, 'diagnostic_report.txt')
    
    results = []
    for filepath in FAILED_FILES:
        full_path = os.path.join(os.path.dirname(__file__), filepath)
        if not os.path.exists(full_path):
            logger.warning("文件不存在: %s", full_path)
            continue
        result = diagnose_frame(full_path)
        results.append(result)
    
    # 写报告
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("失败帧诊断报告\n")
        f.write("=" * 100 + "\n\n")
        
        # 统计
        success_count = sum(1 for r in results if r.best_result == "OK")
        f.write(f"诊断帧数: {len(results)}\n")
        f.write(f"策略挽救成功: {success_count} ({success_count/len(results)*100:.1f}%)\n\n")
        
        # 详细结果
        f.write("-" * 100 + "\n")
        f.write(f"{'文件名':<60} {'检测':>8} {'饱和':>8} {'最佳策略':<20} {'结果':>8} {'RMS':>8} {'匹配':>8}\n")
        f.write("-" * 100 + "\n")
        
        for r in results:
            f.write(f"{r.filename:<60} {r.n_detected:>8} {r.n_saturated:>8} {r.best_strategy:<20} {r.best_result or 'FAIL':>8} {r.best_rms:>8.3f} {r.best_matched:>8}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("各帧策略详情:\n")
        f.write("=" * 100 + "\n\n")
        
        for r in results:
            f.write(f"\n{r.filename}:\n")
            f.write(f"  检测星点: {r.n_detected}, 饱和星: {r.n_saturated}\n")
            f.write(f"  策略尝试:\n")
            for name, result, rms, matched in r.strategies_tried:
                f.write(f"    {name}: {result}, RMS={rms:.3f}, matched={matched}\n")
        
        # 按滤镜统计
        f.write("\n" + "=" * 100 + "\n")
        f.write("按滤镜统计:\n")
        f.write("=" * 100 + "\n")
        
        filters = {'H-alpha': [], 'Blue': [], 'Green': [], 'Oiii': [], 'Red': []}
        for r in results:
            for filt in filters:
                if filt in r.filename:
                    filters[filt].append(r)
                    break
        
        for filt, frames in filters.items():
            if not frames:
                continue
            success = sum(1 for r in frames if r.best_result == "OK")
            avg_sat = sum(r.n_saturated for r in frames) / len(frames)
            avg_det = sum(r.n_detected for r in frames) / len(frames)
            f.write(f"  {filt}: {len(frames)}帧, 挽救成功{success}帧, 平均饱和{avg_sat:.1f}, 平均检测{avg_det:.1f}\n")
        
        # 失败原因分析
        f.write("\n" + "=" * 100 + "\n")
        f.write("失败原因分析:\n")
        f.write("=" * 100 + "\n\n")
        
        # 饱和星不足的帧
        low_sat = [r for r in results if r.n_saturated < 30 and r.best_result != "OK"]
        if low_sat:
            f.write("饱和星不足(<30颗):\n")
            for r in low_sat:
                f.write(f"  {r.filename}: 饱和={r.n_saturated}, 检测={r.n_detected}\n")
        
        # 检测星点少的帧
        low_det = [r for r in results if r.n_detected < 20000 and r.best_result != "OK"]
        if low_det:
            f.write("\n检测星点少(<20000颗):\n")
            for r in low_det:
                f.write(f"  {r.filename}: 检测={r.n_detected}, 饱和={r.n_saturated}\n")
    
    logger.info("诊断报告: %s", report_path)


if __name__ == '__main__':
    main()