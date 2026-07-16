# -*- coding: utf-8 -*-
"""
4SADQ-KV 盲解析管线 端到端测试 (Task 8)
功能: 在 testdata 多望远镜帧上验证 4SADQ-KV 盲解析核心机制(k-vector+6绝对角距+四边形匹配)
用途: Task 8 验收测试, 输出每帧解析结果并与 FITS 头 WCS/已知天体位置对比

测试帧覆盖:
    - T2 中焦: M20 (4096x4096, s0≈0.967, 有 WCS)
    - T1 同 T2 焦距: LDN43 (4096x4096, s0≈0.967, 有 WCS)
    - T3 中焦: NGC55 (4096x4096, s0≈0.989, 无 WCS, 用已知天体位置)
    - T4 短焦宽场: 银心 (4500x3600, s0≈6.188, 无 WCS, 用已知天体位置)

验收标准 (spec):
    - 解析 CRVAL 与基线角距离 < 30"
    - 解析 RMS < 3"

运行:
    cd <project_root>
    python -m lib.plate_solve.blind_index.tests.test_pipeline
    或
    python lib/plate_solve/blind_index/tests/test_pipeline.py
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 将项目根目录加入 sys.path (兼容直接运行与 -m 运行)
_PROJECT_ROOT = os.path.normpath(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.plate_solve.blind_index.python.pipeline import solve_blind, SolveResult
from lib.plate_solve.blind_index.python.wcs_solver import angular_separation_arcsec

# 模块路径
_MODULE_ROOT = os.path.normpath(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
_LOG_DIR = os.path.join(_MODULE_ROOT, "logs")
_REPORT_FILE = os.path.join(_LOG_DIR, "test_report.txt")


# ════════════════════════════════════════════════════════════════════
# 测试帧定义
# ════════════════════════════════════════════════════════════════════
# 每帧字段:
#   path: 相对项目根的图像路径
#   name: 简短标识
#   s0: 像素尺度(arcsec/pixel), None=从 FITS 头读取
#   query_ra/query_dec: DR3 查询中心(度), None=从 FITS 头读取
#   expected_crval: (ra, dec) 基线位置(度), 用于角距离对比; None=无法对比
#   note: 备注
TEST_FRAMES = [
    # T2 中焦, 有 WCS, 完全可对比
    {
        "path": r"testdata\lights\M20_T2_flying_dutchman-20250719@005043-300S-Green.fts",
        "name": "M20_T2_Green",
        "s0": 0.966883202701463,         # 从 WCS pixel_scale 读取
        "query_ra": 270.700029168,
        "query_dec": -22.849924116,
        "expected_crval": (270.700029168, -22.849924116),
        "note": "T2中焦(1917.6mm), 有WCS, M20三叶星云",
    },
    # T1 同 T2 焦距, 有 WCS
    {
        "path": r"testdata\lights\LDN43_LRGBH_flying_dutchman-20250503@031525-600S-Lum.fts",
        "name": "LDN43_T1_Lum",
        "s0": 0.9668267272256788,
        "query_ra": 248.60953524,
        "query_dec": -15.75893659,
        "expected_crval": (248.60953524, -15.75893659),
        "note": "T1同T2焦距(1917.6mm), 有WCS, LDN43暗星云",
    },
    # T3 中焦, 无 WCS, 用已知天体位置 (NGC55≈RA 3.7 Dec -39.2)
    {
        "path": r"testdata\lights\NGC55_T3_flying_dutchman-20250701@074114-600S-Red.fts",
        "name": "NGC55_T3_Red",
        "s0": 0.9893,                    # 206.265 * 9 / 1877
        "query_ra": 3.7,
        "query_dec": -39.2,
        "expected_crval": (3.7, -39.2),
        "note": "T3中焦(1877mm), 无WCS, NGC55星系(已知位置RA 3.7 Dec -39.2)",
    },
    # T4 短焦宽场, 无 WCS, 用已知天体位置 (银心≈RA 266.4 Dec -29.0)
    {
        "path": r"testdata\lights1\panel1\Galaxy_Center_mosaic1_T4_flying_dutchman-20250702@061703-180S-Red.fts",
        "name": "GalaxyCenter_T4_Red",
        "s0": 6.18795,                   # 206.265 * 6 / 200
        "query_ra": 266.4,
        "query_dec": -29.0,
        "expected_crval": (266.4, -29.0),
        "note": "T4短焦宽场(200mm), 无WCS, 银心(已知位置RA 266.4 Dec -29.0)",
    },
]

# 验收阈值
ACCEPT_CRVAL_DEVIATION_ARCSEC = 30.0   # CRVAL 偏差 < 30"
ACCEPT_RMS_ARCSEC = 3.0                # RMS < 3"


# ════════════════════════════════════════════════════════════════════
# 结果容器
# ════════════════════════════════════════════════════════════════════
@dataclass
class FrameResult:
    """单帧测试结果"""
    name: str
    path: str
    s0: float
    query_ra: float
    query_dec: float
    expected_crval: Optional[tuple] = None
    success: bool = False
    rms_arcsec: float = float("inf")
    n_detected: int = 0
    n_reference: int = 0
    n_image_quads: int = 0
    n_reference_quads: int = 0
    n_candidates: int = 0
    best_votes: int = 0
    solved_crval: Optional[tuple] = None
    crval_deviation_arcsec: Optional[float] = None
    total_time: float = 0.0
    stage_timings: dict = field(default_factory=dict)
    message: str = ""
    pass_crval: bool = False
    pass_rms: bool = False


# ════════════════════════════════════════════════════════════════════
# 单帧测试
# ════════════════════════════════════════════════════════════════════
def run_one_frame(frame_cfg: dict, mag_limit: float = 12.0) -> FrameResult:
    """
    对单帧执行 solve_blind 并收集结果。

    Args:
        frame_cfg: TEST_FRAMES 中的字典
        mag_limit: DR3 极限星等, 参考星<50 时升级到 13.0

    Returns:
        FrameResult
    """
    abs_path = os.path.join(_PROJECT_ROOT, frame_cfg["path"])
    res = FrameResult(
        name=frame_cfg["name"],
        path=frame_cfg["path"],
        s0=frame_cfg["s0"],
        query_ra=frame_cfg["query_ra"],
        query_dec=frame_cfg["query_dec"],
        expected_crval=frame_cfg.get("expected_crval"),
    )

    print(f"\n{'=' * 70}")
    print(f"[测试帧] {res.name}")
    print(f"  路径: {res.path}")
    print(f"  s0={res.s0:.4f} arcsec/pix, query=({res.query_ra:.4f}, {res.query_dec:.4f})")
    print(f"  备注: {frame_cfg['note']}")
    print(f"{'=' * 70}")

    if not os.path.exists(abs_path):
        res.message = f"文件不存在: {abs_path}"
        print(f"  [失败] {res.message}")
        return res

    # 第一次尝试 mag_limit=12.0
    t_start = time.time()
    try:
        result: SolveResult = solve_blind(
            image_path=abs_path,
            s0_arcsec_per_pixel=res.s0,
            query_center_ra=res.query_ra,
            query_center_dec=res.query_dec,
            mag_limit=mag_limit,
        )
    except Exception as e:
        res.message = f"solve_blind 异常: {e}"
        res.total_time = time.time() - t_start
        print(f"  [异常] {res.message}")
        import traceback
        traceback.print_exc()
        return res

    # 参考星<50 时升级 mag_limit 到 13.0 重试
    if result.n_reference_stars < 50 and not result.success and mag_limit < 13.0:
        print(f"  [升级] 参考星仅 {result.n_reference_stars} 颗, mag_limit {mag_limit}->13.0 重试")
        mag_limit = 13.0
        try:
            result = solve_blind(
                image_path=abs_path,
                s0_arcsec_per_pixel=res.s0,
                query_center_ra=res.query_ra,
                query_center_dec=res.query_dec,
                mag_limit=mag_limit,
            )
        except Exception as e:
            res.message = f"升级后 solve_blind 异常: {e}"
            res.total_time = time.time() - t_start
            print(f"  [异常] {res.message}")
            return res

    res.total_time = time.time() - t_start
    res.success = result.success
    res.rms_arcsec = result.best_rms_arcsec
    res.n_detected = result.n_detected_stars
    res.n_reference = result.n_reference_stars
    res.n_image_quads = result.n_image_quads
    res.n_reference_quads = result.n_reference_quads
    res.n_candidates = result.n_candidates_total
    res.best_votes = result.best_votes
    res.stage_timings = dict(result.stage_timings)
    res.message = result.message

    if result.wcs is not None:
        res.solved_crval = (result.wcs.crval1, result.wcs.crval2)
        # CRVAL 偏差对比
        if res.expected_crval is not None:
            ra1 = np.array([res.solved_crval[0]])
            dec1 = np.array([res.solved_crval[1]])
            ra2 = np.array([res.expected_crval[0]])
            dec2 = np.array([res.expected_crval[1]])
            sep = float(angular_separation_arcsec(ra1, dec1, ra2, dec2)[0])
            res.crval_deviation_arcsec = sep
            res.pass_crval = sep < ACCEPT_CRVAL_DEVIATION_ARCSEC
        # RMS 验收
        res.pass_rms = res.rms_arcsec < ACCEPT_RMS_ARCSEC

    # 打印单帧摘要
    print(f"\n  --- 单帧结果 [{res.name}] ---")
    print(f"  success={res.success}, message={res.message}")
    print(f"  检测星={res.n_detected}, 参考星={res.n_reference}, "
          f"图像四边形={res.n_image_quads}, 参考四边形={res.n_reference_quads}")
    print(f"  候选总数={res.n_candidates}, 最佳票数={res.best_votes}")
    print(f"  RMS={res.rms_arcsec:.3f}\" (阈值<{ACCEPT_RMS_ARCSEC}\", 通过={res.pass_rms})")
    if res.solved_crval:
        print(f"  solved CRVAL=({res.solved_crval[0]:.5f}, {res.solved_crval[1]:.5f})")
    if res.expected_crval:
        print(f"  expected CRVAL=({res.expected_crval[0]:.5f}, {res.expected_crval[1]:.5f})")
    if res.crval_deviation_arcsec is not None:
        print(f"  CRVAL 偏差={res.crval_deviation_arcsec:.2f}\" "
              f"(阈值<{ACCEPT_CRVAL_DEVIATION_ARCSEC}\", 通过={res.pass_crval})")
    print(f"  总耗时={res.total_time:.3f}s")
    if res.stage_timings:
        timing_str = ", ".join(f"{k}={v:.3f}s" for k, v in res.stage_timings.items())
        print(f"  阶段耗时: {timing_str}")

    return res


# ════════════════════════════════════════════════════════════════════
# 报告生成
# ════════════════════════════════════════════════════════════════════
def format_report(results: list) -> str:
    """生成完整测试报告字符串"""
    lines = []
    lines.append("=" * 90)
    lines.append("4SADQ-KV 盲解析管线 端到端测试报告 (Task 8)")
    lines.append("=" * 90)
    lines.append(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"测试帧数: {len(results)}")
    lines.append(f"验收标准: CRVAL 偏差 < {ACCEPT_CRVAL_DEVIATION_ARCSEC}\", RMS < {ACCEPT_RMS_ARCSEC}\"")
    lines.append("")

    # 每帧详细
    lines.append("-" * 90)
    lines.append("【单帧详细】")
    lines.append("-" * 90)
    for r in results:
        lines.append(f"\n■ {r.name}")
        lines.append(f"  路径: {r.path}")
        lines.append(f"  s0={r.s0:.4f} arcsec/pix")
        lines.append(f"  query_center=({r.query_ra:.4f}, {r.query_dec:.4f})")
        lines.append(f"  success={r.success}, message={r.message}")
        lines.append(f"  n_detected={r.n_detected}, n_reference={r.n_reference}, "
                     f"n_image_quads={r.n_image_quads}, n_reference_quads={r.n_reference_quads}")
        lines.append(f"  n_candidates={r.n_candidates}, best_votes={r.best_votes}")
        lines.append(f"  RMS={r.rms_arcsec:.3f}\" (通过={r.pass_rms}, 阈值<{ACCEPT_RMS_ARCSEC}\")")
        if r.solved_crval:
            lines.append(f"  solved_crval=({r.solved_crval[0]:.5f}, {r.solved_crval[1]:.5f})")
        if r.expected_crval:
            lines.append(f"  expected_crval=({r.expected_crval[0]:.5f}, {r.expected_crval[1]:.5f})")
        if r.crval_deviation_arcsec is not None:
            lines.append(f"  CRVAL偏差={r.crval_deviation_arcsec:.2f}\" "
                         f"(通过={r.pass_crval}, 阈值<{ACCEPT_CRVAL_DEVIATION_ARCSEC}\")")
        lines.append(f"  total_time={r.total_time:.3f}s")
        if r.stage_timings:
            timing_str = ", ".join(f"{k}={v:.3f}s" for k, v in r.stage_timings.items())
            lines.append(f"  stage_timings: {timing_str}")

    # 汇总表
    lines.append("")
    lines.append("-" * 90)
    lines.append("【汇总表】")
    lines.append("-" * 90)
    header = f"{'name':<22} {'s0':>7} {'n_det':>6} {'n_ref':>6} {'n_quads':>7} {'n_cand':>7} {'votes':>6} {'RMS\"':>7} {'dev\"':>8} {'time(s)':>8} {'crval':>6} {'rms':>5} {'succ':>5}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        crval_ok = "✓" if r.pass_crval else ("—" if r.crval_deviation_arcsec is None else "✗")
        rms_ok = "✓" if r.pass_rms else "✗"
        succ = "✓" if r.success else "✗"
        dev_str = f"{r.crval_deviation_arcsec:.2f}" if r.crval_deviation_arcsec is not None else "N/A"
        rms_str = f"{r.rms_arcsec:.3f}" if r.rms_arcsec != float("inf") else "inf"
        n_quads_str = f"{r.n_image_quads}/{r.n_reference_quads}"
        lines.append(
            f"{r.name:<22} {r.s0:>7.4f} {r.n_detected:>6d} {r.n_reference:>6d} "
            f"{n_quads_str:>7} {r.n_candidates:>7d} {r.best_votes:>6d} "
            f"{rms_str:>7} {dev_str:>8} {r.total_time:>8.3f} {crval_ok:>6} {rms_ok:>5} {succ:>5}"
        )

    # 统计
    lines.append("")
    lines.append("-" * 90)
    lines.append("【统计】")
    lines.append("-" * 90)
    n = len(results)
    n_success = sum(1 for r in results if r.success)
    n_pass_crval = sum(1 for r in results if r.pass_crval)
    n_pass_rms = sum(1 for r in results if r.pass_rms)
    n_pass_both = sum(1 for r in results if r.pass_crval and r.pass_rms)
    rms_list = [r.rms_arcsec for r in results if r.success and r.rms_arcsec != float("inf")]
    time_list = [r.total_time for r in results if r.success]
    cand_list = [r.n_candidates for r in results if r.success]
    lines.append(f"成功率 (success)        : {n_success}/{n} = {n_success/n*100:.1f}%")
    lines.append(f"CRVAL 通过率 (<{ACCEPT_CRVAL_DEVIATION_ARCSEC}\")  : {n_pass_crval}/{n} = {n_pass_crval/n*100:.1f}%")
    lines.append(f"RMS 通过率 (<{ACCEPT_RMS_ARCSEC}\")        : {n_pass_rms}/{n} = {n_pass_rms/n*100:.1f}%")
    lines.append(f"双通过率 (CRVAL+RMS)    : {n_pass_both}/{n} = {n_pass_both/n*100:.1f}%")
    if rms_list:
        lines.append(f"成功帧 RMS: mean={np.mean(rms_list):.3f}\", median={np.median(rms_list):.3f}\", "
                     f"min={np.min(rms_list):.3f}\", max={np.max(rms_list):.3f}\"")
    if time_list:
        lines.append(f"成功帧 耗时: mean={np.mean(time_list):.3f}s, median={np.median(time_list):.3f}s")
    if cand_list:
        lines.append(f"成功帧 候选数: mean={np.mean(cand_list):.1f}, median={np.median(cand_list):.1f}")

    # 结论
    lines.append("")
    lines.append("-" * 90)
    lines.append("【结论】")
    lines.append("-" * 90)
    if n_pass_both == n:
        lines.append("✓ 所有帧均通过 CRVAL+RMS 双验收, 4SADQ-KV 核心机制验证通过。")
    elif n_pass_both > 0:
        lines.append(f"△ {n_pass_both}/{n} 帧通过双验收, 部分通过。需分析失败帧原因。")
    else:
        lines.append("✗ 无帧通过双验收, 4SADQ-KV 核心机制存在问题, 需调试。")

    return "\n".join(lines)


def save_report(text: str) -> None:
    """保存报告到 logs/test_report.txt (UTF-8)"""
    os.makedirs(_LOG_DIR, exist_ok=True)
    with open(_REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n[报告已保存] {_REPORT_FILE}")


# ════════════════════════════════════════════════════════════════════
# 主入口
# ════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    """主测试入口

    用法:
        python test_pipeline.py            # 全部帧
        python test_pipeline.py M20 LDN43  # 仅匹配 name 关键字的帧
    """
    argv = argv if argv is not None else sys.argv[1:]
    # 选择测试帧 (支持关键字过滤)
    if argv:
        selected = [f for f in TEST_FRAMES if any(kw.lower() in f["name"].lower() for kw in argv)]
        if not selected:
            print(f"[警告] 关键字 {argv} 未匹配任何帧, 运行全部")
            selected = TEST_FRAMES
    else:
        selected = TEST_FRAMES

    print(f"[开始] 4SADQ-KV 盲解析端到端测试, 共 {len(selected)} 帧")
    results = []
    for i, frame in enumerate(selected, 1):
        print(f"\n{'#' * 90}")
        print(f"# [{i}/{len(selected)}] 开始测试: {frame['name']}")
        print(f"{'#' * 90}")
        r = run_one_frame(frame)
        results.append(r)

    # 生成并保存报告
    report = format_report(results)
    print("\n" + "=" * 90)
    print(report)
    print("=" * 90)
    save_report(report)

    # 退出码: 全部 success 才 0
    return 0 if all(r.success for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
