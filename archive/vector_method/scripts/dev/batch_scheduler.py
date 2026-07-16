"""
向量匹配 V2 批量流水线调度器

架构:
    主线程: 星检测(串行) ──检测完成──→ 放入队列
                                      ↓
    解析线程: 从队列取帧 → 解析(Gaia+RANSAC) → 收集结果

关键:
    - 星检测串行，一个结束立刻下一个
    - 检测完成即发信号，解析线程有数据就执行
    - 检测和解析完全并行（流水线重叠）
"""

import os, sys, time, logging, threading
from queue import Queue
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "cpp", "vector_match_v2"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v2_cpp import VectorMatch as VectorMatchV2Cpp
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

logger = logging.getLogger("batch_scheduler")


@dataclass
class DetectedFrame:
    index: int
    name: str
    img_x: np.ndarray
    img_y: np.ndarray
    img_flux: np.ndarray
    img_saturated: np.ndarray
    center_ra: float
    center_dec: float
    focal_length: float
    pixel_size: float
    width: int
    height: int
    t_detect: float = 0.0


@dataclass
class FrameResult:
    index: int
    name: str
    success: bool
    t_detect: float = 0.0
    t_solve: float = 0.0
    t_total: float = 0.0
    rms_px: float = 0.0
    rms_arcsec: float = 0.0
    matched: int = 0
    scale: float = 0.0
    rotation: float = 0.0
    flip_mode: int = -1
    center_ra: float = 0.0
    center_dec: float = 0.0
    err: str = ""


class BatchScheduler:
    """V2向量匹配批量流水线调度器

    主线程串行检测，检测完一帧立刻放入队列
    解析线程从队列取帧，有数据就执行
    两者完全并行
    """

    def __init__(self, gaia_data_dir: str, db_type: int = 0,
                 on_progress: Optional[Callable] = None):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._on_progress = on_progress
        self._stop_event = threading.Event()
        self._detect_times: List[float] = []
        self._solve_times: List[float] = []

    def run(self, frame_paths: List[str],
            focal_length: float = 200.0,
            pixel_size: float = 6.0) -> List[FrameResult]:
        n_frames = len(frame_paths)
        if n_frames == 0:
            return []

        logger.info("批量调度器启动: %d帧", n_frames)
        self._stop_event.clear()

        results: Dict[int, FrameResult] = {}
        results_lock = threading.Lock()

        # 检测完成队列：主线程放，解析线程取
        detected_queue: Queue[Optional[DetectedFrame]] = Queue(maxsize=2)

        # ── 解析线程：有数据就执行 ──
        def solve_worker():
            vm = VectorMatchV2Cpp(self._gaia_dir, db_type=self._db_type)
            while not self._stop_event.is_set():
                frame = detected_queue.get()  # 阻塞等待
                if frame is None:
                    break

                t0 = time.perf_counter()
                result = FrameResult(index=frame.index, name=frame.name, success=False)
                result.t_detect = frame.t_detect

                try:
                    if len(frame.img_x) < 2:
                        result.err = "星点不足"
                    else:
                        vm_result = vm.solve(
                            frame.img_x, frame.img_y, frame.img_flux, frame.img_saturated,
                            frame.center_ra, frame.center_dec,
                            frame.focal_length, frame.pixel_size,
                            frame.width, frame.height,
                        )
                        if vm_result:
                            result.success = True
                            result.rms_px = vm_result.rms_px
                            result.rms_arcsec = vm_result.rms_arcsec
                            result.matched = vm_result.matched_count
                            result.scale = vm_result.scale_arcsec_px
                            result.rotation = vm_result.rotation_deg
                            result.flip_mode = vm_result.flip_mode
                            result.center_ra = vm_result.center_ra
                            result.center_dec = vm_result.center_dec
                        else:
                            result.err = "解析失败"
                except Exception as e:
                    result.err = str(e)

                t_solve = time.perf_counter() - t0
                result.t_solve = t_solve
                result.t_total = frame.t_detect + t_solve
                self._solve_times.append(t_solve)

                with results_lock:
                    results[frame.index] = result

                if self._on_progress:
                    self._on_progress(len(results), n_frames, result)

                logger.info("[解析] #%d %s %s %.2fs RMS=%.3f matched=%d",
                            frame.index, frame.name,
                            "OK" if result.success else "FAIL",
                            t_solve, result.rms_px, result.matched)

            vm.close()

        # 启动解析线程
        solve_thread = threading.Thread(target=solve_worker, name="solve", daemon=True)
        solve_thread.start()

        # ── 主线程：串行检测，一个结束立刻下一个 ──
        t_start = time.perf_counter()
        reader = ImageReader()
        detector = StarDetector(params=SDetParamsPy(fitRadius=0))

        for i, path in enumerate(frame_paths):
            if self._stop_event.is_set():
                break

            fname = os.path.basename(path)

            try:
                img = reader.read(path)
                width, height = img.width, img.height
                center_ra, center_dec = 0.0, 0.0
                fl, ps = focal_length, pixel_size

                if img.metadata.wcs and img.metadata.wcs.has_wcs:
                    center_ra = img.metadata.wcs.crval1
                    center_dec = img.metadata.wcs.crval2
                if img.metadata.observation:
                    if img.metadata.observation.focallen is not None:
                        fl = img.metadata.observation.focallen
                    if img.metadata.observation.xpixsz is not None:
                        ps = img.metadata.observation.xpixsz
                if center_ra == 0.0 and center_dec == 0.0:
                    center_ra, center_dec = self._extract_center(img.keywords)

                t_d0 = time.perf_counter()
                det_result = detector.detect_ex(img.data)
                t_detect = time.perf_counter() - t_d0
                self._detect_times.append(t_detect)

                # 检测完成，放入队列（解析线程立刻开始）
                detected_queue.put(DetectedFrame(
                    index=i, name=fname,
                    img_x=np.array(det_result.x, dtype=np.float64),
                    img_y=np.array(det_result.y, dtype=np.float64),
                    img_flux=np.array(det_result.flux, dtype=np.float64),
                    img_saturated=np.array(det_result.saturated, dtype=np.int32),
                    center_ra=center_ra, center_dec=center_dec,
                    focal_length=fl, pixel_size=ps,
                    width=width, height=height,
                    t_detect=t_detect,
                ))

                logger.info("[检测] #%d %s %.2fs", i, fname, t_detect)

            except Exception as e:
                logger.error("[检测] #%d %s 失败: %s", i, fname, e)
                detected_queue.put(DetectedFrame(
                    index=i, name=fname,
                    img_x=np.array([]), img_y=np.array([]),
                    img_flux=np.array([]), img_saturated=np.array([]),
                    center_ra=0, center_dec=0,
                    focal_length=focal_length, pixel_size=pixel_size,
                    width=0, height=0,
                ))

        # 检测全部完成，发哨兵
        detected_queue.put(None)

        # 等待解析完成
        solve_thread.join()
        t_total = time.perf_counter() - t_start

        # 按序收集结果
        ordered = []
        for i in range(n_frames):
            if i in results:
                ordered.append(results[i])
            else:
                ordered.append(FrameResult(index=i, name=os.path.basename(frame_paths[i]),
                                          success=False, err="未处理"))

        n_ok = sum(1 for r in ordered if r.success)
        logger.info("批量完成: %d/%d成功 (%.1f%%), 总耗时%.1fs",
                     n_ok, n_frames, n_ok / n_frames * 100 if n_frames else 0, t_total)
        return ordered

    def stop(self):
        self._stop_event.set()

    @staticmethod
    def _extract_center(keywords) -> tuple:
        ra, dec = 0.0, 0.0
        for kw in keywords:
            name = kw.name.upper()
            if name in ("OBJCTRA", "RA"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("h", " ").replace("m", " ").replace("s", " ").split()
                    if len(parts) >= 3:
                        ra = (float(parts[0]) + float(parts[1]) / 60 + float(parts[2]) / 3600) * 15
            elif name in ("OBJCTDEC", "DEC"):
                val = kw.value
                if isinstance(val, str):
                    parts = val.replace("d", " ").replace("'", " ").replace('"', " ").split()
                    if len(parts) >= 3:
                        sign = -1 if parts[0].startswith("-") else 1
                        dec = sign * (abs(float(parts[0])) + float(parts[1]) / 60 + float(parts[2]) / 3600)
        return ra, dec


# ============================================================================
# 报告
# ============================================================================

def extract_filter(filename: str) -> str:
    for f in ["H-alpha", "Oiii", "Red", "Green", "Blue"]:
        if f in filename:
            return f
    return "Unknown"


def print_report(results: List[FrameResult], t_total: float):
    n = len(results)
    n_ok = sum(1 for r in results if r.success)

    print(f"\n{'='*80}")
    print(f"批量流水线调度器 - 测试报告 (检测串行 + 解析并行)")
    print(f"{'='*80}")
    print(f"\n成功率: {n_ok}/{n} ({n_ok/n*100:.1f}%)")
    print(f"总耗时: {t_total:.1f}s (均值{t_total/n:.2f}s/帧)")

    ok_results = [r for r in results if r.success]
    if ok_results:
        dets = [r.t_detect for r in ok_results]
        solves = [r.t_solve for r in ok_results]
        totals = [r.t_total for r in ok_results]
        print(f"\n耗时统计:")
        print(f"  {'阶段':<10} {'中位':>8} {'均值':>8} {'P25':>8} {'P75':>8}")
        print(f"  {'星检测':<10} {np.median(dets):>7.2f}s {np.mean(dets):>7.2f}s "
              f"{np.percentile(dets,25):>7.2f}s {np.percentile(dets,75):>7.2f}s")
        print(f"  {'解析':<10} {np.median(solves):>7.2f}s {np.mean(solves):>7.2f}s "
              f"{np.percentile(solves,25):>7.2f}s {np.percentile(solves,75):>7.2f}s")
        print(f"  {'总计':<10} {np.median(totals):>7.2f}s {np.mean(totals):>7.2f}s "
              f"{np.percentile(totals,25):>7.2f}s {np.percentile(totals,75):>7.2f}s")

        # 流水线重叠分析
        sum_detect = sum(dets)
        sum_solve = sum(solves)
        overlap = sum_detect + sum_solve - t_total
        print(f"\n流水线分析:")
        print(f"  检测总耗时: {sum_detect:.1f}s")
        print(f"  解析总耗时: {sum_solve:.1f}s")
        print(f"  串行理论: {sum_detect+sum_solve:.1f}s")
        print(f"  实际耗时: {t_total:.1f}s")
        print(f"  重叠节省: {overlap:.1f}s ({overlap/(sum_detect+sum_solve)*100:.1f}%)")

    print(f"\n按滤镜:")
    print(f"  {'滤镜':<10} {'成功':>6} {'失败':>4} {'成功率':>7} {'中位检测':>8} {'中位解析':>8}")
    for filt in ["Red", "Green", "Blue", "H-alpha", "Oiii"]:
        fr = [r for r in results if extract_filter(r.name) == filt]
        if not fr:
            continue
        fok = [r for r in fr if r.success]
        med_det = np.median([r.t_detect for r in fok]) if fok else 0
        med_slv = np.median([r.t_solve for r in fok]) if fok else 0
        print(f"  {filt:<10} {len(fok):>5} {len(fr)-len(fok):>4} "
              f"{len(fok)/len(fr)*100:>6.1f}% {med_det:>7.2f}s {med_slv:>7.2f}s")

    if ok_results:
        rms = [r.rms_px for r in ok_results]
        print(f"\nRMS: 中位={np.median(rms):.3f}px 均值={np.mean(rms):.3f}px "
              f"P75={np.percentile(rms,75):.3f}px")

    print(f"\n逐帧结果:")
    print(f"  {'#':>3} {'滤镜':<7} {'检测':>6} {'解析':>6} {'总计':>6} {'RMS':>7} {'匹配':>5} {'结果':>4}")
    for r in results:
        filt = extract_filter(r.name)
        status = "OK" if r.success else "FAIL"
        rms = f"{r.rms_px:.3f}" if r.success else "---"
        matched = f"{r.matched}" if r.success else "---"
        print(f"  {r.index+1:>3} {filt:<7} {r.t_detect:>5.2f}s {r.t_solve:>5.2f}s "
              f"{r.t_total:>5.2f}s {rms:>7} {matched:>5} {status:>4}")

    n_fail = n - n_ok
    if n_fail > 0:
        print(f"\n失败帧:")
        for r in results:
            if not r.success:
                print(f"  #{r.index+1} {r.name}: {r.err}")


def main():
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    panel_dir = os.path.join(PROJECT_ROOT, "testdata", "lights", "panel1")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3SP")

    files = sorted([f for f in os.listdir(panel_dir) if f.endswith('.fts')])
    paths = [os.path.join(panel_dir, f) for f in files]

    print(f"Panel1批量测试: {len(paths)}帧")
    print(f"调度策略: 检测串行(主线程) + 解析并行(独立线程)")

    def on_progress(done, total, result):
        filt = extract_filter(result.name)
        status = "OK" if result.success else "FAIL"
        rms = f"{result.rms_px:.3f}" if result.success else "---"
        print(f"  [{done}/{total}] {result.name} {filt} {status} "
              f"检测={result.t_detect:.2f}s 解析={result.t_solve:.2f}s RMS={rms}")

    scheduler = BatchScheduler(gaia_dir, db_type=0, on_progress=on_progress)
    t0 = time.perf_counter()
    results = scheduler.run(paths)
    t_total = time.perf_counter() - t0
    print_report(results, t_total)


if __name__ == '__main__':
    main()
