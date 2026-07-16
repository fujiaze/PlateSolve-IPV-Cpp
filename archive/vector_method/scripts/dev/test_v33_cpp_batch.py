"""
V3.3 Record-and-Filter
testdata

CSV
"""

import os, sys, time, logging, threading, re, csv
from queue import Queue
from dataclasses import dataclass, fields
from typing import List, Optional, Callable, Dict
import numpy as np

PROJECT_ROOT = r"F:\Astro dev\Astro CS Normalization Database"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "plate_solve", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "astro_image_io", "python"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "lib", "star_detector", "python"))

from vector_match_v3_3_cpp import VectorMatch as VectorMatchV33Cpp
from astro_image_io import ImageReader
from star_detector import StarDetector, SDetParamsPy

logger = logging.getLogger("v33_robustness_test")


def parse_filename(fname):
    info = {"telescope": "", "filter": "", "exposure": 0}
    m = re.search(r'T(\d)', fname)
    if m:
        info["telescope"] = f"T{m.group(1)}"
    elif "LDN43" in fname:
        info["telescope"] = "T1"
    for f in ["H-alpha", "OIII", "Oiii", "Sii", "Lum", "Red", "Green", "Blue"]:
        if f in fname:
            info["filter"] = "Oiii" if f in ("OIII", "Oiii") else f
            break
    m = re.search(r'-(\d+)S-', fname)
    if m:
        info["exposure"] = int(m.group(1))
    return info


def scan_frames(root_dir):
    frames = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.fit', '.fits', '.fts')):
                frames.append(os.path.join(dirpath, f))
    return sorted(frames)


@dataclass
class TestResult:
    filename: str
    telescope: str
    filter_name: str
    exposure_s: int
    width: int
    height: int
    focal_length_mm: float
    pixel_size_um: float
    n_stars: int
    n_saturated: int
    t_detect_s: float
    t_solve_s: float
    rms_px: float
    rms_arcsec: float
    scale_arcsec_px: float
    rotation_deg: float
    flip_mode: int
    matched_count: int
    center_ra: float
    center_dec: float
    success: bool
    fail_reason: str
    theta_snr: float
    theta_peak_deg: float
    best_n_range: int
    median_noise: float
    n_phaseb_pairs: int
    n_phaseb_corr: int
    n_phasea_records: int
    solve_tx: float
    solve_ty: float
    solve_s: float
    s0: float
    original_ra: float
    original_dec: float


CSV_HEADER = [f.name for f in fields(TestResult)]


class RobustnessTester:
    def __init__(self, gaia_data_dir: str, db_type: int = 0,
                 on_result: Optional[Callable] = None):
        self._gaia_dir = gaia_data_dir
        self._db_type = db_type
        self._on_result = on_result

    def run(self, frame_paths: List[str]) -> List[TestResult]:
        n_frames = len(frame_paths)
        if n_frames == 0:
            return [], 0.0

        logger.info("V3.3: %d", n_frames)

        results: Dict[int, TestResult] = {}
        results_lock = threading.Lock()
        detected_queue: Queue = Queue(maxsize=2)

        def solve_worker():
            vm = VectorMatchV33Cpp(self._gaia_dir, db_type=self._db_type)
            while True:
                item = detected_queue.get()
                if item is None:
                    break

                idx, fname, info, img_x, img_y, img_flux, img_saturated, \
                    center_ra, center_dec, fl, ps, w, h, n_stars, n_sat, t_det = item

                t0 = time.perf_counter()
                result = TestResult(
                    filename=os.path.basename(fname), telescope=info["telescope"],
                    filter_name=info["filter"], exposure_s=info["exposure"],
                    width=w, height=h, focal_length_mm=fl, pixel_size_um=ps,
                    n_stars=n_stars, n_saturated=n_sat, t_detect_s=t_det,
                    t_solve_s=0, rms_px=0, rms_arcsec=0, scale_arcsec_px=0,
                    rotation_deg=0, flip_mode=-1, matched_count=0,
                    center_ra=0, center_dec=0, success=False, fail_reason="",
                    theta_snr=0, theta_peak_deg=0, best_n_range=0,
                    median_noise=0, n_phaseb_pairs=0, n_phaseb_corr=0,
                    n_phasea_records=0,
                    solve_tx=0, solve_ty=0, solve_s=0, s0=0,
                    original_ra=0, original_dec=0,
                )

                try:
                    if len(img_x) < 2:
                        result.fail_reason = f"({len(img_x)})"
                    else:
                        vm_result = vm.solve(
                            img_x, img_y, img_flux, img_saturated,
                            center_ra, center_dec, fl, ps, w, h,
                        )
                        if vm_result:
                            result.success = True
                            result.rms_px = vm_result.rms_px
                            result.rms_arcsec = vm_result.rms_arcsec
                            result.scale_arcsec_px = vm_result.scale_arcsec_px
                            result.rotation_deg = vm_result.rotation_deg
                            result.flip_mode = vm_result.flip_mode
                            result.matched_count = vm_result.matched_count
                            result.center_ra = vm_result.center_ra
                            result.center_dec = vm_result.center_dec
                            result.theta_snr = getattr(vm_result, 'theta_snr', 0)
                            result.theta_peak_deg = getattr(vm_result, 'theta_peak_deg', 0)
                            result.best_n_range = getattr(vm_result, 'best_n_range', 0)
                            result.median_noise = getattr(vm_result, 'median_noise', 0)
                            result.n_phaseb_pairs = getattr(vm_result, 'n_phaseb_pairs', 0)
                            result.n_phaseb_corr = getattr(vm_result, 'n_phaseb_corr', 0)
                            result.n_phasea_records = getattr(vm_result, 'n_phasea_records', 0)
                            result.solve_tx = getattr(vm_result, 'solve_tx', 0)
                            result.solve_ty = getattr(vm_result, 'solve_ty', 0)
                            result.solve_s = getattr(vm_result, 'solve_s', 0)
                            result.s0 = getattr(vm_result, 's0', 0)
                            result.original_ra = getattr(vm_result, 'original_ra', 0)
                            result.original_dec = getattr(vm_result, 'original_dec', 0)
                        else:
                            result.fail_reason = "V3.3"
                except Exception as e:
                    result.fail_reason = str(e)[:80]

                result.t_solve_s = time.perf_counter() - t0

                with results_lock:
                    results[idx] = result

                if self._on_result:
                    self._on_result(len(results), n_frames, result)

            vm.close()

        solve_thread = threading.Thread(target=solve_worker, name="solve_v33", daemon=True)
        solve_thread.start()

        t_start = time.perf_counter()
        reader = ImageReader()
        detector = StarDetector(params=SDetParamsPy(fitRadius=0))

        for i, path in enumerate(frame_paths):
            fname = os.path.basename(path)
            info = parse_filename(fname)

            try:
                img = reader.read(path)
                w, h = img.width, img.height
                center_ra, center_dec = 0.0, 0.0
                fl, ps = 200.0, 6.0

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
                det = detector.detect_ex(img.data)
                t_det = time.perf_counter() - t_d0

                n_stars = len(det.x)
                n_sat = int(np.sum(np.array(det.saturated)))

                img_x = np.array(det.x, dtype=np.float64)
                img_y = np.array(det.y, dtype=np.float64)
                img_flux = np.array(det.flux, dtype=np.float64)
                img_saturated = np.array(det.saturated, dtype=np.int32)

                detected_queue.put((
                    i, path, info, img_x, img_y, img_flux, img_saturated,
                    center_ra, center_dec, fl, ps, w, h, n_stars, n_sat, t_det,
                ))

            except Exception as e:
                result = TestResult(
                    filename=fname, telescope=info["telescope"],
                    filter_name=info["filter"], exposure_s=info["exposure"],
                    width=0, height=0, focal_length_mm=0, pixel_size_um=0,
                    n_stars=0, n_saturated=0, t_detect_s=0, t_solve_s=0,
                    rms_px=0, rms_arcsec=0, scale_arcsec_px=0,
                    rotation_deg=0, flip_mode=-1, matched_count=0,
                    center_ra=0, center_dec=0, success=False,
                    fail_reason=f": {str(e)[:60]}",
                    theta_snr=0, theta_peak_deg=0, best_n_range=0,
                    median_noise=0, n_phaseb_pairs=0, n_phaseb_corr=0,
                    n_phasea_records=0,
                    solve_tx=0, solve_ty=0, solve_s=0, s0=0,
                    original_ra=0, original_dec=0,
                )
                with results_lock:
                    results[i] = result
                if self._on_result:
                    self._on_result(len(results), n_frames, result)

        detected_queue.put(None)
        solve_thread.join()
        t_total = time.perf_counter() - t_start

        def _default(idx):
            return TestResult(
                filename=os.path.basename(frame_paths[idx]),
                telescope="", filter_name="", exposure_s=0,
                width=0, height=0, focal_length_mm=0, pixel_size_um=0,
                n_stars=0, n_saturated=0, t_detect_s=0, t_solve_s=0,
                rms_px=0, rms_arcsec=0, scale_arcsec_px=0,
                rotation_deg=0, flip_mode=-1, matched_count=0,
                center_ra=0, center_dec=0, success=False, fail_reason="",
                theta_snr=0, theta_peak_deg=0, best_n_range=0,
                median_noise=0, n_phaseb_pairs=0, n_phaseb_corr=0,
                n_phasea_records=0,
                solve_tx=0, solve_ty=0, solve_s=0, s0=0,
                original_ra=0, original_dec=0,
            )
        ordered = [results.get(i, _default(i)) for i in range(n_frames)]

        n_ok = sum(1 for r in ordered if r.success)
        logger.info("V3.3: %d/%d (%.1f%%) %.1fs",
                     n_ok, n_frames, n_ok / n_frames * 100 if n_frames else 0, t_total)
        return ordered, t_total

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


def save_csv(results: List[TestResult], path: str):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in results:
            writer.writerow({k: getattr(r, k) for k in CSV_HEADER})


def print_report(results: List[TestResult], t_total: float):
    n = len(results)
    n_ok = sum(1 for r in results if r.success)
    ok = [r for r in results if r.success]
    fail = [r for r in results if not r.success]

    print(f"\n{'='*90}")
    print(f"V3.3 Record-and-Filter (s-in-range)")
    print(f"{'='*90}")
    print(f"\nTotal: {n} | OK: {n_ok} ({n_ok/n*100:.1f}%) | Fail: {len(fail)} ({len(fail)/n*100:.1f}%) | {t_total:.1f}s")

    print(f"\nBy telescope:")
    print(f"  {'Tel':<10} {'Total':>5} {'OK':>5} {'Rate':>7} {'rms_px':>8} {'match':>8} {'solve_s':>8} {'SNR':>8} {'best_n':>6} {'corr':>5}")
    for tel in sorted(set(r.telescope for r in results)):
        tr = [r for r in results if r.telescope == tel]
        tok = [r for r in tr if r.success]
        med_rms = np.median([r.rms_px for r in tok]) if tok else 0
        med_m = np.median([r.matched_count for r in tok]) if tok else 0
        med_t = np.median([r.t_solve_s for r in tok]) if tok else 0
        med_snr = np.median([r.theta_snr for r in tok]) if tok else 0
        med_best = np.median([r.best_n_range for r in tok]) if tok else 0
        med_corr = np.median([r.n_phaseb_corr for r in tok]) if tok else 0
        print(f"  {tel:<10} {len(tr):>5} {len(tok):>5} "
              f"{len(tok)/len(tr)*100:>6.1f}% {med_rms:>7.3f}px {med_m:>7.0f} {med_t:>7.2f}s {med_snr:>7.0f}x {int(med_best):>5d} {int(med_corr):>4d}")

    if ok:
        rms = [r.rms_px for r in ok]
        t = [r.t_solve_s for r in ok]
        print(f"\nRMS: med={np.median(rms):.3f}px mean={np.mean(rms):.3f}px P25={np.percentile(rms,25):.3f} P75={np.percentile(rms,75):.3f} max={np.max(rms):.3f}")
        print(f"Time: med={np.median(t):.2f}s mean={np.mean(t):.2f}s P25={np.percentile(t,25):.2f} P75={np.percentile(t,75):.2f} max={np.max(t):.2f}")

    if fail:
        reasons = {}
        for r in fail:
            key = r.fail_reason.split(":")[0][:30]
            reasons[key] = reasons.get(key, 0) + 1
        print(f"\nFail reasons:")
        for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {cnt}")


def main():
    logging.basicConfig(level=logging.WARNING, format='[%(asctime)s][%(name)s][%(levelname)s] %(message)s',
                        datefmt='%H:%M:%S')

    testdata_dir = os.path.join(PROJECT_ROOT, "testdata")
    gaia_dir = os.path.join(PROJECT_ROOT, "GaiaDR3")
    csv_path = os.path.join(PROJECT_ROOT, "v33_robustness_test_results.csv")

    frames = scan_frames(testdata_dir)
    print(f"Scanned {len(frames)} frames")

    def on_result(done, total, result):
        status = "OK" if result.success else "FAIL"
        rms = f"{result.rms_px:.3f}" if result.success else "---"
        print(f"  [{done}/{total}] {result.filename} [{result.telescope}/{result.filter_name}] "
              f"{status} det={result.t_detect_s:.2f}s sol={result.t_solve_s:.2f}s "
              f"stars={result.n_stars} sat={result.n_saturated} RMS={rms} "
              f"SNR={result.theta_snr:.0f}x best_n={result.best_n_range} corr={result.n_phaseb_corr}")

    tester = RobustnessTester(gaia_dir, db_type=1, on_result=on_result)
    results, t_total = tester.run(frames)

    save_csv(results, csv_path)
    print(f"\nCSV saved: {csv_path}")
    print_report(results, t_total)


if __name__ == '__main__':
    main()
