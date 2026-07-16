"""
功能: 验证 record_sampling_metrics 抽样记录脚本的规格行为
用途: 防止帧发现、Gaia搜索半径、圆形过滤和全抽样记录格式回归
"""
import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).with_name("record_sampling_metrics.py")


class RecordSamplingMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("record_sampling_metrics", SCRIPT_PATH)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_parse_frame_metadata_from_nested_telescope_path(self):
        path = Path("lights") / "T3" / "M31" / "M31_T3_scope-20250101@010101-120S-Ha.fts"
        meta = self.mod.parse_frame_metadata(path)
        self.assertEqual(meta["optical_system"], "T3")
        self.assertEqual(meta["target"], "M31")
        self.assertEqual(meta["filter"], "Ha")
        self.assertEqual(meta["frame_name"], "M31_T3_scope-20250101@010101-120S-Ha.fts")

    def test_gaia_radius_uses_half_fov_diagonal(self):
        radius = self.mod.gaia_query_radius_deg(4000, 3000, 2.0)
        self.assertAlmostEqual(radius, 5000.0 / 3600.0)

    def test_catalog_mask_is_circular_half_diagonal_not_box(self):
        xi = np.array([0.0, 4.9, 5.1, 4.0], dtype=np.float64)
        eta = np.array([0.0, 0.0, 0.0, 4.0], dtype=np.float64)
        valid = np.array([True, True, True, True])
        mask = self.mod.circular_catalog_mask(xi, eta, valid, 10.0)
        self.assertEqual(mask.tolist(), [True, True, False, False])

    def test_sampling_records_keep_out_of_scale_measured_samples(self):
        U = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]], dtype=np.float64)
        W = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        records, summary = self.mod.sample_mode_records(
            U,
            W,
            mode=0,
            n_samples=1,
            halfW=20.0,
            halfH=20.0,
            rng=self.mod.SequenceRng([(0, 0)]),
            scale_window=(0.85, 1.15),
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_status"], "measured")
        self.assertFalse(records[0]["is_in_scale_window"])
        self.assertAlmostEqual(records[0]["s"], 10.0)
        self.assertEqual(summary["attempted_samples"], 1)
        self.assertEqual(summary["measured_samples"], 1)

    def test_sampling_records_zero_w_norm_reject_reason(self):
        U = np.array([[10.0, 0.0]], dtype=np.float64)
        W = np.array([[0.0, 0.0]], dtype=np.float64)
        records, summary = self.mod.sample_mode_records(
            U,
            W,
            mode=0,
            n_samples=1,
            halfW=20.0,
            halfH=20.0,
            rng=self.mod.SequenceRng([(0, 0)]),
        )
        self.assertEqual(records[0]["sample_status"], "zero_w_norm")
        self.assertEqual(records[0]["reject_reason"], "zero_w_norm")
        self.assertEqual(summary["measured_samples"], 0)

    def test_detail_csv_contains_required_stable_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "details.csv"
            row = {name: "" for name in self.mod.SAMPLE_DETAIL_COLUMNS}
            row.update({"optical_system": "T1", "mode": 0, "sample_index": 0})
            self.mod.write_csv(path, self.mod.SAMPLE_DETAIL_COLUMNS, [row])
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, self.mod.SAMPLE_DETAIL_COLUMNS)
                loaded = list(reader)
            self.assertEqual(loaded[0]["optical_system"], "T1")

    def test_limit_frames_by_group_keys_on_optical_system_filter(self):
        paths = [
            Path("lights/T1/M20/M20_T1_scope-20250101@010101-120S-Red.fts"),
            Path("lights/T1/M20/M20_T1_scope-20250102@010101-120S-Red.fts"),
            Path("lights/T1/M20/M20_T1_scope-20250103@010101-120S-Red.fts"),
            Path("lights/T1/M20/M20_T1_scope-20250104@010101-120S-Red.fts"),
            Path("lights/T1/NGC6302/NGC6302_T1_scope-20250101@010101-120S-Ha.fts"),
            Path("lights/T2/M20/M20_T2_scope-20250101@010101-120S-Red.fts"),
        ]
        selected = self.mod.limit_frames_by_group(paths, 3)
        os_filter_counts = {}
        for p in selected:
            meta = self.mod.parse_frame_metadata(p)
            key = (meta["optical_system"], meta["filter"])
            os_filter_counts[key] = os_filter_counts.get(key, 0) + 1
        for key, count in os_filter_counts.items():
            self.assertLessEqual(count, 3)
        self.assertEqual(len(selected), 5)

    def test_plot_best_overlay_uses_vector_lines_from_center(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            meta = {"optical_system": "T1", "target": "M20", "filter": "Red", "frame_name": "test.fts"}
            frame_data = {
                "U": np.array([[5.0, 0.0], [0.0, 5.0]], dtype=np.float64),
                "W": np.array([[4.0, 0.0], [0.0, 4.0]], dtype=np.float64),
                "halfW": 10.0,
                "halfH": 10.0,
                "nsat": 2,
                "N_image": 2,
                "M_catalog": 2,
            }
            best_rows = [{
                "mode": "0",
                "ui": "0",
                "wi": "0",
                "s": "1.0",
                "theta_rad": "0.0",
                "theta_deg": "0.0",
                "metric": "density_corr",
                "density_corr": "0.5",
                "coverage_iou": "0.3",
                "score_dc_ci": "0.15",
                "wt_clip_count": "2",
                "peak_theta_deg": "0",
                "peak_s": "1.0",
                "peak_count": "1",
                "attempted_samples": "10",
                "measured_samples": "8",
            }]
            self.mod.plot_best_overlay(out_dir, meta, frame_data, best_rows)
            overlay_dir = out_dir / "overlays"
            self.assertTrue(overlay_dir.exists())
            pngs = list(overlay_dir.glob("*.png"))
            self.assertEqual(len(pngs), 1)

    def test_analyze_outputs_writes_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            best = {name: "" for name in self.mod.BEST_COLUMNS}
            best.update({
                "optical_system": "T1",
                "target": "M20",
                "filter": "Red",
                "frame_name": "frame.fts",
                "metric": "density_corr",
                "mode": "0",
                "density_corr": "0.2",
                "coverage_iou": "0.1",
                "score_dc_ci": "0.02",
            })
            mode = {name: "" for name in self.mod.MODE_STATS_COLUMNS}
            mode.update({
                "optical_system": "T1",
                "target": "M20",
                "filter": "Red",
                "frame_name": "frame.fts",
                "mode": "0",
                "density_corr_peak_zscore": "2.0",
            })
            self.mod.write_csv(out / "frame_best_metrics.csv", self.mod.BEST_COLUMNS, [best])
            self.mod.write_csv(out / "frame_mode_stats.csv", self.mod.MODE_STATS_COLUMNS, [mode])
            analysis_rows, issue_rows = self.mod.analyze_outputs(out)
            self.assertTrue((out / "analysis_summary.csv").exists())
            self.assertTrue((out / "analysis_summary.txt").exists())
            self.assertGreaterEqual(len(analysis_rows), 1)
            self.assertEqual(issue_rows, [])


if __name__ == "__main__":
    unittest.main()
