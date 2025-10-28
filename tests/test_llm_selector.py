#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 23:19:22 2025

@author: habbas
"""

import shutil
import sys
import unittest
from pathlib import Path
from typing import List


sys.path.append(str(Path(__file__).resolve().parent.parent))

from llm_selector import _csv_column_stats, _maybe_apply_historical_winner


class HistoricalOverrideTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checkpoint_root = Path("checkpoint")
        self.checkpoint_root.mkdir(exist_ok=True)
        self.created_runs: List[Path] = []

    def tearDown(self) -> None:
        for run in self.created_runs:
            if run.exists():
                shutil.rmtree(run, ignore_errors=True)

    def _write_history(self, run_name: str, tag: str, dataset: str, values: list[float]) -> None:
        run_dir = self.checkpoint_root / run_name / "compare"
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / f"{tag}_summary_{dataset}.csv"
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("source,target,improvement\n")
            for idx, val in enumerate(values):
                f.write(f"src{idx},tgt{idx},{val}\n")
        self.created_runs.append(run_dir.parent)

    def test_csv_column_stats_reports_mean_and_bounds(self) -> None:
        dataset = "unit_dataset_stats"
        self._write_history("llm_run_stats", "deterministic_cnn", dataset, [0.2, 0.1, 0.4, 0.3])
        csv_file = self.checkpoint_root / "llm_run_stats" / "compare" / f"deterministic_cnn_summary_{dataset}.csv"
        stats = _csv_column_stats(csv_file, "improvement")
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(int(stats["count"]), 4)
        self.assertAlmostEqual(stats["mean"], 0.25, places=6)
        self.assertAlmostEqual(stats["median"], 0.25, places=6)
        self.assertGreater(stats["lower_bound"], 0.0)

    def test_override_applied_when_history_is_confident(self) -> None:
        dataset = "unit_dataset_confident"
        self._write_history("llm_run_confident", "deterministic_cnn", dataset, [0.05, 0.04, 0.03, 0.02])
        cfg = {"model_name": "cnn_features_1d", "lambda_src": 1.0}
        num_summary = {"dataset": dataset}
        updated, note = _maybe_apply_historical_winner(cfg, num_summary)
        self.assertNotEqual(updated, cfg)
        self.assertIsNotNone(note)
        self.assertIn("deterministic_cnn", note or "")
        self.assertIn("mean Δ", note or "")
        self.assertGreater(updated.get("batch_size", 0), 0)

    def test_override_skipped_when_variance_high(self) -> None:
        dataset = "unit_dataset_uncertain"
        self._write_history("llm_run_uncertain", "sngp_wrn_sa", dataset, [0.25, -0.2, 0.05, 0.01])
        cfg = {"model_name": "WideResNet_sa", "lambda_src": 0.8}
        num_summary = {"dataset": dataset}
        updated, note = _maybe_apply_historical_winner(cfg, num_summary)
        self.assertEqual(updated, cfg)
        self.assertIsNone(note)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()