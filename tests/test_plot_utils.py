#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 15:06:11 2026

@author: habbas
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.plot_utils import find_latest_compare_csv


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("source,target,improvement\na,b,0.1\n", encoding="utf-8")


def test_find_latest_compare_csv_is_case_insensitive_for_dataset_tag(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    run = root / "llm_run_20251212_131825" / "compare"
    csv_path = run / "llm_pick_summary_1212_141551_Battery_inconsistent.csv"
    _write_csv(csv_path)

    found = find_latest_compare_csv(root, prefix="llm_pick_summary", dataset_tag="battery_inconsistent")

    assert found == csv_path


def test_find_latest_compare_csv_picks_newest_match_across_runs(tmp_path: Path) -> None:
    root = tmp_path / "checkpoint"
    old = root / "llm_run_20251211_090000" / "compare" / "llm_pick_summary_1211_090000_battery.csv"
    new = root / "llm_run_20251212_090000" / "compare" / "llm_pick_summary_1212_090000_battery.csv"
    _write_csv(old)
    _write_csv(new)

    old.touch()
    new.touch()

    found = find_latest_compare_csv(root, prefix="llm_pick_summary", dataset_tag="battery")

    assert found == new
