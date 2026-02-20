#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 12:25:19 2025

@author: habbas
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _extract_sort_stamp(path: Path) -> float:
    """Return a robust timestamp for sorting result artefacts.

    Prefer filesystem mtime, but gracefully handle unavailable stat metadata.
    """

    try:
        return path.stat().st_mtime
    except OSError:
        return float("-inf")


def _is_matching_compare_csv(path: Path, prefix: str, dataset_tag: str) -> bool:
    name = path.name.lower()
    return name.endswith(".csv") and name.startswith(prefix.lower()) and dataset_tag.lower() in name



def find_latest_llm_run(checkpoint_root: Path) -> Path:
    runs = sorted(
        [p for p in checkpoint_root.glob("llm_run_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        raise FileNotFoundError(f"No llm_run_* directories found under {checkpoint_root}")
    return runs[-1]


def find_latest_compare_csv(
    checkpoint_root: Path,
    prefix: str,
    dataset_tag: str,
    *,
    run_dir: Path | None = None,
) -> Path:
    latest_match: Path | None = None
    
    if run_dir is not None:
        compare_dirs: Iterable[Path] = [Path(run_dir) / "compare"]
    else:
        compare_dirs = [
            p / "compare"
            for p in sorted(
                [r for r in checkpoint_root.glob("llm_run_*") if r.is_dir()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ]

    for compare_dir in compare_dirs:
        if not compare_dir.exists():
            continue
        for candidate in compare_dir.glob("*.csv"):
            if not _is_matching_compare_csv(candidate, prefix, dataset_tag):
                continue
            if latest_match is None or _extract_sort_stamp(candidate) > _extract_sort_stamp(latest_match):
                latest_match = candidate

    if latest_match is not None:
        return latest_match
    raise FileNotFoundError(
        f"No compare CSVs found for prefix={prefix} dataset={dataset_tag} under {checkpoint_root}"
    )
    
def find_latest_compare_csv_optional(
    checkpoint_root: Path,
    prefix: str,
    dataset_tag: str,
    *,
    run_dir: Path | None = None,
) -> Path | None:
    """Best-effort variant of :func:`find_latest_compare_csv`.

    This is useful for partially complete LLM runs where not every model summary
    has finished yet. The newest matching CSV is returned when available,
    otherwise ``None``.
    """

    try:
        return find_latest_compare_csv(
            checkpoint_root,
            prefix,
            dataset_tag,
            run_dir=run_dir,
        )
    except FileNotFoundError:
        return None