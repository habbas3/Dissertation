#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 12:25:19 2025

@author: habbas
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


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
        matches = sorted(
            compare_dir.glob(f"{prefix}*{dataset_tag}*.csv"),
            key=lambda p: p.stat().st_mtime,
        )
        if matches:
            return matches[-1]
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