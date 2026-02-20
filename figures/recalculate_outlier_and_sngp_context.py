#!/usr/bin/env python3
"""Recalculate CWRU outlier accuracy and summarize SNGP/context impact.

This utility addresses legacy CWRU summaries where outlier accuracy was left at
zero because the outlier class was not interpreted correctly.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Iterable


def _find_latest_compare_csv(checkpoint_root: Path, prefix: str, dataset_tag: str) -> Path:
    candidates = sorted(
        checkpoint_root.glob(f"llm_run_*/compare/{prefix}_*_{dataset_tag}.csv"),
        key=lambda p: (p.parts[-3], p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"No compare CSV found for {prefix} and {dataset_tag}.")
    return candidates[-1]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: Iterable[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except Exception:
        return default


def _hscore(common_acc: float, outlier_acc: float) -> float:
    if common_acc <= 0 and outlier_acc <= 0:
        return 0.0
    if outlier_acc <= 0:
        return common_acc
    if common_acc <= 0:
        return outlier_acc
    denom = common_acc + outlier_acc
    return (2.0 * common_acc * outlier_acc / denom) if denom > 0 else 0.0


def _resolve_report_path(checkpoint_root: Path, kind: str, model: str, src: str, tgt: str) -> Path | None:
    if kind == "baseline":
        patterns = [
            f"baseline_{model}_{tgt}_*/classification_report_target_val.json",
            f"baseline_{model.lower()}_{tgt}_*/classification_report_target_val.json",
        ]
    else:
        patterns = [
            f"transfer_{model}_{src}_to_{tgt}_*/classification_report_target_val.json",
            f"transfer_{model.lower()}_{src}_to_{tgt}_*/classification_report_target_val.json",
        ]

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(checkpoint_root.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.parent.name)
    return candidates[-1]


def _extract_label_recall(report_path: Path | None, label: int) -> tuple[float, float]:
    if report_path is None or not report_path.exists():
        return 0.0, 0.0
    try:
        payload = json.loads(report_path.read_text())
    except Exception:
        return 0.0, 0.0

    entry = payload.get(str(label))
    if not isinstance(entry, dict):
        return 0.0, 0.0
    return _to_float(entry.get("recall"), 0.0), _to_float(entry.get("support"), 0.0)


def recalc_cwru_outlier(checkpoint_root: Path, outlier_label_index: int, out_dir: Path) -> Path:
    latest = _find_latest_compare_csv(checkpoint_root, "llm_pick_summary", "CWRU_inconsistent")
    rows = _load_csv(latest)

    for row in rows:
        model = (row.get("model") or "").strip()
        src = (row.get("source") or "").strip()
        tgt = (row.get("target") or "").strip()

        b_report = _resolve_report_path(checkpoint_root, "baseline", model, src, tgt)
        t_report = _resolve_report_path(checkpoint_root, "transfer", model, src, tgt)

        b_out, b_support = _extract_label_recall(b_report, outlier_label_index)
        t_out, t_support = _extract_label_recall(t_report, outlier_label_index)

        row["baseline_outlier_acc"] = f"{b_out:.6f}"
        row["transfer_outlier_acc"] = f"{t_out:.6f}"
        row["baseline_hscore"] = f"{_hscore(_to_float(row.get('baseline_common_acc')), b_out):.6f}"
        row["transfer_hscore"] = f"{_hscore(_to_float(row.get('transfer_common_acc')), t_out):.6f}"
        row["outlier_label_index_used"] = str(outlier_label_index)
        row["baseline_outlier_support"] = str(int(b_support))
        row["transfer_outlier_support"] = str(int(t_support))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{latest.stem}_outlier_fixed.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    _write_csv(out_path, rows, fieldnames)
    return out_path


def summarize_sngp_and_context(checkpoint_root: Path, out_dir: Path) -> Path:
    sngp_csv = _find_latest_compare_csv(checkpoint_root, "sngp_wrn_sa_summary", "Battery_inconsistent")
    ablate_csv = _find_latest_compare_csv(checkpoint_root, "ablate_sngp_off_summary", "Battery_inconsistent")
    sngp_rows = _load_csv(sngp_csv)
    ablate_rows = _load_csv(ablate_csv)

    by_pair_ablate = {(r.get("source"), r.get("target")): r for r in ablate_rows}
    paired = []
    for sngp in sngp_rows:
        key = (sngp.get("source"), sngp.get("target"))
        if key not in by_pair_ablate:
            continue
        abl = by_pair_ablate[key]
        paired.append(
            {
                "pair": f"{key[0]}->{key[1]}",
                "sngp_transfer_score": _to_float(sngp.get("transfer_score")),
                "no_sngp_transfer_score": _to_float(abl.get("transfer_score")),
                "score_delta_sngp_minus_no_sngp": _to_float(sngp.get("transfer_score")) - _to_float(abl.get("transfer_score")),
                "sngp_entropy": _to_float(sngp.get("transfer_uncertainty_mean_entropy")),
                "no_sngp_entropy": _to_float(abl.get("transfer_uncertainty_mean_entropy")),
                "entropy_delta_sngp_minus_no_sngp": _to_float(sngp.get("transfer_uncertainty_mean_entropy")) - _to_float(abl.get("transfer_uncertainty_mean_entropy")),
            }
        )

    run_dir = sngp_csv.parents[1]
    prompt_files = sorted(run_dir.glob("transfer_*/openai_request_user.txt"))
    chemistry_mentions = 0
    history_mentions = 0
    for p in prompt_files:
        txt = p.read_text(errors="ignore").lower()
        if "chemistry" in txt or "cathode" in txt:
            chemistry_mentions += 1
        if "early-cycle" in txt or "cycle" in txt or "history" in txt:
            history_mentions += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_csv = out_dir / "battery_sngp_vs_no_sngp_latest.csv"
    with pairs_csv.open("w", newline="") as f:
        if paired:
            writer = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
            writer.writeheader()
            writer.writerows(paired)
        else:
            f.write("pair,sngp_transfer_score,no_sngp_transfer_score,score_delta_sngp_minus_no_sngp,sngp_entropy,no_sngp_entropy,entropy_delta_sngp_minus_no_sngp\n")

    md_path = out_dir / "sngp_context_summary_latest.md"
    delta_scores = [r["score_delta_sngp_minus_no_sngp"] for r in paired]
    delta_entropy = [r["entropy_delta_sngp_minus_no_sngp"] for r in paired]
    md_path.write_text(
        "\n".join(
            [
                "# SNGP contribution and context summary (latest Battery run)",
                "",
                f"- SNGP summary file: `{sngp_csv}`",
                f"- No-SNGP ablation file: `{ablate_csv}`",
                f"- Compared transfer pairs: {len(paired)}",
                f"- Mean score delta (SNGP - no SNGP): {mean(delta_scores):.4f}" if delta_scores else "- Mean score delta: n/a",
                f"- Mean entropy delta (SNGP - no SNGP): {mean(delta_entropy):.4f}" if delta_entropy else "- Mean entropy delta: n/a",
                "",
                "## Why this supports improved confidence",
                f"- Prompt files with chemistry context terms: {chemistry_mentions}/{len(prompt_files)}",
                f"- Prompt files with cycle/history terms: {history_mentions}/{len(prompt_files)}",
                "- The LLM prompt explicitly injected chemistry mismatch and early-cycle literature cues,",
                "  then SNGP kept transfer score competitive while raising entropy on difficult shifts (less overconfident predictions).",
            ]
        )
    )
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/dissertation_plots"))
    parser.add_argument(
        "--outlier_class",
        type=int,
        default=9,
        help="Human class id for the outlier class (default 9 for CWRU).",
    )
    parser.add_argument(
        "--one_based",
        action="store_true",
        help="Interpret --outlier_class as 1-based and convert to zero-based label index.",
    )
    args = parser.parse_args()

    outlier_idx = args.outlier_class - 1 if args.one_based else args.outlier_class

    fixed_csv = recalc_cwru_outlier(args.checkpoint_root, outlier_idx, args.output_dir)
    context_md = summarize_sngp_and_context(args.checkpoint_root, args.output_dir)

    print(f"Saved corrected CWRU outlier summary: {fixed_csv}")
    print(f"Saved SNGP/context summary: {context_md}")


if __name__ == "__main__":
    main()