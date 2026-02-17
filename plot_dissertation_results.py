#!/usr/bin/env python3
"""Generate dissertation-ready plots from real LLM ablation run outputs.

This utility reads the existing `checkpoint/llm_run_*` results and creates:
- Ablation comparison per dataset
- Accuracy-aspect deltas (overall/common/outlier/H-score)
- H-score distributions by ablation tag
- SNGP vs deterministic improvement comparison
- SNGP confidence (entropy) distributions
- Confidence vs H-score scatter for SNGP predictions
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


KNOWN_DATASETS = ("Battery_inconsistent", "CWRU_inconsistent")


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _dataset_from_text(text: str) -> str | None:
    low = text.lower()
    for ds in KNOWN_DATASETS:
        if ds.lower() in low:
            return ds
    return None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_latest_runs_by_dataset(checkpoint_root: Path) -> Dict[str, Path]:
    chosen: Dict[str, Path] = {}
    for run in sorted(checkpoint_root.glob("llm_run_*")):
        lb_path = run / "llm_leaderboard.csv"
        if not lb_path.exists():
            continue
        rows = _load_csv(lb_path)
        if not rows:
            continue

        found_dataset = None
        for row in rows:
            ds = _dataset_from_text(row.get("summary_csv", ""))
            if ds:
                found_dataset = ds
                break
        if not found_dataset:
            compare_dir = run / "compare"
            if compare_dir.exists():
                for f in compare_dir.glob("*.csv"):
                    ds = _dataset_from_text(f.name)
                    if ds:
                        found_dataset = ds
                        break
        if found_dataset:
            chosen[found_dataset] = run
    return chosen


def _load_run_payload(run_dir: Path) -> Tuple[List[Dict[str, str]], Dict[str, List[Dict[str, str]]]]:
    lb_rows = _load_csv(run_dir / "llm_leaderboard.csv")
    compare_payload: Dict[str, List[Dict[str, str]]] = {}
    for row in lb_rows:
        tag = row.get("tag", "")
        summary_csv = row.get("summary_csv", "")
        path = Path(summary_csv)
        if not path.is_absolute():
            path = Path(summary_csv)
        if not path.exists():
            alt = run_dir / "compare" / Path(summary_csv).name
            path = alt
        if path.exists() and tag:
            compare_payload[tag] = _load_csv(path)
    return lb_rows, compare_payload


def _clean_tag(tag: str) -> str:
    return tag.replace("_", " ")


def plot_ablation_improvement(lb_rows: List[Dict[str, str]], out_path: Path, dataset: str) -> None:
    records = []
    for row in lb_rows:
        imp = _safe_float(row.get("avg_improvement"))
        tag = row.get("tag")
        if tag and imp is not None:
            records.append((tag, imp * 100.0))
    if not records:
        return
    records.sort(key=lambda x: x[1], reverse=True)

    tags = [r[0] for r in records]
    vals = [r[1] for r in records]
    colors = ["#2E86AB" if v >= 0 else "#D64550" for v in vals]

    plt.figure(figsize=(max(9, len(tags) * 0.75), 5.5))
    bars = plt.bar(range(len(tags)), vals, color=colors)
    plt.axhline(0, color="black", linewidth=0.9)
    plt.xticks(range(len(tags)), [_clean_tag(t) for t in tags], rotation=35, ha="right")
    plt.ylabel("Average improvement vs baseline (%)")
    plt.title(f"Ablation comparison ({dataset})")
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + (0.2 if v >= 0 else -0.6), f"{v:+.2f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _mean_delta(compare_rows: List[Dict[str, str]], metric: str) -> float | None:
    deltas: List[float] = []
    for row in compare_rows:
        b = _safe_float(row.get(f"baseline_{metric}"))
        t = _safe_float(row.get(f"transfer_{metric}"))
        if b is not None and t is not None:
            deltas.append((t - b) * 100.0)
    return mean(deltas) if deltas else None


def plot_accuracy_aspects(compare_payload: Dict[str, List[Dict[str, str]]], out_path: Path, dataset: str) -> None:
    aspects = ["accuracy", "common_acc", "outlier_acc", "hscore"]
    aspect_labels = ["Accuracy", "Common acc", "Outlier acc", "H-score"]

    tags = list(compare_payload.keys())
    if not tags:
        return
    tags.sort()

    x = list(range(len(tags)))
    width = 0.18
    plt.figure(figsize=(max(10, len(tags) * 0.8), 6))

    for idx, (aspect, label) in enumerate(zip(aspects, aspect_labels)):
        vals = []
        for tag in tags:
            m = _mean_delta(compare_payload[tag], aspect)
            vals.append(m if m is not None else 0.0)
        shift = (idx - 1.5) * width
        plt.bar([v + shift for v in x], vals, width=width, label=label)

    plt.axhline(0, color="black", linewidth=0.9)
    plt.xticks(x, [_clean_tag(t) for t in tags], rotation=35, ha="right")
    plt.ylabel("Transfer - baseline (percentage points)")
    plt.title(f"Ablation impact by metric aspect ({dataset})")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_hscore_distribution(compare_payload: Dict[str, List[Dict[str, str]]], out_path: Path, dataset: str) -> None:
    tags = []
    distributions = []
    for tag, rows in sorted(compare_payload.items()):
        vals = [_safe_float(r.get("transfer_hscore")) for r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            tags.append(tag)
            distributions.append(vals)
    if not distributions:
        return

    plt.figure(figsize=(max(10, len(tags) * 0.8), 5.5))
    plt.boxplot(distributions, tick_labels=[_clean_tag(t) for t in tags], showmeans=True)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Transfer H-score")
    plt.title(f"H-score distribution across ablations ({dataset})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_sngp_vs_det(lb_rows: List[Dict[str, str]], out_path: Path, dataset: str) -> None:
    groups: Dict[str, List[float]] = defaultdict(list)
    for row in lb_rows:
        imp = _safe_float(row.get("avg_improvement"))
        if imp is None:
            continue
        method = (row.get("method") or "").lower()
        if method == "sngp":
            groups["SNGP"].append(imp * 100.0)
        elif method:
            groups["Deterministic/other"].append(imp * 100.0)

    if not groups:
        return

    labels = list(groups.keys())
    means = [mean(groups[k]) for k in labels]
    errs = []
    for k in labels:
        vals = groups[k]
        if len(vals) < 2:
            errs.append(0.0)
        else:
            m = mean(vals)
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            errs.append(math.sqrt(var))

    plt.figure(figsize=(6.5, 4.8))
    bars = plt.bar(labels, means, yerr=errs, capsize=6, color=["#3D9970", "#FF851B"])
    plt.axhline(0, color="black", linewidth=0.9)
    plt.ylabel("Average improvement (%)")
    plt.title(f"SNGP improvement advantage ({dataset})")
    for b, v in zip(bars, means):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:+.2f}%", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_sngp_confidence(compare_payload: Dict[str, List[Dict[str, str]]], out_path: Path, dataset: str) -> None:
    tags = []
    dists = []
    for tag, rows in sorted(compare_payload.items()):
        entropy = [_safe_float(r.get("transfer_uncertainty_mean_entropy")) for r in rows]
        entropy = [e for e in entropy if e is not None]
        if entropy:
            tags.append(tag)
            dists.append(entropy)
    if not dists:
        return

    plt.figure(figsize=(max(10, len(tags) * 0.8), 5.5))
    plt.violinplot(dists, showmeans=True, showmedians=True)
    plt.xticks(range(1, len(tags) + 1), [_clean_tag(t) for t in tags], rotation=35, ha="right")
    plt.ylabel("Mean predictive entropy (lower is more confident)")
    plt.title(f"SNGP confidence profile by ablation ({dataset})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_confidence_vs_hscore(compare_payload: Dict[str, List[Dict[str, str]]], out_path: Path, dataset: str) -> None:
    plt.figure(figsize=(6.5, 5.2))
    any_points = False
    for tag, rows in sorted(compare_payload.items()):
        xs = []
        ys = []
        for r in rows:
            x = _safe_float(r.get("transfer_uncertainty_mean_entropy"))
            y = _safe_float(r.get("transfer_hscore"))
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
        if xs:
            any_points = True
            plt.scatter(xs, ys, alpha=0.75, label=_clean_tag(tag), s=28)
    if not any_points:
        plt.close()
        return

    plt.xlabel("Mean predictive entropy")
    plt.ylabel("Transfer H-score")
    plt.title(f"Confidence vs H-score ({dataset})")
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def write_dataset_summary(lb_rows: List[Dict[str, str]], out_path: Path, dataset: str) -> None:
    entries = []
    for r in lb_rows:
        imp = _safe_float(r.get("avg_improvement"))
        if imp is None:
            continue
        entries.append((r.get("tag", "unknown"), imp * 100.0, r.get("method", "")))
    if not entries:
        return
    entries.sort(key=lambda x: x[1], reverse=True)
    top = entries[0]

    sngp_vals = [v for _, v, m in entries if (m or "").lower() == "sngp"]
    det_vals = [v for _, v, m in entries if (m or "").lower() != "sngp"]

    lines = [
        f"# Plot summary for {dataset}",
        "",
        f"- Best configuration: **{top[0]}** with **{top[1]:+.2f}%** average improvement.",
        f"- Number of evaluated ablations/configurations: **{len(entries)}**.",
    ]
    if sngp_vals:
        lines.append(f"- Mean SNGP improvement: **{mean(sngp_vals):+.2f}%**.")
    if det_vals:
        lines.append(f"- Mean deterministic/other improvement: **{mean(det_vals):+.2f}%**.")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_root", type=Path, default=Path("checkpoint"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/dissertation_plots"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = _find_latest_runs_by_dataset(args.checkpoint_root)
    if not runs:
        raise SystemExit("No usable llm_run_* results found under checkpoint root.")

    for dataset, run_dir in sorted(runs.items()):
        lb_rows, compare_payload = _load_run_payload(run_dir)
        slug = dataset.lower().replace("_", "-")

        plot_ablation_improvement(lb_rows, args.output_dir / f"ablation-comparison-{slug}.png", dataset)
        plot_accuracy_aspects(compare_payload, args.output_dir / f"accuracy-aspects-{slug}.png", dataset)
        plot_hscore_distribution(compare_payload, args.output_dir / f"hscore-distribution-{slug}.png", dataset)
        plot_sngp_vs_det(lb_rows, args.output_dir / f"sngp-vs-deterministic-{slug}.png", dataset)
        plot_sngp_confidence(compare_payload, args.output_dir / f"sngp-confidence-{slug}.png", dataset)
        plot_confidence_vs_hscore(compare_payload, args.output_dir / f"confidence-vs-hscore-{slug}.png", dataset)
        write_dataset_summary(lb_rows, args.output_dir / f"summary-{slug}.md", dataset)

        print(f"Generated dissertation plots for {dataset} using {run_dir}")


if __name__ == "__main__":
    main()