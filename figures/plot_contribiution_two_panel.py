#!/usr/bin/env python3
"""Create a 2-panel dissertation contribution figure from prepared CSV summaries.

Panel A (Battery): SNGP vs no-SNGP transfer score delta and entropy delta by transfer pair.
Panel B (CWRU): Transfer gains in outlier accuracy and H-score.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def _to_float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _find_latest_by_glob(pattern: str) -> Path:
    candidates = sorted(Path().glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No file matched pattern: {pattern}")
    return candidates[-1]


def _prep_battery_rows(path: Path) -> tuple[list[str], list[float], list[float]]:
    rows = _read_csv(path)
    pairs = [r.get("pair", "?") for r in rows]
    score_delta_pp = [_to_float(r.get("score_delta_sngp_minus_no_sngp")) * 100.0 for r in rows]
    entropy_delta = [_to_float(r.get("entropy_delta_sngp_minus_no_sngp")) for r in rows]
    return pairs, score_delta_pp, entropy_delta


def _prep_cwru_rows(path: Path) -> tuple[list[str], list[float], list[float]]:
    rows = _read_csv(path)
    labels: list[str] = []
    outlier_gain_pp: list[float] = []
    hscore_gain_pp: list[float] = []

    for row in rows:
        src = row.get("source", "?")
        tgt = row.get("target", "?")
        labels.append(f"{src}→{tgt}")

        b_out = _to_float(row.get("baseline_outlier_acc"))
        t_out = _to_float(row.get("transfer_outlier_acc"))
        b_h = _to_float(row.get("baseline_hscore"))
        t_h = _to_float(row.get("transfer_hscore"))

        outlier_gain_pp.append((t_out - b_out) * 100.0)
        hscore_gain_pp.append((t_h - b_h) * 100.0)

    return labels, outlier_gain_pp, hscore_gain_pp


def make_figure(battery_csv: Path, cwru_csv: Path, out_fig: Path, title: str) -> None:
    b_pairs, b_score_delta_pp, b_entropy_delta = _prep_battery_rows(battery_csv)
    c_labels, c_outlier_gain_pp, c_hscore_gain_pp = _prep_cwru_rows(cwru_csv)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Panel A: Battery contribution (SNGP uncertainty effect vs score tradeoff)
    ax = axes[0]
    x = list(range(len(b_pairs)))
    bars = ax.bar(x, b_score_delta_pp, color="#4C72B0", alpha=0.85, label="Score delta (pp)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(b_pairs, rotation=35, ha="right")
    ax.set_ylabel("Transfer score delta (SNGP - no SNGP, pp)")
    ax.set_title("A) Battery: SNGP vs no-SNGP")

    ax2 = ax.twinx()
    ax2.plot(x, b_entropy_delta, color="#DD8452", marker="o", linewidth=2.0, label="Entropy delta")
    ax2.set_ylabel("Entropy delta (SNGP - no SNGP, uncertainty proxy)")

    mean_score = mean(b_score_delta_pp) if b_score_delta_pp else 0.0
    mean_entropy = mean(b_entropy_delta) if b_entropy_delta else 0.0
    ax.text(
        0.02,
        0.97,
        (
            f"mean score Δ: {mean_score:+.2f} pp\n"
            f"mean entropy Δ: {mean_entropy:+.3f}\n"
            "(SNGP contribution: higher uncertainty with safer transfer behavior)"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    # Combined legend for twin axes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=8,
        framealpha=0.95,
    )

    # Panel B: CWRU outlier/H-score gains
    ax = axes[1]
    x = list(range(len(c_labels)))
    width = 0.42
    ax.bar([i - width / 2 for i in x], c_outlier_gain_pp, width=width, color="#55A868", label="Outlier acc gain (pp)")
    ax.bar([i + width / 2 for i in x], c_hscore_gain_pp, width=width, color="#C44E52", label="H-score gain (pp)")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(c_labels, rotation=35, ha="right")
    ax.set_ylabel("Transfer - baseline gain (pp)")
    ax.set_title("B) CWRU: outlier-driven gains")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)

    mean_out = mean(c_outlier_gain_pp) if c_outlier_gain_pp else 0.0
    mean_h = mean(c_hscore_gain_pp) if c_hscore_gain_pp else 0.0
    ax.text(
        0.02,
        0.97,
        f"mean outlier Δ: {mean_out:+.2f} pp\nmean H-score Δ: {mean_h:+.2f} pp",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.suptitle(title)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)
    print(f"Saved contribution figure: {out_fig}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--battery_csv",
        type=Path,
        default=Path("dissertation_plots/battery_sngp_vs_no_sngp_latest.csv"),
        help="Battery SNGP-vs-no-SNGP paired summary CSV.",
    )
    parser.add_argument(
        "--cwru_csv",
        type=Path,
        default=None,
        help="Optional explicit corrected CWRU outlier CSV path.",
    )
    parser.add_argument(
        "--cwru_glob",
        default="dissertation_plots/*_CWRU_inconsistent_outlier_fixed.csv",
        help="Glob used when --cwru_csv is omitted (latest match is used).",
    )
    parser.add_argument(
        "--out_fig",
        type=Path,
        default=Path("dissertation_plots/contribution_figure_two_panel.png"),
    )
    parser.add_argument(
        "--title",
        default="Contribution Figure: Context + SNGP Improves Robust Transfer Behavior",
    )
    args = parser.parse_args()

    cwru_csv = args.cwru_csv if args.cwru_csv is not None else _find_latest_by_glob(args.cwru_glob)

    if not args.battery_csv.exists():
        raise FileNotFoundError(f"Battery CSV not found: {args.battery_csv}")
    if not cwru_csv.exists():
        raise FileNotFoundError(f"CWRU CSV not found: {cwru_csv}")

    make_figure(args.battery_csv, cwru_csv, args.out_fig, args.title)


if __name__ == "__main__":
    main()