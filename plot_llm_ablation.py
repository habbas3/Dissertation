#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualise LLM ablation comparisons (history, cycle limits, head toggles)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _find_latest_llm_run(root: Path) -> Path | None:
    runs = sorted(root.glob("llm_run_*/llm_leaderboard.csv"))
    if not runs:
        return None
    return runs[-1].parent


def _load_ablation_json(run_dir: Path) -> list[dict]:
    path = run_dir / "llm_ablation.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def plot_leaderboard(df: pd.DataFrame, out_path: Path) -> None:
    group_rules = [
        ("cycle limits", "cycles_"),
        ("history toggles", "history_"),
        ("head toggles", "head_"),
        ("regularization", "ablate_"),
    ]
    
    def _display_tag(tag: str) -> tuple[str, bool]:
        highlight = "cnn_openmax" in tag.lower()
        label = f"{tag} [LLM pick]" if highlight else tag
        return label, highlight

    def _assign_group(tag: str) -> str:
        for group, key in group_rules:
            if key in tag:
                return group
        return "other"

    grouped = df.copy()
    grouped["group"] = grouped["tag"].astype(str).map(_assign_group)
    group_order = [group for group, _ in group_rules] + ["other"]
    grouped["group_order"] = grouped["group"].apply(
        lambda group: group_order.index(group) if group in group_order else len(group_order)
    )
    grouped["cycle_order"] = grouped["cycle_limit"].fillna(float("inf"))
    grouped["order_within_group"] = grouped.apply(
        lambda row: row["cycle_order"] if row["group"] == "cycle limits" else -row["avg_improvement"],
        axis=1,
    )
    ordered = grouped.sort_values(["group_order", "order_within_group"], ascending=[True, True])
    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    yerr = None
    if "improvement_ci95" in ordered.columns:
        try:
            yerr = (ordered["improvement_ci95"].fillna(0.0) * 100.0).to_list()
        except Exception:
            yerr = None
    palette = {
        "cycle limits": "#4C78A8",
        "history toggles": "#F58518",
        "head toggles": "#54A24B",
        "regularization": "#E45756",
        "other": "#999999",
    }
    colors = [palette.get(group, "#999999") for group in ordered["group"]]
    x_positions = list(range(len(ordered)))
    avg_improvement_pct = ordered["avg_improvement"] * 100.0
    ax.bar(x_positions, avg_improvement_pct, color=colors, yerr=yerr, capsize=4)
    group_bounds: list[tuple[int, int, str]] = []
    start = 0
    for idx, group in enumerate(ordered["group"]):
        if idx == 0:
            continue
        prev_group = ordered["group"].iloc[idx - 1]
        if group != prev_group:
            group_bounds.append((start, idx - 1, prev_group))
            start = idx
    if len(ordered):
        group_bounds.append((start, len(ordered) - 1, ordered["group"].iloc[-1]))
    for band_index, (start_idx, end_idx, group) in enumerate(group_bounds):
        if band_index % 2 == 0:
            ax.axvspan(start_idx - 0.5, end_idx + 0.5, color="#000000", alpha=0.04, zorder=0)
        if end_idx + 1 < len(ordered):
            ax.axvline(end_idx + 0.5, color="#666666", alpha=0.4, linewidth=0.8)
    legend_handles = [
        plt.Line2D([0], [0], color=palette[group], lw=6, label=group)
        for group in group_order
        if group in ordered["group"].values
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Ablation group", loc="upper right", frameon=False)
    labels, highlights = zip(*(_display_tag(str(tag)) for tag in ordered["tag"]))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for tick_label, highlight in zip(ax.get_xticklabels(), highlights):
        if highlight:
            tick_label.set_fontweight("bold")
    plt.ylabel("Avg. improvement vs. baseline (%)")
    plt.title("LLM comparison + ablation leaderboard")
    for i, (_, row) in enumerate(ordered.iterrows()):
        v = row["avg_improvement"] * 100.0
        n = row.get("improvement_count")
        note = f"{v:+.2f}%"
        if pd.notna(n):
            note += f"\nn={int(n)}"
        plt.text(i, v + 0.1, note, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cycle_limits(df: pd.DataFrame, out_path: Path) -> None:
    scoped = df.dropna(subset=["cycle_limit"])  # type: ignore[arg-type]
    if scoped.empty:
        return
    plt.figure(figsize=(8, 4))
    scoped = scoped.sort_values("cycle_limit")
    improvement_pct = scoped["avg_improvement"] * 100.0
    plt.plot(scoped["cycle_limit"], improvement_pct, marker="o")
    plt.xlabel("Cycle horizon exposed")
    plt.ylabel("Avg. improvement vs. baseline (%)")
    plt.title("Early-cycle EOL ablation")
    for x, y in zip(scoped["cycle_limit"], improvement_pct):
        plt.text(x, y, f"{y:+.2f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def _collect_ablation_pairs(df: pd.DataFrame) -> list[dict]:
    tags = set(df["tag"].dropna().astype(str))
    pairs = []
    used = set()
    for tag in sorted(tags):
        if tag.endswith("_off"):
            counterpart = tag[:-4] + "_on"
            if counterpart in tags:
                key = tuple(sorted([tag, counterpart]))
                if key in used:
                    continue
                used.add(key)
                pairs.append(
                    {
                        "label": tag[:-4],
                        "baseline": tag,
                        "ablated": counterpart,
                    }
                )
    return pairs


def plot_pairwise_deltas(df: pd.DataFrame, out_path: Path) -> None:
    pairs = _collect_ablation_pairs(df)
    if not pairs:
        return

    avg_map = (df.set_index("tag")["avg_improvement"] * 100.0).to_dict()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(pairs))))

    y_positions = list(range(len(pairs)))
    for y, pair in zip(y_positions, pairs):
        baseline = avg_map.get(pair["baseline"])
        ablated = avg_map.get(pair["ablated"])
        if baseline is None or ablated is None:
            continue
        ax.plot([baseline, ablated], [y, y], color="slategray", linewidth=2)
        ax.scatter([baseline], [y], color="steelblue", zorder=3, label="baseline" if y == 0 else "")
        ax.scatter([ablated], [y], color="darkorange", zorder=3, label="ablated" if y == 0 else "")
        delta = ablated - baseline
        mid = (baseline + ablated) / 2
        ax.text(mid, y + 0.05, f"{delta:+.2f}%", ha="center", va="bottom", fontsize=9)


    ax.set_yticks(y_positions)
    ax.set_yticklabels([pair["label"] for pair in pairs])
    ax.set_xlabel("Avg. improvement vs. baseline (%)")
    ax.set_title("Baseline vs ablated pairs")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, out_path: Path) -> None:
    """Persist a short markdown summary of the ablation sweep."""

    if df.empty:
        print("No ablation rows available for summary.")
        return
    ordered = df.sort_values("avg_improvement", ascending=False)
    winner = ordered.iloc[0]

    cycle_df = ordered.dropna(subset=["cycle_limit"]).sort_values("cycle_limit")
    plateau_cycle: int | None = None
    if not cycle_df.empty:
        prev = None
        for _, row in cycle_df.iterrows():
            cur = float(row["avg_improvement"])
            if prev is not None and cur <= prev + 1e-3:
                plateau_cycle = int(row["cycle_limit"])
                break
            prev = cur

    lines = [
        "# LLM ablation summary",
        "",
        f"Top candidate: **{winner['tag']}** (Δ={winner['avg_improvement'] * 100.0:+.2f}%)",
    ]
    if "improvement_median" in winner and not pd.isna(winner.get("improvement_median", float("nan"))):
        med = float(winner.get("improvement_median")) * 100.0
        ci = winner.get("improvement_ci95")
        count = winner.get("improvement_count")
        ci_str = f"{float(ci) * 100.0:+.2f}%" if ci is not None and not pd.isna(ci) else "n/a"
        count_str = int(count) if count is not None and not pd.isna(count) else "?"
        lines.append(
            f"Median Δ={med:+.2f}% "
            f"(n={count_str}, ±95% CI≈{ci_str})."
        )
    if not cycle_df.empty:
        best_cycle = cycle_df.loc[cycle_df["avg_improvement"].idxmax(), "cycle_limit"]
        lines.append(f"Best cycle-limited prompt: **{int(best_cycle)} cycles**")
        if plateau_cycle:
            lines.append(f"Improvement plateau begins by **{plateau_cycle} cycles** (≤1e-3 gain vs. prior).")
            
    lines.extend(
        [
            "",
            "Definitions:",
            "- Δ improvement is reported as a percentage: (transfer metric − Zhao CNN baseline) × 100.",
            "- history_on includes prior leaderboard context in the prompt; history_off is a cold-start prompt.",
            "- chemistry_off removes battery chemistry hints; load_off removes CWRU load/HP/rpm transfer metadata.",
            "- cycle-limited prompts cap the LLM context to the first N cycles; hyperparameters stay locked to the",
            "  primary LLM pick when the ablation suite is configured with locked hyperparameters.",
        ]
    )

    out_path.write_text("\n".join(lines))
    print(f"Saved ablation summary to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=Path("checkpoint"),
        help="Path to an llm_run_* folder or the checkpoint root (auto-picks latest run).",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip writing ablation_summary.md next to the plots.",
    )
    args = parser.parse_args()

    run_dir: Path
    if args.run_dir.name.startswith("llm_run_") and args.run_dir.is_dir():
        run_dir = args.run_dir
    else:
        latest = _find_latest_llm_run(args.run_dir)
        if latest is None:
            raise SystemExit("No llm_run_* leaderboard found under the provided path.")
        run_dir = latest

    leaderboard_path = run_dir / "llm_leaderboard.csv"
    if not leaderboard_path.exists():
        raise SystemExit(f"Missing leaderboard at {leaderboard_path}")

    df = pd.read_csv(leaderboard_path)
    # Ablation deltas are computed vs. the Zhao CNN baseline; drop llm_pick for readability.
    df = df[df["tag"].astype(str) != "llm_pick"].copy()
    if df.empty:
        raise SystemExit("Leaderboard only contains llm_pick; no ablation rows to plot.")
    ablation_records = _load_ablation_json(run_dir)
    if ablation_records:
        print(f"Loaded {len(ablation_records)} ablation prompt variants from {run_dir}")

    lb_plot = run_dir / "ablation_leaderboard.png"
    plot_leaderboard(df, lb_plot)
    print(f"Saved leaderboard plot to {lb_plot}")

    cycle_plot = run_dir / "ablation_cycles.png"
    plot_cycle_limits(df, cycle_plot)
    if cycle_plot.exists():
        print(f"Saved cycle-ablation plot to {cycle_plot}")
        
    pair_plot = run_dir / "ablation_pairs.png"
    plot_pairwise_deltas(df, pair_plot)
    if pair_plot.exists():
        print(f"Saved ablation-pairs plot to {pair_plot}")
        
    if not args.no_summary:
        _write_summary(df, run_dir / "ablation_summary.md")


if __name__ == "__main__":
    main()