#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualise CWRU ablation comparisons (history, cycle limits, head toggles)."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_GROUP_RULES = [
    ("cycle limits", "cycles_"),
    ("history toggles", "history_"),
    ("head toggles", "head_"),
    ("regularization", "ablate_"),
    ("chemistry toggles", "chemistry_"),
]

_DEFAULT_CWRU_RUN = Path(
    "/Users/moondiab/Documents/Dissertation/UDTL_Lable_Inconsistent-main/checkpoint/"
    "llm_run_20260126_213942"
)

def _find_latest_dataset_run(root: Path, dataset_tag: str) -> Path | None:
    runs = list(root.glob("llm_run_*/llm_leaderboard.csv"))
    matches: list[Path] = []
    for run in runs:
        try:
            df = pd.read_csv(run)
        except Exception:
            continue
        if "summary_csv" not in df.columns:
            continue
        if df["summary_csv"].astype(str).str.contains(dataset_tag, na=False).any():
            matches.append(run.parent)
    if not matches:
        return None
    ts_regex = re.compile(r"llm_run_(\d{8})_(\d{6})")

    def sort_key(path: Path) -> tuple[int, float]:
        match = ts_regex.search(path.name)
        if match:
            stamp = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").timestamp()
            return (1, stamp)
        return (0, path.stat().st_mtime)

    return max(matches, key=sort_key)


def _load_ablation_json(run_dir: Path) -> list[dict]:
    path = run_dir / "llm_ablation.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _assign_group(tag: str) -> str:
    for group, key in _GROUP_RULES:
        if key in tag:
            return group
    return "other"

def _infer_improvement_column(df: pd.DataFrame) -> str | None:
    for col in ("improvement", "delta_common", "delta_metric"):
        if col in df.columns:
            return col
    return None


def _load_compare_summaries(run_dir: Path, dataset_tag: str) -> pd.DataFrame:
    compare_dir = run_dir / "compare"
    if not compare_dir.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for path in sorted(compare_dir.glob(f"*summary*{dataset_tag}*.csv")):
        try:
            summary = pd.read_csv(path)
        except Exception:
            continue
        imp_col = _infer_improvement_column(summary)
        if imp_col is None:
            continue
        improvements = pd.to_numeric(summary[imp_col], errors="coerce").dropna()
        if improvements.empty:
            continue
        stem = path.stem
        tag = stem.split("_summary")[0] if "_summary" in stem else stem
        mean = float(improvements.mean())
        median = float(improvements.median())
        count = int(improvements.shape[0])
        if count > 1:
            ci95 = float(1.96 * improvements.std(ddof=1) / (count**0.5))
        else:
            ci95 = 0.0
        rows.append(
            {
                "tag": tag,
                "avg_improvement": mean,
                "improvement_median": median,
                "improvement_count": count,
                "improvement_ci95": ci95,
                "summary_csv": str(path),
            }
        )
    return pd.DataFrame(rows)



def _infer_cycle_limit(df: pd.DataFrame) -> pd.Series:
    if "cycle_limit" in df.columns:
        parsed = pd.to_numeric(df["cycle_limit"], errors="coerce")
    else:
        parsed = pd.Series([float("nan")] * len(df), index=df.index)
    cycle_regex = re.compile(r"cycles_(\d+)")
    for idx, tag in df["tag"].astype(str).items():
        if pd.notna(parsed.loc[idx]):
            continue
        match = cycle_regex.search(tag)
        if match:
            parsed.loc[idx] = float(match.group(1))
    return parsed


def plot_leaderboard(df: pd.DataFrame, out_path: Path) -> None:
    grouped = df.copy()
    grouped["group"] = grouped["tag"].astype(str).map(_assign_group)
    group_order = [group for group, _ in _GROUP_RULES] + ["other"]
    grouped["group_order"] = grouped["group"].apply(
        lambda group: group_order.index(group) if group in group_order else len(group_order)
    )
    ordered = grouped.sort_values(["group_order", "avg_improvement"], ascending=[True, False])
    if ordered.empty:
        return

    llm_pick_mask = ordered["tag"].astype(str).eq("llm_pick")
    if llm_pick_mask.any():
        highlight_idx = llm_pick_mask.idxmax()
    else:
        highlight_idx = ordered["avg_improvement"].idxmax()
    highlights = ordered.index == highlight_idx

    def _format_label(row: pd.Series, highlight: bool) -> str:
        tag = str(row["tag"])
        if not highlight:
            return tag
        if tag == "llm_pick":
            model_name = row.get("model_name") or "model"
            return f"{tag} ({model_name}, history+load/chem on)"
        return f"{tag} [best]"
    labels = [
        _format_label(row, highlight)
        for (_, row), highlight in zip(ordered.iterrows(), highlights)
    ]

    plt.figure(figsize=(11, 6))
    ax = plt.gca()
    yerr = None
    if "improvement_ci95" in ordered.columns:
        try:
            yerr = ordered["improvement_ci95"].fillna(0.0).to_list()
        except Exception:
            yerr = None

    palette = {
        "cycle limits": "#4C78A8",
        "history toggles": "#F58518",
        "head toggles": "#54A24B",
        "regularization": "#E45756",
        "chemistry toggles": "#9C6ADE",
        "other": "#999999",
    }
    colors = [palette.get(group, "#999999") for group in ordered["group"]]
    x_positions = list(range(len(ordered)))

    edgecolors = ["black" if highlight else "none" for highlight in highlights]
    linewidths = [1.5 if highlight else 0 for highlight in highlights]
    ax.bar(
        x_positions,
        ordered["avg_improvement"],
        color=colors,
        yerr=yerr,
        capsize=4,
        edgecolor=edgecolors,
        linewidth=linewidths,
    )

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
    for band_index, (start_idx, end_idx, _) in enumerate(group_bounds):
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

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for tick_label, highlight in zip(ax.get_xticklabels(), highlights):
        if highlight:
            tick_label.set_fontweight("bold")
    plt.ylabel("Avg. improvement vs. baseline")
    plt.title("CWRU ablation leaderboard")
    for i, (_, row) in enumerate(ordered.iterrows()):
        v = row["avg_improvement"]
        n = row.get("improvement_count")
        note = f"{v:+.3f}"
        if pd.notna(n):
            note += f"\nn={int(n)}"
        plt.text(i, v + 0.001, note, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_cycle_limits(df: pd.DataFrame, out_path: Path) -> None:
    scoped = df.copy()
    scoped["cycle_limit"] = _infer_cycle_limit(scoped)
    scoped = scoped.dropna(subset=["cycle_limit"])  # type: ignore[arg-type]
    if scoped.empty:
        return
    plt.figure(figsize=(8, 4))
    scoped = scoped.sort_values("cycle_limit")
    plt.plot(scoped["cycle_limit"], scoped["avg_improvement"], marker="o")
    plt.xlabel("Cycle horizon exposed")
    plt.ylabel("Avg. improvement vs. baseline")
    plt.title("CWRU early-cycle ablation")
    for x, y in zip(scoped["cycle_limit"], scoped["avg_improvement"]):
        plt.text(x, y, f"{y:+.3f}", ha="center", va="bottom", fontsize=8)
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

    avg_map = df.set_index("tag")["avg_improvement"].to_dict()
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
        ax.text(mid, y + 0.05, f"{delta:+.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([pair["label"] for pair in pairs])
    ax.set_xlabel("Avg. improvement vs. baseline")
    ax.set_title("CWRU baseline vs ablated pairs")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _write_summary(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        print("No ablation rows available for summary.")
        return

    ordered = df.sort_values("avg_improvement", ascending=False)
    llm_pick_rows = ordered[ordered["tag"].astype(str) == "llm_pick"]
    winner = llm_pick_rows.iloc[0] if not llm_pick_rows.empty else ordered.iloc[0]

    cycle_df = ordered.copy()
    cycle_df["cycle_limit"] = _infer_cycle_limit(cycle_df)
    cycle_df = cycle_df.dropna(subset=["cycle_limit"]).sort_values("cycle_limit")
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
        "# CWRU ablation summary",
        "",
        (
            f"Top candidate (LLM pick): **{winner['tag']}** (Δ={winner['avg_improvement']:+.4f})"
            if not llm_pick_rows.empty
            else f"Top candidate: **{winner['tag']}** (Δ={winner['avg_improvement']:+.4f})"
        ),
    ]
    if "improvement_median" in winner and not pd.isna(winner.get("improvement_median", float("nan"))):
        med = float(winner.get("improvement_median"))
        ci = winner.get("improvement_ci95")
        count = winner.get("improvement_count")
        ci_str = f"{float(ci):+.4f}" if ci is not None and not pd.isna(ci) else "n/a"
        count_str = int(count) if count is not None and not pd.isna(count) else "?"
        lines.append(f"Median Δ={med:+.4f} (n={count_str}, ±95% CI≈{ci_str}).")
    if not cycle_df.empty:
        best_cycle = cycle_df.loc[cycle_df["avg_improvement"].idxmax(), "cycle_limit"]
        lines.append(f"Best cycle-limited prompt: **{int(best_cycle)} cycles**")
        if plateau_cycle:
            lines.append(f"Improvement plateau begins by **{plateau_cycle} cycles** (≤1e-3 gain vs. prior).")

    out_path.write_text("\n".join(lines))
    print(f"Saved ablation summary to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help=(
            "Path to an llm_run_* folder (defaults to the latest known CWRU run, or "
            "auto-picks the latest CWRU run when omitted)."
        ),
    )
    parser.add_argument(
        "--checkpoint_root",
        type=Path,
        default=Path("checkpoint"),
        help="Root folder that stores llm_run_* directories.",
    )
    parser.add_argument(
        "--dataset_tag",
        type=str,
        default="CWRU_inconsistent",
        help="Dataset tag to filter leaderboard rows.",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip writing ablation_summary.md next to the plots.",
    )
    args = parser.parse_args()

    run_dir: Path
    if args.run_dir:
        run_dir = args.run_dir
    elif _DEFAULT_CWRU_RUN.exists():
        run_dir = _DEFAULT_CWRU_RUN
    else:
        latest = _find_latest_dataset_run(args.checkpoint_root, args.dataset_tag)
        if latest is None:
            raise SystemExit(f"No llm_run_* leaderboard found for {args.dataset_tag} under {args.checkpoint_root}.")
        run_dir = latest

    leaderboard_path = run_dir / "llm_leaderboard.csv"
    if leaderboard_path.exists():
        df = pd.read_csv(leaderboard_path)
        if "summary_csv" in df.columns:
            df = df[df["summary_csv"].astype(str).str.contains(args.dataset_tag, na=False)].copy()
    else:
        df = _load_compare_summaries(run_dir, args.dataset_tag)
        if df.empty:
            raise SystemExit(
                f"Missing leaderboard at {leaderboard_path} and no compare summaries found in {run_dir / 'compare'}."
            )
        print("Leaderboard missing; using compare summary CSVs for ablation aggregation.")

    
    if df.empty:
        raise SystemExit(f"No {args.dataset_tag} rows available to plot.")

    ablation_records = _load_ablation_json(run_dir)
    if ablation_records:
        print(f"Loaded {len(ablation_records)} ablation prompt variants from {run_dir}")

    lb_plot = run_dir / "cwru_ablation_leaderboard.png"
    plot_leaderboard(df, lb_plot)
    print(f"Saved leaderboard plot to {lb_plot}")

    cycle_plot = run_dir / "cwru_ablation_cycles.png"
    plot_cycle_limits(df, cycle_plot)
    if cycle_plot.exists():
        print(f"Saved cycle-ablation plot to {cycle_plot}")

    pair_plot = run_dir / "cwru_ablation_pairs.png"
    plot_pairwise_deltas(df, pair_plot)
    if pair_plot.exists():
        print(f"Saved ablation-pairs plot to {pair_plot}")

    if not args.no_summary:
        _write_summary(df, run_dir / "cwru_ablation_summary.md")


if __name__ == "__main__":
    main()
