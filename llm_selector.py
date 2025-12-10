#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 13:07:22 2025

@author: habbas
"""

# llm_selector.py
import os, json, math, textwrap, csv, statistics
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

# ---- Optional providers -------------------------------------------------------
# Use OpenAI if OPENAI_API_KEY is present (official SDK); else try Ollama (local).
# OpenAI "Responses API" is the current recommended path and supports structured outputs. 
# Docs: https://platform.openai.com/docs/api-reference/responses  :contentReference[oaicite:0]{index=0}
# Official SDK install: pip install openai  (Libraries page) :contentReference[oaicite:1]{index=1}
OPENAI_OK = False
try:
    from openai import OpenAI  # official SDK (2024+)  :contentReference[oaicite:2]{index=2}
    OPENAI_OK = True
except Exception:
    pass

OLLAMA_OK = False
try:
    import ollama  # official Ollama Python SDK  :contentReference[oaicite:3]{index=3}
    OLLAMA_OK = True
except Exception:
    # fallback to REST later if needed
    pass

try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=False)
except Exception:
    pass



import re
from pathlib import Path

# -------------------- helpers (architectures + normalization) --------------------

_ALLOWED_MODELS = [
    "cnn_features_1d", "cnn_features_1d_sa", "cnn_openmax",
    "WideResNet", "WideResNet_sa", "WideResNet_edited"
]

_ALLOWED_ARCH = [
    "cnn_1d", "cnn_1d_sa", "cnn_openmax",
    "wideresnet", "wideresnet_sa", "wideresnet_edited"
]

def _normalize_arch(s: str) -> str:
    if not s:
        return ""
    t = s.strip().lower().replace("-", "_")
    aliases = {
        "cnn": "cnn_1d",
        "cnn1d": "cnn_1d",
        "cnn_features_1d": "cnn_1d",
        "cnn_features_1d_sa": "cnn_1d_sa",
        "wrn": "wideresnet",
        "wideresnet_sa": "wideresnet_sa",
        "wideresnet_edited": "wideresnet_edited",
    }
    if t in aliases:
        t = aliases[t]
    # keep only our known set; otherwise empty
    return t if t in _ALLOWED_ARCH else ""

def _normalize_model_name(s: str) -> str:
    if not s:
        return ""
    t = s.strip().replace("-", "_")
    # match allowed names case-insensitively
    for m in _ALLOWED_MODELS:
        if t.lower() == m.lower():
            return m
    # common aliases
    alias = {
        "cnn": "cnn_features_1d",
        "cnn_1d": "cnn_features_1d",
        "cnn_sa": "cnn_features_1d_sa",
        "openmax": "cnn_openmax",
        "wrn": "WideResNet",
        "wrn_sa": "WideResNet_sa",
        "wideresnet": "WideResNet",
        "wideresnetsa": "WideResNet_sa",
        "wideresnet_edited": "WideResNet_edited",
    }
    t2 = t.lower().replace("_", "")
    return alias.get(t2, "")

def _arch_to_model_name(arch: str, self_attention: bool, openmax: bool) -> str:
    """Map architecture + toggles to one of your concrete model_name strings."""
    a = (arch or "").lower()
    if a.startswith("cnn"):
        if openmax:
            return "cnn_openmax"
        return "cnn_features_1d_sa" if self_attention else "cnn_features_1d"
    if a.startswith("wideresnet_edited"):
        return "WideResNet_edited"
    if a.startswith("wideresnet"):
        return "WideResNet_sa" if self_attention else "WideResNet"
    # unknown arch -> fallback (lightweight default)
    return "cnn_features_1d_sa"

def _clamp(v, lo, hi, default):
    try:
        if isinstance(default, int):
            return int(max(lo, min(hi, int(v))))
        return float(max(lo, min(hi, float(v))))
    except Exception:
        return default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _split_total_examples(split: Optional[dict]) -> int:
    if not split:
        return 0
    counts = split.get("class_distribution") or {}
    total = 0
    for v in counts.values():
        total += _safe_int(v)
    if total:
        return total
    batch_shape = split.get("batch_shape") or []
    if batch_shape:
        return _safe_int(batch_shape[0])
    return 0


def _collect_cycle_stats(num_summary: Optional[dict]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    if not num_summary:
        return stats
    raw_stats = num_summary.get("cycle_stats") or {}
    for key, value in raw_stats.items():
        stats[key] = _safe_int(value)
    splits = (num_summary or {}).get("splits") or {}
    if "target_train_cycles" not in stats:
        stats["target_train_cycles"] = _split_total_examples(splits.get("target_train"))
    if "source_train_cycles" not in stats:
        stats["source_train_cycles"] = _split_total_examples(splits.get("source_train"))
    return stats


def _csv_column_stats(path: Path, column: str) -> Optional[dict[str, float]]:
    values: list[float] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = row.get(column)
                if raw is None:
                    continue
                try:
                    values.append(float(raw))
                except ValueError:
                    continue
    except Exception:
        return None
    if not values:
        return None
    count = len(values)
    mean_val = statistics.fmean(values) if hasattr(statistics, "fmean") else float(sum(values) / count)
    median_val = statistics.median(values)
    std_val = statistics.pstdev(values) if count > 1 else 0.0
    stderr = std_val / math.sqrt(count) if count > 1 else 0.0
    lower_bound = mean_val - 1.96 * stderr if count > 1 else mean_val
    positive_count = sum(1 for v in values if v > 0)
    return {
        "mean": float(mean_val),
        "median": float(median_val),
        "std": float(std_val),
        "count": float(count),
        "positive_count": float(positive_count),
        "lower_bound": float(lower_bound),
    }


def _mean_csv_column(path: Path, column: str) -> Optional[float]:
    stats = _csv_column_stats(path, column)
    return None if stats is None else stats["mean"]


_HISTORICAL_TAG_OVERRIDES: dict[str, dict[str, Any]] = {
    "deterministic_cnn": {
        "architecture": "cnn_1d",
        "self_attention": False,
        "sngp": False,
        "openmax": False,
        "dropout": 0.25,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "lambda_src": 1.0,
        "model_name": "cnn_features_1d",
    },
    "sngp_wrn_sa": {
        "architecture": "wideresnet_sa",
        "self_attention": True,
        "sngp": True,
        "openmax": False,
        "dropout": 0.3,
        "learning_rate": 5e-4,
        "batch_size": 96,
        "lambda_src": 0.8,
        "model_name": "WideResNet_sa",
    },
}


def _maybe_apply_historical_winner(
    cfg: dict,
    num_summary: Optional[dict],
    allow_history: bool = True,
) -> Tuple[dict, Optional[str]]:
    if not allow_history:
        return cfg, None
    dataset = ""
    if num_summary:
        dataset = str(num_summary.get("dataset") or num_summary.get("dataset_variant") or "").strip()
    if not dataset:
        return cfg, None

    checkpoint_root = Path("checkpoint")
    if not checkpoint_root.exists():
        return cfg, None

    best_tag: Optional[str] = None
    best_score = float("-inf")
    best_stats: Optional[dict[str, float]] = None
    min_runs = int(os.environ.get("LLM_HIST_MIN_RUNS", 3))
    min_positive_frac = float(os.environ.get("LLM_HIST_MIN_POSITIVE_FRAC", 0.6))
    pattern = f"*_{dataset}.csv"
    for run_dir in sorted(checkpoint_root.glob("llm_run_*")):
        compare_dir = run_dir / "compare"
        if not compare_dir.is_dir():
            continue
        for csv_path in compare_dir.glob(pattern):
            tag = csv_path.stem.split("_summary_", 1)[0]
            stats = _csv_column_stats(csv_path, "improvement")
            if stats is None:
                continue
            count = int(stats.get("count", 0))
            positive_count = float(stats.get("positive_count", 0.0))
            if count < min_runs:
                continue
            if stats.get("median", 0.0) <= 0:
                continue
            frac_positive = positive_count / count if count > 0 else 0.0
            if frac_positive < min_positive_frac:
                continue
            if stats.get("lower_bound", float("-inf")) <= 0:
                continue
            avg = float(stats.get("mean", float("-inf")))
            if avg > best_score:
                best_score = avg
                best_tag = tag
                best_stats = stats

    if not best_tag or best_tag == "llm_pick":
        return cfg, None
    if best_score <= 0:
        return cfg, None

    override = _HISTORICAL_TAG_OVERRIDES.get(best_tag)
    if not override:
        return cfg, None

    updated = dict(cfg)
    updated.update({k: v for k, v in override.items() if k not in {"architecture", "model_name"}})

    if "architecture" in override:
        updated["architecture"] = override["architecture"]
    if "openmax" in override:
        updated["openmax"] = override["openmax"]
        updated["use_unknown_head"] = bool(override["openmax"])
    updated["self_attention"] = override.get("self_attention", updated.get("self_attention", False))
    updated["sngp"] = override.get("sngp", updated.get("sngp", False))

    arch = updated.get("architecture", cfg.get("architecture", ""))
    updated["model_name"] = override.get(
        "model_name",
        _arch_to_model_name(arch, updated.get("self_attention", False), updated.get("openmax", False)),
    )
    
    if best_stats is None:
        return cfg, None

    note = (
        f"Historical leaderboard favoured {best_tag} on {dataset} "
        f"(mean Δ={best_stats['mean']:+.4f}, median Δ={best_stats['median']:+.4f}, "
        f"n={int(best_stats['count'])}, 95% LCB={best_stats['lower_bound']:+.4f}); "
        "applied its hyperparameters."
    )
    return updated, note




def _numeric_heuristic_adjust(
    cfg: dict,
    num_summary: Optional[dict],
    allow_history: bool = True,
) -> Tuple[dict, list[str]]:
    """Refine the config using numeric signals so the selection is data-aware."""

    if not num_summary:
        return cfg, []

    splits = num_summary.get("splits") or {}
    seq_len = _safe_int(num_summary.get("seq_len") or num_summary.get("sequence_length_requested") or 128, 128)
    channels = _safe_int(num_summary.get("channels") or (splits.get("source_train") or {}).get("channels") or 1, 1)
    dataset_name = str(num_summary.get("dataset") or "").lower()
    dataset_variant = str(num_summary.get("dataset_variant") or dataset_name).lower()
    cycle_stats = _collect_cycle_stats(num_summary)
    tgt_cycles = cycle_stats.get("target_train_cycles", 0)
    src_cycles = cycle_stats.get("source_train_cycles", 0)
    total_target_examples = _split_total_examples(splits.get("target_train"))
    metrics = num_summary.get("metrics") or {}
    src_acc = metrics.get("source_acc") or metrics.get("source_val_acc")
    tgt_acc = metrics.get("target_acc") or metrics.get("target_val_acc")
    transfer_gap = None
    if src_acc is not None and tgt_acc is not None:
        try:
            transfer_gap = float(src_acc) - float(tgt_acc)
        except Exception:
            transfer_gap = None
    notes = str(num_summary.get("notes", "")).lower()

    arch_scores: Dict[str, float] = {arch: 0.0 for arch in _ALLOWED_ARCH}
    complexity = seq_len * max(channels, 1)
    
    cwru_bias_note = None

    for arch in arch_scores:
        if arch.endswith("_sa"):
            arch_scores[arch] += 0.4 if seq_len >= 256 else -0.2
        if arch.startswith("wideresnet"):
            arch_scores[arch] += 0.6 if complexity >= 4096 else -0.3
            arch_scores[arch] += 0.4 if channels >= 6 else 0.0
        if arch.startswith("cnn"):
            arch_scores[arch] += 0.5 if complexity <= 4096 else -0.2
            arch_scores[arch] += 0.3 if tgt_cycles <= 400 else 0.0

        if tgt_cycles and tgt_cycles < 150:
            arch_scores[arch] -= 0.4 if arch.startswith("wideresnet") else 0.0
            
    if "cwru" in dataset_name or "cwru" in dataset_variant or "bearing" in dataset_variant:
        arch_scores["wideresnet_sa"] += 0.7
        arch_scores["wideresnet"] += 0.45
        arch_scores["cnn_1d"] -= 0.2
        if not cfg.get("sngp"):
            cfg["sngp"] = True
        cwru_bias_note = (
            "CWRU bearings rewarded wideresnet capacity and SNGP calibration over the Zhao CNN baseline."
        )

    current_arch = cfg.get("architecture") or ""
    if current_arch in arch_scores:
        arch_scores[current_arch] += 0.25  # retain some prior weight

    best_arch = max(arch_scores.items(), key=lambda kv: kv[1])[0]

    heuristic_notes: list[str] = []

    if best_arch != current_arch:
        cfg["architecture"] = best_arch
        cfg["model_name"] = _arch_to_model_name(best_arch, cfg.get("self_attention", False), cfg.get("openmax", False))
        heuristic_notes.append(
            f"Numeric complexity score favored {best_arch} for {channels}×{seq_len} windows."
        )
        
    if cwru_bias_note:
        heuristic_notes.append(cwru_bias_note)

    # Toggle reasoning
    wants_attention = seq_len >= 256 or channels >= 6
    cfg["self_attention"] = bool(cfg.get("self_attention") or best_arch.endswith("_sa") or wants_attention and best_arch.startswith("wide"))
    if cfg["self_attention"]:
        arch = cfg.get("architecture", best_arch)
        if arch.startswith("cnn") and not arch.endswith("_sa") and arch != "cnn_openmax":
            cfg["architecture"] = "cnn_1d_sa"
            cfg["model_name"] = _arch_to_model_name("cnn_1d_sa", True, cfg.get("openmax", False))
        elif arch.startswith("wideresnet") and not arch.endswith("_sa") and "edited" not in arch:
            cfg["architecture"] = "wideresnet_sa"
            cfg["model_name"] = _arch_to_model_name("wideresnet_sa", True, cfg.get("openmax", False))

    open_set = "label_inconsistent" in notes or "open" in notes
    if open_set and not cfg.get("sngp", False):
        cfg["sngp"] = True
        heuristic_notes.append("Enabled SNGP to stabilise open-set transfer risk detected in metadata.")
    if open_set and num_summary.get("dataset") == "CWRU_inconsistent" and not cfg.get("openmax", False):
        cfg["openmax"] = True
        heuristic_notes.append("OpenMax head activated for explicit unknown rejection on inconsistent labels.")

    if cfg.get("openmax"):
        cfg["architecture"] = "cnn_openmax"
        cfg["model_name"] = "cnn_openmax"
        cfg["self_attention"] = False
        cfg["use_unknown_head"] = True
    else:
        cfg["use_unknown_head"] = bool(cfg.get("openmax", False))
        
    if transfer_gap is not None:
        if transfer_gap > 0.08:
            cfg["warmup_epochs"] = max(cfg.get("warmup_epochs", 3), 3)
            cfg["lambda_src"] = max(cfg.get("lambda_src", 1.0), 1.2)
            cfg["sngp"] = True if cfg.get("sngp") is False else cfg.get("sngp")
            heuristic_notes.append(
                "Source accuracy notably higher than target → keep head-only warmup and calibrated SNGP for safer transfer."
            )
        elif transfer_gap < -0.05:
            cfg["warmup_epochs"] = min(cfg.get("warmup_epochs", 3), 2)
            heuristic_notes.append(
                "Target already outperforms source → shorten warmup to accelerate full fine-tuning."
            )

    # Hyperparameter tuning
    if tgt_cycles and tgt_cycles < 180:
        cfg["dropout"] = max(cfg.get("dropout", 0.3), 0.35)
        cfg["learning_rate"] = min(cfg.get("learning_rate", 3e-4), 5e-4)
        cfg["batch_size"] = min(cfg.get("batch_size", 64), 32)
        heuristic_notes.append(
            "Low target-cycle count → tightened lr/dropout and batch size to curb overfitting."
        )
    elif tgt_cycles and tgt_cycles > 600:
        cfg["learning_rate"] = min(max(cfg.get("learning_rate", 3e-4), 7e-4), 2e-3)
        cfg["batch_size"] = min(max(cfg.get("batch_size", 64), 96), 196)
        heuristic_notes.append(
            "Rich target coverage allows larger batch and step size for faster adaptation."
        )

    if total_target_examples and total_target_examples < cfg.get("batch_size", 64):
        cfg["batch_size"] = max(8, 2 ** int(math.log2(max(4, total_target_examples // 2))))

    if src_cycles and tgt_cycles:
        ratio = tgt_cycles / max(src_cycles, 1)
        if ratio < 0.4:
            cfg["lambda_src"] = max(cfg.get("lambda_src", 1.0), 1.5)
            heuristic_notes.append("Target has far fewer cycles than source → keeping stronger source supervision.")
        elif ratio > 1.2:
            cfg["lambda_src"] = min(cfg.get("lambda_src", 1.0), 0.6)
            heuristic_notes.append("Target richer than source → down-weighting source loss for quicker domain fit.")
            
    cfg, hist_note = _maybe_apply_historical_winner(cfg, num_summary, allow_history=allow_history)
    if hist_note:
        heuristic_notes.append(hist_note)
    return cfg, heuristic_notes


def _autofill_rationale(cfg: dict, num_summary: dict, provider_reason: str = "", heuristic_notes: Optional[list[str]] = None) -> str:
    ch = num_summary.get("channels")
    sl = num_summary.get("seq_len")
    notes = str(num_summary.get("notes", "")).lower()
    cycle_stats = _collect_cycle_stats(num_summary)
    tgt_cycles = cycle_stats.get("target_train_cycles")
    src_cycles = cycle_stats.get("source_train_cycles")
    metrics = num_summary.get("metrics") or {}
    src_acc = metrics.get("source_acc") or metrics.get("source_val_acc")
    tgt_acc = metrics.get("target_acc") or metrics.get("target_val_acc")
    transfer_gap = None
    if src_acc is not None and tgt_acc is not None:
        try:
            transfer_gap = float(src_acc) - float(tgt_acc)
        except Exception:
            transfer_gap = None
    parts: list[str] = []
    dataset_variant = str(num_summary.get("dataset_variant") or "").lower()

    if provider_reason:
        parts.append(provider_reason.strip().rstrip(".") + ".")
    if ch is not None and sl is not None:
        parts.append(f"Evaluated {ch} channels × {sl}-step windows and matched them with {cfg.get('architecture')} capacity.")

    if tgt_cycles or src_cycles:
        if tgt_cycles and src_cycles:
            parts.append(
                f"Target fine-tune spans ≈{tgt_cycles} cycles versus {src_cycles} source cycles, guiding lr={cfg['learning_rate']:.2e} and dropout={cfg['dropout']:.2f}."
            )
        elif tgt_cycles:
            parts.append(
                f"Observed ≈{tgt_cycles} target cycles and set lr={cfg['learning_rate']:.2e} with batch size {cfg['batch_size']} to stay stable."
            )
            
    if transfer_gap is not None:
        direction = "lags" if transfer_gap > 0 else "exceeds"
        parts.append(
            f"Transfer gap detected: target {direction} source by {abs(transfer_gap):.2%}; warmup_epochs={cfg.get('warmup_epochs', 0)} keeps pretrained features steady."
        )

    if cfg.get("warmup_epochs", 0):
        parts.append(
            f"Backbone frozen for {cfg['warmup_epochs']} epoch(s) before full fine-tuning to stabilise feature reuse."
        )

    if ("label_inconsistent" in notes or "open" in notes) and (cfg.get("sngp") or cfg.get("openmax")):
        parts.append("Label inconsistency flagged → calibrated heads (SNGP/OpenMax) stay enabled for open-set robustness.")
        
    if "cwru" in dataset_variant and len(parts) < 4:
        parts.append("Benchmarking against Zhao et al.'s CNN baseline to highlight transfer-aware gains.")

    if heuristic_notes:
        for msg in heuristic_notes:
            if len(parts) >= 4:
                break
            parts.append(msg.rstrip(".") + ".")

    # Ensure between 2-4 sentences by padding with architecture summary if required
    if not parts:
        parts = ["Configuration selected from data shape and task setup."]
    if len(parts) < 2 and cfg.get("architecture"):
        parts.append(f"Architecture {cfg['architecture']} keeps self_attention={cfg.get('self_attention')} and sngp={cfg.get('sngp')} for stability.")

    return " ".join(parts[:4])
# -------------------------------------------------------------------------------



def _list_model_architectures() -> list[str]:
    """Return available model architectures by parsing models/__init__.py."""
    init_py = Path(__file__).resolve().parent / "models" / "__init__.py"
    names: list[str] = []
    try:
        with init_py.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("from") or "import" not in line:
                    continue
                target = line.split("import", 1)[1].split("#", 1)[0]
                for part in target.split(","):
                    part = part.strip()
                    if " as " in part:
                        part = part.split(" as ", 1)[1].strip()
                    if re.search(
                        r"Adversarial|sngp|spectral|attention|classifier|auxiliary",
                        part,
                        re.IGNORECASE,
                    ):
                        continue
                    names.append(part)
    except Exception:
        pass
    fallback = [
        "cnn_features_1d",
        "cnn_features_1d_sa",
        "cnn_openmax",
        "WideResNet",
        "WideResNet_sa",
        "WideResNet_edited",
    ]
    return sorted(set(names)) or fallback


MODEL_ARCHS = _list_model_architectures()

_ARCHITECTURE_GUIDE = {
    "cnn_1d": {
        "headline": "Lightweight 1-D temporal CNN",
        "best_for": (
            "Short-to-mid sequences (<256 steps) with <=8 channels,"
            " fast iteration, or when GPU memory is limited."
        ),
        "strengths": [
            "Low latency and easy to regularize",
            "Works well on CWRU bearings with narrow frequency bands",
            "Stable baseline when data is mostly closed-set",
        ],
        "watchouts": [
            "May underfit long-horizon battery curves",
            "Less expressive for high-channel sensor arrays",
        ],
    },
    "cnn_1d_sa": {
        "headline": "CNN with lightweight self-attention tail",
        "best_for": (
            "Sequences needing local feature extraction plus some long-range"
            " mixing (Argonne battery cycle slices, multi-axis sensors)."
        ),
        "strengths": [
            "Keeps CNN efficiency while capturing multi-cycle correlations",
            "Balances between latency and expressivity",
        ],
        "watchouts": [
            "Slightly higher memory than pure CNN",
            "Attention layer can overfit tiny datasets without dropout",
        ],
    },
    "cnn_openmax": {
        "headline": "CNN with OpenMax calibrated tail",
        "best_for": (
            "Closed-set CNN workloads where explicit unknown rejection is"
            " mandatory (e.g., CWRU with novel fault types)."
        ),
        "strengths": [
            "Adds open-set calibration to the fast CNN backbone",
            "Pairs well with label-inconsistent or anomaly-heavy splits",
        ],
        "watchouts": [
            "Assumes CNN-appropriate sequence lengths",
            "Requires tuning OpenMax thresholds for noisy labels",
        ],
    },
    "wideresnet": {
        "headline": "WideResNet 1-D (high capacity)",
        "best_for": (
            "Longer sequences (>=256 steps) or >8 channel inputs needing"
            " stronger representation power (battery chemo-mechanical signals)."
        ),
        "strengths": [
            "Deep residual blocks capture rich frequency structure",
            "Handles transfer tasks with large domain gap",
        ],
        "watchouts": [
            "Higher compute/memory footprint",
            "Can overfit small CWRU subsets without augmentation",
        ],
    },
    "wideresnet_sa": {
        "headline": "WideResNet with attention head",
        "best_for": (
            "Hybrid scenarios: long multi-channel sequences where global"
            " context matters (battery degradation trajectories)."
        ),
        "strengths": [
            "Residual depth + attention for regime shifts",
            "Works well when label inconsistency requires context",
        ],
        "watchouts": [
            "Largest memory use; ensure batch size is feasible",
            "Needs dropout to avoid memorizing small source domains",
        ],
    },
    "wideresnet_edited": {
        "headline": "WideResNet variant tuned for battery",
        "best_for": (
            "Argonne battery cycles with physics-inspired feature edits"
            " (e.g., state-of-health regression buckets)."
        ),
        "strengths": [
            "Pre-activation blocks favor smooth degradation patterns",
            "Often strongest when transfer source ≠ target chemistry",
        ],
        "watchouts": [
            "Less validated on vibration data",
            "Expect longer warm-up and sensitivity to learning rate",
        ],
    },
}

_TOGGLE_GUIDE = {
    "self_attention": {
        "purpose": "Captures long-range interactions beyond convolutional receptive fields.",
        "use_when": "Sequence length is large or cross-channel alignment matters.",
        "avoid_when": "Sequence is extremely short or data volume is tiny (risk of overfit).",
    },
    "sngp": {
        "purpose": "Spectral-normalized Gaussian Process head for calibrated uncertainty.",
        "use_when": "Expect domain shift (transfer) or need reliable OOD detection.",
        "avoid_when": "Latency budget is strict and domain is stable closed-set.",
    },
    "openmax": {
        "purpose": "Adds OpenMax logits for explicit unknown-class rejection.",
        "use_when": "Have outliers/label inconsistency and want hard unknown bucket.",
        "avoid_when": "Dataset is clean closed-set and decision boundary is tight.",
    },
}



# ------------------------------------------------------------------------------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "model_name": {"type": "string", "enum": MODEL_ARCHS},
        "self_attention": {"type": "boolean"},
        "sngp": {"type": "boolean"},
        "openmax": {"type": "boolean"},
        "use_unknown_head": {"type": "boolean"},
        "bottleneck": {"type": "integer", "minimum": 16, "maximum": 1024},
        "dropout": {"type": "number", "minimum": 0.0, "maximum": 0.8},
        "learning_rate": {"type": "number", "minimum": 1e-5, "maximum": 1e-2},
        "batch_size": {"type": "integer", "minimum": 4, "maximum": 256},
        "lambda_src": {"type": "number", "minimum": 0.0, "maximum": 5.0},
        "warmup_epochs": {"type": "integer", "minimum": 0, "maximum": 20},
        "rationale": {"type": "string"}
    },
    "required": ["model_name"],
    "additionalProperties": False
}

_ARCH_LIST_STR = ", ".join(MODEL_ARCHS)

SYSTEM_PROMPT = """\
You are a model-selection assistant for time-series fault diagnosis / battery health and CWRU vibration data.
Given a SHORT dataset description, a MODEL REFERENCE cheat-sheet, and SMALL numeric summary, recommend ONE configuration likely to perform best.

Leverage the architecture/toggle guidance to weigh trade-offs (capacity vs. sequence length, open-set risk, transfer gap, compute limits).

The baseline to beat is Zhao et al.'s deterministic 1-D CNN without transfer learning; bias choices toward configs that can surpass it on the provided dataset statistics.

Return STRICT JSON with these fields (no prose outside JSON):
- architecture: one of [cnn_1d, cnn_1d_sa, cnn_openmax, wideresnet, wideresnet_sa, wideresnet_edited]
- model_name: one of [cnn_features_1d, cnn_features_1d_sa, cnn_openmax, WideResNet, WideResNet_sa, WideResNet_edited]
- self_attention: boolean
- sngp: boolean
- openmax: boolean
- use_unknown_head: boolean
- bottleneck: integer (16..1024)
- dropout: float (0..0.8)
- learning_rate: float (1e-5..1e-2)
- batch_size: integer (4..256)
- lambda_src: float (0..5)
- warmup_epochs: integer (0..20)
- rationale: 2–4 sentences referencing channels, seq_len, label consistency/open-set, and trade-offs among SA/SNGP/OpenMax.

Return VALID JSON ONLY (no extra text).
"""


def _summarize_numeric(num_summary: Dict[str, Any]) -> str:
    """"Render a compact-yet-informative summary for the LLM prompt."""

    def _fmt(value: Any, depth: int = 0) -> str:
        if isinstance(value, (int, float)):
            return f"{value}"
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            head = [
                _fmt(v, depth + 1)
                for v in value[:4]
            ]
            suffix = ""
            if len(value) > 4:
                suffix = f", … (+{len(value) - 4})"
            return "[" + ", ".join(head) + suffix + "]"
        if isinstance(value, dict):
            items = list(value.items())
            parts = []
            for i, (k, v) in enumerate(items[:6]):
                parts.append(f"{k}: {_fmt(v, depth + 1)}")
            if len(items) > 6:
                parts.append(f"… (+{len(items) - 6} more)")
            return "{" + ", ".join(parts) + "}"
        return textwrap.shorten(str(value), width=160, placeholder="…")
    lines = []
    for key in sorted(num_summary.keys()):
        lines.append(f"{key}: {_fmt(num_summary[key])}")
    return "\n".join(lines)

def _format_architecture_reference() -> str:
    lines: list[str] = []
    for arch in _ALLOWED_ARCH:
        guide = _ARCHITECTURE_GUIDE.get(arch)
        if not guide:
            continue
        lines.append(f"{arch} — {guide['headline']}")
        lines.append(f"  Best for: {guide['best_for']}")
        if guide.get("strengths"):
            strengths = "; ".join(guide["strengths"])
            lines.append(f"  Strengths: {strengths}")
        if guide.get("watchouts"):
            watch = "; ".join(guide["watchouts"])
            lines.append(f"  Watch-outs: {watch}")
        lines.append("")

    lines.append("Toggles and heads:")
    for toggle, info in _TOGGLE_GUIDE.items():
        lines.append(f"  {toggle}: {info['purpose']}")
        lines.append(f"    Use when: {info['use_when']}")
        lines.append(f"    Avoid when: {info['avoid_when']}")
    return "\n".join(lines).strip()


def _limit_summary_cycles(num_summary: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Deep copy a numeric summary and clamp cycle hints to an early horizon."""

    import copy as _copy

    scoped = _copy.deepcopy(num_summary)
    scoped["cycle_limit_hint"] = int(limit)
    cycle_stats = scoped.get("cycle_stats") or {}
    for key in [
        "target_train_cycles",
        "source_train_cycles",
        "target_val_cycles",
        "source_val_cycles",
    ]:
        if key in cycle_stats:
            try:
                cycle_stats[key] = min(int(cycle_stats[key]), int(limit))
            except Exception:
                cycle_stats[key] = int(limit)
    scoped["cycle_stats"] = cycle_stats

    for split in (scoped.get("splits") or {}).values():
        shape = split.get("batch_shape") or []
        if isinstance(shape, list) and shape:
            try:
                shape[0] = min(int(shape[0]), int(limit))
            except Exception:
                shape[0] = int(limit)
        if "preview" in split:
            split["preview"]["cycle_limit_hint"] = int(limit)
    return scoped


def run_ablation_suite(
    text_context: str,
    num_summary: Dict[str, Any],
    backend: str = "auto",
    model: Optional[str] = None,
    debug_dir: Optional[str] = None,
    cycle_horizons: Optional[list[int]] = None,
) -> list[dict[str, Any]]:
    """
    Generate configs for a small ablation grid:
    - history_on: full context (default behaviour)
    - history_off: cold-start without checkpoint leaderboard hints
    - cycle-limited variants: truncated early-cycle exposure
    """

    records: list[dict[str, Any]] = []

    primary = select_config(
        text_context,
        num_summary,
        backend=backend,
        model=model,
        debug_dir=debug_dir,
        allow_history=True,
    )
    records.append({
        "tag": "history_on",
        "config": primary,
        "cycle_limit": None,
    })

    cold = select_config(
        text_context,
        num_summary,
        backend=backend,
        model=model,
        debug_dir=debug_dir,
        allow_history=False,
    )
    records.append({
        "tag": "history_off",
        "config": cold,
        "cycle_limit": None,
    })

    for horizon in cycle_horizons or []:
        scoped_summary = _limit_summary_cycles(num_summary, horizon)
        cfg = select_config(
            text_context,
            scoped_summary,
            backend=backend,
            model=model,
            debug_dir=debug_dir,
            allow_history=True,
        )
        records.append({
            "tag": f"cycles_{horizon}",
            "config": cfg,
            "cycle_limit": int(horizon),
            "num_summary": scoped_summary,
        })

    return records


def _build_user_prompt(text_context: str, num_summary: Dict[str, Any]) -> str:
    summary = _summarize_numeric(num_summary)
    schema_str = json.dumps(JSON_SCHEMA, indent=2)
    arch_reference = textwrap.shorten(
        _format_architecture_reference(), width=2400, placeholder="..."
    )
    return f"""\
DATASET CONTEXT
---------------
{textwrap.shorten(text_context.strip(), width=2000, placeholder='...')}

MODEL REFERENCE
---------------
{arch_reference}


NUMERIC SUMMARY
---------------
{summary}

REQUIRED JSON SCHEMA
--------------------
{schema_str}

Return ONLY a JSON object that matches the schema.
"""

def _validate_or_default(payload, num_summary=None, allow_history: bool = True) -> dict:
    """
    Tolerant validator: parse provider JSON (or text), normalize fields,
    honor 'architecture' if provided, and compose a specific rationale.
    """
    import json
    if num_summary is None:
        num_summary = {}

    # Parse payload -> dict
    obj = {}
    try:
        obj = json.loads(payload) if isinstance(payload, str) else payload
        if not isinstance(obj, dict):
            obj = {}
    except Exception:
        obj = {}

    # 1) read raw fields
    raw_arch = str(obj.get("architecture", obj.get("arch", "")) or "").strip()
    raw_model = str(obj.get("model_name", "")).strip()
    raw_self_att = obj.get("self_attention", None)
    raw_sngp = obj.get("sngp", None)
    raw_openmax = obj.get("openmax", None)
    raw_use_unk = obj.get("use_unknown_head", None)

    # 2) normalize architecture and model_name
    arch = _normalize_arch(raw_arch)
    model_name = _normalize_model_name(raw_model)

    # 3) heuristic defaults if missing
    ch = num_summary.get("channels", None)
    sl = num_summary.get("seq_len", None)

    if not arch and model_name:
        # infer arch from model_name
        mn = model_name.lower()
        if "wideresnet_edited" in mn:
            arch = "wideresnet_edited"
        elif "wideresnet" in mn:
            arch = "wideresnet_sa" if "sa" in mn else "wideresnet"
        elif "openmax" in mn:
            arch = "cnn_openmax"
        elif "cnn_features_1d_sa" in mn:
            arch = "cnn_1d_sa"
        else:
            arch = "cnn_1d"

    if not arch and not model_name:
        # choose a sensible default arch from data shape
        if (sl and sl >= 256) or (ch and ch >= 16):
            arch = "wideresnet_sa"
        else:
            arch = "cnn_1d_sa"

    # 4) toggles (prefer provider values; else infer from arch/model)
    self_attention = bool(raw_self_att) if raw_self_att is not None else (
        arch.endswith("_sa") or "sa" in model_name.lower()
    )
    sngp = bool(raw_sngp) if raw_sngp is not None else False
    openmax = bool(raw_openmax) if raw_openmax is not None else (arch == "cnn_openmax")
    use_unknown_head = bool(raw_use_unk) if raw_use_unk is not None else False

    # 5) model_name resolution (architecture + toggles -> concrete model name)
    if not model_name:
        model_name = _arch_to_model_name(arch, self_attention, openmax)

    # 6) numeric hyperparams (with safe clamps)
    bottleneck = _clamp(obj.get("bottleneck", 256), 16, 1024, 256)
    dropout = _clamp(obj.get("dropout", 0.3), 0.0, 0.8, 0.3)
    lr = _clamp(obj.get("learning_rate", 3e-4), 1e-5, 1e-2, 3e-4)
    bs = _clamp(obj.get("batch_size", 64), 4, 256, 64)
    lam_src = _clamp(obj.get("lambda_src", 1.0), 0.0, 5.0, 1.0)
    warmup_epochs = _clamp(obj.get("warmup_epochs", 3), 0, 20, 3)

    # 7) rationale
    prov_rat = str(obj.get("rationale", "")).strip()
    if not prov_rat or prov_rat.lower().startswith("fallback"):
        prov_rat = ""
        
    

    cfg = {
        "architecture": arch,
        "model_name": model_name,
        "self_attention": bool(self_attention),
        "sngp": bool(sngp),
        "openmax": bool(openmax),
        "use_unknown_head": bool(use_unknown_head),
        "bottleneck": bottleneck,
        "dropout": dropout,
        "learning_rate": lr,
        "batch_size": bs,
        "lambda_src": lam_src,
        "warmup_epochs": warmup_epochs,
    }
    
    cfg, heuristic_notes = _numeric_heuristic_adjust(cfg, num_summary, allow_history=allow_history)

    cfg["rationale"] = _autofill_rationale(
        cfg,
        num_summary or {},
        provider_reason=prov_rat,
        heuristic_notes=heuristic_notes,
    )

    return cfg


# ---------------------------- Provider adapters --------------------------------

def call_openai(text_context: str,
                num_summary: Dict[str, Any],
                model: str = "gpt-4o-mini",
                debug_dir: Optional[str] = None,
                allow_history: bool = True) -> Dict[str, Any]:
    """OpenAI backend using Chat Completions (v1 SDK)."""
    import os, json
    from openai import OpenAI

    # Build prompts
    sys = SYSTEM_PROMPT
    user = _build_user_prompt(text_context, num_summary)

    # Init client (picks up OPENAI_API_KEY / OPENAI_PROJECT / OPENAI_ORG_ID)
    client = OpenAI()

    # Try JSON mode; if the model/SDK doesn’t support response_format, retry without it
    content = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content or ""
    except TypeError:
        # No JSON mode → ask for JSON in the system prompt instead
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sys + "\nReturn strictly JSON."},
                      {"role": "user", "content": user}],
            temperature=0.2,
        )
        content = resp.choices[0].message.content or ""

    # Debug dumps
    if debug_dir:
        _safe_write(f"{debug_dir}/openai_request_user.txt", user)
        _safe_write(f"{debug_dir}/openai_raw.txt", content)

    # Robust JSON parse (exact → fallback extract)
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        try:
            s, e = content.find("{"), content.rfind("}")
            if s != -1 and e != -1 and e > s:
                obj = json.loads(content[s:e+1])
        except Exception:
            obj = None

    cfg = _validate_or_default(
        obj if obj is not None else content,
        num_summary=num_summary,
        allow_history=allow_history,
    )
    cfg["_provider"] = "openai"
    cfg["_raw"] = content
    return cfg




# Save text/JSON safely; ignore errors in debug mode
def _safe_write(path: str, data) -> None:
    try:
        import os, json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2)
            else:
                f.write(data if isinstance(data, str) else str(data))
    except Exception:
        # Debug writes should never crash the run
        pass


def call_ollama(text_context: str,
                num_summary: Dict[str, Any],
                model: str = "llama3.1:8b",
                debug_dir: Optional[str] = None,
                allow_history: bool = True) -> Dict[str, Any]:
    """
    Use local Ollama with JSON-only mode. Saves raw request/response if debug_dir is given.
    """
    import json, os, requests
    sys = SYSTEM_PROMPT
    user = _build_user_prompt(text_context, num_summary)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        "format": "json",     # enforce JSON output
        "stream": False,
        "options": {"temperature": 0.2}
    }

    try:
        host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        url = host.rstrip("/") + "/api/chat"
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        content = r.json()["message"]["content"]
    except Exception as e:
        content = ""
        if debug_dir:
            _safe_write(f"{debug_dir}/ollama_error.txt", f"{type(e).__name__}: {e}")

    if debug_dir:
        _safe_write(f"{debug_dir}/ollama_request_user.txt", user)
        _safe_write(f"{debug_dir}/ollama_request_payload.json", payload)
        _safe_write(f"{debug_dir}/ollama_raw.txt", content)

    # Parse JSON (provider should already be JSON because of format='json')
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        # crude extraction of first {...}
        try:
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                obj = json.loads(content[start:end+1])
        except Exception:
            obj = None

    parsed = _validate_or_default(
        json.dumps(obj) if obj is not None else content,
        num_summary=num_summary,
        allow_history=allow_history,
    )
    parsed["_provider"] = "ollama"
    parsed["_raw"] = content
    return parsed



# ------------------------------ Public API ------------------------------------

def select_config(text_context: str,
                  num_summary: Dict[str, Any],
                  backend: str = "auto",
                  model: Optional[str] = None,
                  debug_dir: Optional[str] = None,
                  allow_history: bool = True) -> Dict[str, Any]:
    if backend == "openai" or (backend == "auto" and os.getenv("OPENAI_API_KEY")):
        return call_openai(
            text_context,
            num_summary,
            model or "gpt-4.1-mini",
            debug_dir=debug_dir,
            allow_history=allow_history,
        )
    if backend == "ollama" or backend == "auto":
        return call_ollama(
            text_context,
            num_summary,
            model or "llama3.1:8b",
            debug_dir=debug_dir,
            allow_history=allow_history,
        )
    raise RuntimeError("No LLM backend available. Set OPENAI_API_KEY or run Ollama locally.")