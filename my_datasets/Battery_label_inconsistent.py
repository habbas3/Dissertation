from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, StratifiedGroupKFold, ShuffleSplit
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import torch
from .sequence_aug import RandomAddGaussian, RandomScale, RandomStretch, RandomTimeShift

# --- Data transforms ---
class Reshape:
    def __call__(self, x):
        return x.T  # (seq_len, features) -> (features, seq_len)

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


# --- Splitting utilities ---
from typing import Optional, Tuple, Sequence



def _safe_stratified_split(y, groups=None, test_size=0.2, seed=42, min_val=20):
    """
    Robust split for tiny/imbalanced Battery slices:
      - Prefer stratified (group-aware if groups provided)
      - If any class has < 2 samples, put those singletons in TRAIN and stratify the rest
      - If stratification is still impossible, fall back to group/random split
      - Ensure a minimum validation size via test_size adjustment
    """
    import numpy as np

    y = np.asarray(y)
    n = len(y)
    eff_test = min(0.5, max(test_size, min_val / max(1, n)))  # keep val reasonably sized

    classes, counts = np.unique(y, return_counts=True)

    # Case 1: only one class present → no stratification possible
    if len(classes) <= 1:
        if groups is not None:
            gss = GroupShuffleSplit(n_splits=1, test_size=eff_test, random_state=seed)
            tr, va = next(gss.split(np.zeros(n), groups=groups))
        else:
            ss = ShuffleSplit(n_splits=1, test_size=eff_test, random_state=seed)
            tr, va = next(ss.split(np.zeros(n)))
        print("⚠️  Split fallback: single-class set → using GroupShuffle/Shuffle split.")
        return tr, va

    # Case 2: some classes are singletons → send them to TRAIN, stratify the rest
    singletons = set(classes[counts < 2].tolist())
    if singletons:
        mask_single = np.isin(y, list(singletons))
        idx_single = np.where(mask_single)[0]          # force to TRAIN
        idx_rest   = np.where(~mask_single)[0]         # eligible for stratified split
        y_rest     = y[idx_rest]
        groups_rest = groups[idx_rest] if groups is not None else None

        rest_classes, rest_counts = np.unique(y_rest, return_counts=True)
        can_stratify_rest = (len(rest_classes) >= 2) and np.all(rest_counts >= 2)

        if can_stratify_rest:
            if groups_rest is not None:
                # Use SGKF and pick a fold closest to eff_test proportion
                n_splits = max(2, int(round(1.0 / eff_test)))
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                best = None
                target = eff_test * len(idx_rest)
                for tr_r, va_r in sgkf.split(np.zeros_like(y_rest), y_rest, groups_rest):
                    if best is None or abs(len(va_r) - target) < abs(len(best[1]) - target):
                        best = (tr_r, va_r)
                tr_r, va_r = best
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=eff_test, random_state=seed)
                tr_r, va_r = next(sss.split(np.zeros_like(y_rest), y_rest))
            tr_idx = idx_rest[tr_r]
            va_idx = idx_rest[va_r]
            # add singleton samples to TRAIN
            tr_idx = np.concatenate([tr_idx, idx_single])
            print(f"⚠️  Split fallback: moved {len(idx_single)} singleton samples to TRAIN; stratified the rest.")
            return tr_idx, va_idx
        else:
            # Can't stratify the rest either → group/random split on rest, then add singletons to TRAIN
            if groups_rest is not None:
                gss = GroupShuffleSplit(n_splits=1, test_size=eff_test, random_state=seed)
                tr_r, va_r = next(gss.split(np.zeros(len(idx_rest)), groups=groups_rest))
            else:
                ss = ShuffleSplit(n_splits=1, test_size=eff_test, random_state=seed)
                tr_r, va_r = next(ss.split(np.zeros(len(idx_rest))))
            tr_idx = np.concatenate([idx_rest[tr_r], idx_single])
            va_idx = idx_rest[va_r]
            print(f"⚠️  Split fallback: no stratify possible; using GroupShuffle/Shuffle on rest. "
                  f"Singletons ({len(idx_single)}) kept in TRAIN.")
            return tr_idx, va_idx

    # Case 3: clean stratified split (preferred)
    if groups is not None:
        n_splits = max(2, int(round(1.0 / eff_test)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        best = None
        target = eff_test * n
        for tr, va in sgkf.split(np.zeros_like(y), y, groups):
            if best is None or abs(len(va) - target) < abs(len(best[1]) - target):
                best = (tr, va)
        return best
    else:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=eff_test, random_state=seed)
        return next(sss.split(np.zeros_like(y), y))

    
# --- Sequence generation ---
def build_sequences(df, feature_cols, label_col, seq_len=32, group_col=None):
    sequences, labels, groups = [], [], []
    for _, group in df.groupby("filename"):
        group = group.sort_values("cycle_number")
        features = group[feature_cols].values
        labels_all = group[label_col].values
        group_label = group[group_col].iloc[0] if group_col else None
        if len(features) < seq_len:
            # Pad by repeating the last row so CNNs still receive a minimum length
            pad_len = seq_len - len(features)
            pad = np.repeat(features[-1:], pad_len, axis=0)
            seq = np.concatenate([features, pad], axis=0)
            label = labels_all[-1]
            sequences.append(seq)
            labels.append(label)
            if group_col:
                groups.append(group_label)
        else:
            for i in range(len(features) - seq_len + 1):  # +1 to get last window
                seq = features[i:i + seq_len]
                label = labels_all[i + seq_len - 1]
                sequences.append(seq)
                labels.append(label)
                if group_col:
                    groups.append(group_label)
    sequences = np.array(sequences)
    labels = np.array(labels)
    if group_col:
        return sequences, labels, np.array(groups)
    return sequences, labels


class BatteryDataset(Dataset):
    def __init__(self, sequences, labels, transform=None):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx]  # shape: (seq_len, n_features) or (n_features,)
        y = self.labels[idx]
        
        if self.transform is not None:
            x = self.transform(x)
        else:
            if x.ndim == 1:
                x = np.expand_dims(x, axis=1)
            if x.ndim == 2:
                x = x.T
        
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()
        return x, y


# --- Main loader ---
def load_battery_dataset(
    csv_path,
    source_cathodes,
    target_cathodes=None,
    classification_label="eol_class",
    batch_size=64,
    sequence_length=32,
    num_classes=None,
    cycles_per_file=50,
    # cycles_per_file=None,
    sample_random_state=42,
):
    
    df = pd.read_csv(csv_path)
    
    # Normalise string columns once at load time so downstream filtering by
    # cathode/cell names is reliable regardless of stray whitespace in the raw
    # CSV.  This mirrors the trimming performed when the cycle-level file is
    # generated and fixes cases where groups such as "NMC532" were previously
    # missed because the value was stored as " NMC532".
    string_cols = [
        "filename",
        "battery_name",
        "batch",
        "cell_id",
        "anode",
        "cathode",
        "electrolyte",
        "dataset_name",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
            
    cycle_counts = df.groupby("filename")["cycle_number"].max()
    print("\U0001F501 Total cycles per cell:\n", cycle_counts)
    
    per_cathode_cycles = (
        df.groupby(["cathode", "filename"])["cycle_number"].nunique().rename("cycle_count").sort_values()
    )
    if len(per_cathode_cycles) <= 40:
        print("🔁 Cycle counts by cathode/filename:\n", per_cathode_cycles)
    else:
        print("🔁 Cycle counts by cathode/filename (first 10):\n", per_cathode_cycles.head(10))
        print("⋮")
        print("🔁 Cycle counts by cathode/filename (last 10):\n", per_cathode_cycles.tail(10))

    def _limit_cycles_per_cell(df_subset: pd.DataFrame, limit: Optional[int]) -> pd.DataFrame:
        if df_subset is None or df_subset.empty or limit is None or limit <= 0:
            return df_subset.copy()

        parts = []
        truncated = []
        for name, group in df_subset.groupby("filename"):
            ordered = group.sort_values("cycle_number")
            parts.append(ordered.head(limit))
            if len(ordered) > limit:
                truncated.append(name)

        if truncated:
            print(
                f"✂️  Truncated {len(truncated)} cells to the first {limit} cycles "
                "so baseline and transfer compare the same early-life horizon."
            )
            
        return pd.concat(parts, ignore_index=True) if parts else df_subset.iloc[0:0].copy()


    def _train_val_split_by_group(
        df_subset: pd.DataFrame,
        group_col: str,
        label_col: str,
        val_fraction: float,
        seed: int,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Sequence, Sequence]:
        if df_subset is None or df_subset.empty:
            return df_subset.iloc[0:0].copy(), df_subset.iloc[0:0].copy(), [], []

        base = (
            df_subset[[group_col, label_col]]
            .drop_duplicates(subset=[group_col])
            .set_index(group_col)[label_col]
        )
        group_names = base.index.to_numpy()
        y_groups = base.to_numpy()

        if len(group_names) <= 1:
            return df_subset.copy(), df_subset.iloc[0:0].copy(), group_names, []

        val_fraction = min(0.5, max(val_fraction, 1.0 / max(2, len(group_names))))
        min_val = max(1, int(round(val_fraction * len(group_names))))
        
        tr_idx, va_idx = _safe_stratified_split(
            y_groups,
            groups=group_names,
            test_size=val_fraction,
            seed=seed,
            min_val=min_val,
        )

        train_groups = group_names[tr_idx]
        val_groups = group_names[va_idx]

        if len(val_groups) == 0 and len(train_groups) > 1:
            val_groups = train_groups[:1]
            train_groups = train_groups[1:]

        train_df = df_subset[df_subset[group_col].isin(train_groups)].copy()
        val_df = df_subset[df_subset[group_col].isin(val_groups)].copy()
        return train_df, val_df, train_groups, val_groups

    def _count_cycles(df_subset: pd.DataFrame) -> int:
        if df_subset is None or df_subset.empty:
            return 0
        if "cycle_number" in df_subset.columns:
            return int(pd.to_numeric(df_subset["cycle_number"], errors="coerce").notna().sum())
        return int(len(df_subset))
    
    def _describe_split(name: str, df_subset: pd.DataFrame, groups: Sequence) -> None:
        if df_subset is None or df_subset.empty:
            print(f'⚠️ {name}: 0 cells available.')
            return

        group_count = len(set(groups)) if groups else 0
        per_cell_counts = df_subset.groupby('filename')['cycle_number'].nunique()
        stats = per_cell_counts.describe()
        median = stats.get('50%', float('nan'))
        print(
            f'🧮 {name}: {group_count} cells, {df_subset.shape[0]} cycles, '
            f'median per-cell horizon {median:.1f} (min {stats.get("min", float("nan")):.0f}, '
            f'max {stats.get("max", float("nan")):.0f})'
        )
    
    if classification_label not in df.columns:
        raise ValueError(f"Missing classification label: {classification_label}")
    if "cathode" not in df.columns:
        raise ValueError("'cathode' column missing in CSV.")

    # Optionally rebin the target column into a different number of classes.
    # This is useful when the raw label is numeric and a user wants to
    # evaluate a coarser or finer-grained classification task without
    # regenerating the CSV file.
    if num_classes is not None:
        if pd.api.types.is_numeric_dtype(df[classification_label]):
            try:
                df[classification_label] = pd.qcut(
                    df[classification_label],
                    q=num_classes,
                    labels=False,
                    duplicates="drop",
                )
    
    
    
            except Exception as exc:  # pragma: no cover - defensive, feature optional
                raise ValueError(
                    f"Failed to bin '{classification_label}' into {num_classes} classes"
                ) from exc
        else:
            print(
                f"⚠️ classification label '{classification_label}' is non-numeric; "
                "skipping quantile binning and using existing categories."
            )

    # Encode labels
    df[classification_label + "_encoded"] = LabelEncoder().fit_transform(
        df[classification_label]
    )

    print("🔢 Class distribution:\n", df[classification_label + "_encoded"].value_counts())
    print("🔬 Cathode distribution:\n", df["cathode"].value_counts())
    df["cathode"] = df["cathode"].astype(str).str.strip()

    # ✅ Handle both pretraining and transfer modes
    source_df = df[df["cathode"].isin(source_cathodes)].reset_index(drop=True)
    if source_df.empty:
        raise ValueError(f"❌ No rows found for source cathodes: {source_cathodes}")
    if target_cathodes is None or len(target_cathodes) == 0:
        print("🛠 Loading pretraining mode (source only, no target)")
        target_df = pd.DataFrame(columns=df.columns)
    else:
        target_df = df[df["cathode"].isin(target_cathodes)].reset_index(drop=True)
        if target_df.empty:
            print(f"⚠️ No rows found for target cathodes: {target_cathodes}")

    source_limited = _limit_cycles_per_cell(source_df, cycles_per_file)
    target_limited = _limit_cycles_per_cell(target_df, cycles_per_file) if not target_df.empty else target_df.copy()

    src_group_col = "cell_id" if "cell_id" in source_limited.columns else "filename"
    tgt_group_col = "cell_id" if (not target_limited.empty and "cell_id" in target_limited.columns) else "filename"

    source_train_cycles, source_eval_cycles, src_train_groups, src_val_groups = _train_val_split_by_group(
        source_limited,
        group_col=src_group_col,
        label_col=classification_label + "_encoded",
        val_fraction=0.25,
        seed=sample_random_state,
    )

    if not target_limited.empty:
        target_train_cycles, target_eval_cycles, tgt_train_groups, tgt_val_groups = _train_val_split_by_group(
            target_limited,
            group_col=tgt_group_col,
            label_col=classification_label + "_encoded",
            val_fraction=0.3,
            seed=sample_random_state,
        )
    else:
        target_train_cycles = target_limited.copy()
        target_eval_cycles = target_limited.copy()
        tgt_train_groups = []
        tgt_val_groups = []

    src_train_cycle_count = _count_cycles(source_train_cycles)
    src_eval_cycle_count = _count_cycles(source_eval_cycles)
    tgt_train_cycle_count = _count_cycles(target_train_cycles)
    tgt_eval_cycle_count = _count_cycles(target_eval_cycles)
    
    _describe_split('Source train', source_train_cycles, src_train_groups)
    _describe_split('Source val', source_eval_cycles, src_val_groups)
    if not target_limited.empty:
        _describe_split('Target train', target_train_cycles, tgt_train_groups)
        _describe_split('Target val', target_eval_cycles, tgt_val_groups)

    feature_cols = ['cycle_number', 'energy_charge', 'capacity_charge', 'energy_discharge',
                    'capacity_discharge', 'cycle_start', 'cycle_duration']

    scaler = StandardScaler()
    scaler_fit_parts = [frame for frame in [source_train_cycles, source_eval_cycles, target_train_cycles, target_eval_cycles] if frame is not None and not frame.empty]
    if scaler_fit_parts:
        scaler.fit(pd.concat(scaler_fit_parts, ignore_index=True)[feature_cols])
    else:
        scaler.fit(df[feature_cols])

    def _transform(df_subset: pd.DataFrame) -> pd.DataFrame:
        if df_subset is None or df_subset.empty:
            return df_subset
    
        df_subset = df_subset.copy()
        df_subset[feature_cols] = scaler.transform(df_subset[feature_cols])
        return df_subset

    source_train_cycles = _transform(source_train_cycles)
    source_eval_cycles = _transform(source_eval_cycles)
    target_train_cycles = _transform(target_train_cycles)
    target_eval_cycles = _transform(target_eval_cycles)

    label_col = classification_label + "_encoded"

    transform = Compose([Reshape()])

    def _build_arrays(df_subset: pd.DataFrame, group_col: str):
        if df_subset is None or df_subset.empty:
            return np.zeros((0, sequence_length, len(feature_cols))), np.zeros((0,), dtype=int), np.zeros((0,), dtype=object)
        return build_sequences(df_subset, feature_cols, label_col, seq_len=sequence_length, group_col=group_col)

    # --- Source split ---
    X_src_all, y_src_all, g_src_all = _build_arrays(source_train_cycles, src_group_col)
    X_src_val_all, y_src_val_all, g_src_val_all = _build_arrays(source_eval_cycles, src_group_col)

    if X_src_val_all.size == 0 and y_src_all.size > 0:
        tr_idx, va_idx = _safe_stratified_split(y_src_all, groups=g_src_all, test_size=0.2, seed=42, min_val=30)
        X_train = X_src_all[tr_idx]
        y_train = y_src_all[tr_idx]
        g_src_all = g_src_all[tr_idx]
        X_val = X_src_all[va_idx]
        y_val = y_src_all[va_idx]
    else:
        X_train, y_train = X_src_all, y_src_all
        X_val, y_val = X_src_val_all, y_src_val_all

    source_train_dataset = BatteryDataset(X_train, y_train, transform)
    source_val_dataset = BatteryDataset(X_val, y_val, transform)
    source_train = DataLoader(source_train_dataset, batch_size=batch_size, shuffle=True)
    source_val = DataLoader(source_val_dataset, batch_size=batch_size, shuffle=False)

    # --- Target split ---
    target_train_dataset = None
    target_val_dataset = None
    X_tgt_val = np.zeros((0, sequence_length, len(feature_cols)))
    y_tgt_val = np.zeros((0,), dtype=int)

    if not target_limited.empty:
        X_tgt_all, y_tgt_all, g_tgt_all = _build_arrays(target_train_cycles, tgt_group_col)
        X_tgt_val_all, y_tgt_val_all, g_tgt_val_all = _build_arrays(target_eval_cycles, tgt_group_col)

        if X_tgt_val_all.size == 0 and y_tgt_all.size > 0:
            tgt_tr_idx, tgt_va_idx = _safe_stratified_split(y_tgt_all, groups=g_tgt_all, test_size=0.3, seed=42, min_val=20)
            X_tgt_train = X_tgt_all[tgt_tr_idx]
            y_tgt_train = y_tgt_all[tgt_tr_idx]
            g_tgt_all = g_tgt_all[tgt_tr_idx]
            X_tgt_val = X_tgt_all[tgt_va_idx]
            y_tgt_val = y_tgt_all[tgt_va_idx]
        else:
            X_tgt_train, y_tgt_train = X_tgt_all, y_tgt_all
            X_tgt_val, y_tgt_val = X_tgt_val_all, y_tgt_val_all

        if len(y_tgt_train) > 0 and len(y_tgt_train) < 100:
            aug_transform = Compose([
                RandomAddGaussian(sigma=0.05),
                RandomScale(sigma=0.1),
                RandomStretch(sigma=0.3),
                RandomTimeShift(shift_ratio=0.2),
                Reshape(),
            ])
            class_counts = np.bincount(y_tgt_train)
            class_weights = 1.0 / np.clip(class_counts, 1, None)
            cathode_counts = Counter(g_tgt_all)
            cathode_weights = {c: 1.0 / cnt for c, cnt in cathode_counts.items()}
            sample_weights = class_weights[y_tgt_train] * np.array([cathode_weights.get(g, 1.0) for g in g_tgt_all])
            sample_weights = torch.from_numpy(sample_weights).float()
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=max(len(sample_weights) * 10, len(sample_weights)),
                replacement=True,
            )
            target_train_dataset = BatteryDataset(X_tgt_train, y_tgt_train, aug_transform)
            target_train = DataLoader(target_train_dataset, batch_size=batch_size, sampler=sampler)
    
        else:
            target_train_dataset = BatteryDataset(X_tgt_train, y_tgt_train, transform)
            target_train = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)

        target_val_dataset = BatteryDataset(X_tgt_val, y_tgt_val, transform)
        target_val = DataLoader(target_val_dataset, batch_size=batch_size, shuffle=False)

    else:
        target_train = None
        target_val = None

    label_names = sorted(df[classification_label].dropna().unique().tolist())

    src_val_cycle_effective = src_eval_cycle_count if src_eval_cycle_count > 0 else (src_train_cycle_count if len(y_val) > 0 else 0)
    tgt_val_cycle_effective = tgt_eval_cycle_count if tgt_eval_cycle_count > 0 else (tgt_train_cycle_count if len(y_tgt_val) > 0 else 0)

    stats = {
        "source_train_cycles": src_train_cycle_count,
        "source_val_cycles": src_val_cycle_effective,
        "source_val_has_holdout": bool(src_eval_cycle_count > 0),
        "target_train_cycles": tgt_train_cycle_count,
        "target_val_cycles": tgt_val_cycle_effective,
        "target_val_has_holdout": bool(tgt_eval_cycle_count > 0),
        "source_train_sequences": len(source_train_dataset),
        "source_val_sequences": len(source_val_dataset),
        "target_train_sequences": len(target_train_dataset) if target_train_dataset is not None else 0,
        "target_val_sequences": len(target_val_dataset) if target_val_dataset is not None else 0,
        "source_train_cells": len(set(src_train_groups)) if src_train_groups else 0,
        "source_val_cells": len(set(src_val_groups)) if src_val_groups else 0,
        "target_train_cells": len(set(tgt_train_groups)) if tgt_train_groups else 0,
        "target_val_cells": len(set(tgt_val_groups)) if tgt_val_groups else 0,
    }

    for loader, seq_key, cyc_key, cell_key in [
        (source_train, "source_train_sequences", "source_train_cycles", "source_train_cells"),
        (source_val, "source_val_sequences", "source_val_cycles", "source_val_cells"),
        (target_train, "target_train_sequences", "target_train_cycles", "target_train_cells"),
        (target_val, "target_val_sequences", "target_val_cycles", "target_val_cells"),
    ]:
        if loader is not None:
            setattr(loader, "sequence_count", stats[seq_key])
            setattr(loader, "cycle_count", stats[cyc_key])
            setattr(loader, "cell_count", stats.get(cell_key, 0))

    print("📊 Source val class counts:", Counter(y_val))
    if not target_limited.empty:
        print("📊 Target val class counts:", Counter(y_tgt_val))

    return source_train, source_val, target_train, target_val, label_names, df, stats
    
    
    
    
