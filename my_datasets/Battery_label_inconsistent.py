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
from typing import Optional



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
    cycles_per_file=None,
    sample_random_state=42,
):
    
    df = pd.read_csv(csv_path)
    
    cycle_counts = df.groupby("filename")["cycle_number"].max()
    print("\U0001F501 Total cycles per cell:\n", cycle_counts)

    rng = np.random.default_rng(sample_random_state)

    def _sample_cycles(group: pd.DataFrame) -> pd.DataFrame:
        """Optionally select a contiguous block of cycles at random.

        Parameters
        ----------
        group : DataFrame
            All cycles for a single cell.

        Returns
        -------
        DataFrame
            Either the full group (if ``cycles_per_file`` is ``None``) or a
            random contiguous slice of length ``cycles_per_file``.
        """

        if cycles_per_file is None or cycles_per_file <= 0 or len(group) <= cycles_per_file:
            return group
        start_max = len(group) - cycles_per_file
        start = int(rng.integers(0, start_max + 1))
        return group.iloc[start : start + cycles_per_file]

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
        
    # Optionally subsample a fixed number of cycles from each cell
    source_df = source_df.groupby("filename", group_keys=False).apply(_sample_cycles).reset_index(drop=True)
    if not target_df.empty:
        target_df = target_df.groupby("filename", group_keys=False).apply(_sample_cycles).reset_index(drop=True)

    feature_cols = ['cycle_number', 'energy_charge', 'capacity_charge', 'energy_discharge',
                    'capacity_discharge', 'cycle_start', 'cycle_duration']

    scaler = StandardScaler()
    scaler.fit(df[feature_cols])
    source_df[feature_cols] = scaler.transform(source_df[feature_cols])
    if not target_df.empty:
        target_df[feature_cols] = scaler.transform(target_df[feature_cols])

    # --- Source split ---
    label_col = classification_label + "_encoded"
    
    src_group_col = "cell_id" if "cell_id" in source_df.columns else "filename"
    X_src, y_src, g_src = build_sequences(
        source_df, feature_cols, label_col, seq_len=sequence_length, group_col=src_group_col
    )
    
    src_tr_idx, src_va_idx = _safe_stratified_split(y_src, groups=g_src, test_size=0.2, seed=42, min_val=30)
    X_train, y_train = X_src[src_tr_idx], y_src[src_tr_idx]
    X_val, y_val = X_src[src_va_idx], y_src[src_va_idx]
    

    transform = Compose([Reshape()])

    source_train = DataLoader(BatteryDataset(X_train, y_train, transform), batch_size=batch_size, shuffle=True)
    source_val = DataLoader(BatteryDataset(X_val, y_val, transform), batch_size=batch_size, shuffle=False)

    # --- Target split ---
    if not target_df.empty:
        tgt_group_col = "cell_id" if "cell_id" in target_df.columns else "filename"
        X_tgt, y_tgt, g_tgt = build_sequences(
            target_df, feature_cols, label_col, seq_len=sequence_length, group_col=tgt_group_col)
        tgt_tr_idx, tgt_va_idx = _safe_stratified_split(y_tgt, groups=g_tgt, test_size=0.3, seed=42, min_val=20)
        X_tgt_train, y_tgt_train, g_tgt_train = (
            X_tgt[tgt_tr_idx], y_tgt[tgt_tr_idx], g_tgt[tgt_tr_idx]
        )
        X_tgt_val, y_tgt_val = X_tgt[tgt_va_idx], y_tgt[tgt_va_idx]

        # When very few target samples are available, augment and oversample
        if len(y_tgt_train) > 0 and len(y_tgt_train) < 100:
            aug_transform = Compose([
                RandomAddGaussian(sigma=0.05),
                RandomScale(sigma=0.1),
                RandomStretch(sigma=0.3),
                RandomTimeShift(shift_ratio=0.2),
                Reshape(),
            ])
            class_counts = np.bincount(y_tgt_train)
            class_weights = 1.0 / class_counts
            cathode_counts = Counter(g_tgt_train)
            cathode_weights = {c: 1.0 / cnt for c, cnt in cathode_counts.items()}
            sample_weights = class_weights[y_tgt_train] * np.array([cathode_weights[g] for g in g_tgt_train])
            sample_weights = torch.from_numpy(sample_weights).float()
            sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=len(sample_weights) * 10,
                replacement=True,
            )
            target_train_dataset = BatteryDataset(X_tgt_train, y_tgt_train, aug_transform)
            target_train = DataLoader(target_train_dataset, batch_size=batch_size, sampler=sampler)
        else:
            target_train_dataset = BatteryDataset(X_tgt_train, y_tgt_train, transform)
            target_train = DataLoader(target_train_dataset, batch_size=batch_size, shuffle=True)
        
        target_val = DataLoader(BatteryDataset(X_tgt_val, y_tgt_val, transform), batch_size=batch_size, shuffle=False)
    else:
        target_train, target_val = None, None

    label_names = sorted(df[classification_label].dropna().unique().tolist())

    print("📊 Source val class counts:", Counter(y_val))
    if not target_df.empty:
        print("📊 Target val class counts:", Counter(y_tgt_val))

    return source_train, source_val, target_train, target_val, label_names, df



