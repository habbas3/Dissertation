from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
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

# --- Sequence generation ---
def build_sequences(df, feature_cols, label_col, seq_len=32):
    sequences, labels = [], []
    for _, group in df.groupby("filename"):
        group = group.sort_values("cycle_number")
        features = group[feature_cols].values
        labels_all = group[label_col].values
        if len(features) < seq_len:
            # Pad by repeating the last row so CNNs still receive a minimum length
            pad_len = seq_len - len(features)
            pad = np.repeat(features[-1:], pad_len, axis=0)
            seq = np.concatenate([features, pad], axis=0)
            label = labels_all[-1]
            sequences.append(seq)
            labels.append(label)
        else:
            for i in range(len(features) - seq_len + 1):  # +1 to get last window
                seq = features[i:i + seq_len]
                label = labels_all[i + seq_len - 1]
                sequences.append(seq)
                labels.append(label)
    return np.array(sequences), np.array(labels)


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
):
    
    df = pd.read_csv(csv_path)
    
    cycle_counts = df.groupby("filename")["cycle_number"].max()
    print("\U0001F501 Total cycles per cell:\n", cycle_counts)

    def _half_cycles(group: pd.DataFrame) -> pd.DataFrame:
        """Return only the first 40% of cycles for a single cell.

        Uses the ordering of `cycle_number` within each group to select the
        earliest portion of the available cycles, which is more robust when
        cycle numbering does not start at 0 or 1."""
        group = group.sort_values("cycle_number")
        n_cycles = len(group)
        cutoff_idx = int(np.ceil(n_cycles * 0.50))
        return group.iloc[:cutoff_idx]

    if classification_label not in df.columns:
        raise ValueError(f"Missing classification label: {classification_label}")
    if "cathode" not in df.columns:
        raise ValueError("'cathode' column missing in CSV.")

    # Encode labels
    df[classification_label + "_encoded"] = LabelEncoder().fit_transform(df[classification_label])

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
        
    # Use only the first 50% of cycles for each cell in both source and target
    source_df = source_df.groupby("filename", group_keys=False).apply(_half_cycles).reset_index(drop=True)
    if not target_df.empty:
        target_df = target_df.groupby("filename", group_keys=False).apply(_half_cycles).reset_index(drop=True)

    feature_cols = ['cycle_number', 'energy_charge', 'capacity_charge', 'energy_discharge',
                    'capacity_discharge', 'cycle_start', 'cycle_duration']

    scaler = StandardScaler()
    source_df[feature_cols] = scaler.fit_transform(source_df[feature_cols])
    if not target_df.empty:
        target_df[feature_cols] = scaler.transform(target_df[feature_cols])

    # --- Source split ---
    label_col = classification_label + "_encoded"
    file_labels = source_df.groupby("filename")[label_col].first()
    files = file_labels.index.to_numpy()
    labels = file_labels.values

    if len(np.unique(labels)) == 1 or np.min(np.bincount(labels)) < 2:
        print("⚠️ Not enough samples to stratify source files. Using random split.")
        if len(files) < 2:
            train_files, val_files = files, []
        else:
            train_files, val_files = train_test_split(
                files,
                test_size=0.2,
                random_state=42,
            )

    else:
        train_files, val_files = train_test_split(
            files,
            test_size=0.2,
            stratify=labels,
            random_state=42,
        )

    train_df = source_df[source_df["filename"].isin(train_files)].reset_index(drop=True)
    val_df = source_df[source_df["filename"].isin(val_files)].reset_index(drop=True)
    X_train, y_train = build_sequences(train_df, feature_cols, label_col, seq_len=sequence_length)
    X_val, y_val = build_sequences(val_df, feature_cols, label_col, seq_len=sequence_length)
    

    transform = Compose([Reshape()])  

    source_train = DataLoader(BatteryDataset(X_train, y_train, transform), batch_size=batch_size, shuffle=True)
    source_val   = DataLoader(BatteryDataset(X_val, y_val, transform), batch_size=batch_size, shuffle=False)

    # --- Target split ---
    if not target_df.empty:
        label_col = classification_label + "_encoded"
        tgt_file_labels = target_df.groupby("filename")[label_col].first()
        tgt_files = tgt_file_labels.index.to_numpy()
        tgt_labels = tgt_file_labels.values

        if len(np.unique(tgt_labels)) == 1 or np.min(np.bincount(tgt_labels)) < 2:
            print(f"⚠️ Not enough target samples to stratify. Using random split.")
            if len(tgt_files) < 2:
                tgt_train_files, tgt_val_files = tgt_files, []
            else:
                tgt_train_files, tgt_val_files = train_test_split(
                    tgt_files,
                    test_size=0.2,
                    random_state=42,
                )
        else:
            tgt_train_files, tgt_val_files = train_test_split(
                tgt_files,
                test_size=0.2,
                stratify=tgt_labels,
                random_state=42,
            )

        tgt_train_df = target_df[target_df["filename"].isin(tgt_train_files)].reset_index(drop=True)
        tgt_val_df = target_df[target_df["filename"].isin(tgt_val_files)].reset_index(drop=True)

        X_tgt_train, y_tgt_train = build_sequences(tgt_train_df, feature_cols, label_col, seq_len=sequence_length)
        X_tgt_val, y_tgt_val = build_sequences(tgt_val_df, feature_cols, label_col, seq_len=sequence_length)
        
        
        

        # When very few target samples are available, augment and oversample
        if len(y_tgt_train) > 0 and len(y_tgt_train) < 100:
            aug_transform = Compose([
                RandomAddGaussian(sigma=0.05),
                RandomScale(sigma=0.1),
                RandomStretch(sigma=0.3),
                Reshape(),
            ])
            class_counts = np.bincount(y_tgt_train)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_tgt_train]
            sample_weights = torch.from_numpy(sample_weights).float()
            sampler = WeightedRandomSampler(sample_weights,
                                            num_samples=len(sample_weights) * 10,
                                            replacement=True)
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



