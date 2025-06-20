from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch

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
        for i in range(len(features) - seq_len + 1):   # fix: +1 to get last window
            seq = features[i:i+seq_len]                # shape [seq_len, features]
            label = labels_all[i+seq_len-1]            # fix: last label in the window
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

    if classification_label not in df.columns:
        raise ValueError(f"❌ Missing classification label: {classification_label}")
    if "cathode" not in df.columns:
        raise ValueError("❌ 'cathode' column missing in CSV.")

    # Encode labels
    df[classification_label + "_encoded"] = LabelEncoder().fit_transform(df[classification_label])

    print("🔢 Class distribution:\n", df[classification_label + "_encoded"].value_counts())
    print("🔬 Cathode distribution:\n", df["cathode"].value_counts())

    # ✅ Handle both pretraining and transfer modes
    source_df = df[df["cathode"].isin(source_cathodes)].reset_index(drop=True)
    if target_cathodes is None or len(target_cathodes) == 0:
        print("🛠 Loading pretraining mode (source only, no target)")
        target_df = pd.DataFrame(columns=df.columns)
    else:
        target_df = df[df["cathode"].isin(target_cathodes)].reset_index(drop=True)

    feature_cols = ['cycle_number', 'energy_charge', 'capacity_charge', 'energy_discharge',
                    'capacity_discharge', 'cycle_start', 'cycle_duration']

    scaler = StandardScaler()
    source_df[feature_cols] = scaler.fit_transform(source_df[feature_cols])
    if not target_df.empty:
        target_df[feature_cols] = scaler.transform(target_df[feature_cols])

    # --- Source split ---
    label_col = classification_label + "_encoded"
    if len(np.unique(source_df[label_col])) == 1 or np.min(np.bincount(source_df[label_col])) < 2:
        print("⚠️ Not enough samples to stratify source data. Using full data for training.")
        train_df = source_df
        val_df = source_df.iloc[:0]

    else:
        train_df, val_df = train_test_split(
            source_df,
            test_size=0.2,
            stratify=source_df[label_col],
            random_state=42,
        )

    X_train, y_train = build_sequences(train_df, feature_cols, label_col, seq_len=sequence_length)
    X_val, y_val = build_sequences(val_df, feature_cols, label_col, seq_len=sequence_length)

    transform = Compose([Reshape()])  

    source_train = DataLoader(BatteryDataset(X_train, y_train, transform), batch_size=batch_size, shuffle=True)
    source_val   = DataLoader(BatteryDataset(X_val, y_val, transform), batch_size=batch_size, shuffle=False)

    # --- Target split ---
    if not target_df.empty:
        label_col = classification_label + "_encoded"
        if len(np.unique(target_df[label_col])) == 1 or np.min(np.bincount(target_df[label_col])) < 2:
            print(f"⚠️ Not enough target samples to stratify. Using full data for training.")
            tgt_train_df = target_df
            tgt_val_df = target_df.iloc[:0]
        else:
            tgt_train_df, tgt_val_df = train_test_split(
                target_df,
                test_size=0.2,
                stratify=target_df[label_col],
                random_state=42,
            )

        X_tgt_train, y_tgt_train = build_sequences(tgt_train_df, feature_cols, label_col, seq_len=sequence_length)
        X_tgt_val, y_tgt_val = build_sequences(tgt_val_df, feature_cols, label_col, seq_len=sequence_length)

        target_train = DataLoader(BatteryDataset(X_tgt_train, y_tgt_train, transform), batch_size=batch_size, shuffle=True)
        target_val   = DataLoader(BatteryDataset(X_tgt_val, y_tgt_val, transform), batch_size=batch_size, shuffle=False)
    else:
        target_train, target_val = None, None

    label_names = sorted(df[classification_label].dropna().unique().tolist())

    print("📊 Source val class counts:", Counter(y_val))
    if not target_df.empty:
        print("📊 Target val class counts:", Counter(y_tgt_val))

    return source_train, source_val, target_train, target_val, label_names, df



