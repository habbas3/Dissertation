import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from my_datasets.SequenceDatasets import dataset
from my_datasets.sequence_aug import *
from tqdm import tqdm
signal_size = 1024
datasetname = ["12k Drive End Bearing Fault Data", "12k Fan End Bearing Fault Data", "48k Drive End Bearing Fault Data",
                   "Normal Baseline Data"]
axis = ["_DE_time", "_FE_time", "_BA_time"]
dataname= {0:["97.mat","105.mat", "118.mat", "130.mat", "169.mat", "185.mat", "197.mat", "209.mat", "222.mat","234.mat"],  # 1797rpm
           1:["98.mat","106.mat", "119.mat", "131.mat", "170.mat", "186.mat", "198.mat", "210.mat", "223.mat","235.mat"],  # 1772rpm
           2:["99.mat","107.mat", "120.mat", "132.mat", "171.mat", "187.mat", "199.mat", "211.mat", "224.mat","236.mat"],  # 1750rpm
           3:["100.mat","108.mat", "121.mat","133.mat", "172.mat", "188.mat", "200.mat", "212.mat", "225.mat","237.mat"]}  # 1730rpm
label = [i for i in range(0, 10)]

# Human-friendly class codes for reporting/plotting.
#
# The CWRU naming convention in this loader is:
#   0 = Normal operating condition
#   1–9 = Fault classes ordered by increasing damage size
#   9 is also reused as a mask/outlier label in the label-inconsistent
#       tasks (UAN/OSBP), so plots should avoid implying chronological order.
#
# The alphabetical prefixes (A–J) mirror the CWRU tables while explicitly
# separating the normal class (0) from the masked/outlier bucket (9).
CWRU_FAULT_CODES = {
    0: "A (Normal)",
    1: "B (IF, 7 mil)",
    2: "C (BF, 7 mil)",
    3: "D (OF, 7 mil)",
    4: "E (IF, 14 mil)",
    5: "F (BF, 14 mil)",
    6: "G (OF, 14 mil)",
    7: "H (IF, 21 mil)",
    8: "I (BF, 21 mil)",
    9: "J (OF, 21 mil / Masked-outlier placeholder)",
}


def get_fault_code(label_id: int) -> str:
    """Return the alphabetical fault code used for plots/reports."""

    return CWRU_FAULT_CODES.get(label_id, f"Class {label_id}")


def dataset_information(source_N, target_N, label_inconsistent):
    default_names = list(range(len(dataname[0])))
    default_labels = list(range(len(dataname[0])))

    name_source = default_names
    name_target = default_names
    label_source = default_labels
    label_target = default_labels
    if label_inconsistent == 'PADA':
        label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        name_source = label_source
        name_target = label_source
        if (source_N == [0] and target_N == [1]) or (source_N == [0] and target_N == [2]) or (source_N == [0] and target_N == [3]):
            label_target = [0, 1, 2, 4, 5, 7, 8, 9]
            name_target = label_target
    elif label_inconsistent == 'OSBP':
        name_target = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if (source_N == [0] and target_N == [1]) or (source_N == [0] and target_N == [2]) or (source_N == [0] and target_N == [3]):
            name_source = [0, 2, 3, 5, 6, 7, 8, 9]
            label_source = [0, 1, 2, 3, 4, 5, 6, 7]
            label_target = [0, 8, 1, 2, 8, 3, 4, 5, 6, 7]
    elif label_inconsistent == 'UAN':
                
        
        if (source_N == [0] and target_N == [1]) or (source_N == [0] and target_N == [2]):
            name_source = [0, 1, 2, 4, 5, 6, 7, 8, 9]
            label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            name_target = [1, 2, 3, 4, 5, 7, 8, 9]
            label_target = [1, 2, 9, 3, 4, 6, 7, 8]
            
        elif (source_N == [0] and target_N == [3]) or (source_N == [1] and target_N == [0]):
            name_source = [0, 1, 2, 3, 4, 5, 7, 8]
            label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            name_target = [0, 2, 3, 4, 5, 6, 7, 8, 9]
            label_target = [0, 2, 3, 4, 5, 9, 6, 7, 9]
        elif (source_N == [1] and target_N == [2]) or (source_N == [1] and target_N == [3]):
            
            name_source = [1, 2, 4, 5, 7, 8, 9]
            label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            name_target = [1, 3, 6, 8]
            label_target = [0, 9, 9, 5]
            
        elif (source_N == [2] and target_N == [0]) or (source_N == [2] and target_N == [1]):
            name_source = [1, 3, 4, 6, 7, 8]
            label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            name_target = [0, 1, 2, 3, 5, 6, 8, 9]
            label_target = [9, 0, 9, 1, 9, 3, 5, 9]
            
        elif (source_N == [2] and target_N == [3]) or (source_N == [3] and target_N == [0]):
            name_source = [0, 1, 2, 7, 8]
            label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            name_target = [1, 2, 6, 9]
            label_target = [1, 2, 9, 9]
            
        elif (source_N == [3] and target_N == [1]) or (source_N == [3] and target_N == [2]):
            name_source = [0, 1, 5, 7, 8]
            label_source = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            name_target = [0, 4, 8]
            label_target = [0, 9, 4]
            
    num_classes = len(set(label_source)) if label_source else 0
    return name_source, name_target, label_source, label_target, num_classes

def get_files(root, N, name, label):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''
    data = []
    lab =[]
    for k in range(len(N)):
        for i, n in enumerate(name):
            #print(n)
            if int(dataname[N[k]][n].split(".")[0])<101:
                path1 =os.path.join(root,datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])
            data1, lab1 = data_load(path1,dataname[N[k]][n],label=label[i])
            data += data1
            lab +=lab1

    return [data, lab]


def _resolve_axis_key(axisname: str) -> str:
    datanumber = axisname.split(".")
    if eval(datanumber[0]) < 100:
        return "X0" + datanumber[0] + axis[0]
    return "X" + datanumber[0] + axis[0]



def data_load(filename, axisname, label):
    """
    This function is mainly used to generate test data and training data.
    filename:Data location
    axisname:Select which channel's data,---->"_DE_time","_FE_time","_BA_time"
    """
    realaxis = _resolve_axis_key(axisname)
    fl = loadmat(filename)[realaxis]
    data = []
    lab = []
    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size

    return data, lab


def build_window_index(root, N, name, label):
    """Build a lightweight index of CWRU windows to avoid loading all data into memory."""
    window_index = []
    for k in range(len(N)):
        for i, n in enumerate(name):
            if int(dataname[N[k]][n].split(".")[0]) < 101:
                path1 = os.path.join(root, datasetname[3], dataname[N[k]][n])
            else:
                path1 = os.path.join(root, datasetname[0], dataname[N[k]][n])
            axisname = dataname[N[k]][n]
            realaxis = _resolve_axis_key(axisname)
            fl = loadmat(path1)[realaxis]
            max_windows = fl.shape[0] // signal_size
            for window_idx in range(max_windows):
                start = window_idx * signal_size
                window_index.append((path1, axisname, start, label[i]))
            del fl
    return window_index


class CWRUWindowDataset(torch.utils.data.Dataset):
    """Lazy-loading dataset that reads CWRU windows on demand."""

    def __init__(self, window_index, transform=None, sequence_length=1, cache_size=4):
        self.window_index = window_index
        self.sequence_length = sequence_length
        self.transforms = transform if transform else Compose([Reshape()])
        self._labels = np.asarray([entry[3] for entry in window_index], dtype=int)
        # A tiny LRU cache avoids repeated MAT parsing churn and keeps memory bounded.
        self.cache_size = max(1, int(cache_size))
        self._cache = OrderedDict()

    @property
    def labels(self):
        if self._labels.size == 0:
            return np.zeros((0,), dtype=int)
        if self.sequence_length <= 1:
            return self._labels
        num_sequences = max(0, len(self.window_index) - self.sequence_length + 1)
        indices = np.arange(num_sequences) + self.sequence_length - 1
        indices = np.clip(indices, 0, len(self._labels) - 1)
        return self._labels[indices]

    def _load_signal(self, path, axisname):
        cache_key = (path, axisname)
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._cache.move_to_end(cache_key)
            return cached
        realaxis = _resolve_axis_key(axisname)
        data = np.asarray(loadmat(path)[realaxis], dtype=np.float32).reshape(-1)
        self._cache[cache_key] = data
        self._cache.move_to_end(cache_key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return data

    def __len__(self):
        return max(0, len(self.window_index) - self.sequence_length + 1)

    def __getitem__(self, item):
        windows = []
        for offset in range(self.sequence_length):
            path, axisname, start, _ = self.window_index[item + offset]
            signal = self._load_signal(path, axisname)
            window = np.asarray(signal[start:start + signal_size]).squeeze()
            windows.append(window)
        seq = np.stack(windows)
        seq = self.transforms(seq)
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)
        if self._labels.size == 0:
            label = -1
        elif self.sequence_length <= 1:
            label = int(self._labels[item])
        else:
            idx = min(item + self.sequence_length - 1, len(self._labels) - 1)
            label = int(self._labels[idx])
        return seq, int(label)




#--------------------------------------------------------------------------------------------------------------------
class CWRU_inconsistent(object):
    #num_classes = 10
    inputchannel = 1
    def __init__(self, data_dir, transfer_task,inconsistent, normlizetype="0-1"):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.inconsistent = inconsistent
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScale(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train and val
            name_source, name_target, label_source, label_target, num_classes = dataset_information(self.source_N,self.target_N,self.inconsistent)
            source_index = build_window_index(self.data_dir, self.source_N, name_source, label_source)
            source_labels = [entry[3] for entry in source_index]
            train_idx, val_idx = train_test_split(
                np.arange(len(source_index)),
                test_size=0.2,
                random_state=40,
                stratify=source_labels,
            )
            source_train = CWRUWindowDataset(
                [source_index[i] for i in train_idx],
                transform=self.data_transforms['train'],
                sequence_length=1,
            )
            source_val = CWRUWindowDataset(
                [source_index[i] for i in val_idx],
                transform=self.data_transforms['val'],
                sequence_length=1,
            )

            # get target train and val
            target_index = build_window_index(self.data_dir, self.target_N, name_target, label_target)
            target_labels = [entry[3] for entry in target_index]
            train_idx, val_idx = train_test_split(
                np.arange(len(target_index)),
                test_size=0.2,
                random_state=40,
                stratify=target_labels,
            )
            target_train = CWRUWindowDataset(
                [target_index[i] for i in train_idx],
                transform=self.data_transforms['train'],
                sequence_length=1,
            )
            target_val = CWRUWindowDataset(
                [target_index[i] for i in val_idx],
                transform=self.data_transforms['val'],
                sequence_length=1,
            )
            return source_train, source_val, target_train, target_val, num_classes
        else:
            # Baseline: load all classes with consistent labeling
            all_names = list(range(len(dataname[0])))
            all_labels = list(range(len(dataname[0])))

            # get source train and val
            source_index = build_window_index(self.data_dir, self.source_N, all_names, all_labels)
            source_labels = [entry[3] for entry in source_index]
            train_idx, val_idx = train_test_split(
                np.arange(len(source_index)),
                data_pd,
                test_size=0.2,
                random_state=40,
                stratify=source_labels,
            )
            source_train = CWRUWindowDataset(
                [source_index[i] for i in train_idx],
                transform=self.data_transforms['train'],
                sequence_length=1,
            )
            source_val = CWRUWindowDataset(
                [source_index[i] for i in val_idx],
                transform=self.data_transforms['val'],
                sequence_length=1,
            )
            

            # get target val (entire target domain with same labels)
            target_index = build_window_index(self.data_dir, self.target_N, all_names, all_labels)
            target_val = CWRUWindowDataset(
                target_index,
                transform=self.data_transforms['val'],
                sequence_length=1,
            )
            return source_train, source_val, target_val


"""
    def data_split(self):

"""