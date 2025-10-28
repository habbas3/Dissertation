#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from my_datasets.sequence_aug import *

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None, sequence_length=32):
        self.test = bool(test)
        self.sequence_length = sequence_length
        self.transforms = transform if transform else Compose([Reshape()])

        # ``list_data`` may arrive as a pandas ``DataFrame`` (typical for our
        # loaders) or as a simple ``dict``/list pair.  Converting the ``data``
        # column of a DataFrame directly via ``np.array`` produces an ``object``
        # dtype array, which later breaks ``torch.tensor`` conversion.  Instead
        # we explicitly stack the underlying numpy arrays to obtain a numeric
        # ``float`` array.  This keeps the dataset backend agnostic while
        # preventing the ``numpy.object_`` error reported during CWRU runs.
        if isinstance(list_data, pd.DataFrame):
            data_series = list_data['data']
            # Some sources store each window as ``(length, 1)`` which would
            # otherwise introduce an extra singleton dimension when stacked.
            self.seq_data = np.stack([np.asarray(x).squeeze() for x in data_series.to_list()])
            raw_labels = list_data['label'].to_numpy() if 'label' in list_data else None
        elif isinstance(list_data, dict):
            self.seq_data = np.stack([np.asarray(x).squeeze() for x in list_data['data']])
            raw_labels = np.array(list_data.get('label')) if list_data.get('label') is not None else None
        else:  # assume a tuple/list like [data, labels]
            self.seq_data = np.stack([np.asarray(x).squeeze() for x in list_data[0]])
            raw_labels = np.array(list_data[1]) if len(list_data) > 1 else None

        self._raw_labels = raw_labels

        if raw_labels is None:
            num_sequences = max(0, len(self.seq_data) - self.sequence_length + 1)
            self.labels = np.full((num_sequences,), -1, dtype=int)
        else:
            # ``raw_labels`` typically stores one label per raw window.  When we
            # build sequences of ``sequence_length`` consecutive windows the
            # number of samples exposed to the DataLoader becomes
            # ``len(seq_data) - sequence_length + 1``.  Samplers (especially the
            # class-balanced wrapper) expect the ``labels`` attribute to match
            # ``__len__``.  We therefore precompute the label for each sequence
            # using the label of the last window in the slice.
            num_sequences = max(0, len(self.seq_data) - self.sequence_length + 1)
            if num_sequences == 0:
                self.labels = np.zeros((0,), dtype=int)
            else:
                indices = np.arange(num_sequences) + self.sequence_length - 1
                indices = np.clip(indices, 0, len(raw_labels) - 1)
                self.labels = np.asarray(raw_labels)[indices]



    def __len__(self):
        """Return the number of available sequences.

        The previous implementation returned ``len(self.seq_data) - self.sequence_length``
        which fails when ``len(self.seq_data)`` is equal to ``sequence_length`` and
        also yields negative values for very short sequences.  The DataLoader
        expects ``__len__`` to be non-negative, so we clamp the result and include
        the last possible window.
        """

        return len(self.labels)

    def __getitem__(self, item):
        seq = self.seq_data[item:item + self.sequence_length]
        seq = self.transforms(seq)

        # ``sequence_aug`` transformations operate on numpy arrays.  After the
        # pipeline finishes we convert to a tensor unless a tensor was already
        # produced (e.g. by a user-provided transform).
        if not isinstance(seq, torch.Tensor):
            seq = torch.tensor(seq, dtype=torch.float32)

        if self.test:
            return seq, item
        else:
            if self.labels.size == 0:
                # No labels available (e.g. unlabeled target data)
                return seq, -1

            label = self.labels[item]
            return seq, label



