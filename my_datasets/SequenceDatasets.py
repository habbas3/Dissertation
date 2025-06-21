#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from my_datasets.sequence_aug import *

class dataset(Dataset):

    def __init__(self, list_data, test=False, transform=None, sequence_length=32):
        self.test = bool(test)
        self.sequence_length = sequence_length
        self.transforms = transform if transform else Compose([Reshape()])
        self.labels = list_data.get('label')
        self.seq_data = list_data['data']

        # Convert to numpy array to easily slice sequences
        self.seq_data = np.array(self.seq_data)
        self.labels = np.array(self.labels) if self.labels is not None else None

    def __len__(self):
        """Return the number of available sequences.

        The previous implementation returned ``len(self.seq_data) - self.sequence_length``
        which fails when ``len(self.seq_data)`` is equal to ``sequence_length`` and
        also yields negative values for very short sequences.  The DataLoader
        expects ``__len__`` to be non-negative, so we clamp the result and include
        the last possible window.
        """

        return max(0, len(self.seq_data) - self.sequence_length + 1)

    def __getitem__(self, item):
        seq = self.seq_data[item:item + self.sequence_length]
        seq = self.transforms(seq)

        if self.test:
            return seq, item
        else:
            label = self.labels[item + self.sequence_length - 1]  # predict last label
            return seq, label


