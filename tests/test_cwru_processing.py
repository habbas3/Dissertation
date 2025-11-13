#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 15:52:54 2025

@author: habbas
"""

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

import my_datasets.CWRU_label_inconsistent as cwru
from my_datasets.sequence_aug import Normalize


def test_normalize_constant_signal_produces_finite_output():
    norm = Normalize("0-1")
    signal = np.ones((2, 8), dtype=np.float32)
    out = norm(signal)
    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))
    assert np.all(out == 0.0)


def test_cwru_sequence_length_and_tensor_shape(monkeypatch):
    def _fake_get_files(root, N, name, labels):
        data = []
        target_labels = []
        for lbl in labels:
            for _ in range(5):
                data.append(np.full((1024,), float(lbl), dtype=np.float32))
                target_labels.append(lbl)
        return [data, target_labels]

    monkeypatch.setattr(cwru, "get_files", _fake_get_files)

    dataset = cwru.CWRU_inconsistent("/tmp", ([0], [1]), "UAN")
    src_train, src_val, tgt_train, tgt_val, num_classes = dataset.data_split(transfer_learning=True)

    assert src_train.sequence_length == 1
    sample, label = src_train[0]
    assert sample.shape[0] == 1
    assert sample.shape[-1] == 1024
    assert torch.isfinite(sample).all()
    assert isinstance(label, (int, np.integer))
    assert num_classes >= 1