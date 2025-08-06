#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 18:29:37 2023

@author: habbas3
"""

# settings.py
import global_habbas3 as _core

def init():
    """Initialise the underlying global containers."""

    return _core.init()


def __getattr__(name):  # pragma: no cover - simple delegation
    return getattr(_core, name)


def __setattr__(name, value):  # pragma: no cover - simple delegation
    setattr(_core, name, value)