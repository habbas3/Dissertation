#!/usr/bin/python
# -*- coding:utf-8 -*-

import logging

def setlogger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 🔁 Clear existing handlers before adding new ones
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Optional: also log to console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
