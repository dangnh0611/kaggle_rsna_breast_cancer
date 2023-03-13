# YOLOX: https://github.com/Megvii-BaseDetection/YOLOX/blob/5c110a25596ad8385e955ac4fc992ba040043fe6/yolox/exp/build.py
#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import importlib
import os
import sys


def get_exp_by_file(exp_file):
    try:
        sys.path.append(os.path.dirname(exp_file))
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        exp = current_exp.Exp
    except Exception:
        raise ImportError("{} doesn't contains class named 'Exp'".format(exp_file))
    return exp


def get_exp_fn(exp_file):
    """
    Args:
        exp_file (str): file path of experiment.
    """
    assert exp_file is not None, "Please provide exp file or exp name."
    return get_exp_by_file(exp_file)