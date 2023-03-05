#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.data_dir = "/home/dangnh36/datasets/coco_val"
        self.train_ann = "/home/dangnh36/datasets/coco_val/annotations/instances_train2017.json"
        self.val_ann = "/home/dangnh36/datasets/coco_val/annotations/instances_val2017.json"
