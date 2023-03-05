#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 1
        # factor of model depth
        self.depth = 0.33
        # factor of model width
        self.width = 0.375
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 4
        self.input_size = (768, 768)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 0
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.data_dir = "/home/dangnh36/datasets/.comp/rsna/roi_det/v2/yolox_coco"
        # name of annotation file for training
        self.train_ann = "/home/dangnh36/datasets/.comp/rsna/roi_det/v2/yolox_coco/annotations/instances_train2017.json"
        # name of annotation file for evaluation
        self.val_ann = "/home/dangnh36/datasets/.comp/rsna/roi_det/v2/yolox_coco/annotations/instances_val2017.json"
        # name of annotation file for testing
        self.test_ann = "instances_test2017.json"

        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 0.
        # prob of applying mixup aug
        self.mixup_prob = 0.
        # prob of applying hsv aug
        self.hsv_prob = 0.
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 0.
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.0
        self.mosaic_scale = (0.5, 1.5)
        # apply mixup aug or not
        self.enable_mixup = False
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 0.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 20
        # max training epoch
        self.max_epoch = 400
        # minimum learning rate during warmup
        self.warmup_lr = 1e-4
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.001 / 32.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 5
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 1
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (768, 768)
        # self.test_size = (1024, 1024)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.9
        self.cache_dataset = None
        self.dataset = None
