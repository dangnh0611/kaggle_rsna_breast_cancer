import logging
import math
import os

import albumentations as A
import cv2
import numpy as np
import sklearn
import torch
import torch.distributed as dist
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from ignite.distributed import DistributedProxySampler
from projects.rsna import datasets
from projects.rsna.exps.base_exp import Exp as _BaseExp
from projects.rsna.utils import augs
from projects.rsna.utils import metrics as rsna_metrics
from projects.rsna.utils import samplers
from timm import utils
from timm.data import (AugMixDataset, FastCollateMixup, Mixup, create_dataset,
                       create_loader, resolve_data_config)
from timm.loss import (BinaryCrossEntropy, JsdCrossEntropy,
                       LabelSmoothingCrossEntropy, SoftTargetCrossEntropy,
                       BinaryCrossEntropyPosSmoothOnly)
from timm.metrics import compute_usual_metrics, pfbeta_np
from timm.models import (create_model, load_checkpoint, model_parameters,
                         resume_checkpoint, safe_model_name)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from torch.utils.data import Sampler, WeightedRandomSampler


class ValAugment:

    def __init__(self):
        pass

    def __call__(self, img):
        return img


class TrainAugment:

    def __init__(self):
        self.transform_fn = A.Compose(
            [
                # crop
                augs.CustomRandomSizedCropNoResize(scale=(0.5, 1.0),
                                                   ratio=(0.5, 0.8),
                                                   always_apply=False,
                                                   p=0.4),

                # flip
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                # downscale
                A.OneOf([
                    A.Downscale(scale_min=0.75,
                                scale_max=0.95,
                                interpolation=dict(upscale=cv2.INTER_LINEAR,
                                                   downscale=cv2.INTER_AREA),
                                always_apply=False,
                                p=0.1),
                    A.Downscale(scale_min=0.75,
                                scale_max=0.95,
                                interpolation=dict(upscale=cv2.INTER_LANCZOS4,
                                                   downscale=cv2.INTER_AREA),
                                always_apply=False,
                                p=0.1),
                    A.Downscale(scale_min=0.75,
                                scale_max=0.95,
                                interpolation=dict(upscale=cv2.INTER_LINEAR,
                                                   downscale=cv2.INTER_LINEAR),
                                always_apply=False,
                                p=0.8),
                ],
                        p=0.125),

                # contrast
                # relative dark/bright between region, like HDR
                A.OneOf([
                    A.RandomToneCurve(scale=0.3, always_apply=False, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2),
                                               contrast_limit=(-0.4, 0.5),
                                               brightness_by_max=True,
                                               always_apply=False,
                                               p=0.5)
                ],
                        p=0.5),

                # affine
                A.OneOf(
                    [
                        A.ShiftScaleRotate(shift_limit=None,
                                           scale_limit=[-0.15, 0.15],
                                           rotate_limit=[-30, 30],
                                           interpolation=cv2.INTER_LINEAR,
                                           border_mode=cv2.BORDER_CONSTANT,
                                           value=0,
                                           mask_value=None,
                                           shift_limit_x=[-0.1, 0.1],
                                           shift_limit_y=[-0.2, 0.2],
                                           rotate_method='largest_box',
                                           always_apply=False,
                                           p=0.6),

                        # one of with other affine
                        A.ElasticTransform(alpha=1,
                                           sigma=20,
                                           alpha_affine=10,
                                           interpolation=cv2.INTER_LINEAR,
                                           border_mode=cv2.BORDER_CONSTANT,
                                           value=0,
                                           mask_value=None,
                                           approximate=False,
                                           same_dxdy=False,
                                           always_apply=False,
                                           p=0.2),

                        # distort
                        A.GridDistortion(num_steps=5,
                                         distort_limit=0.3,
                                         interpolation=cv2.INTER_LINEAR,
                                         border_mode=cv2.BORDER_CONSTANT,
                                         value=0,
                                         mask_value=None,
                                         normalized=True,
                                         always_apply=False,
                                         p=0.2),
                    ],
                    p=0.5),

                # random erase
                A.CoarseDropout(max_holes=6,
                                max_height=0.15,
                                max_width=0.25,
                                min_holes=1,
                                min_height=0.05,
                                min_width=0.1,
                                fill_value=0,
                                mask_fill_value=None,
                                always_apply=False,
                                p=0.25),
            ],
            p=0.9)

        print('TRAIN AUG:\n', self.transform_fn)

    def __call__(self, img):
        return self.transform_fn(image=img)['image']


class ValTransform:

    def __init__(self, input_size, interpolation=cv2.INTER_LINEAR):
        self.input_size = input_size
        self.interpolation = interpolation
        self.max_h, self.max_w = input_size

        def _fit_resize(image, **kwargs):
            img_h, img_w = image.shape[:2]
            r = min(self.max_h / img_h, self.max_w / img_w)
            new_h, new_w = int(img_h * r), int(img_w * r)
            new_image = cv2.resize(image, (new_w, new_h),
                                   interpolation=interpolation)
            return new_image

        self.transform_fn = A.Compose([
            A.Lambda(name="FitResize",
                     image=_fit_resize,
                     always_apply=True,
                     p=1.0),
            A.PadIfNeeded(min_height=self.max_h,
                          min_width=self.max_w,
                          pad_height_divisor=None,
                          pad_width_divisor=None,
                          position=A.augmentations.geometric.transforms.
                          PadIfNeeded.PositionType.CENTER,
                          border_mode=cv2.BORDER_CONSTANT,
                          value=0,
                          mask_value=None,
                          always_apply=True,
                          p=1.0),
            ToTensorV2(transpose_mask=True)
        ])

    def __call__(self, img):
        return self.transform_fn(image=img)['image']


TrainTransform = ValTransform


class Exp(_BaseExp):

    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.meta = {
            'csv_root_dir': '/home/dangnh36/datasets/.comp/rsna/cv/v1/',
            # 'img_root_dir': '/home/dangnh36/datasets/.comp/rsna/crops/uint8_voilut_png@yolox_nano_bre_416_datav2',
            'img_root_dir':
            '/raid/.comp/uint8_voilut_png@yolox_nano_bre_416_datav2',
            'fold_idx': 0,
            'num_sched_epochs': 6,
            'num_epochs': 50,
            'start_ratio': 1 / 3,
            'end_ratio': 1 / 7,
            'one_pos_mode': True,
        }
        old_meta_len = len(self.meta)
        self.meta.update(self.args.exp_kwargs)
        assert len(self.meta) == old_meta_len
        print('\n------\nEXP METADATA:\n', self.meta)

    def build_train_dataset(self):
        assert self.data_config is not None
        fold_idx = self.meta['fold_idx']
        csv_root_dir = self.meta['csv_root_dir']
        img_root_dir = self.meta['img_root_dir']
        csv_path = os.path.join(csv_root_dir, f'train_fold_{fold_idx}.csv')
        augment_fn = TrainAugment()
        transform_fn = TrainTransform(self.data_config['input_size'][1:])
        train_dataset = datasets.v1.RSNADataset(csv_path, img_root_dir,
                                                augment_fn, transform_fn, n_channels=self.args.input_size[0])
        return train_dataset

    def build_train_loader(self, collate_fn=None):
        train_dataset = self.build_train_dataset()

        # wrap dataset in AugMix helper
        if self.args.num_aug_splits > 1:
            train_dataset = AugMixDataset(train_dataset,
                                          num_splits=self.args.num_aug_splits)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = self.args.train_interpolation
        if self.args.no_aug or not train_interpolation:
            train_interpolation = self.data_config['interpolation']

        if self.args.distributed:
            assert not isinstance(train_dataset,
                                  torch.utils.data.IterableDataset)
            sampler = DistributedProxySampler(
                samplers.BalanceSamplerV3(
                    train_dataset,
                    batch_size=self.args.batch_size,
                    num_sched_epochs=self.meta['num_sched_epochs'],
                    num_epochs=self.meta['num_epochs'],
                    start_ratio=self.meta['start_ratio'],
                    end_ratio=self.meta['end_ratio'],
                    one_pos_mode=self.meta['one_pos_mode'],
                    seed=self.args.seed))
        else:
            sampler = samplers.BalanceSamplerV3(
                train_dataset,
                batch_size=self.args.batch_size,
                num_sched_epochs=self.meta['num_sched_epochs'],
                num_epochs=self.meta['num_epochs'],
                start_ratio=self.meta['start_ratio'],
                end_ratio=self.meta['end_ratio'],
                one_pos_mode=self.meta['one_pos_mode'],
                seed=self.args.seed)

        train_loader = create_loader(
            train_dataset,
            input_size=self.data_config['input_size'],
            batch_size=self.args.batch_size,
            is_training=True,
            use_prefetcher=self.args.prefetcher,
            no_aug=self.args.no_aug,
            re_prob=self.args.reprob,
            re_mode=self.args.remode,
            re_count=self.args.recount,
            re_split=self.args.resplit,
            scale=self.args.scale,
            ratio=self.args.ratio,
            hflip=self.args.hflip,
            vflip=self.args.vflip,
            color_jitter=self.args.color_jitter,
            auto_augment=self.args.aa,
            num_aug_repeats=self.args.aug_repeats,
            num_aug_splits=self.args.num_aug_splits,
            interpolation=train_interpolation,
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            num_workers=self.args.workers,
            distributed=self.args.distributed,
            collate_fn=collate_fn,
            pin_memory=self.args.pin_mem,
            device=self.args.device,
            use_multi_epochs_loader=self.args.use_multi_epochs_loader,
            worker_seeding=self.args.worker_seeding,
            sampler=sampler,
        )
        return train_loader

    def build_val_dataset(self):
        assert self.data_config is not None

        fold_idx = self.meta['fold_idx']
        csv_root_dir = self.meta['csv_root_dir']
        img_root_dir = self.meta['img_root_dir']
        csv_path = os.path.join(csv_root_dir, f'val_fold_{fold_idx}.csv')

        augment_fn = ValAugment()
        transform_fn = ValTransform(self.data_config['input_size'][1:])
        val_dataset = datasets.v1.RSNADataset(csv_path, img_root_dir,
                                              augment_fn, transform_fn, n_channels= self.args.input_size[0])
        return val_dataset

    def build_train_loss_fn(self):
        pos_weight = self.args.pos_weight
        pos_weight = None if pos_weight <= 0 else pos_weight
        assert self.args.smoothing > 0 and self.args.bce_loss
        # setup loss function
        if self.args.jsd_loss:
            assert self.args.num_aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=self.args.num_aug_splits, smoothing=self.args.smoothing)
        elif self.args.mixup_active:
            # smoothing is handled with mixup target transform which outputs sparse, soft targets
            if self.args.bce_loss:
                train_loss_fn = BinaryCrossEntropy(target_threshold=self.args.bce_target_thresh)
            else:
                train_loss_fn = SoftTargetCrossEntropy()
        elif self.args.smoothing:
            if self.args.bce_loss:
                if pos_weight is not None:
                    assert self.args.num_classes == 1
                    pos_weight = torch.Tensor([pos_weight])
                print('Using pos weight:', pos_weight)
                train_loss_fn = BinaryCrossEntropyPosSmoothOnly(smoothing=self.args.smoothing, target_threshold=self.args.bce_target_thresh, pos_weight=pos_weight)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=self.args.device)
        return train_loss_fn
