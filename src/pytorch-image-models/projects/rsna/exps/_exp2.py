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
from projects.rsna.utils import metrics as rsna_metrics
from timm import utils
from timm.data import (AugMixDataset, FastCollateMixup, Mixup, create_dataset,
                       create_loader, resolve_data_config)
from timm.exp.base_exp import Exp as _BaseExp
from timm.loss import (BinaryCrossEntropy, JsdCrossEntropy,
                       LabelSmoothingCrossEntropy, SoftTargetCrossEntropy)
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
        self.transform_fn = A.Compose([
            A.HorizontalFlip(p=0.5),
        ])

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


class BalanceSampler(Sampler):

    def __init__(self, dataset, ratio=8):
        self.r = ratio - 1
        self.dataset = dataset
        labels = dataset.get_labels()
        self.pos_index = np.where(labels > 0)[0]
        self.neg_index = np.where(labels == 0)[0]
        print('Num pos:', len(self.pos_index))
        print('Num neg:', len(self.neg_index))

        self.neg_length = self.r * int(np.floor(len(self.neg_index) / self.r))
        self.len = self.neg_length + self.neg_length // self.r

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.neg_length].reshape(-1, self.r)
        pos_index = np.random.choice(pos_index,
                                     self.neg_length // self.r).reshape(-1, 1)

        index = np.concatenate([pos_index, neg_index], -1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.len


class Exp(_BaseExp):

    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.args.csv_root_dir = '/home/dangnh36/datasets/.comp/rsna/cv/v1/'
        self.args.img_root_dir = '/home/dangnh36/datasets/.comp/rsna/crops/uint8_voilut_png@yolox_nano_bre_416_datav2'
        self.args.fold_idx = 0
        self.ratio = 4

    def build_train_dataset(self):
        assert self.data_config is not None
        csv_path = os.path.join(self.args.csv_root_dir,
                                f'train_fold_{self.args.fold_idx}.csv')
        augment_fn = TrainAugment()
        transform_fn = TrainTransform(self.data_config['input_size'][1:])
        train_dataset = datasets.v1.RSNADataset(csv_path,
                                                self.args.img_root_dir,
                                                augment_fn, transform_fn)
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
                BalanceSampler(train_dataset, ratio=self.ratio))
        else:
            sampler = BalanceSampler(train_dataset, ratio=self.ratio)

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
        csv_path = os.path.join(self.args.csv_root_dir,
                                f'val_fold_{self.args.fold_idx}.csv')
        augment_fn = ValAugment()
        transform_fn = ValTransform(self.data_config['input_size'][1:])
        train_dataset = datasets.v1.RSNADataset(csv_path,
                                                self.args.img_root_dir,
                                                augment_fn, transform_fn)
        return train_dataset

    def compute_metrics(self,
                        df,
                        plot_save_path,
                        thres_range=(0, 1, 0.01),
                        sort_by='pfbeta',
                        additional_info=False):
        preds = df['preds'].to_numpy()
        gts = df['targets'].to_numpy()
        sample_weights = df['sample_weights']
        metrics = self._compute_metrics(gts, preds, sample_weights,
                                        thres_range, sort_by)

        # rank 0 only
        if additional_info:
            rsna_metrics.compute_all(df, plot_save_path)

        return metrics

    def _compute_metrics(self,
                         gts,
                         preds,
                         sample_weights=None,
                         thres_range=(0, 1, 0.01),
                         sort_by='pfbeta'):
        if isinstance(gts, torch.Tensor):
            gts = gts.cpu().numpy()
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()
        assert isinstance(gts, np.ndarray) and isinstance(preds, np.ndarray)
        assert len(preds) == len(gts)

        ##### METRICS FOR PROBABILISTIC PREDICTION #####
        # log loss for pos/neg/overall
        pos_preds = preds[gts == 1]
        neg_preds = preds[gts == 0]
        if len(pos_preds) > 0:
            pos_loss = sklearn.metrics.log_loss(np.ones_like(pos_preds),
                                                pos_preds,
                                                eps=1e-15,
                                                normalize=True,
                                                sample_weight=None,
                                                labels=[0, 1])
        else:
            pos_loss = 99999.
        if len(neg_preds) > 0:
            neg_loss = sklearn.metrics.log_loss(np.zeros_like(neg_preds),
                                                neg_preds,
                                                eps=1e-15,
                                                normalize=True,
                                                sample_weight=None,
                                                labels=[0, 1])
        else:
            neg_loss = 99999.
        total_loss = sklearn.metrics.log_loss(gts,
                                              preds,
                                              eps=1e-15,
                                              normalize=True,
                                              sample_weight=None,
                                              labels=[0, 1])

        # Probabilistic-fbeta
        pfbeta = pfbeta_np(gts, preds, beta=1.0)
        # AUC
        fpr, tpr, _thresholds = sklearn.metrics.roc_curve(gts,
                                                          preds,
                                                          pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)

        ##### METRICS FOR CATEGORICAL PREDICTION #####
        # PER THRESHOLD METRIC
        per_thres_metrics = []
        for thres in np.arange(*thres_range):
            bin_preds = (preds > thres).astype(np.uint8)
            metric_at_thres = compute_usual_metrics(gts, bin_preds, beta=1.0)
            pfbeta_at_thres = pfbeta_np(gts, bin_preds, beta=1.0)
            metric_at_thres['pfbeta'] = pfbeta_at_thres

            if sample_weights is not None:
                w_metric_at_thres = compute_usual_metrics(gts,
                                                          bin_preds,
                                                          beta=1.0)
                w_metric_at_thres = {
                    f'w_{k}': v
                    for k, v in w_metric_at_thres.items()
                }
                metric_at_thres.update(w_metric_at_thres)
            per_thres_metrics.append((thres, metric_at_thres))

        per_thres_metrics.sort(key=lambda x: x[1][sort_by], reverse=True)

        # handle multiple thresholds with same scores
        top_score = per_thres_metrics[0][1][sort_by]
        same_scores = []
        for j, (thres, metric_at_thres) in enumerate(per_thres_metrics):
            if metric_at_thres[sort_by] == top_score:
                same_scores.append(abs(thres - 0.5))
            else:
                assert metric_at_thres[sort_by] < top_score
                break
        if len(same_scores) == 1:
            best_thres, best_metric = per_thres_metrics[0]
        else:
            # the nearer 0.5 threshold is --> better
            best_idx = np.argmin(np.array(same_scores))
            best_thres, best_metric = per_thres_metrics[best_idx]

        # best thres, best results, all results
        return {
            'best_thres': best_thres,
            'best_metric': best_metric,
            'all_metrics': per_thres_metrics,
            'pfbeta': pfbeta,
            'auc': auc,
            'pos_log_loss': pos_loss,
            'neg_log_loss': neg_loss,
            'log_loss': total_loss,
        }