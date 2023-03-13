import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import sklearn
import torch
import torch.distributed as dist
import torch.nn as nn
from albumentations.augmentations.crops import functional as F
from albumentations.augmentations.geometric import functional as FGeometric
from albumentations.core.bbox_utils import union_of_bboxes
from albumentations.core.transforms_interface import (BoxInternalType,
                                                      DualTransform,
                                                      KeypointInternalType,
                                                      to_tuple)
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


TrainAugment = ValAugment


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

    def compute_metrics(self,
                        df,
                        plot_save_path,
                        thres_range=(0, 1, 0.01),
                        sort_by='pfbeta',
                        additional_info=False):
        ori_df = df[[
            'site_id', 'patient_id', 'laterality', 'cancer', 'preds', 'targets'
        ]]
        all_metrics = {}

        reducer_single = lambda df: df
        reducer_gbmean = lambda df: df.groupby(['patient_id', 'laterality']
                                               ).mean()
        reducer_gbmax = lambda df: df.groupby(['patient_id', 'laterality']
                                              ).mean()
        reducer_gbmean_site1 = lambda df: df[df.site_id == 1].reset_index(
            drop=True).groupby(['patient_id', 'laterality']).mean()
        reducer_gbmean_site2 = lambda df: df[df.site_id == 2].reset_index(
            drop=True).groupby(['patient_id', 'laterality']).mean()

        reducers = {
            'single': reducer_single,
            'gbmean': reducer_gbmean,
            'gbmean_site1': reducer_gbmean_site1,
            'gbmean_site2': reducer_gbmean_site2,
            'gbmax': reducer_gbmax,
        }

        for reducer_name, reducer in reducers.items():
            df = reducer(ori_df.copy())
            preds = df['preds'].to_numpy()
            gts = df['targets'].to_numpy()
            # mean_sample_weights = mean_df['sample_weights']
            _metrics = self._compute_metrics(gts, preds, None, thres_range,
                                             sort_by)
            all_metrics[f'{reducer_name}_best_thres'] = _metrics['best_thres']
            all_metrics.update({
                f'{reducer_name}_best_{k}': v
                for k, v in _metrics['best_metric'].items()
            })
            all_metrics[f'{reducer_name}_pfbeta'] = _metrics['pfbeta']
            all_metrics[f'{reducer_name}_auc'] = _metrics['auc']
            all_metrics[f'{reducer_name}_prauc'] = _metrics['prauc']

        # rank 0 only
        if additional_info:
            rsna_metrics.compute_all(ori_df, plot_save_path)
        return all_metrics

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

        # ##### METRICS FOR PROBABILISTIC PREDICTION #####
        # # log loss for pos/neg/overall
        # pos_preds = preds[gts == 1]
        # neg_preds = preds[gts == 0]
        # if len(pos_preds) > 0:
        #     pos_loss = sklearn.metrics.log_loss(np.ones_like(pos_preds),
        #                                         pos_preds,
        #                                         eps=1e-15,
        #                                         normalize=True,
        #                                         sample_weight=None,
        #                                         labels=[0, 1])
        # else:
        #     pos_loss = 99999.
        # if len(neg_preds) > 0:
        #     neg_loss = sklearn.metrics.log_loss(np.zeros_like(neg_preds),
        #                                         neg_preds,
        #                                         eps=1e-15,
        #                                         normalize=True,
        #                                         sample_weight=None,
        #                                         labels=[0, 1])
        # else:
        #     neg_loss = 99999.
        # total_loss = sklearn.metrics.log_loss(gts,
        #                                       preds,
        #                                       eps=1e-15,
        #                                       normalize=True,
        #                                       sample_weight=None,
        #                                       labels=[0, 1])

        # Probabilistic-fbeta
        pfbeta = pfbeta_np(gts, preds, beta=1.0)
        # AUC
        fpr, tpr, _thresholds = sklearn.metrics.roc_curve(gts,
                                                          preds,
                                                          pos_label=1)
        auc = sklearn.metrics.auc(fpr, tpr)

        # PR-AUC
        precisions, recalls, _thresholds = sklearn.metrics.precision_recall_curve(
            gts, preds)
        pr_auc = sklearn.metrics.auc(recalls, precisions)

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
            'prauc': pr_auc,
            # 'pos_log_loss': pos_loss,
            # 'neg_log_loss': neg_loss,
            # 'log_loss': total_loss,
        }