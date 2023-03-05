from timm.exp.base_exp import Exp as _BaseExp
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from projects.rsna import datasets
from projects.rsna.utils import metrics as rsna_metrics
import numpy as np
import sklearn
import torch
from timm.metrics import compute_usual_metrics, pfbeta_np
import os

from timm.models import create_model
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm import utils
import logging
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
import torch.nn as nn

from torch.utils.data import WeightedRandomSampler
from exhaustive_weighted_random_sampler import ExhaustiveWeightedRandomSampler
from ignite.distributed import DistributedProxySampler


class TrainAugment:

    def __init__(self):
        pass

    def __call__(self, img):
        return img


class TrainTransform:

    def __init__(self, input_size):
        input_size = input_size
        self.transform_fn = A.Compose([
            A.LongestMaxSize(input_size,
                             interpolation=cv2.INTER_AREA,
                             always_apply=True,
                             p=1.0),
            A.PadIfNeeded(min_height=input_size,
                          min_width=input_size,
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


ValAugment = TrainAugment
ValTransform = TrainTransform


class Exp(_BaseExp):

    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.args.csv_root_dir = '/home/dangnh36/datasets/.comp/rsna/cv/v1/'
        self.args.img_root_dir = '/home/dangnh36/datasets/.comp/rsna/crops/uint8_voilut_png@yolox_nano_bre_416_datav2'
        self.args.fold_idx = 0
        self.pos_neg_ratio = 1.0 / 8

    def build_train_dataset(self):
        assert self.data_config is not None
        csv_path = os.path.join(self.args.csv_root_dir, f'train_fold_{self.args.fold_idx}.csv')
        augment_fn = TrainAugment()
        transform_fn = TrainTransform(self.data_config['input_size'][-1])
        train_dataset = datasets.v1.RSNADataset(csv_path, self.args.img_root_dir, augment_fn,
                                      transform_fn)
        return train_dataset

    def build_train_loader(self, collate_fn = None):
        train_dataset = self.build_train_dataset()

        # wrap dataset in AugMix helper
        if self.args.num_aug_splits > 1:
            train_dataset = AugMixDataset(train_dataset, num_splits=self.args.num_aug_splits)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = self.args.train_interpolation
        if self.args.no_aug or not train_interpolation:
            train_interpolation = self.data_config['interpolation']

        sampler_weights = train_dataset.get_sampler_weights(self.pos_neg_ratio)
        if self.args.distributed:
            assert not isinstance(train_dataset, torch.utils.data.IterableDataset)
            sampler = DistributedProxySampler(
                ExhaustiveWeightedRandomSampler(sampler_weights, num_samples=len(train_dataset))
            )
        else:
            sampler = ExhaustiveWeightedRandomSampler(sampler_weights, num_samples=len(train_dataset))

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
            sampler = sampler,
        )
        return train_loader


    def build_val_dataset(self):
        assert self.data_config is not None
        csv_path = os.path.join(self.args.csv_root_dir, f'val_fold_{self.args.fold_idx}.csv')
        augment_fn = ValAugment()
        transform_fn = ValTransform(self.data_config['input_size'][-1])
        train_dataset = datasets.v1.RSNADataset(csv_path, self.args.img_root_dir, augment_fn,
                                      transform_fn)
        return train_dataset

    
    def compute_metrics(self,
                        df,
                        plot_save_path,
                        thres_range=(0, 1, 0.01),
                        sort_by='pfbeta'):
        preds = df['preds']
        gts = df['targets']
        sample_weights = df['sample_weights']

        rsna_metrics.compute_all(df, plot_save_path)

        metrics = self._compute_metrics(gts, preds, sample_weights, thres_range, sort_by)
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