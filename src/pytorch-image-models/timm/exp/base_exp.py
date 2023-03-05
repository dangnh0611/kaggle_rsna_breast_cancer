from timm.models import create_model
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm import utils
import logging
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
import torch.nn as nn

_logger = logging.getLogger(__name__)



class Exp:
    def __init__(self, args):
        # synchorize with change in args (alias of same obj)
        self.args = args
        self.data_config = None

        # infer num channels
        in_chans = 3
        if self.args.in_chans is not None:
            in_chans = args.in_chans
        elif self.args.input_size is not None:
            in_chans = args.input_size[0]
        self.args.in_chans = in_chans


    def build_model(self):
        model = create_model(
            self.args.model,
            pretrained=self.args.pretrained,
            in_chans=self.args.in_chans,
            num_classes=self.args.num_classes,
            drop_rate=self.args.drop,
            drop_path_rate=self.args.drop_path,
            drop_block_rate=self.args.drop_block,
            global_pool=self.args.gp,
            bn_momentum=self.args.bn_momentum,
            bn_eps=self.args.bn_eps,
            scriptable=self.args.torchscript,
            checkpoint_path=self.args.initial_checkpoint,
            **self.args.model_kwargs,
        )

        # if utils.is_primary(self.args):
        #     _logger.info(
        #     f'Model {safe_model_name(self.args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')
        
        assert self.data_config is None
        self.data_config = resolve_data_config(vars(self.args), model=model, verbose=utils.is_primary(self.args))

        return model


    def build_train_dataset(self):
        train_dataset = create_dataset(
            self.args.dataset,
            root=self.args.data_dir,
            split=self.args.train_split,
            is_training=True,
            class_map=self.args.class_map,
            download=self.args.dataset_download,
            batch_size=self.args.batch_size,
            seed=self.args.seed,
            repeats=self.args.epoch_repeats,
        )
        return train_dataset


    def build_val_dataset(self):
        val_dataset = create_dataset(
            self.args.dataset,
            root=self.args.data_dir,
            split=self.args.val_split,
            is_training=False,
            class_map=self.args.class_map,
            download=self.args.dataset_download,
            batch_size=self.args.batch_size,
        )
        return val_dataset


    def build_train_loader(self, collate_fn = None):
        train_dataset = self.build_train_dataset()

        # wrap dataset in AugMix helper
        if self.args.num_aug_splits > 1:
            train_dataset = AugMixDataset(train_dataset, num_splits=self.args.num_aug_splits)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = self.args.train_interpolation
        if self.args.no_aug or not train_interpolation:
            train_interpolation = self.data_config['interpolation']

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
        )
        return train_loader


    def build_val_loader(self):
        val_dataset = self.build_val_dataset()

        val_workers = self.args.workers
        if self.args.distributed and ('tfds' in self.args.dataset or 'wds' in self.args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            val_workers = min(2, self.args.workers)
        val_loader = create_loader(
            val_dataset,
            input_size=self.data_config['input_size'],
            batch_size=self.args.validation_batch_size or self.args.batch_size,
            is_training=False,
            use_prefetcher=self.args.prefetcher,
            interpolation=self.data_config['interpolation'],
            mean=self.data_config['mean'],
            std=self.data_config['std'],
            num_workers=val_workers,
            distributed=self.args.distributed,
            crop_pct=self.data_config['crop_pct'],
            pin_memory=self.args.pin_mem,
            device=self.args.device,
        )
        return val_loader


    def build_train_loss_fn(self):
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
                train_loss_fn = BinaryCrossEntropy(smoothing=self.args.smoothing, target_threshold=self.args.bce_target_thresh)
            else:
                train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.args.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(device=self.args.device)
        return train_loss_fn
        

    def build_val_loss_fn(self):
        val_loss_fn = nn.CrossEntropyLoss().to(device=self.args.device)
        return val_loss_fn


    def build_optimizer(self, model):
        optimizer =  create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=self.args),
        **self.args.opt_kwargs,
        )
        return optimizer

    
    def build_lr_scheduler(self, optimizer):
        lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(self.args),
        updates_per_epoch=self.args.updates_per_epoch,
        )
        return lr_scheduler, num_epochs


