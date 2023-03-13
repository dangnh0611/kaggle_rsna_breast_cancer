



# train fold 3 + soft_pos = 0.9
python3 src/pytorch-image-models/train_exp.py -f src/pytorch-image-models/projects/rsna/exps/exp7.py \
    -expn train_fold3 --model convnext_small.fb_in22k_ft_in1k_384 \
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \
    --opt sgd --lr 3e-3 --min-lr 5e-5 --sched cosine --warmup-lr 3e-5 \
    --epochs 30 --warmup-epoch 4 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --log-wandb --exp-kwargs fold_idx=3 num_sched_epochs=10 num_epochs=35 \
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# train fold 2 + soft_pos = 0.9
python3 src/pytorch-image-models/train_exp.py -f src/pytorch-image-models/projects/rsna/exps/exp7.py \
    -expn train_fold2 --model convnext_small.fb_in22k_ft_in1k_384 \
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \
    --opt sgd --lr 3e-3 --min-lr 5e-5 --sched cosine --warmup-lr 3e-5 \
    --epochs 30 --warmup-epoch 4 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --log-wandb --exp-kwargs fold_idx=2 num_sched_epochs=10 num_epochs=35 \
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# train fold 0 + soft_pos = 0.8
python3 src/pytorch-image-models/train_exp.py -f src/pytorch-image-models/projects/rsna/exps/exp7.py \
    -expn stage1_train_fold0 --model convnext_small.fb_in22k_ft_in1k_384 \
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \
    --opt sgd --lr 1e-3 --min-lr 1e-5 --sched cosine --warmup-lr 1e-5 \
    --epochs 24 --warmup-epoch 4 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.2 --log-interval 500 \
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --log-wandb --exp-kwargs fold_idx=0 num_sched_epochs=10 num_epochs=35 \
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# train fold 1 + soft_pos = 0.8
python3 src/pytorch-image-models/train_exp.py -f src/pytorch-image-models/projects/rsna/exps/exp7.py \
    -expn stage1_train_fold1 --model convnext_small.fb_in22k_ft_in1k_384 \
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \
    --opt sgd --lr 1e-3 --min-lr 1e-5 --sched cosine --warmup-lr 1e-5 \
    --epochs 24 --warmup-epoch 4 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.2 --log-interval 500 \
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --log-wandb --exp-kwargs fold_idx=1 num_sched_epochs=10 num_epochs=35 \
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# finetune fold 0 + soft_pos = 0.9
python3 src/pytorch-image-models/train_exp.py -f src/pytorch-image-models/projects/rsna/exps/exp7.py \
    -expn stage2_finetune_train_fold0 --model convnext_small.fb_in22k_ft_in1k_384 --pretrained \
    --initial-checkpoint output/train/20230224-153342-convnext_small_fb_in22k_ft_in1k_384-fold0-3x2048x1024_alldata_fold0/checkpoint-24.pth.tar \
    --num-classes 1 --batch-size 8 --validation-batch-size 16 \
    --opt sgd --lr 1e-3 --min-lr 5e-5 --sched cosine --warmup-lr 1e-3 \
    --epochs 16 --warmup-epoch 1 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --log-wandb --exp-kwargs fold_idx=0 num_sched_epochs=10 num_epochs=35 \
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 17 18 \
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# finetune fold 1 + soft_pos = 0.9
python3 src/pytorch-image-models/train_exp.py -f src/pytorch-image-models/projects/rsna/exps/exp7.py \
    -expn stage2_finetune_train_fold1 --model convnext_small.fb_in22k_ft_in1k_384 --pretrained \
    --initial-checkpoint output/train/20230224-154520-convnext_small_fb_in22k_ft_in1k_384-fold1-3x2048x1024_alldta_fold1/checkpoint-24.pth.tar \
    --num-classes 1 --batch-size 8 --validation-batch-size 16 \
    --opt sgd --lr 1e-3 --min-lr 5e-5 --sched cosine --warmup-lr 1e-3 \
    --epochs 16 --warmup-epoch 1 --cooldown-epochs 1 \
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \
    --save-images --log-wandb --exp-kwargs fold_idx=1 num_sched_epochs=10 num_epochs=35 \
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 17 18 \
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max
