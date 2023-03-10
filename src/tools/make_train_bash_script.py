import argparse
import os

from settings import SETTINGS


def parse_args():
    parser = argparse.ArgumentParser(
        'Generate and write training bash script.')
    parser.add_argument('--mode',
                        type=str,
                        default='fully_reproduce',
                        choices=['fully_reproduce', 'reproduce'],
                        help='Reproduce mode')
    parser.add_argument('--save-path',
                        type=str,
                        default='./_train_script_auto_generated.sh',
                        help='Path to save train script.')
    args = parser.parse_args()
    return args


def main(args):
    if args.mode == 'fully_reproduce':
        EXP_CFG_PATH = 'src/pytorch-image-models/projects/rsna/exps/exp7_fully_reproduce.py'
        EXP_NAME_PREFIX = 'fully_reproduce_train'
    elif args.mode == 'reproduce':
        EXP_NAME_PREFIX = 'reproduce_train'
        EXP_CFG_PATH = 'src/pytorch-image-models/projects/rsna/exps/exp7.py'
    else:
        raise AssertionError()

    EXP_NAME_FOLD_2 = f'{EXP_NAME_PREFIX}_fold_2'
    EXP_NAME_FOLD_3 = f'{EXP_NAME_PREFIX}_fold_3'
    STAGE1_EXP_NAME_FOLD_0 = f'stage1_{EXP_NAME_PREFIX}_fold_0'
    STAGE1_EXP_NAME_FOLD_1 = f'stage1_{EXP_NAME_PREFIX}_fold_1'
    STAGE2_EXP_NAME_FOLD_0 = f'stage2_{EXP_NAME_PREFIX}_fold_0'
    STAGE2_EXP_NAME_FOLD_1 = f'stage2_{EXP_NAME_PREFIX}_fold_1'

    SAVE_CKPT_DIR = os.path.join(SETTINGS.MODEL_CHECKPOINT_DIR,
                                 'timm_classification')
    FINETUNE_INIT_CKPT_PATH_FOLD_0 = os.path.abspath(
        os.path.join(SAVE_CKPT_DIR, STAGE1_EXP_NAME_FOLD_0,
                     'checkpoint-24.pth.tar'))
    FINETUNE_INIT_CKPT_PATH_FOLD_1 = os.path.abspath(
        os.path.join(SAVE_CKPT_DIR, STAGE1_EXP_NAME_FOLD_1,
                     'checkpoint-24.pth.tar'))

    TRAIN_SCRIPT_CONTENT = f"""
#! /bin/bash
###########################################################
# This script was generated by {os.path.abspath(__file__)}
###########################################################

# Before running this script, make sure {os.path.abspath(SAVE_CKPT_DIR)} is empty
# Training procedure contains 6 STEPS, more detail at https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449
# Note that STEP 3 need to be finished before start STEP 5
# and STEP 4 need to be finished before start STEP 6
# For ones who have multiple GPUs, you can run each step in parallel, each step on a single GPU.
# For example, if you have 4 GPU (>=40 GB Memory), you can do STEP 1 + STEP 2 + STEP 3 + STEP 4 in parallel first.
# Once STEP 3 and STEP 4 finished, you can start STEP 5 and STEP 6



# [STEP 1] train fold 2 + soft_pos = 0.9
# this will create new directory and save checkpoints to {os.path.abspath(os.path.join(SAVE_CKPT_DIR, EXP_NAME_FOLD_2))} 
python3 src/pytorch-image-models/train_exp.py \\
    -f {EXP_CFG_PATH} \\
    -expn {EXP_NAME_FOLD_2} --model convnext_small.fb_in22k_ft_in1k_384 \\
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \\
    --opt sgd --lr 3e-3 --min-lr 5e-5 --sched cosine --warmup-lr 3e-5 \\
    --epochs 30 --warmup-epoch 4 --cooldown-epochs 1 \\
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \\
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \\
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \\
    --save-images --log-wandb --exp-kwargs fold_idx=2 num_sched_epochs=10 num_epochs=35 \\
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \\
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# [STEP 2] train fold 3 + soft_pos = 0.9
# this will create new directory and save checkpoints to {os.path.abspath(os.path.join(SAVE_CKPT_DIR, EXP_NAME_FOLD_3))} 
python3 src/pytorch-image-models/train_exp.py \\
    -f {EXP_CFG_PATH} \\
    -expn {EXP_NAME_FOLD_3} --model convnext_small.fb_in22k_ft_in1k_384 \\
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \\
    --opt sgd --lr 3e-3 --min-lr 5e-5 --sched cosine --warmup-lr 3e-5 \\
    --epochs 30 --warmup-epoch 4 --cooldown-epochs 1 \\
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \\
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \\
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \\
    --save-images --log-wandb --exp-kwargs fold_idx=3 num_sched_epochs=10 num_epochs=35 \\
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \\
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# [STEP 3] train (stage 1) fold 0 + soft_pos = 0.8
# this will create new directory and save checkpoints to {os.path.abspath(os.path.join(SAVE_CKPT_DIR, STAGE1_EXP_NAME_FOLD_0))} 
python3 src/pytorch-image-models/train_exp.py \\
    -f {EXP_CFG_PATH} \\
    -expn {STAGE1_EXP_NAME_FOLD_0} --model convnext_small.fb_in22k_ft_in1k_384 \\
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \\
    --opt sgd --lr 1e-3 --min-lr 1e-5 --sched cosine --warmup-lr 1e-5 \\
    --epochs 24 --warmup-epoch 4 --cooldown-epochs 1 \\
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.2 --log-interval 500 \\
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \\
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \\
    --save-images --log-wandb --exp-kwargs fold_idx=0 num_sched_epochs=10 num_epochs=35 \\
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \\
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# [STEP 4] train (stage 1) fold 1 + soft_pos = 0.8
# this will create new directory and save checkpoints to {os.path.abspath(os.path.join(SAVE_CKPT_DIR, STAGE1_EXP_NAME_FOLD_1))} 
python3 src/pytorch-image-models/train_exp.py \\
    -f {EXP_CFG_PATH} \\
    -expn {STAGE1_EXP_NAME_FOLD_1} --model convnext_small.fb_in22k_ft_in1k_384 \\
    --pretrained --num-classes 1 --batch-size 8 --validation-batch-size 16 \\
    --opt sgd --lr 1e-3 --min-lr 1e-5 --sched cosine --warmup-lr 1e-5 \\
    --epochs 24 --warmup-epoch 4 --cooldown-epochs 1 \\
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.2 --log-interval 500 \\
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \\
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \\
    --save-images --log-wandb --exp-kwargs fold_idx=1 num_sched_epochs=10 num_epochs=35 \\
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 10 18 \\
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# [STEP 5] finetune (stage 2) fold 0 + soft_pos = 0.9
# this will create new directory and save checkpoints to {os.path.abspath(os.path.join(SAVE_CKPT_DIR, STAGE2_EXP_NAME_FOLD_0))} 
python3 src/pytorch-image-models/train_exp.py \\
    -f {EXP_CFG_PATH} \\
    -expn {STAGE2_EXP_NAME_FOLD_0} --model convnext_small.fb_in22k_ft_in1k_384 --pretrained \\
    --initial-checkpoint {FINETUNE_INIT_CKPT_PATH_FOLD_0} \\
    --num-classes 1 --batch-size 8 --validation-batch-size 16 \\
    --opt sgd --lr 1e-3 --min-lr 5e-5 --sched cosine --warmup-lr 1e-3 \\
    --epochs 16 --warmup-epoch 1 --cooldown-epochs 1 \\
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \\
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \\
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \\
    --save-images --log-wandb --exp-kwargs fold_idx=0 num_sched_epochs=10 num_epochs=35 \\
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 17 18 \\
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max


# [STEP 6] finetune (stage 2) fold 1 + soft_pos = 0.9
# this will create new directory and save checkpoints to {os.path.abspath(os.path.join(SAVE_CKPT_DIR, STAGE2_EXP_NAME_FOLD_1))} 
python3 src/pytorch-image-models/train_exp.py \\
    -f {EXP_CFG_PATH} \\
    -expn {STAGE2_EXP_NAME_FOLD_1} --model convnext_small.fb_in22k_ft_in1k_384 --pretrained \\
    --initial-checkpoint {FINETUNE_INIT_CKPT_PATH_FOLD_1} \\
    --num-classes 1 --batch-size 8 --validation-batch-size 16 \\
    --opt sgd --lr 1e-3 --min-lr 5e-5 --sched cosine --warmup-lr 1e-3 \\
    --epochs 16 --warmup-epoch 1 --cooldown-epochs 1 \\
    --no-aug --crop-pct 1.0 --bce-loss --smoothing 0.1 --log-interval 500 \\
    --workers 8 --input-size 3 2048 1024 --eval-metric gbmean_best_pfbeta \\
    --checkpoint-hist 100 --drop 0.5 --drop-path 0.2 --amp --amp-impl native \\
    --save-images --log-wandb --exp-kwargs fold_idx=1 num_sched_epochs=10 num_epochs=35 \\
    start_ratio=0.1429 end_ratio=0.1429 one_pos_mode=True --dense-ckpt-epochs 17 18 \\
    --dense-ckpt-bins 2 --model-ema --model-ema-decay 0.9998 --gp max
    """

    print(TRAIN_SCRIPT_CONTENT)

    with open(args.save_path, 'w') as f:
        f.write(TRAIN_SCRIPT_CONTENT)
    print(f'Writed train bash script to {args.save_path}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
