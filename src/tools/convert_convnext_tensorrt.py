"""Sorry for this smelled code. I have no time refactor it :("""
import os
import sys
import time

import cv2
import numpy as np
import tensorrt as trt
import torch

try:
    import torch_tensorrt
except:
    print('torch_tensorrt is not installed yet.')
import argparse

from timm.data import resolve_data_config
from timm.models import create_model
from torch import nn
from torch2trt import TRTModule
from torch2trt import torch2trt as torch2trt
from tqdm import tqdm

from settings import SETTINGS
from src.submit.model import KFoldEnsembleModel

MODEL_INFO = {
    'model_name': 'convnext_small.fb_in22k_ft_in1k_384',
    'num_classes': 1,
    'in_chans': 3,
    'global_pool': 'max',
}

BATCH_SIZE = 2
PRECISION = 'fp32'
HALF_INPUT = False
TRT_BACKEND = 'torch2trt'
assert TRT_BACKEND in ['torch_tensorrt', 'torch2trt']
TRT_SAVE_NAME = f'best_ensemble_{MODEL_INFO["model_name"].split(".")[0]}_batch{BATCH_SIZE}_{PRECISION}.engine'


def get_sample_batch():
    return (torch.rand(BATCH_SIZE, 3, 2048, 1024) * 255).float().cuda()


def convert_with_torch2trt(torch_ckpt_paths, trt_save_path):
    if PRECISION == 'fp32':
        use_fp16 = False
    elif PRECISION == 'fp16':
        use_fp16 = True
    else:
        raise AssertionError()

    model = KFoldEnsembleModel(MODEL_INFO, torch_ckpt_paths)
    # print(model)
    model.eval()
    model.cuda()
    sample_batch = get_sample_batch()
    print('Sample input shape:', sample_batch.shape)
    with torch.inference_mode():
        prob = model(sample_batch)
        print(prob.shape, prob)

    # CONVERT TO TENSORRT
    with torch.inference_mode():
        print('START CONVERT')
        model_trt = torch2trt(
            model,
            inputs=[sample_batch],
            input_names=None,
            output_names=None,
            log_level=trt.Logger.ERROR,
            fp16_mode=use_fp16,
            max_workspace_size=1 << 32,
            strict_type_constraints=False,
            keep_network=True,
            use_onnx=True,
            default_device_type=trt.DeviceType.GPU,
            dla_core=0,
            gpu_fallback=True,
            device_types={},
            min_shapes=[(1, 3, 2048, 1024)],
            max_shapes=[(BATCH_SIZE, 3, 2048, 1024)],
            opt_shapes=[(BATCH_SIZE, 3, 2048, 1024)],
            onnx_opset=15,
            max_batch_size=BATCH_SIZE,
        )
    torch.save(model_trt.state_dict(), trt_save_path)
    print('ALL DONE!')


def convert_with_torch_tensorrt(torch_ckpt_paths, trt_save_path):
    if PRECISION == 'fp32':
        use_fp16 = False
    elif PRECISION == 'fp16':
        use_fp16 = True
    else:
        raise AssertionError()

    model = KFoldEnsembleModel(MODEL_INFO, torch_ckpt_paths)
    # print(model)
    model.eval()
    model.cuda()

    inputs = [
        torch_tensorrt.Input(
            min_shape=(1, 3, 2048, 1024),
            opt_shape=(BATCH_SIZE, 3, 2048, 1024),
            max_shape=(BATCH_SIZE, 3, 2048, 1024),
            dtype=torch.half if HALF_INPUT else torch.float32,
            #         format=torch_tensorrt.TensorFormat.NCHW
        )
    ]
    if use_fp16:
        enable_precisions = torch.half
    else:
        enable_precisions = torch.float32


#     sample_batch = inputs[0].example_tensor
    sample_batch = get_sample_batch()
    print('Sample input shape:', sample_batch.shape)
    with torch.inference_mode():
        prob = model(sample_batch)
        print(prob.shape, prob)

    # CONVERT TO TENSORRT
    with torch.inference_mode():
        print('START CONVERT')
        trt_model = torch_tensorrt.compile(
            model,
            inputs=inputs,
            enabled_precisions=enable_precisions,  # Run with FP32
            workspace_size=1 << 32,
        )
    torch.jit.save(trt_model, trt_save_path)
    print('ALL DONE!')


def convert(torch_ckpt_paths, trt_save_path):
    print(f'USING BACKEND {TRT_BACKEND}')
    if TRT_BACKEND == 'torch_tensorrt':
        convert_with_torch_tensorrt(torch_ckpt_paths, trt_save_path)
    elif TRT_BACKEND == 'torch2trt':
        convert_with_torch2trt(torch_ckpt_paths, trt_save_path)
    else:
        raise Exception()


def parse_args():
    parser = argparse.ArgumentParser(
        'Generate and write training bash script.')
    parser.add_argument('--mode',
                        type=str,
                        default='trained',
                        choices=['reproduce', 'trained'],
                        help='')
    args = parser.parse_args()
    return args


def main(args):
    if args.mode == 'trained':
        torch_model_ckpt_paths = [
            os.path.join(SETTINGS.ASSETS_DIR, 'trained',
                         f'best_convnext_fold_{fold_idx}.pth.tar')
            for fold_idx in range(4)
        ]
        trt_save_path = os.path.join(SETTINGS.ASSETS_DIR, 'trained',
                                     TRT_SAVE_NAME)

    elif args.mode == 'reproduce':
        MODEL_FINAL_SELECTION_DIR = SETTINGS.MODEL_FINAL_SELECTION_DIR
        torch_model_ckpt_paths = [
            os.path.join(MODEL_FINAL_SELECTION_DIR,
                         f'best_convnext_fold_{fold_idx}.pth.tar')
            for fold_idx in range(4)
        ]
        trt_save_path = os.path.join(MODEL_FINAL_SELECTION_DIR, TRT_SAVE_NAME)

    print('MODEL CHECKPOINTS:\n', torch_model_ckpt_paths)
    print('TENSORRT ENGINE WILL BE SAVED TO', trt_save_path)
    convert(torch_model_ckpt_paths, trt_save_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
