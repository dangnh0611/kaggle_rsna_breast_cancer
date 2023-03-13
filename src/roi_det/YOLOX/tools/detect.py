#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from setproctitle import setproctitle
setproctitle("python3 detect.py")

import argparse
import os
import random
import warnings
from loguru import logger

import torch
_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import numpy as np
from yolox.core import launch
from yolox.exp import get_exp
from yolox.utils import (
    configure_module,
    configure_nccl,
    fuse_model,
    postprocess,
    get_local_rank,
    get_model_info,
    setup_logger,
    vis,
)

from torch2trt import TRTModule
from tqdm import tqdm
from yolox.data.datasets import COCO_CLASSES



def make_parser():
    parser = argparse.ArgumentParser("YOLOX Detect")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("--input", default=None, type=str, help="Input dir")
    parser.add_argument("--output", default=None, type=str, help="Output dir")

    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--workers", default=8, type=int, help="Dataloader num workers")
    parser.add_argument("--seed", default=42, type=int, help="eval seed")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--speed",
        dest="speed",
        default=False,
        action="store_true",
        help="speed test only.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True
    warnings.warn(
        "You have chosen to seed testing. This will turn on the CUDNN deterministic setting, "
    )


def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)


class TRTModel:
    def __init__(self, engine_path, hw = None, strides = None, exp = None):
        model = TRTModule()
        model.load_state_dict(torch.load(engine_path))
        self.model = model
        if hw is None or strides is None:
            assert exp is not None
            self._set_meta(exp)
        else:
            # [torch.Size([80, 80]), torch.Size([40, 40]), torch.Size([20, 20])]
            self.hw = hw
            # [8, 16, 32]
            self.strides = strides


    def _set_meta(self, exp):
        logger.info("Start probing model metadata..")
        # dummy infer
        torch_model = exp.get_model().cuda().eval()
        _dummy = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        torch_model(_dummy)
        # set attributes
        self.hw = torch_model.head.hw
        self.strides = torch_model.head.strides
        # cleanup
        del torch_model, _dummy
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info('Done probbing model metadata..')


    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs = torch.cat([
            (outputs[..., 0:2] + grids) * strides,
            torch.exp(outputs[..., 2:4]) * strides,
            outputs[..., 4:]
        ], dim=-1)
        return outputs


    def __call__(self, inputs):
        outputs =  self.model(inputs)
        outputs = self.decode_outputs(outputs, dtype=outputs.type())
        return outputs


class TorchModel:
    def __init__(self, exp, ckpt_path, rank, half, num_gpu = 1):
        is_distributed = num_gpu > 1

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
        logger.info("Model Structure:\n{}".format(str(model)))

        model.cuda(rank)
        model.eval()

        logger.info("Loading checkpoint from {}".format(ckpt_path))
        loc = "cuda:{}".format(rank)
        ckpt = torch.load(ckpt_path, map_location=loc)
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded checkpoint successfully.")

        if is_distributed:
            model = DDP(model, device_ids=[rank])

        # fuse conv + bn
        if args.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)
        
        # half
        model = model.eval()
        if half:
            logger.info("\tUsing FP16 precision (half)..")
            model = model.half()

        self.model = model
        self.tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor

    
    def __call__(self, inputs):
        inputs = inputs.type(self.tensor_type)
        return self.model(inputs)


class DirDataset(Dataset):
    def __init__(self, dir, input_size, interpolation = cv2.INTER_LINEAR):
        self.input_size = input_size
        self.interpolation = interpolation
        names = os.listdir(dir)
        paths = [os.path.join(dir, name) for name in names]
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        ori_img = cv2.imread(img_path)
        img, r = self.preprocess(ori_img)
        # img = torch.from_numpy(img).float()
        return img, r

    def preprocess(self, img, swap=(2, 0, 1)):
        input_size = self.input_size
        if len(img.shape) == 3:
            padded_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
        else:
            padded_img = np.full(input_size, 114, dtype=np.uint8)

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=self.interpolation,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        # padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        padded_img = padded_img.astype(np.float32)
        return padded_img, r


def visualize(img, bboxes, scores, classes, ratio, class_names = COCO_CLASSES):
    vis_res = vis(img, bboxes, scores, classes, 0.0, class_names)
    return vis_res



@logger.catch
def main(exp, args, num_gpu):
    seed_all(args.seed)
    # set environment variables for distributed training
    configure_nccl()
    cudnn.benchmark = True
    rank = get_local_rank()
    torch.cuda.set_device(rank)
    save_dir = os.path.join(exp.output_dir, args.experiment_name)
    if rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    setup_logger(save_dir, distributed_rank=rank, filename="val_log.txt", mode="a")
    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = None
    if args.trt:
        model = TRTModel(args.ckpt, hw = None, strides = None, exp = exp)
    else:
        model = TorchModel(exp, args.ckpt, rank, args.fp16, num_gpu)

    print(model)

    dataset = DirDataset(args.input, input_size = exp.test_size, interpolation = exp.interpolation)
    print('DATASET LENGTH:', len(dataset))
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=False,
                     num_workers= args.workers, pin_memory=True, drop_last=False, pin_memory_device='cuda')

    # for _ in range(10):
    #     dummy_input = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
    #     outputs = model(dummy_input)
    #     outputs = postprocess(
    #                 outputs, exp.num_classes, exp.test_conf, exp.nmsthre
    #             )
    #     print(outputs)

    img_paths = dataset.paths

    output_dir = args.output
    save_viz_dir = os.path.join(output_dir, 'viz')
    save_coco_txt_dir = os.path.join(output_dir, 'coco_txt')
    save_coco_txt_conf_dir = os.path.join(output_dir, 'coco_txt_with_conf')
    save_miss_dir = os.path.join(output_dir, 'miss')
    save_fail_dir = os.path.join(output_dir, 'fail')
    for dir_path in [save_viz_dir, save_coco_txt_dir, save_coco_txt_conf_dir, save_miss_dir, save_fail_dir]:
        os.makedirs(dir_path, exist_ok= True)
    idx = -1
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_imgs, batch_ratios = batch
            batch_ratios = batch_ratios.cpu().numpy()
            batch_imgs = batch_imgs.cuda()
            outputs = model(batch_imgs)
            outputs = postprocess(
                        outputs, exp.num_classes, exp.test_conf, exp.nmsthre
                    )
        
            for i, output in enumerate(outputs):
                idx += 1 
                img_path = img_paths[idx]
                img_name = os.path.basename(img_path)
                save_path = os.path.join(save_viz_dir, img_name)
                txt_name = ''.join(img_name.split('.')[:-1]) + '.txt'
                save_coco_txt_path = os.path.join(save_coco_txt_dir, txt_name)
                save_coco_txt_conf_path = os.path.join(save_coco_txt_conf_dir, txt_name)
                save_miss_path = os.path.join(save_miss_dir, img_name)
                save_fail_path = os.path.join(save_fail_dir, img_name)

                raw_img = cv2.imread(img_path)
                ori_img_h, ori_img_w = raw_img.shape[:2]
                ratio = batch_ratios[i]
                
                if output is None:
                    cv2.imwrite(save_miss_path, raw_img)
                    continue

                output = output.cpu().numpy()
                # real coord
                bboxes = output[:, 0:4].copy()
                # preprocessing: resize
                bboxes /= ratio

                coco_bboxes = bboxes.copy()
                # xyxy --> ltwh --> xywh
                coco_bboxes[:, 2:] = coco_bboxes[:, 2:] - coco_bboxes[:, :2]
                coco_bboxes[:, :2] += coco_bboxes[:, 2:] / 2
                # norm to 0-1
                coco_bboxes[:, 0] /= ori_img_w
                coco_bboxes[:, 1] /= ori_img_h
                coco_bboxes[:, 2] /= ori_img_w
                coco_bboxes[:, 3] /= ori_img_h

                classes = output[:, 6]
                scores = output[:, 4] * output[:, 5]

                # save coco txt
                for cls, xywh, box_conf, cls_conf in zip(classes, coco_bboxes, output[:, 4], output[:, 5]):
                    line = (cls, *xywh, box_conf, cls_conf, ori_img_h, ori_img_w)  
                    with open(save_coco_txt_conf_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    line = (cls, *xywh)
                    with open(save_coco_txt_path, 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

                # save viz images
                viz = visualize(raw_img, bboxes, scores, classes, ratio, class_names = ['ROI'])
                if len(output) == 0:
                    cv2.imwrite(save_miss_path, viz)
                elif len(output) > 1:
                    cv2.imwrite(save_fail_path, viz)
                else:
                    # cv2.imwrite(save_path, viz)
                    pass


    assert idx == len(img_paths) - 1
            

    




    

if __name__ == "__main__":
    configure_module()
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = torch.cuda.device_count() if args.devices is None else args.devices
    assert num_gpu <= torch.cuda.device_count()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args, num_gpu),
    )



# python -m yolox.tools.detect -f exps/example/custom/yolox_nano_bre_416.py \
#     -c YOLOX_outputs/yolox_nano_bre_416/best_ckpt.pth -b 16 -d 1 --conf 0.5 --nms 0.9 \
#     --tsize 416 --speed --fuse --workers 2 --input image_dir --output runs/detect/yolox_nano_bre_416