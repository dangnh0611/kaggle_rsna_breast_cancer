import os
import sys

import cv2
import numpy as np
import tensorrt as trt
import torch
from torch import nn
from torch2trt import torch2trt as torch2trt

# sys.path.append('/home/dangnh36/projects/.comp/rsna/src/pytorch-image-models/')
from timm.data import resolve_data_config
from timm.models import create_model

MODEL_NAME = 'tf_efficientnet_b4.ns_jft_in1k'
MODEL_CKPTS = [
#     '/kaggle/input/kaggle-rsna-ckpts/fold0_convnext-small_exp4_ratio8_ep11.pth.tar',
    '/kaggle/input/kaggle-rsna-ckpts/fold0_effb4_ep3.tar'
]
NUM_CLASSES = 2
IN_CHANS = 3
MEAN = np.array([0.485, 0.456, 0.406]) * 255
STD = np.array([0.229, 0.224, 0.225]) * 255
GLOBAL_POOL = 'avg'
BATCH_SIZE = 4


class KFoldEnsembleModel(nn.Module):

    def __init__(self, model_name, num_classes, in_chans, ckpt_paths):
        super(KFoldEnsembleModel, self).__init__()
        fmodels = []
        for i, ckpt_path in enumerate(ckpt_paths):
            fmodel = create_model(
                model_name,
                num_classes=num_classes,
                in_chans=in_chans,
                pretrained=False,
                checkpoint_path=ckpt_path,
                global_pool=GLOBAL_POOL,
            ).eval()
            # print(fmodel)
            data_config = resolve_data_config({}, model=fmodel)
            print('Data config:', data_config)
            fmodels.append(fmodel)
        self.fmodels = nn.ModuleList(fmodels)

        self.register_buffer('mean',
                             torch.FloatTensor(MEAN).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(STD).reshape(1, 3, 1, 1))

    def forward(self, x):
#         x = x.sub(self.mean).div(self.std)
        x = (x - self.mean) / self.std
        probs = []
        for fmodel in self.fmodels:
            logits = fmodel(x)
            prob = logits.softmax(dim=1)[:, 1]
            probs.append(prob)
        probs = torch.stack(probs, dim=1)
        return probs


def get_sample_batch():
    # return (torch.rand(8, 3, 2048, 1024) * 255).float().cuda()
    img_dir = '/kaggle/tmp/pngs'
    img_names = os.listdir(img_dir)
    imgs = []
    for img_name in img_names[:BATCH_SIZE]:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        imgs.append(img)
    # B H W 3
    batch = np.stack(imgs, axis=0)
    batch = torch.from_numpy(batch)
    batch = batch.permute(0, 3, 1, 2)
    batch = batch.cuda().float()
    return batch


if __name__ == '__main__':
    model = KFoldEnsembleModel(MODEL_NAME, NUM_CLASSES, IN_CHANS, MODEL_CKPTS)
    # print(model)
    model.eval()
    model.cuda()
    sample_batch = get_sample_batch()
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
            fp16_mode=True,
            max_workspace_size=1 << 32,
            strict_type_constraints=False,
            keep_network=True,
            use_onnx=False,
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
    torch.save(model_trt.state_dict(), 'fold0_effb4_ep3_th054_trt_fp16.pth')
    print('ALL DONE!')