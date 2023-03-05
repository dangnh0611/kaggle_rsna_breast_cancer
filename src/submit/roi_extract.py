import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import sys

import cv2
import numpy as np
import torch
import torchvision

sys.path.append('/kaggle/tmp/libs/')
from torch2trt import TRTModule
from torch.nn import functional as F

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]
_TORCH11X = (_TORCH_VER >= [1, 10])


def meshgrid(*tensors):
    if _TORCH11X:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)


def extract_roi_otsu(img, gkernel=(5, 5)):
    """WARNING: this function modify input image inplace."""
    ori_h, ori_w = img.shape[:2]
    # clip percentile: implant, white lines
    upper = np.percentile(img, 95)
    img[img > upper] = np.min(img)
    # Gaussian filtering to reduce noise (optional)
    if gkernel is not None:
        img = cv2.GaussianBlur(img, gkernel, 0)
    _, img_bin = cv2.threshold(img, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # dilation to improve contours connectivity
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (-1, -1))
    img_bin = cv2.dilate(img_bin, element)
    cnts, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return None, None, None
    areas = np.array([cv2.contourArea(cnt) for cnt in cnts])
    select_idx = np.argmax(areas)
    cnt = cnts[select_idx]
    area_pct = areas[select_idx] / (img.shape[0] * img.shape[1])
    x0, y0, w, h = cv2.boundingRect(cnt)
    # min-max for safety only
    # x0, y0, x1, y1
    x1 = min(max(int(x0 + w), 0), ori_w)
    y1 = min(max(int(y0 + h), 0), ori_h)
    x0 = min(max(int(x0), 0), ori_w)
    y0 = min(max(int(y0), 0), ori_h)
    return [x0, y0, x1, y1], area_pct, None


class RoiExtractor:

    def __init__(self,
                 engine_path,
                 input_size,
                 num_classes,
                 conf_thres=0.5,
                 nms_thres=0.9,
                 class_agnostic=False,
                 area_pct_thres=0.04,
                 hw=None,
                 strides=None,
                 exp=None):
        self.input_size = input_size
        self.input_h, self.input_w = input_size
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.class_agnostic = class_agnostic
        self.area_pct_thres = area_pct_thres

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
        assert exp is not None
        print("Start probing model metadata..")
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
        print('Done probbing model metadata..')

    def decode_outputs(self, outputs):
        dtype = outputs.type()
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

        outputs = torch.cat(
            [(outputs[..., 0:2] + grids) * strides,
             torch.exp(outputs[..., 2:4]) * strides, outputs[..., 4:]],
            dim=-1)
        return outputs

    def post_process(self,
                     pred,
                     conf_thres=0.5,
                     nms_thres=0.9,
                     class_agnostic=False):
        box_corner = pred.new(pred.shape)
        box_corner[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        box_corner[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        box_corner[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        box_corner[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
        pred[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(pred))]
        for i, image_pred in enumerate(pred):

            # If none are remaining => process next image
            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence
            class_conf, class_pred = torch.max(image_pred[:, 5:5 +
                                                          self.num_classes],
                                               1,
                                               keepdim=True)

            conf_mask = (image_pred[:, 4] * class_conf.squeeze() >=
                         conf_thres).squeeze()
            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float()), 1)
            detections = detections[conf_mask]
            if not detections.size(0):
                continue

            if class_agnostic:
                nms_out_index = torchvision.ops.nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    nms_thres,
                )
            else:
                nms_out_index = torchvision.ops.batched_nms(
                    detections[:, :4],
                    detections[:, 4] * detections[:, 5],
                    detections[:, 6],
                    nms_thres,
                )
            detections = detections[nms_out_index]
            if output[i] is None:
                output[i] = detections
            else:
                output[i] = torch.cat((output[i], detections))
        return output

    def preprocess_single(self, img: torch.Tensor):
        ori_h = img.size(0)
        ori_w = img.size(1)
        ratio = min(self.input_h / ori_h, self.input_w / ori_w)
        # resize
        resized_img = F.interpolate(img.view(1, 1, ori_h, ori_w),
                                    mode="bilinear",
                                    scale_factor=ratio,
                                    recompute_scale_factor=True)[0, 0]
        # padding
        padded_img = torch.full((self.input_h, self.input_w),
                                114,
                                dtype=resized_img.dtype,
                                device='cuda')
        padded_img[:resized_img.size(0), :resized_img.size(1)] = resized_img
        padded_img = padded_img.unsqueeze(-1).expand(-1, -1, 3)
        # HWC --> CHW
        padded_img = padded_img.permute(2, 0, 1)
        padded_img = padded_img.float()
        return padded_img, resized_img, ratio, ori_h, ori_w

    def detect_single(self, img):
        padded_img, resized_img, ratio, ori_h, ori_w = self.preprocess_single(
            img)
        padded_img = padded_img.unsqueeze(0)
        output = self.model(padded_img)
        output = self.decode_outputs(output)
        # x0, y0, x1, y1, box_conf, cls_conf, cls_id
        output = self.post_process(output, self.conf_thres, self.nms_thres)[0]
        if output is not None:
            output[:, :4] = output[:, :4] / ratio
            # re-compute: conf = box_conf * cls_conf
            output[:, 4] = output[:, 4] * output[:, 5]
            # select box with highest confident
            output = output[output[:, 4].argmax()]
            x0 = min(max(int(output[0]), 0), ori_w)
            y0 = min(max(int(output[1]), 0), ori_h)
            x1 = min(max(int(output[2]), 0), ori_w)
            y1 = min(max(int(output[3]), 0), ori_h)
            area_pct = (x1 - x0) * (y1 - y0) / (ori_h * ori_w)
            if area_pct >= self.area_pct_thres:
                # xyxy, area_pct, conf
                return [x0, y0, x1, y1], area_pct, output[4]

        # if YOLOX fail, try Otsu thresholding + find contours
        xyxy, area_pct, _ = extract_roi_otsu(
            resized_img.to(torch.uint8).cpu().numpy())
        # if both fail, use full frame
        if xyxy is not None:
            if area_pct >= self.area_pct_thres:
                print('ROI detection: using Otsu.')
                x0, y0, x1, y1 = xyxy
                x0 = min(max(int(x0 / ratio), 0), ori_w)
                y0 = min(max(int(y0 / ratio), 0), ori_h)
                x1 = min(max(int(x1 / ratio), 0), ori_w)
                y1 = min(max(int(y1 / ratio), 0), ori_h)
                return [x0, y0, x1, y1], area_pct, None
        print('ROI detection: both fail.')
        return None, area_pct, None
            