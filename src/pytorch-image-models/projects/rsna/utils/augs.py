import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from albumentations.augmentations.crops import functional as F
from albumentations.augmentations.geometric import functional as FGeometric
from albumentations.augmentations.utils import (_maybe_process_in_chunks,
                                                preserve_shape)
from albumentations.core.transforms_interface import (DualTransform,
                                                      ImageOnlyTransform)
from albumentations import random_utils


class _CustomBaseRandomSizedCropNoResize(DualTransform):
    # Base class for RandomSizedCrop and RandomResizedCrop

    def __init__(self, always_apply=False, p=1.0):
        super(_CustomBaseRandomSizedCropNoResize,
              self).__init__(always_apply, p)

    def apply(self,
              img,
              crop_height=0,
              crop_width=0,
              h_start=0,
              w_start=0,
              interpolation=cv2.INTER_LINEAR,
              **params):
        return F.random_crop(img, crop_height, crop_width, h_start, w_start)

    def apply_to_bbox(self,
                      bbox,
                      crop_height=0,
                      crop_width=0,
                      h_start=0,
                      w_start=0,
                      rows=0,
                      cols=0,
                      **params):
        return F.bbox_random_crop(bbox, crop_height, crop_width, h_start,
                                  w_start, rows, cols)

    def apply_to_keypoint(self,
                          keypoint,
                          crop_height=0,
                          crop_width=0,
                          h_start=0,
                          w_start=0,
                          rows=0,
                          cols=0,
                          **params):
        keypoint = F.keypoint_random_crop(keypoint, crop_height, crop_width,
                                          h_start, w_start, rows, cols)
        scale_x = self.width / crop_width
        scale_y = self.height / crop_height
        keypoint = FGeometric.keypoint_scale(keypoint, scale_x, scale_y)
        return keypoint


class CustomRandomSizedCropNoResize(_CustomBaseRandomSizedCropNoResize):
    """Torchvision's variant of crop a random part of the input and rescale it to some size.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        scale ((float, float)): range of size of the origin size cropped
        ratio ((float, float)): range of aspect ratio of the origin aspect ratio cropped
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
            self,
            scale=(0.08, 1.0),
            ratio=(0.75, 1.3333333333333333),
            always_apply=False,
            p=1.0,
    ):

        super(CustomRandomSizedCropNoResize,
              self).__init__(always_apply=always_apply, p=p)
        self.scale = scale
        self.ratio = ratio

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        area = img.shape[0] * img.shape[1]

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area *
                                    aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area /
                                    aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return "scale", "ratio"


# @TODO: support other native dtype: float, uint16,.. or support higher bit-depth (current 8 bits, LUT size = 256)
# cv2.LUT is tricky and other implementation with higher bit-depth can cause performance (speed) drop
# https://stackoverflow.com/questions/27098831/lookup-table-for-16-bit-mat-efficient-way
# https://stackoverflow.com/questions/71734861/opencv-python-lut-for-16bit-image
# https://stackoverflow.com/questions/28423701/efficient-way-to-loop-through-pixels-of-16-bit-mat-in-opencv
# https://answers.opencv.org/question/206755/lut-for-16bit-image/
@preserve_shape
def move_tone_curve_allow_float_img(img, low_y, high_y):
    """Rescales the relationship between bright and dark areas of the image by manipulating its tone curve.
    Args:
        img (numpy.ndarray): RGB or grayscale image.
        low_y (float): y-position of a Bezier control point used
            to adjust the tone curve, must be in range [0, 1]
        high_y (float): y-position of a Bezier control point used
            to adjust image tone curve, must be in range [0, 1]
    """
    input_dtype = img.dtype

    if low_y < 0 or low_y > 1:
        raise ValueError("low_shift must be in range [0, 1]")
    if high_y < 0 or high_y > 1:
        raise ValueError("high_shift must be in range [0, 1]")

    if input_dtype != np.uint8:
        # raise ValueError("Unsupported image type {}".format(input_dtype))
        assert input_dtype == np.float32
        img = (img * 255).astype(np.uint8)

    t = np.linspace(0.0, 1.0, 256)

    # Defines responze of a four-point bezier curve
    def evaluate_bez(t):
        return 3 * (1 - t)**2 * t * low_y + 3 * (1 - t) * t**2 * high_y + t**3

    evaluate_bez = np.vectorize(evaluate_bez)
    remapping = np.rint(evaluate_bez(t) * 255).astype(np.uint8)

    lut_fn = _maybe_process_in_chunks(cv2.LUT, lut=remapping)
    img = lut_fn(img)
    # convert back to float image in range [0, 1]
    if input_dtype != np.uint8:
        img = (img / 255).astype(np.float32)
    return img


class RandomToneCurveAllowFloatImage(ImageOnlyTransform):
    """Randomly change the relationship between bright and dark areas of the image by manipulating its tone curve.
    Args:
        scale (float): standard deviation of the normal distribution.
            Used to sample random distances to move two control points that modify the image's curve.
            Values should be in range [0, 1]. Default: 0.1
    Targets:
        image
    Image types:
        uint8, float32
    Note: if image in float32 dtype, first convert it to uint8, modify tone curve then convert back to float32.
    """

    def __init__(
        self,
        scale=0.1,
        always_apply=False,
        p=0.5,
    ):
        super(RandomToneCurveAllowFloatImage, self).__init__(always_apply, p)
        self.scale = scale

    def apply(self, image, low_y, high_y, **params):
        return move_tone_curve_allow_float_img(image, low_y, high_y)

    def get_params(self):
        return {
            "low_y":
            np.clip(random_utils.normal(loc=0.25, scale=self.scale), 0, 1),
            "high_y":
            np.clip(random_utils.normal(loc=0.75, scale=self.scale), 0, 1),
        }

    def get_transform_init_args_names(self):
        return ("scale", )
