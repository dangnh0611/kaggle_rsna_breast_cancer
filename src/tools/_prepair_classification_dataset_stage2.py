import warnings

warnings.filterwarnings("ignore")
import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import gc
import os
import time

import cv2
import dicomsdl
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi
from tqdm import tqdm

from src.roi_det import roi_extract
from src.utils.dicom import (DicomsdlMetadata, PydicomMetadata, feed_ndarray,
                             min_max_scale, percentile_min_max_scale)
from src.utils.misc import load_img_from_file, save_img_to_file
from src.utils.windowing import apply_windowing

__all__ = [
    'stage2_process_rsna', 'stage2_process_vindr', 'stage2_process_miniddsm',
    'stage2_process_cmmd', 'stage2_process_cddcesm', 'stage2_process_bmcd'
]

J2K_SUID = '1.2.840.10008.1.2.4.90'
J2K_HEADER = b"\x00\x00\x00\x0C"
JLL_SUID = '1.2.840.10008.1.2.4.70'
JLL_HEADER = b"\xff\xd8\xff\xe0"
SUID2HEADER = {J2K_SUID: J2K_HEADER, JLL_SUID: JLL_HEADER}
VOILUT_FUNCS_MAP = {'LINEAR': 0, 'LINEAR_EXACT': 1, 'SIGMOID': 2}
VOILUT_FUNCS_INV_MAP = {v: k for k, v in VOILUT_FUNCS_MAP.items()}

# roi detection
ROI_YOLOX_INPUT_SIZE = [416, 416]
ROI_YOLOX_CONF_THRES = 0.5
ROI_YOLOX_NMS_THRES = 0.9
ROI_YOLOX_HW = [(52, 52), (26, 26), (13, 13)]
ROI_YOLOX_STRIDES = [8, 16, 32]
ROI_AREA_PCT_THRES = 0.04


########################## COMPETITION DATA + VINDR + CMMD + BMCD ##########################
def _stage2_process_single_rsna(roi_extractor,
                                dcm_path,
                                save_path,
                                save_backend='cv2',
                                index=0):
    dcm = dicomsdl.open(dcm_path)
    ds = dcmread(dcm_path, stop_before_pixels=True)
    meta = DicomsdlMetadata(dcm)
    info = dcm.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]

    ori_dtype = info['dtype']
    img = np.empty(shape, dtype=ori_dtype)
    dcm.copyFrameData(index, img)
    # assert img.max() < 32768
    img_torch = torch.from_numpy(img.astype(np.int16)).cuda()

    # YOLOX for ROI extraction
    img_yolox = min_max_scale(img_torch)
    img_yolox = (img_yolox * 255)  # float32
    # @TODO: subtract on large array --> should move after F.interpolate()
    if meta.invert:
        img_yolox = 255 - img_yolox
    # YOLOX infer
    try:
        xyxy, _area_pct, _conf = roi_extractor.detect_single(img_yolox)
        if xyxy is not None:
            x0, y0, x1, y1 = xyxy
            crop = img_torch[y0:y1, x0:x1]
        else:
            crop = img_torch
    except:
        print('ROI extract exception!')
        crop = img_torch

    # apply windowing
    if meta.window_widths:
        crop = apply_windowing(crop,
                               window_width=meta.window_widths[0],
                               window_center=meta.window_centers[0],
                               voi_func=meta.voilut_func,
                               y_min=0,
                               y_max=255,
                               backend='torch')
    else:
        print('No windowing param!')
        crop = min_max_scale(crop)
        crop = crop * 255

    if meta.invert:
        crop = 255 - crop
    crop = crop.to(torch.uint8)
    crop = crop.cpu().numpy()
    save_img_to_file(save_path, crop, backend=save_backend)


# share the same pipeline
_stage2_process_single_vindr = _stage2_process_single_cmmd = _stage2_process_single_bmcd = _stage2_process_single_rsna


########################## MINI-DDSM ##########################
def _stage2_process_single_miniddsm(roi_extractor,
                                    dcm_path,
                                    save_path,
                                    save_backend='cv2',
                                    index=0):
    img = cv2.imread(dcm_path, cv2.IMREAD_ANYDEPTH)
    assert img.dtype == np.uint16

    # YOLOX for ROI extraction
    uint16_img = img.copy()
    img = percentile_min_max_scale(img)
    img_yolox = torch.from_numpy(img).cuda()  # float32
    img_yolox = (img_yolox * 255)  # float32
    # @TODO: subtract on large array --> should move after F.interpolate()
    # YOLOX infer
    try:
        xyxy, _area_pct, _conf = roi_extractor.detect_single(img_yolox)
        if xyxy is not None:
            x0, y0, x1, y1 = xyxy
            crop = uint16_img[y0:y1, x0:x1]
        else:
            crop = uint16_img
    except:
        print('ROI extract exception!')
        crop = uint16_img

    crop = percentile_min_max_scale(crop)
    crop = crop * 255
    crop = crop.astype(np.uint8)
    save_img_to_file(save_path, crop, backend=save_backend)


########################## CDD-CESM ##########################
def _stage2_process_single_cddcesm(roi_extractor,
                                   dcm_path,
                                   save_path,
                                   save_backend='cv2',
                                   index=0):
    img = cv2.imread(dcm_path, cv2.IMREAD_ANYDEPTH)
    assert img.dtype == np.uint8

    # YOLOX for ROI extraction
    img_yolox = torch.from_numpy(img).cuda().float()  # float32
    # @TODO: subtract on large array --> should move after F.interpolate()
    # YOLOX infer
    try:
        xyxy, _area_pct, _conf = roi_extractor.detect_single(img_yolox)
        if xyxy is not None:
            x0, y0, x1, y1 = xyxy
            crop = img[y0:y1, x0:x1]
        else:
            crop = img
    except:
        print('ROI extract exception!')
        crop = img

    crop = percentile_min_max_scale(crop)
    crop = crop * 255
    crop = crop.astype(np.uint8)
    save_img_to_file(save_path, crop, backend=save_backend)


def decode_crop_save(decode_crop_save_single_func,
                     roi_yolox_engine_path,
                     dcm_paths,
                     save_paths,
                     save_backend='cv2'):
    assert len(dcm_paths) == len(save_paths)
    print(f'Loading YOLOX from {roi_yolox_engine_path}')
    roi_detector = roi_extract.RoiExtractor(engine_path=roi_yolox_engine_path,
                                            input_size=ROI_YOLOX_INPUT_SIZE,
                                            num_classes=1,
                                            conf_thres=ROI_YOLOX_CONF_THRES,
                                            nms_thres=ROI_YOLOX_NMS_THRES,
                                            class_agnostic=False,
                                            area_pct_thres=ROI_AREA_PCT_THRES,
                                            hw=ROI_YOLOX_HW,
                                            strides=ROI_YOLOX_STRIDES,
                                            exp=None)
    print('ROI extractor (YOLOX) loaded!')
    for i in tqdm(range(len(dcm_paths))):
        decode_crop_save_single_func(roi_detector, dcm_paths[i], save_paths[i],
                                     save_backend)

    del roi_detector
    gc.collect()
    torch.cuda.empty_cache()
    return


def decode_crop_save_parallel(process_single_func,
                              roi_yolox_engine_path,
                              dcm_paths,
                              save_paths,
                              save_backend='cv2',
                              parallel_n_jobs=2,
                              parallel_n_chunks=4,
                              joblib_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return decode_crop_save(process_single_func, roi_yolox_engine_path,
                                dcm_paths, save_paths, save_backend)
    else:
        num_samples = len(dcm_paths)
        num_samples_per_chunk = num_samples // parallel_n_chunks
        if num_samples % parallel_n_chunks > 0:
            num_samples_per_chunk += 1
        starts = [num_samples_per_chunk * i for i in range(parallel_n_chunks)]
        ends = [
            min(start + num_samples_per_chunk, num_samples) for start in starts
        ]

        print(
            f'Starting {parallel_n_jobs} jobs with backend `{joblib_backend}`, {parallel_n_chunks} chunks...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
            delayed(decode_crop_save)
            (process_single_func, roi_yolox_engine_path, dcm_paths[start:end],
             save_paths[start:end], save_backend)
            for start, end in zip(starts, ends))


def stage2_process(
        decode_crop_save_single_func,
        stage1_image_extension,  # dcm, dicom, png, jpg, jpeg
        roi_yolox_engine_path,
        stage1_images_dir,
        cleaned_csv_path,
        cleaned_images_dir,
        n_jobs=8,
        n_chunks=8):
    os.makedirs(cleaned_images_dir, exist_ok=True)
    df = pd.read_csv(cleaned_csv_path)
    src_paths = []
    dst_paths = []
    for i in range(len(df)):
        patient_id = df.at[i, 'patient_id']
        image_id = df.at[i, 'image_id']
        src_path = os.path.join(stage1_images_dir, str(patient_id),
                                f'{image_id}.{stage1_image_extension}')
        dst_path = os.path.join(cleaned_images_dir,
                                f'{patient_id}@{image_id}.png')
        src_paths.append(src_path)
        dst_paths.append(dst_path)

    start = time.time()
    # CPU decode all others (exceptions) with dicomsdl
    decode_crop_save_parallel(decode_crop_save_single_func,
                                roi_yolox_engine_path,
                                src_paths,
                                dst_paths,
                                save_backend='cv2',
                                parallel_n_jobs=n_jobs,
                                parallel_n_chunks=n_chunks,
                                joblib_backend='loky')
    end = time.time()
    gc.collect()
    torch.cuda.empty_cache()
    print(f'Convert done in {end - start} sec')


def stage2_process_rsna(*args, **kwargs):
    return stage2_process(_stage2_process_single_rsna, 'dcm', *args, **kwargs)


def stage2_process_vindr(*args, **kwargs):
    return stage2_process(_stage2_process_single_vindr, 'dicom', *args,
                          **kwargs)


def stage2_process_miniddsm(*args, **kwargs):
    return stage2_process(_stage2_process_single_miniddsm, 'png', *args,
                          **kwargs)


def stage2_process_cmmd(*args, **kwargs):
    return stage2_process(_stage2_process_single_cmmd, 'dcm', *args, **kwargs)


def stage2_process_cddcesm(*args, **kwargs):
    return stage2_process(_stage2_process_single_cddcesm, 'jpg', *args,
                          **kwargs)


def stage2_process_bmcd(*args, **kwargs):
    return stage2_process(_stage2_process_single_bmcd, 'dcm', *args, **kwargs)
