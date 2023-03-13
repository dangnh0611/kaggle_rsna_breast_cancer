import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
# Must import tensorrt before torch: https://github.com/NVIDIA/TensorRT/issues/1945
import tensorrt
import warnings
warnings.filterwarnings("ignore")
import argparse
import gc
import multiprocessing as mp
import os
import shutil
import time

import albumentations as A
import cv2
import dicomsdl
import numpy as np
import nvidia.dali as dali
import pandas as pd
import pydicom
import torch
from albumentations.pytorch.transforms import ToTensorV2
from joblib import Parallel, delayed
from torch2trt import TRTModule
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from settings import SETTINGS
from src.roi_det import roi_extract
from src.utils.dicom import (DicomsdlMetadata, PydicomMetadata, feed_ndarray,
                             min_max_scale)
from src.utils.misc import load_img_from_file, save_img_to_file
from src.utils.windowing import apply_windowing

BATCH_SIZE = 2
THRES = 0.31
AUTO_THRES = False
AUTO_THRES_PERCENTILE = 0.97935

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
# model
MODEL_INPUT_SIZE = [2048, 1024]
N_CPUS = min(4, mp.cpu_count())
N_CHUNKS = 2
RM_DONE_CHUNK = True

#############################################


def resize_and_pad(img, input_size=MODEL_INPUT_SIZE):
    input_h, input_w = input_size
    ori_h, ori_w = img.shape[:2]
    ratio = min(input_h / ori_h, input_w / ori_w)
    # resize
    img = F.interpolate(img.view(1, 1, ori_h, ori_w),
                        mode="bilinear",
                        scale_factor=ratio,
                        recompute_scale_factor=True)[0, 0]
    # padding
    padded_img = torch.zeros((input_h, input_w),
                             dtype=img.dtype,
                             device='cuda')
    cur_h, cur_w = img.shape
    y_start = (input_h - cur_h) // 2
    x_start = (input_w - cur_w) // 2
    padded_img[y_start:y_start + cur_h, x_start:x_start + cur_w] = img
    padded_img = padded_img.unsqueeze(-1).expand(-1, -1, 3)
    return padded_img


class _JStreamExternalSource:

    def __init__(self, dcm_paths, batch_size=1):
        self.dcm_paths = dcm_paths
        self.len = len(dcm_paths)
        self.batch_size = batch_size

    def __call__(self, batch_info):
        idx = batch_info.iteration
        # print('IDX:', batch_info.iteration, batch_info.epoch_idx)
        start = idx * self.batch_size
        end = min(self.len, start + self.batch_size)
        if end <= start:
            raise StopIteration()

        batch_dcm_paths = self.dcm_paths[start:end]
        j_streams = []
        inverts = []
        windowing_params = []
        voilut_funcs = []

        for dcm_path in batch_dcm_paths:
            ds = pydicom.dcmread(dcm_path)
            pixel_data = ds.PixelData
            offset = pixel_data.find(
                SUID2HEADER[ds.file_meta.TransferSyntaxUID])
            j_stream = np.array(bytearray(pixel_data[offset:]), np.uint8)
            invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
            meta = PydicomMetadata(ds)
            windowing_param = np.array(
                [meta.window_centers, meta.window_widths], np.float16)
            voilut_func = VOILUT_FUNCS_MAP[meta.voilut_func]
            j_streams.append(j_stream)
            inverts.append(invert)
            windowing_params.append(windowing_param)
            voilut_funcs.append(voilut_func)
        return j_streams, np.array(inverts, dtype=np.bool_), \
            windowing_params, np.array(voilut_funcs, dtype=np.uint8)


@dali.pipeline_def
def _dali_pipeline(eii):
    jpeg, invert, windowing_param, voilut_func = dali.fn.external_source(
        source=eii,
        num_outputs=4,
        dtype=[
            dali.types.UINT8, dali.types.BOOL, dali.types.FLOAT16,
            dali.types.UINT8
        ],
        batch=True,
        batch_info=True,
        parallel=True)
    ori_img = dali.fn.experimental.decoders.image(
        jpeg,
        device='mixed',
        output_type=dali.types.ANY_DATA,
        dtype=dali.types.UINT16)
    return ori_img, invert, windowing_param, voilut_func


def decode_crop_save_dali(roi_yolox_engine_path,
                          dcm_paths,
                          save_paths,
                          save_backend='cv2',
                          batch_size=1,
                          num_threads=1,
                          py_num_workers=1,
                          py_start_method='fork',
                          device_id=0):

    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']
    num_dcms = len(dcm_paths)

    # dali to process with chunk in-memory
    external_source = _JStreamExternalSource(dcm_paths, batch_size=batch_size)
    pipe = _dali_pipeline(
        external_source,
        py_num_workers=py_num_workers,
        py_start_method=py_start_method,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        debug=False,
    )
    pipe.build()

    roi_extractor = roi_extract.RoiExtractor(engine_path=roi_yolox_engine_path,
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

    num_batchs = num_dcms // batch_size
    last_batch_size = batch_size
    if num_dcms % batch_size > 0:
        num_batchs += 1
        last_batch_size = num_dcms % batch_size

    cur_idx = -1
    for _batch_idx in tqdm(range(num_batchs)):
        try:
            outs = pipe.run()
        except Exception as e:
            #             print('DALI exception occur:', e)
            print(
                f'Exception: One of {dcm_paths[_batch_idx * batch_size: (_batch_idx + 1) * batch_size]} can not be decoded.'
            )
            # ignore this batch and re-build pipeline
            if _batch_idx < num_batchs - 1:
                cur_idx += batch_size
                del external_source, pipe
                gc.collect()
                torch.cuda.empty_cache()
                external_source = _JStreamExternalSource(
                    dcm_paths[(_batch_idx + 1) * batch_size:],
                    batch_size=batch_size)
                pipe = _dali_pipeline(
                    external_source,
                    py_num_workers=py_num_workers,
                    py_start_method=py_start_method,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    device_id=device_id,
                    debug=False,
                )
                pipe.build()
            else:
                cur_idx += last_batch_size
            continue

        imgs = outs[0]
        inverts = outs[1]
        windowing_params = outs[2]
        voilut_funcs = outs[3]
        for j in range(len(inverts)):
            cur_idx += 1
            save_path = save_paths[cur_idx]
            img_dali = imgs[j]
            img_torch = torch.empty(img_dali.shape(),
                                    dtype=torch.int16,
                                    device='cuda')
            feed_ndarray(img_dali,
                         img_torch,
                         cuda_stream=torch.cuda.current_stream(device=0))
            # @TODO: test whether copy uint16 to int16 pointer is safe in this case
            if 0:
                img_np = img_dali.as_cpu().squeeze(-1)  # uint16
                print(type(img_np), img_np.shape)
                img_np = torch.from_numpy(img_np, dtype=torch.int16)
                diff = torch.max(torch.abs(img_np - img_torch))
                assert diff == 0, f'{img_torch.shape}, {img_np.shape}, {diff}'

            invert = inverts.at(j).item()
            windowing_param = windowing_params.at(j)
            voilut_func = voilut_funcs.at(j).item()
            voilut_func = VOILUT_FUNCS_INV_MAP[voilut_func]

            # YOLOX for ROI extraction
            img_yolox = min_max_scale(img_torch)
            img_yolox = (img_yolox * 255)  # float32
            if invert:
                img_yolox = 255 - img_yolox
            # YOLOX infer
            # who know if exception happen in hidden test ?
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
            if windowing_param.shape[1] != 0:
                default_window_center = windowing_param[0, 0]
                default_window_width = windowing_param[1, 0]
                crop = apply_windowing(crop,
                                       window_width=default_window_width,
                                       window_center=default_window_center,
                                       voi_func=voilut_func,
                                       y_min=0,
                                       y_max=255,
                                       backend='torch')
            # if no window center/width in dcm file
            # do simple min-max scaling
            else:
                print('No windowing param!')
                crop = min_max_scale(crop)
                crop = crop * 255
            if invert:
                crop = 255 - crop
            crop = resize_and_pad(crop, MODEL_INPUT_SIZE)
            crop = crop.to(torch.uint8)
            crop = crop.cpu().numpy()
            save_img_to_file(save_path, crop, backend=save_backend)


#     assert cur_idx == len(
#         save_paths) - 1, f'{cur_idx} != {len(save_paths) - 1}'
    try:
        del external_source, pipe, roi_extractor
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return


def decode_and_save_dali_parallel(
        roi_yolox_engine_path,
        dcm_paths,
        save_paths,
        save_backend='cv2',
        batch_size=1,
        num_threads=1,
        py_num_workers=1,
        py_start_method='fork',
        device_id=0,
        parallel_n_jobs=2,
        parallel_n_chunks=4,
        parallel_backend='joblib',  # joblib or multiprocessing
        joblib_backend='loky'):
    assert parallel_backend in ['joblib', 'multiprocessing']
    assert joblib_backend in ['threading', 'multiprocessing', 'loky']
    # py_num_workers > 0 means using multiprocessing worker
    # 'fork' multiprocessing after CUDA init is not work (we must use 'spawn' instead)
    # since our pipeline can be re-build (when a dicom can't be decoded on GPU),
    # 2 options:
    #       (py_num_workers = 0, py_start_method=?)
    #       (py_num_workers > 0, py_start_method = 'spawn')
    assert not (py_num_workers > 0 and py_start_method == 'fork')

    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return decode_crop_save_dali(roi_yolox_engine_path,
                                     dcm_paths,
                                     save_paths,
                                     save_backend=save_backend,
                                     batch_size=batch_size,
                                     num_threads=num_threads,
                                     py_num_workers=py_num_workers,
                                     py_start_method=py_start_method,
                                     device_id=device_id)
    else:
        num_samples = len(dcm_paths)
        num_samples_per_chunk = num_samples // parallel_n_chunks
        if num_samples % parallel_n_chunks > 0:
            num_samples_per_chunk += 1
        starts = [num_samples_per_chunk * i for i in range(parallel_n_chunks)]
        ends = [
            min(start + num_samples_per_chunk, num_samples) for start in starts
        ]
        if isinstance(device_id, list):
            assert len(device_id) == parallel_n_chunks
        elif isinstance(device_id, int):
            device_id = [device_id] * parallel_n_chunks

        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`, {parallel_n_chunks} chunks ...'
        )
        if parallel_backend == 'joblib':
            _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
                delayed(decode_crop_save_dali)(
                    roi_yolox_engine_path,
                    dcm_paths[start:end],
                    save_paths[start:end],
                    save_backend=save_backend,
                    batch_size=batch_size,
                    num_threads=num_threads,
                    py_num_workers=py_num_workers,  # ram_v3
                    py_start_method=py_start_method,
                    device_id=worker_device_id,
                ) for start, end, worker_device_id in zip(
                    starts, ends, device_id))
        else:  # manually start multiprocessing's processes
            workers = []
            daemon = False if py_num_workers > 0 else True
            for i in range(parallel_n_jobs):
                start = starts[i]
                end = ends[i]
                worker_device_id = device_id[i]
                worker = mp.Process(group=None,
                                    target=decode_crop_save_dali,
                                    args=(
                                        roi_yolox_engine_path,
                                        dcm_paths[start:end],
                                        save_paths[start:end],
                                    ),
                                    kwargs={
                                        'save_backend': save_backend,
                                        'batch_size': batch_size,
                                        'num_threads': num_threads,
                                        'py_num_workers': py_num_workers,
                                        'py_start_method': py_start_method,
                                        'device_id': worker_device_id,
                                    },
                                    daemon=daemon)
                workers.append(worker)
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
    return


def _single_decode_crop_save_sdl(roi_extractor,
                                 dcm_path,
                                 save_path,
                                 save_backend='cv2',
                                 index=0):
    dcm = dicomsdl.open(dcm_path)
    meta = DicomsdlMetadata(dcm)
    info = dcm.getPixelDataInfo()
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]

    ori_dtype = info['dtype']
    img = np.empty(shape, dtype=ori_dtype)
    dcm.copyFrameData(index, img)
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

    # apply voi lut
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
    crop = resize_and_pad(crop, MODEL_INPUT_SIZE)
    crop = crop.to(torch.uint8)
    crop = crop.cpu().numpy()
    save_img_to_file(save_path, crop, backend=save_backend)


def decode_crop_save_sdl(roi_yolox_engine_path,
                         dcm_paths,
                         save_paths,
                         save_backend='cv2'):
    assert len(dcm_paths) == len(save_paths)
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
        _single_decode_crop_save_sdl(roi_detector, dcm_paths[i], save_paths[i],
                                     save_backend)

    del roi_detector
    gc.collect()
    torch.cuda.empty_cache()
    return


def decode_crop_save_sdl_parallel(roi_yolox_engine_path,
                                  dcm_paths,
                                  save_paths,
                                  save_backend='cv2',
                                  parallel_n_jobs=2,
                                  parallel_n_chunks=4,
                                  joblib_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return decode_crop_save_sdl(roi_yolox_engine_path, dcm_paths,
                                    save_paths, save_backend)
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
            delayed(decode_crop_save_sdl)(roi_yolox_engine_path,
                                          dcm_paths[start:end],
                                          save_paths[start:end], save_backend)
            for start, end in zip(starts, ends))


def make_uid_transfer_dict(df, dcm_root_dir):
    machine_id_to_transfer = {}
    machine_id = df.machine_id.unique()
    for i in machine_id:
        row = df[df.machine_id == i].iloc[0]
        sample_dcm_path = os.path.join(dcm_root_dir, str(row.patient_id),
                                       f'{row.image_id}.dcm')
        dicom = pydicom.dcmread(sample_dcm_path)
        machine_id_to_transfer[i] = dicom.file_meta.TransferSyntaxUID
    return machine_id_to_transfer


class ValTransform:

    def __init__(self):
        self.transform_fn = A.Compose([ToTensorV2(transpose_mask=True)])

    def __call__(self, img):
        return self.transform_fn(image=img)['image']


class RSNADataset(Dataset):

    def __init__(self, df, img_root_dir, transform_fn=None):
        self.img_paths = []
        self.transform_fn = transform_fn
        self.df = df
        for i in tqdm(range(len(df))):
            patient_id = df.at[i, 'patient_id']
            image_id = df.at[i, 'image_id']
            img_name = f'{patient_id}@{image_id}.png'
            img_path = os.path.join(img_root_dir, img_name)
            self.img_paths.append(img_path)
        print(f'Done loading dataset with {len(self.img_paths)} samples.')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print('ERROR:', img_path)
        if self.transform_fn:
            img = self.transform_fn(img)
        return img

    def get_df(self):
        return self.df


def parse_args():
    parser = argparse.ArgumentParser(
        'Generate and write training bash script.')
    parser.add_argument('--mode',
                        type=str,
                        default='trained',
                        choices=['reproduce', 'trained', 'partial_reproduce'],
                        help='')
    parser.add_argument('--trt',
                        action='store_true',
                        help='Using TensorRT or not')
    args = parser.parse_args()
    return args


def main(args):
    # Mode
    # trained: YOLOX trained + Convnext trained
    # partial_reproduce: YOLOX trained + Convnext retrain
    # reproduce: YOLOX retrain + Convnext retrain

    if args.mode in ['trained', 'partial_reproduce']:
        ROI_YOLOX_ENGINE_PATH = os.path.join(SETTINGS.ASSETS_DIR, 'trained',
                                             'yolox_nano_416_roi_trt.pth')
    elif args.mode == 'reproduce':
        ROI_YOLOX_ENGINE_PATH = os.path.join(
            SETTINGS.MODEL_FINAL_SELECTION_DIR, 'yolox_nano_416_roi_trt.pth')

    if args.mode == 'trained':
        TORCH_MODEL_CKPT_PATHS = [
            os.path.join(SETTINGS.ASSETS_DIR, 'trained',
                         f'best_convnext_fold_{fold_idx}.pth.tar')
            for fold_idx in range(4)
        ]
        TRT_MODEL_PATH = os.path.join(
            SETTINGS.ASSETS_DIR, 'trained',
            'best_ensemble_convnext_small_batch2_fp32.engine')
    elif args.mode in ['reproduce', 'partial_reproduce']:
        MODEL_FINAL_SELECTION_DIR = SETTINGS.MODEL_FINAL_SELECTION_DIR
        TORCH_MODEL_CKPT_PATHS = [
            os.path.join(MODEL_FINAL_SELECTION_DIR,
                         f'best_convnext_fold_{fold_idx}.pth.tar')
            for fold_idx in range(4)
        ]
        TRT_MODEL_PATH = os.path.join(
            MODEL_FINAL_SELECTION_DIR,
            'best_ensemble_convnext_small_batch2_fp32.engine')
    else:
        raise ValueError()

    CSV_PATH = os.path.join(SETTINGS.RAW_DATA_DIR,
                            'rsna-breast-cancer-detection', 'test.csv')
    DCM_ROOT_DIR = os.path.join(SETTINGS.RAW_DATA_DIR,
                                'rsna-breast-cancer-detection', 'test_images')
    SAVE_IMG_ROOT_DIR = os.path.join(SETTINGS.TEMP_DIR, 'pngs')
    shutil.rmtree(SAVE_IMG_ROOT_DIR)

    ###########################################################################
    global_df = pd.read_csv(CSV_PATH)
    MACHINE_TO_SUID = make_uid_transfer_dict(global_df, DCM_ROOT_DIR)
    all_patients = list(global_df.patient_id.unique())
    num_patients = len(all_patients)

    num_patients_per_chunk = num_patients // N_CHUNKS + 1
    chunk_patients = [
        all_patients[num_patients_per_chunk * i:num_patients_per_chunk *
                     (i + 1)] for i in range(N_CHUNKS)
    ]
    print(f'PATIENT CHUNKS: {[len(c) for c in chunk_patients]}')

    pred_dfs = []
    for chunk_idx, chunk_patients in enumerate(chunk_patients):
        os.makedirs(SAVE_IMG_ROOT_DIR, exist_ok=True)
        df = global_df[global_df.patient_id.isin(chunk_patients)].reset_index(
            drop=True)
        print(
            f'Processing chunk {chunk_idx} with {len(chunk_patients)} patients, {len(df)} images'
        )
        if len(df) == 0:
            continue
        dcm_paths = []
        save_paths = []
        dali_dcm_paths = []
        dali_save_paths = []
        for i in range(len(df)):
            patient_id = df.at[i, 'patient_id']
            image_id = df.at[i, 'image_id']
            suid = MACHINE_TO_SUID[df.at[i, 'machine_id']]
            dcm_path = os.path.join(DCM_ROOT_DIR, str(patient_id),
                                    f'{image_id}.dcm')
            save_path = os.path.join(SAVE_IMG_ROOT_DIR,
                                     f'{patient_id}@{image_id}.png')
            # if os.path.isfile(save_path):
            #     continue
            dcm_paths.append(dcm_path)
            save_paths.append(save_path)
            if suid == J2K_SUID or suid == JLL_SUID:
                dali_dcm_paths.append(dcm_path)
                dali_save_paths.append(save_path)

        if 1:
            t0 = time.time()
            # try to decode all .90 and .70 with DALI
            decode_and_save_dali_parallel(
                ROI_YOLOX_ENGINE_PATH,
                dali_dcm_paths,
                dali_save_paths,
                save_backend='cv2',
                batch_size=1,
                num_threads=1,
                py_num_workers=0,
                py_start_method='fork',
                device_id=0,
                parallel_n_jobs=N_CPUS + 1,
                parallel_n_chunks=N_CPUS + 1,
                parallel_backend='joblib',  # joblib or multiprocessing
                joblib_backend='loky')
            gc.collect()
            torch.cuda.empty_cache()
            t1 = time.time()
            print(f'DALI done in {t1 - t0} sec')

            # CPU decode all others (exceptions) with dicomsdl
            done_img_names = os.listdir(SAVE_IMG_ROOT_DIR)
            save_img_names = [os.path.basename(p) for p in save_paths]
            remain_img_names = list(set(save_img_names) - set(done_img_names))
            remain_img_paths = [
                os.path.join(SAVE_IMG_ROOT_DIR, name)
                for name in remain_img_names
            ]
            remain_dcm_paths = []
            for name in remain_img_names:
                patient_id, image_id = os.path.basename(name).split(
                    '.')[0].split('@')
                remain_dcm_paths.append(
                    os.path.join(DCM_ROOT_DIR, patient_id, f'{image_id}.dcm'))
            num_remain = len(remain_dcm_paths)
            print(f'Number of undecoded files: {num_remain}')
            #         print(f'Remains: {remain_img_names}')
            if num_remain > 0:
                # 16 or just any > 0 number
                if num_remain > 32 * N_CPUS:
                    sdl_n_jobs = N_CPUS
                    sdl_n_chunks = N_CPUS
                else:
                    sdl_n_jobs = 1
                    sdl_n_chunks = 1
                decode_crop_save_sdl_parallel(ROI_YOLOX_ENGINE_PATH,
                                              remain_dcm_paths,
                                              remain_img_paths,
                                              save_backend='cv2',
                                              parallel_n_jobs=sdl_n_jobs,
                                              parallel_n_chunks=sdl_n_chunks,
                                              joblib_backend='loky')
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print('No remain files to decode.')
            t2 = time.time()
            print(f'SDL done in { t2 - t1} sec')
            print(f'TOTAL DECODING TIME: {t2 - t0} sec')

        dataset = RSNADataset(df,
                              SAVE_IMG_ROOT_DIR,
                              transform_fn=ValTransform())
        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=False,
        )

        if args.trt:
            model = TRTModule()
            model.load_state_dict(torch.load(TRT_MODEL_PATH))
            assert os.path.isfile(TRT_MODEL_PATH)
        else:
            from model import KFoldEnsembleModel
            model_info = {
                'model_name': 'convnext_small.fb_in22k_ft_in1k_384',
                'num_classes': 1,
                'in_chans': 3,
                'global_pool': 'max',
            }
            model = KFoldEnsembleModel(model_info, TORCH_MODEL_CKPT_PATHS)
            model.eval()
            model.cuda()

        all_probs = []
        with torch.inference_mode():
            for batch in tqdm(dataloader):
                batch = batch.cuda().float()
                probs = model(batch)
                probs = probs.cpu().numpy()
                all_probs.append(probs)

        # N * num_models
        all_probs = np.concatenate(all_probs, axis=0)
        all_probs = np.nan_to_num(all_probs, nan=0.0, posinf=None, neginf=None)
        all_probs = all_probs.mean(axis=-1)
        assert all_probs.shape[0] == len(df)

        df['preds'] = all_probs
        pred_dfs.append(df)
        print(f'DONE CHUNK {chunk_idx} with {len(df)} samples')
        del model
        gc.collect()
        torch.cuda.empty_cache()
        if RM_DONE_CHUNK:
            shutil.rmtree(SAVE_IMG_ROOT_DIR)
            print(f'Removed save image directory {SAVE_IMG_ROOT_DIR}')
        print('-----------------------------\n\n')

    pred_df = pd.concat(pred_dfs).reset_index(drop=True)
    if 'prediction_id' not in pred_df.columns:
        pred_df['prediction_id'] = pred_df.apply(
            lambda row: str(row.patient_id) + '_' + row.laterality, axis=1)
    submit_df = pred_df[['prediction_id', 'preds']]
    print(submit_df)
    submit_df = pred_df.groupby('prediction_id').mean()
    if AUTO_THRES:
        thres = np.quantile(submit_df['preds'].values, AUTO_THRES_PERCENTILE)
    else:
        thres = THRES
    print(f'BINARIZED USING THRESHOLD={thres}')
    submit_df['cancer'] = (submit_df['preds'].values > thres).astype(int)
    submit_df = submit_df['cancer']

    SAVE_SUBMISSION_CSV_PATH = os.path.join(SETTINGS.SUBMISSION_DIR,
                                            'submission.csv')
    os.makedirs(SETTINGS.SUBMISSION_DIR, exist_ok=True)
    print('Saving submission to', SAVE_SUBMISSION_CSV_PATH)
    submit_df.to_csv(SAVE_SUBMISSION_CSV_PATH)


if __name__ == '__main__':
    args = parse_args()
    main(args)