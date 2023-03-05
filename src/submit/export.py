import warnings

warnings.filterwarnings("ignore")
import os

from metrics import compute_metrics

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import ctypes
import gc
import importlib
import multiprocessing as mp
import os
import shutil

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
from nvidia.dali import types
from nvidia.dali.backend import TensorGPU, TensorListGPU
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time


BATCH_SIZE = 4
THRES = 0.52
AUTO_THRES = True
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

MODE = 'LOCAL-ALL'
assert MODE in ['LOCAL-VAL', 'KAGGLE-VAL', 'KAGGLE-TEST', 'LOCAL-ALL']

if MODE == 'KAGGLE-VAL':
    ROI_YOLOX_ENGINE_PATH = '/kaggle/input/kaggle-rsna-ckpts/yolox_nano_bre_416_v2_trt.pth'
    CSV_PATH = '/kaggle/input/kaggle-rsna-ckpts/val_fold_0.csv'
    DCM_ROOT_DIR = '/kaggle/input/rsna-breast-cancer-detection/train_images'
    SAVE_IMG_ROOT_DIR = '/kaggle/tmp/pngs'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = False
elif MODE == 'KAGGLE-TEST':
    ROI_YOLOX_ENGINE_PATH = '/kaggle/input/kaggle-rsna-ckpts/yolox_nano_bre_416_v2_trt.pth'
    CSV_PATH = '/kaggle/input/rsna-breast-cancer-detection/test.csv'
    DCM_ROOT_DIR = '/kaggle/input/rsna-breast-cancer-detection/test_images'
    SAVE_IMG_ROOT_DIR = '/kaggle/tmp/pngs'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = True
elif MODE == 'LOCAL-VAL':
    ROI_YOLOX_ENGINE_PATH = '../roi_det/YOLOX/YOLOX_outputs/yolox_nano_bre_416/model_trt.pth'
    CSV_PATH = '../../datasets/cv/v1/val_fold_0.csv'
    DCM_ROOT_DIR = '../../datasets/train_images/'
    SAVE_IMG_ROOT_DIR = './temp_save'
    N_CHUNKS = 2
    N_CPUS = 2
    RM_DONE_CHUNK = False
elif MODE == 'LOCAL-ALL':
    ROI_YOLOX_ENGINE_PATH = '../roi_det/YOLOX/YOLOX_outputs/yolox_nano_bre_416/model_trt.pth'
    CSV_PATH = '../../datasets/train.csv'
    DCM_ROOT_DIR = '../../datasets/train_images/'
    SAVE_IMG_ROOT_DIR = '/raid/.comp/export_uint16_png_v2'
    N_CHUNKS = 1
    N_CPUS = 1
    RM_DONE_CHUNK = False

# DALI patch for INT16 support
################################################################################
DALI2TORCH_TYPES = {
    types.DALIDataType.FLOAT: torch.float32,
    types.DALIDataType.FLOAT64: torch.float64,
    types.DALIDataType.FLOAT16: torch.float16,
    types.DALIDataType.UINT8: torch.uint8,
    types.DALIDataType.INT8: torch.int8,
    types.DALIDataType.UINT16: torch.int16,
    types.DALIDataType.INT16: torch.int16,
    types.DALIDataType.INT32: torch.int32,
    types.DALIDataType.INT64: torch.int64
}


# @TODO: dangerous to copy from UINT16 to INT16 (memory layout?)
# little/big endian ?
# @TODO: faster reuse memory without copying: https://github.com/NVIDIA/DALI/issues/4126
def feed_ndarray(dali_tensor, arr, cuda_stream=None):
    """
    Copy contents of DALI tensor to PyTorch's Tensor.

    Parameters
    ----------
    `dali_tensor` : nvidia.dali.backend.TensorCPU or nvidia.dali.backend.TensorGPU
                    Tensor from which to copy
    `arr` : torch.Tensor
            Destination of the copy
    `cuda_stream` : torch.cuda.Stream, cudaStream_t or any value that can be cast to cudaStream_t.
                    CUDA stream to be used for the copy
                    (if not provided, an internal user stream will be selected)
                    In most cases, using pytorch's current stream is expected (for example,
                    if we are copying to a tensor allocated with torch.zeros(...))
    """
    dali_type = DALI2TORCH_TYPES[dali_tensor.dtype]

    assert dali_type == arr.dtype, (
        "The element type of DALI Tensor/TensorList"
        " doesn't match the element type of the target PyTorch Tensor: "
        "{} vs {}".format(dali_type, arr.dtype))
    assert dali_tensor.shape() == list(arr.size()), \
        ("Shapes do not match: DALI tensor has size {0}, but PyTorch Tensor has size {1}".
            format(dali_tensor.shape(), list(arr.size())))
    cuda_stream = types._raw_cuda_stream(cuda_stream)

    # turn raw int to a c void pointer
    c_type_pointer = ctypes.c_void_p(arr.data_ptr())
    if isinstance(dali_tensor, (TensorGPU, TensorListGPU)):
        stream = None if cuda_stream is None else ctypes.c_void_p(cuda_stream)
        dali_tensor.copy_to_external(c_type_pointer, stream, non_blocking=True)
    else:
        dali_tensor.copy_to_external(c_type_pointer)
    return arr


################################################################################


class PydicomMetadata:

    def __init__(self, ds):
        if "WindowWidth" not in ds or "WindowCenter" not in ds:
            self.window_widths = []
            self.window_centers = []
        else:
            ww = ds['WindowWidth']
            wc = ds['WindowCenter']
            self.window_widths = [float(e)
                                  for e in ww] if ww.VM > 1 else [float(ww.value)]

            self.window_centers = [float(e) for e in wc
                                   ] if wc.VM > 1 else [float(wc.value)]
        
        # if nan --> LINEAR
        self.voilut_func = str(ds.get('VOILUTFunction', 'LINEAR')).upper()
        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        assert len(self.window_widths) == len(self.window_centers)


class DicomsdlMetadata:

    def __init__(self, ds):
        self.window_widths = ds.WindowWidth
        self.window_centers = ds.WindowCenter
        if self.window_widths is None or self.window_centers is None:
            self.window_widths = []
            self.window_centers = []
        else:
            try:
                if not isinstance(self.window_widths, list):
                    self.window_widths = [self.window_widths]
                self.window_widths = [float(e) for e in self.window_widths]
                if not isinstance(self.window_centers, list):
                    self.window_centers = [self.window_centers]
                self.window_centers = [float(e) for e in self.window_centers]
            except:
                self.window_widths = []
                self.window_centers = []
                
        # if nan --> LINEAR
        self.voilut_func = ds.VOILUTFunction
        if self.voilut_func is None:
            self.voilut_func = 'LINEAR'
        else:
            self.voilut_func = str(self.voilut_func).upper()
        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        assert len(self.window_widths) == len(self.window_centers)


def min_max_scale(img):
    maxv = img.max()
    minv = img.min()
    if maxv > minv:
        return (img - minv) / (maxv - minv)
    else:
        return img - minv  # ==0


# DEPRECATED: too slow :D
# from Pydicom's source
def apply_windowing_np(arr,
                       window_width=None,
                       window_center=None,
                       voi_func='LINEAR',
                       y_min=0,
                       y_max=255):
    print('WARNING: Deprecated. Using apply_windowing_np_v2() instead.')
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.astype(np.float64)
    arr = arr.astype(np.float32)

    if voi_func in ['LINEAR', 'LINEAR_EXACT']:
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1
        below = arr <= (window_center - window_width / 2)
        above = arr > (window_center + window_width / 2)
        between = np.logical_and(~below, ~above)

        arr[below] = y_min
        arr[above] = y_max
        if between.any():
            arr[between] = ((
                (arr[between] - window_center) / window_width + 0.5) * y_range
                            + y_min)
    elif voi_func == 'SIGMOID':
        arr = y_range / (1 +
                         np.exp(-4 *
                                (arr - window_center) / window_width)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def apply_windowing_np_v2(arr,
                          window_width=None,
                          window_center=None,
                          voi_func='LINEAR',
                          y_min=0,
                          y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.astype(np.float64)
    arr = arr.astype(np.float32)

    if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1

        # simple trick to improve speed
        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        arr = arr * s + b
        arr = np.clip(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        # simple trick to improve speed
        s = -4 / window_width
        arr = y_range / (1 + np.exp((arr - window_center) * s)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def apply_windowing_torch(arr,
                          window_width=None,
                          window_center=None,
                          voi_func='LINEAR',
                          y_min=0,
                          y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed (default) or just float32 ?
    # arr = arr.double()
    arr = arr.float()

    if voi_func == 'LINEAR' or voi_func == 'LINEAR_EXACT':
        # PS3.3 C.11.2.1.2.1 and C.11.2.1.3.2
        if voi_func == 'LINEAR':
            if window_width < 1:
                raise ValueError(
                    "The (0028,1051) Window Width must be greater than or "
                    "equal to 1 for a 'LINEAR' windowing operation")
            window_center -= 0.5
            window_width -= 1

        # simple trick to improve speed
        s = y_range / window_width
        b = (-window_center / window_width + 0.5) * y_range + y_min
        arr = arr * s + b
        arr = torch.clamp(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        # simple trick to improve speed
        s = -4 / window_width
        arr = y_range / (1 + torch.exp((arr - window_center) * s)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


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
    padded_img[y_start:y_start + cur_h, x_start: x_start + cur_w] = img
    padded_img = padded_img.unsqueeze(-1).expand(-1, -1, 3)
    return padded_img


def save_img_to_file(save_path, img, backend='cv2'):
    file_ext = os.path.basename(save_path).split('.')[-1]
    if backend == 'cv2':
        if img.dtype == np.uint16:
            # https://docs.opencv.org/3.4/d4/da8/group__imgcodecs.html#gabbc7ef1aa2edfaa87772f1202d67e0ce
            assert file_ext in ['png', 'jp2', 'tiff', 'tif']
            cv2.imwrite(save_path, img)
        elif img.dtype == np.uint8:
            cv2.imwrite(save_path, img)
        else:
            raise ValueError(
                '`cv2` backend only support uint8 or uint16 images.')
    elif backend == 'np':
        assert file_ext == 'npy'
        np.save(save_path, img)
    else:
        raise ValueError(f'Unsupported backend `{backend}`.')


def load_img_from_file(img_path, backend='cv2'):
    if backend == 'cv2':
        return cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
    elif backend == 'np':
        return np.load(img_path)
    else:
        raise ValueError()


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
            
#             for ss in [10152, 10315, 10342, 1036, 10432, 10439, 12678]:
#                 if str(ss) in dcm_path:
#                     offset += 2
#             if idx % 5 == 4:
#                 offset += 2
                    
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
#         return j_streams, inverts, windowing_params, voilut_funcs
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


def decode_crop_save_dali(dcm_paths,
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

            img = img_torch.cpu().numpy().astype(np.uint16)
            save_img_to_file(save_path, img, backend=save_backend)

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
        dcm_paths,
        save_paths,
        save_backend='cv2',
        batch_size=1,
        num_threads=1,
        py_num_workers=1,
        py_start_method='fork',
        device_id=0,
        parallel_n_jobs=2,
        parallel_n_chunks = 4,
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
        return decode_crop_save_dali(dcm_paths,
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
            min(start + num_samples_per_chunk, num_samples)
            for start in starts
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

    # ori_dtype = info['dtype']
    # img = np.empty(shape, dtype=ori_dtype)
    # assert img.dtype == np.uint16, f'{img.dtype}'
    # dcm.copyFrameData(index, img)
    # img_torch = torch.from_numpy(img.astype(np.int16)).cuda()

    img = np.empty(shape, dtype=np.int16)
    dcm.copyFrameData(index, img)
    img_torch = torch.from_numpy(img).cuda()  # int16
    img = img_torch.cpu().numpy().astype(np.uint16)
    save_img_to_file(save_path, img, backend=save_backend)


def decode_crop_save_sdl(dcm_paths, save_paths, save_backend='cv2'):
    assert len(dcm_paths) == len(save_paths)
    roi_detector = None
    for i in tqdm(range(len(dcm_paths))):
        _single_decode_crop_save_sdl(roi_detector, dcm_paths[i], save_paths[i],
                                     save_backend)

    del roi_detector
    gc.collect()
    torch.cuda.empty_cache()
    return


def decode_crop_save_sdl_parallel(dcm_paths,
                                  save_paths,
                                  save_backend='cv2',
                                  parallel_n_jobs=2,
                                  parallel_n_chunks = 4,
                                  joblib_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return decode_crop_save_sdl(dcm_paths, save_paths, save_backend)
    else:
        num_samples = len(dcm_paths)
        num_samples_per_chunk = num_samples // parallel_n_chunks
        if num_samples % parallel_n_chunks > 0:
            num_samples_per_chunk += 1
        starts = [num_samples_per_chunk * i for i in range(parallel_n_chunks)]
        ends = [
            min(start + num_samples_per_chunk, num_samples)
            for start in starts
        ]

        print(
            f'Starting {parallel_n_jobs} jobs with backend `{joblib_backend}`, {parallel_n_chunks} chunks...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
            delayed(decode_crop_save_sdl)(dcm_paths[start:end],
                                          save_paths[start:end], save_backend)
            for start, end in zip(starts, ends))

######################################################
# MAIN CODE

if MODE == 'KAGGLE-TEST':
    global_df = pd.read_csv(CSV_PATH)
else:
    global_df = pd.read_csv(CSV_PATH)
all_patients = list(global_df.patient_id.unique())
num_patients = len(all_patients)

num_patients_per_chunk = num_patients // N_CHUNKS + 1
chunk_patients = [
    all_patients[num_patients_per_chunk * i:num_patients_per_chunk * (i + 1)]
    for i in range(N_CHUNKS)
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
    for i in range(len(df)):
        patient_id = df.at[i, 'patient_id']
        image_id = df.at[i, 'image_id']
        dcm_path = os.path.join(DCM_ROOT_DIR, str(patient_id),
                                f'{image_id}.dcm')
        save_path = os.path.join(SAVE_IMG_ROOT_DIR,
                                 f'{patient_id}@{image_id}.png')

        if os.path.isfile(save_path):
            continue
        else:
            dcm_paths.append(dcm_path)
            save_paths.append(save_path)
        # dcm_paths.append(dcm_path)
        # save_paths.append(save_path)

    if 1:
        t0 = time.time()
        # try to decode all with DALI
        decode_and_save_dali_parallel(
            dcm_paths,
            save_paths,
            save_backend='cv2',
            batch_size=1,
            num_threads=1,
            py_num_workers=0,
            py_start_method='fork',
            device_id=0,
            parallel_n_jobs=N_CPUS * 2,
            parallel_n_chunks = N_CPUS * 2,
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
            os.path.join(SAVE_IMG_ROOT_DIR, name) for name in remain_img_names
        ]
        remain_dcm_paths = []
        for name in remain_img_names:
            patient_id, image_id = os.path.basename(name).split('.')[0].split(
                '@')
            remain_dcm_paths.append(
                os.path.join(DCM_ROOT_DIR, patient_id, f'{image_id}.dcm'))
        num_remain = len(remain_dcm_paths)
        print(f'Number of undecoded files: {num_remain}')
#         print(f'Remains: {remain_img_names}')
        if num_remain > 0:
            # 16 or just any > 0 number
            if num_remain > 32 * N_CPUS:
                sdl_n_jobs = N_CPUS * 2
                sdl_n_chunks = N_CPUS * 2,
            else:
                sdl_n_jobs = 1
                sdl_n_chunks = 1
            decode_crop_save_sdl_parallel(remain_dcm_paths,
                                          remain_img_paths,
                                          save_backend='cv2',
                                          parallel_n_jobs=sdl_n_jobs,
                                          parallel_n_chunks = sdl_n_chunks,
                                          joblib_backend='loky')
            gc.collect()
            torch.cuda.empty_cache()
        else:
            print('No remain files to decode.')
        t2 = time.time()
        print(f'SDL done in { t2 - t1} sec')
        print(f'TOTAL DECODING TIME: {t2 - t0} sec')