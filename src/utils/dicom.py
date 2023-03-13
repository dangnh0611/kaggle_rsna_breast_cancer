import warnings

warnings.filterwarnings("ignore")
import gc
import multiprocessing
import os
import shutil
import time

import cv2
import dicomsdl
import numpy as np
import nvidia.dali as dali
import pandas as pd
import pydicom
import torch
from joblib import Parallel, delayed
from pydicom.pixel_data_handlers.util import apply_voi_lut, pixel_dtype
from tqdm import tqdm

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import ctypes

from src.utils import misc as misc_utils
import nvidia.dali.types as types
from nvidia.dali import types
from nvidia.dali.backend import TensorGPU, TensorListGPU
from nvidia.dali.experimental import eager
from nvidia.dali.types import DALIDataType
from torch.nn import functional as F
from tqdm import tqdm

from src.utils.windowing import apply_windowing

try:
    import nvjpeg2k
except:
    # print('Fail to import nvjpeg2k')
    # print('If you want to use NVJPEG2K, please install first.')
    pass

J2K_SUID = '1.2.840.10008.1.2.4.90'
J2K_HEADER = b"\x00\x00\x00\x0C"
JLL_SUID = '1.2.840.10008.1.2.4.70'
JLL_HEADER = b"\xff\xd8\xff\xe0"
SUID2HEADER = {J2K_SUID: J2K_HEADER, JLL_SUID: JLL_HEADER}
VOILUT_FUNCS_MAP = {'LINEAR': 0, 'LINEAR_EXACT': 1, 'SIGMOID': 2}
VOILUT_FUNCS_INV_MAP = {v: k for k, v in VOILUT_FUNCS_MAP.items()}

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

TORCH_DTYPES = {
    'uint8': torch.uint8,
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
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


class PydicomMetadata:

    def __init__(self, ds):
        if "WindowWidth" not in ds or "WindowCenter" not in ds:
            self.window_widths = []
            self.window_centers = []
        else:
            ww = ds['WindowWidth']
            wc = ds['WindowCenter']
            self.window_widths = [float(e) for e in ww
                                  ] if ww.VM > 1 else [float(ww.value)]

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


#@TODO: percentile on both min-max ?
# this version is not correctly implemented, but used in the winning submission
def percentile_min_max_scale(img, pct=99):
    if isinstance(img, np.ndarray):
        maxv = np.percentile(img, pct) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv  # ==0
        ret = np.clip(ret, 0, 1)
    elif isinstance(img, torch.Tensor):
        maxv = torch.quantile(img, pct / 100) - 1
        minv = img.min()
        assert maxv >= minv
        if maxv > minv:
            ret = (img - minv) / (maxv - minv)
        else:
            ret = img - minv  # ==0
        ret = torch.clamp(ret, 0, 1)
    else:
        raise ValueError(
            'Invalid img type, should be numpy array or torch.Tensor')
    return ret


# @TODO: support windowing with more bits (>8)
def normalize_dicom_img(img,
                        invert,
                        save_dtype,
                        window_centers,
                        window_widths,
                        window_func,
                        window_index=0,
                        method='windowing',
                        force_use_gpu=True):
    assert method in ['min_max', 'min_max_pct', 'windowing']
    assert save_dtype in ['uint8', 'uint16', 'float16', 'float32', 'float64']
    if save_dtype == 'uint16':
        if invert:
            img = img.max() - img
        return img

    if method == 'windowing':
        assert save_dtype == 'uint8', 'Currently `windowing` normalization only support `uint8` save dtype.'
        # apply windowing
        if len(window_centers) > 0:
            window_center = window_centers[window_index]
            window_width = window_widths[window_index]
            windowing_backend = 'torch' if isinstance(
                img, torch.Tensor) or force_use_gpu else 'np_v2'
            img = apply_windowing(img,
                                  window_width=window_width,
                                  window_center=window_center,
                                  voi_func=window_func,
                                  y_min=0,
                                  y_max=255,
                                  backend=windowing_backend)
        # if no window center/width in dcm file
        # do simple min-max scaling
        else:
            print(
                'No windowing param, perform min-max scaling normalization instead.'
            )
            img = min_max_scale(img)
            img = img * 255
        img = img.to(torch.uint8)
        return img
    elif method == 'min_max':
        # [0, 1]
        img = min_max_scale(img)
    elif method == 'min_max_pct':
        # [0, 1]
        img = percentile_min_max_scale(img)
    else:
        raise ValueError(f'Invalid normalization method `{method}`')
    if invert:
        img = 1.0 - img
    if save_dtype == 'uint8':
        img = img * 255
    # convert to specified dtype: uint8, float
    if isinstance(img, np.ndarray):
        img = img.astype(save_dtype)
    elif isinstance(img, torch.Tensor):
        img = img.to(TORCH_DTYPES[save_dtype])

    return img


class _DaliJpegStreamExternalSource:

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

            # # for debugging purpose only
            # for ss in [10152, 10315, 10342, 1036, 10432, 10439, 12678]:
            #     if str(ss) in dcm_path:
            #         offset += 2
            # if idx % 5 == 4:
            #     offset += 2

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
def _dali_decode_pipeline_ram(eii):
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


def convert_with_dali_ram(dcm_paths,
                          save_paths,
                          save_backend='cv2',
                          save_dtype='uint8',
                          batch_size=1,
                          num_threads=1,
                          py_num_workers=1,
                          py_start_method='fork',
                          device_id=0,
                          normalization='windowing'):

    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']
    assert normalization in ['min_max', 'min_max_pct', 'windowing']
    num_dcms = len(dcm_paths)

    # dali to process with chunk in-memory
    external_source = _DaliJpegStreamExternalSource(dcm_paths,
                                                    batch_size=batch_size)
    pipe = _dali_decode_pipeline_ram(
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
    all_fails = []
    for _batch_idx in tqdm(range(num_batchs)):
        try:
            outs = pipe.run()
        except Exception as e:
            # print('DALI exception occur:', e)
            fails = dcm_paths[_batch_idx * batch_size:(_batch_idx + 1) *
                              batch_size]
            all_fails.extend(fails)
            print(f'Exception: One of {fails} can not be decoded.')
            # ignore this batch and re-build pipeline
            if _batch_idx < num_batchs - 1:
                cur_idx += batch_size
                del external_source, pipe
                gc.collect()
                torch.cuda.empty_cache()
                external_source = _DaliJpegStreamExternalSource(
                    dcm_paths[(_batch_idx + 1) * batch_size:],
                    batch_size=batch_size)
                pipe = _dali_decode_pipeline_ram(
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
            img = torch.empty(img_dali.shape(),
                              dtype=torch.int16,
                              device='cuda')
            feed_ndarray(img_dali,
                         img,
                         cuda_stream=torch.cuda.current_stream(device=0))

            invert = inverts.at(j).item()
            windowing_param = windowing_params.at(j)
            voilut_func = voilut_funcs.at(j).item()
            voilut_func = VOILUT_FUNCS_INV_MAP[voilut_func]

            img = normalize_dicom_img(img,
                                      invert=invert,
                                      save_dtype=save_dtype,
                                      window_centers=windowing_param[0],
                                      window_widths=windowing_param[1],
                                      window_func=voilut_func,
                                      window_index=0,
                                      method=normalization)
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            # save to file
            misc_utils.save_img_to_file(save_path, img, backend=save_backend)

    assert cur_idx == len(
        save_paths) - 1, f'{cur_idx} != {len(save_paths) - 1}'

    try:
        del external_source, pipe
    except:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return all_fails


@dali.pipeline_def
def _dali_decode_pipeline_disk(j2kfiles):
    jpegs, _ = dali.fn.readers.file(files=j2kfiles)
    images = dali.fn.experimental.decoders.image(
        jpegs,
        device='mixed',
        output_type=dali.types.ANY_DATA,
        dtype=dali.types.UINT16)
    return images


def convert_with_dali_disk(dcm_paths,
                           save_paths,
                           j2k_temp_dir,
                           save_backend='cv2',
                           save_dtype='uint8',
                           chunk=64,
                           batch_size=1,
                           num_threads=2,
                           device_id=0,
                           normalization='windowing'):
    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']
    assert normalization in ['min_max', 'min_max_pct', 'windowing']

    if chunk % batch_size > 0:
        print(
            'Warning: set chunk divided by batch_size for maximum performance.'
        )
    num_dcms = len(dcm_paths)

    for start_idx in tqdm(range(0, num_dcms, chunk)):
        end_idx = min(num_dcms - 1, start_idx + chunk)
        if end_idx == start_idx:
            break

        os.makedirs(j2k_temp_dir, exist_ok=True)
        chunk_dcm_paths = dcm_paths[start_idx:end_idx]
        chunk_save_paths = save_paths[start_idx:end_idx]

        temp_j2k_paths = []
        metas = []
        for dcm_path, save_path in zip(chunk_dcm_paths, chunk_save_paths):
            ds = pydicom.dcmread(dcm_path)
            meta = PydicomMetadata(ds)
            pixel_data = ds.PixelData
            offset = pixel_data.find(
                SUID2HEADER[ds.file_meta.TransferSyntaxUID])
            temp_jpeg_name = os.path.basename(save_path).replace(
                '.png', '.temp')
            temp_jpeg_path = os.path.join(j2k_temp_dir, temp_jpeg_name)
            temp_j2k_paths.append(temp_jpeg_path)
            with open(temp_jpeg_path, "wb") as temp_f:
                temp_f.write(bytearray(pixel_data[offset:]))
            metas.append(meta)
        # dali to process with chunk in-memory
        pipe = _dali_decode_pipeline_disk(temp_j2k_paths,
                                          batch_size=batch_size,
                                          num_threads=num_threads,
                                          device_id=device_id,
                                          debug=False)
        pipe.build()

        chunk_size = len(chunk_dcm_paths)
        num_batchs = chunk_size // batch_size
        if chunk_size % batch_size > 0:
            num_batchs += 1

        idx_in_chunk = -1
        for _batch_idx in tqdm(range(num_batchs)):
            try:
                outs = pipe.run()
            except StopIteration as e:
                raise AssertionError('This should not be the case.')
            imgs = outs[0]
            for j in range(len(imgs)):
                idx_in_chunk += 1
                save_path = chunk_save_paths[idx_in_chunk]
                meta = metas[idx_in_chunk]
                img_dali = imgs[j]
                img = torch.empty(img_dali.shape(),
                                  dtype=torch.int16,
                                  device='cuda')
                feed_ndarray(img_dali,
                             img,
                             cuda_stream=torch.cuda.current_stream(device=0))
                img = normalize_dicom_img(img,
                                          invert=meta.invert,
                                          save_dtype=save_dtype,
                                          window_centers=meta.window_centers,
                                          window_widths=meta.window_widths,
                                          window_func=meta.voilut_func,
                                          window_index=0,
                                          method=normalization)
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                # save to file
                misc_utils.save_img_to_file(save_path,
                                            img,
                                            backend=save_backend)

        assert idx_in_chunk == chunk_size - 1, f'{idx_in_chunk} != {chunk_size - 1}'

        shutil.rmtree(j2k_temp_dir)


def convert_with_dali(
        dcm_paths,
        save_paths,
        normalization='windowing',
        save_backend='cv2',
        save_dtype='uint8',
        cache='ram',
        chunk=64,  # cache=disk only
        batch_size=1,
        num_threads=2,
        py_num_workers=1,  # cache=ram only
        py_start_method='fork',  # cache=ram only
        device_id=0,
        j2k_temp_dir=None,  # cache=disk only
):
    print('CONVERT WITH DALI...')
    if cache == 'ram':
        del chunk
        convert_with_dali_ram(dcm_paths,
                              save_paths,
                              save_backend=save_backend,
                              save_dtype=save_dtype,
                              batch_size=batch_size,
                              num_threads=num_threads,
                              py_num_workers=py_num_workers,
                              py_start_method=py_start_method,
                              device_id=device_id,
                              normalization=normalization)
    elif cache == 'disk':
        assert j2k_temp_dir is not None
        convert_with_dali_disk(dcm_paths,
                               save_paths,
                               j2k_temp_dir,
                               save_backend=save_backend,
                               save_dtype=save_dtype,
                               chunk=chunk,
                               batch_size=batch_size,
                               num_threads=num_threads,
                               device_id=device_id,
                               normalization=normalization)
    else:
        raise ValueError(f'Unsupported cache method {cache}')


def convert_with_dali_parallel(
        dcm_paths,
        save_paths,
        normalization='windowing',
        save_backend='cv2',
        save_dtype='uint8',
        cache='ram',
        chunk=64,  # cache=disk only
        batch_size=1,
        num_threads=2,
        py_num_workers=1,  # cache=ram only
        py_start_method='fork',  # cache=ram only
        device_id=0,
        j2k_temp_dir=None,  # cache=disk only
        parallel_n_jobs=2,
        parallel_backend='loky'):
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_dali(dcm_paths,
                                 save_paths,
                                 normalization=normalization,
                                 save_backend=save_backend,
                                 save_dtype=save_dtype,
                                 cache=cache,
                                 chunk=chunk,
                                 batch_size=batch_size,
                                 num_threads=num_threads,
                                 py_num_workers=py_num_workers,
                                 py_start_method=py_start_method,
                                 device_id=device_id,
                                 j2k_temp_dir=j2k_temp_dir)
    else:
        assert cache != 'disk', 'Currently, cache method `disk` can not be used in parallel.'
        num_samples = len(dcm_paths)
        num_samples_per_worker = num_samples // parallel_n_jobs
        if num_samples % parallel_n_jobs > 0:
            num_samples_per_worker += 1
        starts = [num_samples_per_worker * i for i in range(parallel_n_jobs)]
        ends = [
            min(start + num_samples_per_worker, num_samples)
            for start in starts
        ]
        if isinstance(device_id, list):
            assert len(device_id) == parallel_n_jobs
        elif isinstance(device_id, int):
            device_id = [device_id] * parallel_n_jobs

        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(convert_with_dali)(
                dcm_paths[start:end],
                save_paths[start:end],
                normalization=normalization,
                save_backend=save_backend,
                save_dtype=save_dtype,
                cache=cache,
                chunk=chunk,
                batch_size=batch_size,
                num_threads=num_threads,
                py_num_workers=py_num_workers,
                py_start_method=py_start_method,
                device_id=worker_device_id,
                j2k_temp_dir=j2k_temp_dir,
            ) for start, end, worker_device_id in zip(starts, ends, device_id))


#################################### DICOMSDL ####################################
def _convert_single_with_dicomsdl(dcm_path,
                                  save_path,
                                  normalization='windowing',
                                  save_backend='cv2',
                                  save_dtype='uint8',
                                  index=0,
                                  legacy=False):
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

    # legacy: old method (cpu numpy operation), for compatibility only
    # new method: gpu torch operation to improve speed
    if not legacy:
        img = torch.from_numpy(img.astype(np.int16)).cuda()
    img = normalize_dicom_img(img,
                              invert=meta.invert,
                              save_dtype=save_dtype,
                              window_centers=meta.window_centers,
                              window_widths=meta.window_widths,
                              window_func=meta.voilut_func,
                              window_index=0,
                              method=normalization)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    # save to file
    misc_utils.save_img_to_file(save_path, img, backend=save_backend)


def convert_with_dicomsdl(dcm_paths,
                          save_paths,
                          normalization='windowing',
                          save_backend='cv2',
                          save_dtype='uint8',
                          legacy=False):
    assert len(dcm_paths) == len(save_paths)
    for i in tqdm(range(len(dcm_paths))):
        _convert_single_with_dicomsdl(dcm_paths[i],
                                      save_paths[i],
                                      normalization=normalization,
                                      save_backend=save_backend,
                                      save_dtype=save_dtype,
                                      legacy=legacy)
    return


def convert_with_dicomsdl_parallel(dcm_paths,
                                   save_paths,
                                   normalization='windowing',
                                   save_backend='cv2',
                                   save_dtype='uint8',
                                   parallel_n_jobs=2,
                                   joblib_backend='loky',
                                   legacy=False):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_dicomsdl(dcm_paths,
                                     save_paths,
                                     normalization=normalization,
                                     save_backend=save_backend,
                                     save_dtype=save_dtype,
                                     legacy=legacy)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{joblib_backend}`')
        _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
            delayed(_convert_single_with_dicomsdl)(dcm_paths[j],
                                                   save_paths[j],
                                                   normalization=normalization,
                                                   save_backend=save_backend,
                                                   save_dtype=save_dtype,
                                                   legacy=legacy)
            for j in tqdm(range(len(dcm_paths))))


def _convert_single_with_pydicom(dcm_path,
                                 save_path,
                                 normalization='windowing',
                                 save_backend='cv2',
                                 save_dtype='uint8'):
    ds = pydicom.dcmread(dcm_path)
    img = ds.pixel_array
    meta = PydicomMetadata(ds)
    img = normalize_dicom_img(img,
                              invert=meta.invert,
                              save_dtype=save_dtype,
                              window_centers=meta.window_centers,
                              window_widths=meta.window_widths,
                              window_func=meta.voilut_func,
                              window_index=0,
                              method=normalization)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    # save to file
    misc_utils.save_img_to_file(save_path, img, backend=save_backend)


def convert_with_pydicom(dcm_paths,
                         save_paths,
                         normalization='windowing',
                         save_backend='cv2',
                         save_dtype='uint8'):
    assert len(dcm_paths) == len(save_paths)
    for dcm_path, save_path in tqdm(zip(dcm_paths, save_paths)):
        _convert_single_with_pydicom(dcm_path, save_path, normalization,
                                     save_backend, save_dtype)


def convert_with_pydicom_parallel(dcm_paths,
                                  save_paths,
                                  normalization='windowing',
                                  save_backend='cv2',
                                  save_dtype='uint8',
                                  parallel_n_jobs=2,
                                  parallel_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_pydicom(dcm_paths, save_paths, normalization,
                                    save_backend, save_dtype)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(_convert_single_with_pydicom)(
                dcm_paths[j], save_paths[j], normalization, save_backend,
                save_dtype) for j in tqdm(range(len(dcm_paths))))


####################################### NVJPEG2K #######################################
def _convert_single_with_nvjpeg2k(j2k_decoder,
                                  dcm_path,
                                  save_path,
                                  normalization='windowing',
                                  save_backend='cv2',
                                  save_dtype='uint8'):
    ds = pydicom.dcmread(dcm_path)
    # support J2K only
    assert ds.file_meta.TransferSyntaxUID == J2K_SUID
    meta = PydicomMetadata(ds)
    pixel_data = ds.PixelData
    offset = pixel_data.find(SUID2HEADER[ds.file_meta.TransferSyntaxUID])
    j2k_stream = bytearray(pixel_data[offset:])
    img = j2k_decoder.decode(j2k_stream)
    img = normalize_dicom_img(img,
                              invert=meta.invert,
                              save_dtype=save_dtype,
                              window_centers=meta.window_centers,
                              window_widths=meta.window_widths,
                              window_func=meta.voilut_func,
                              window_index=0,
                              method=normalization)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    # save to file
    misc_utils.save_img_to_file(save_path, img, backend=save_backend)


def convert_with_nvjpeg2k(dcm_paths,
                          save_paths,
                          normalization='windowing',
                          save_backend='cv2',
                          save_dtype='uint8'):
    assert len(dcm_paths) == len(save_paths)
    j2k_decoder = nvjpeg2k.Decoder()
    for i in tqdm(range(len(dcm_paths))):
        _convert_single_with_nvjpeg2k(j2k_decoder, dcm_paths[i], save_paths[i],
                                      normalization, save_backend, save_dtype)


def convert_with_nvjpeg2k_parallel(dcm_paths,
                                   save_paths,
                                   normalization='windowing',
                                   save_backend='cv2',
                                   save_dtype='uint8',
                                   parallel_n_jobs=2,
                                   parallel_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_nvjpeg2k(dcm_paths, save_paths, normalization,
                                     save_backend, save_dtype)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`...'
        )

        num_samples = len(dcm_paths)
        num_samples_per_worker = num_samples // parallel_n_jobs
        if num_samples % parallel_n_jobs > 0:
            num_samples_per_worker += 1
        starts = [num_samples_per_worker * i for i in range(parallel_n_jobs)]
        ends = [
            min(start + num_samples_per_worker, num_samples)
            for start in starts
        ]

        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(convert_with_nvjpeg2k)(
                dcm_paths[start:end], save_paths[start:end], normalization,
                save_backend, save_dtype) for start, end in zip(starts, ends))


####################################### DALI EAGER EXECUTION #######################################
def _convert_single_with_dali_eager(dcm_path,
                                    save_path,
                                    normalization='windowing',
                                    save_backend='cv2',
                                    save_dtype='uint8'):
    ds = pydicom.dcmread(dcm_path)
    meta = PydicomMetadata(ds)
    pixel_data = ds.PixelData
    offset = pixel_data.find(SUID2HEADER[ds.file_meta.TransferSyntaxUID])
    j2k_stream = bytearray(pixel_data[offset:])

    j2k_stream = np.array(j2k_stream, dtype=np.uint8)
    output = eager.experimental.decoders.image([j2k_stream],
                                               device='gpu',
                                               output_type=types.ANY_DATA,
                                               dtype=DALIDataType.UINT16)
    # img = output.as_cpu().at(0).squeeze(-1)   # numpy cpu array
    img_dali = output[0][0]  # gpu array
    img = torch.empty(img_dali.shape(), dtype=torch.int16, device='cuda')
    feed_ndarray(img_dali,
                 img,
                 cuda_stream=torch.cuda.current_stream(device=0))
    img = normalize_dicom_img(img,
                              invert=meta.invert,
                              save_dtype=save_dtype,
                              window_centers=meta.window_centers,
                              window_widths=meta.window_widths,
                              window_func=meta.voilut_func,
                              window_index=0,
                              method=normalization)
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    # save to file
    misc_utils.save_img_to_file(save_path, img, backend=save_backend)


def convert_with_dali_eager(dcm_paths,
                            save_paths,
                            normalization='windowing',
                            save_backend='cv2',
                            save_dtype='uint8'):
    assert len(dcm_paths) == len(save_paths)
    for i in tqdm(range(len(dcm_paths))):
        _convert_single_with_dali_eager(dcm_paths[i], save_paths[i],
                                        normalization, save_backend,
                                        save_dtype)


def convert_with_dali_eager_parallel(dcm_paths,
                                     save_paths,
                                     normalization='windowing',
                                     save_backend='cv2',
                                     save_dtype='uint8',
                                     parallel_n_jobs=2,
                                     parallel_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_dali_eager(dcm_paths, save_paths, normalization,
                                       save_backend, save_dtype)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`...'
        )

        num_samples = len(dcm_paths)
        num_samples_per_worker = num_samples // parallel_n_jobs
        if num_samples % parallel_n_jobs > 0:
            num_samples_per_worker += 1
        starts = [num_samples_per_worker * i for i in range(parallel_n_jobs)]
        ends = [
            min(start + num_samples_per_worker, num_samples)
            for start in starts
        ]

        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(convert_with_dali_eager)(
                dcm_paths[start:end], save_paths[start:end], normalization,
                save_backend, save_dtype) for start, end in zip(starts, ends))