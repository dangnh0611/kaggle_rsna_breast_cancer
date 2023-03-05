# https://github.com/pydicom/pydicom/issues/1554
# https://github.com/pydicom/pydicom/issues/539

import torch
import nvidia.dali as dali
import os
import numpy as np
import cv2
import dicomsdl
from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import time
import pydicom
import shutil
import queue
import multiprocessing
multiprocessing.set_start_method('spawn', force = True)
import gc
from pydicom.pixel_data_handlers.util import apply_voi_lut, pixel_dtype
# import nvjpeg2k

J2K_SYNTAX_UID = '1.2.840.10008.1.2.4.90'
J2K_HEADER = b"\x00\x00\x00\x0C"

JLOSSLESS_SYNTAX_UID = '1.2.840.10008.1.2.4.70'
JLOSSLESS_HEADER = b"\xff\xd8\xff\xe0"

def apply_windowing_np(arr,
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

        arr = ((arr - window_center) / window_width + 0.5) * y_range + y_min
        arr = np.clip(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        arr = y_range / (1 +
                         np.exp(-4 *
                                (arr - window_center) / window_width)) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


def apply_windowing_np_v3(arr,
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
        
#         arr = ((arr - window_center) / window_width + 0.5) * y_range + y_min
        # faster ?
        m = y_range / window_width
        n = (- window_center / window_width + 0.5) * y_range + y_min
        arr = arr * m + n

        arr = np.clip(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
#         arr = y_range / (1 + np.exp(-4 * (arr - window_center) / window_width)) + y_min
        m = -4 / window_width
        arr = y_range / (1 + np.exp((arr - window_center) * m) ) + y_min
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

#         arr = ((arr - window_center) / window_width + 0.5) * y_range + y_min
        # faster ?
        m = y_range / window_width
        n = (- window_center / window_width + 0.5) * y_range + y_min
        arr = arr * m + n
        arr = torch.clamp(arr, y_min, y_max)

    elif voi_func == 'SIGMOID':
        m = -4 / window_width
        arr = y_range / (1 + torch.exp((arr - window_center) * m) ) + y_min
    else:
        raise ValueError(
            f"Unsupported (0028,1056) VOI LUT Function value '{voi_func}'")
    return arr


class PydicomMetadata:
    def __init__(self, ds):
        temp = ds['WindowWidth']
        self.window_widths = [float(e) for e in temp] if temp.VM > 1 else [float(temp.value)]
        temp = ds['WindowCenter']
        self.window_centers = [float(e) for e in temp] if temp.VM > 1 else [float(temp.value)]
        # if nan --> LINEAR
        self.voilut_func = ds.get('VOILUTFunction', 'LINEAR')

        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        self.nbit = ds.BitsStored

        assert len(self.window_widths) == len(self.window_widths)


class DicomsdlMetadata:
    def __init__(self, ds):
        self.window_widths = ds.WindowWidth
        if not isinstance(self.window_widths, list):
            self.window_widths = [self.window_widths]

        self.window_centers = ds.WindowCenter
        if not isinstance(self.window_centers, list):
            self.window_centers = [self.window_centers]
        # if nan --> LINEAR
        self.voilut_func = ds.VOILUTFunction
        self.voilut_func = 'LINEAR' if self.voilut_func is None else self.voilut_func

        self.invert = (ds.PhotometricInterpretation == 'MONOCHROME1')
        self.nbit = ds.BitsStored

        assert len(self.window_widths) == len(self.window_widths)
        


def min_max_scale(img):
    maxv = img.max()
    minv = img.min()
    if maxv > minv:
        return (img - minv) / (maxv - minv)
    else:
        return img - minv  # ==0


def apply_voilut(arr,
                  window_width=None,
                  window_center=None,
                  voi_func='linear',
                  y_min=0,
                  y_max=255):
    assert window_width > 0
    y_range = y_max - y_min
    # float64 needed or just float32 ?
    arr = arr.astype('float64')

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


def from_uint16(img, invert=False, dtype='uint8', bit_scale=False, nbit=None):
    assert dtype in ['uint8', 'uint16', 'float16', 'float32', 'float64']

    if dtype == 'uint16':
        if invert:
            img = img.max() - img
        return img

    # first, scaling (uint16 --> float64)
    if bit_scale:
        maxv = 2**nbit - 1
        img = img / maxv
    else:
        img = min_max_scale(img)

    # invert if needed
    if invert:
        img = 1. - img

    # convert to specified dtype
    if dtype == 'uint8':
        return (img * 255).astype(np.uint8)
    else:  # float16/32/64
        return img.astype(dtype)


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


#https://github.com/NVIDIA/DALI/issues/2330
#https://medium.datadriveninvestor.com/gpu-accelerated-data-loading-with-dali-part-2-pipelines-and-data-loaders-99f51548e8a6
#https://github.com/NVIDIA/DALI/blob/main/dali/python/nvidia/dali/plugin/pytorch.py
#https://zhuanlan.zhihu.com/p/518240063
class _DicomToJ2kExternalSourceV1(object):

    def __init__(self, j2k_streams):
        self.j2k_streams = j2k_streams

    def __iter__(self):
        return self

    def __next__(self):
        return self.j2k_streams


@dali.pipeline_def
def _j2k_decode_pipeline_ram_v1(eii, resize=None):
    jpeg = dali.fn.external_source(source=eii,
                                   dtype=dali.types.UINT8,
                                   batch=True,
                                   parallel=False)
    image = dali.fn.experimental.decoders.image(
        jpeg,
        device='mixed',
        output_type=dali.types.ANY_DATA,
        dtype=dali.types.UINT16)
    if resize is not None:
        image = dali.fn.resize(image, size=resize)
    return image


def convert_with_dali_ram_v1(dcm_paths,
                             save_paths,
                             save_backend='cv2',
                             save_dtype='uint8',
                             chunk=-1,
                             batch_size=16,
                             num_threads=2,
                             py_num_workers=0,
                             device_id=0):
    del chunk, py_num_workers
    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']

    num_dcms = len(dcm_paths)

    for start_idx in tqdm(range(0, num_dcms, batch_size)):
        end_idx = min(num_dcms - 1, start_idx + batch_size)
        if end_idx == start_idx:
            break
        batch_dcm_paths = dcm_paths[start_idx:end_idx]
        batch_save_paths = save_paths[start_idx:end_idx]
        j2k_streams = []
        inverts = []
        nbits = []
        for dcm_path in batch_dcm_paths:
            dcm = pydicom.dcmread(dcm_path)
            assert dcm.file_meta.TransferSyntaxUID == J2K_SYNTAX_UID
            # # why need this ?
            ##################################
            # with open(file_path, 'rb') as fp:
            #     raw = DicomBytesIO(fp.read())
            #     dicom = pydicom.dcmread(raw)
            ##################################
            pixel_data = dcm.PixelData
            offset = pixel_data.find(J2K_HEADER)
            # @TODO: different size or not (for each dcm file) ?
            # if same size, buffering (np.copyto) for optimization
            # @ANS: no, different sizes
            j2k_streams.append(
                np.array(bytearray(pixel_data[offset:]), np.uint8))
            inverts.append(dcm.PhotometricInterpretation == 'MONOCHROME1')
            nbits.append(dcm.BitsStored)
        # dali to process with chunk in-memory
        pipe = _j2k_decode_pipeline_ram_v1(
            _DicomToJ2kExternalSourceV1(j2k_streams),
            batch_size=len(j2k_streams),
            num_threads=num_threads,
            device_id=device_id,
            debug=False)
        pipe.build()
        outs = pipe.run()
        outs = outs[0].as_cpu()

        for i, save_path in enumerate(batch_save_paths):
            invert = inverts[i]
            nbit = nbits[i]
            img = outs.at(i).squeeze(-1)  # uint16
            img = from_uint16(img,
                              invert,
                              save_dtype,
                              bit_scale=False,
                              nbit=nbit)
            save_img_to_file(save_path, img, backend=save_backend)


# class _DicomToJ2kExternalSourceV2:

#     def __init__(self, dcm_paths):
#         self.dcm_paths = dcm_paths
#         self.len = len(dcm_paths)

#     def __call__(self, sample_info):
#         idx = sample_info.idx_in_epoch
#         if idx >= self.len:
#             raise StopIteration
#         # print('IDX:', sample_info.idx_in_epoch, sample_info.idx_in_batch)

#         dcm_path = self.dcm_paths[idx]
#         dcm = pydicom.dcmread(dcm_path)
#         assert dcm.file_meta.TransferSyntaxUID == J2K_SYNTAX_UID
#         # # why need this ?
#         ##################################
#         # with open(file_path, 'rb') as fp:
#         #     raw = DicomBytesIO(fp.read())
#         #     dicom = pydicom.dcmread(raw)
#         ##################################
#         pixel_data = dcm.PixelData
#         offset = pixel_data.find(J2K_HEADER)
#         # @TODO: different size or not (for each dcm file) ?
#         # if same size, buffering (np.copyto) for optimization
#         # @ANS: no, different sizes
#         j2k_stream = np.array(bytearray(pixel_data[offset:]), np.uint8)
#         invert = dcm.PhotometricInterpretation == 'MONOCHROME1'
#         invert = np.array([invert], dtype=np.bool_)
#         return (j2k_stream, invert)

# @dali.pipeline_def
# def _j2k_decode_pipeline_ram_v2(eii, resize=None, num_outputs=None):
#     jpeg, invert = dali.fn.external_source(
#         source=eii,
#         num_outputs=num_outputs,
#         dtype=[dali.types.UINT8, dali.types.BOOL],
#         batch=False)
#     image = dali.fn.experimental.decoders.image(
#         jpeg,
#         device='mixed',
#         output_type=dali.types.ANY_DATA,
#         dtype=dali.types.UINT16)
#     if resize is not None:
#         image = dali.fn.resize(image, size=resize)
#     return image, invert

# def convert_with_dali_ram_v2(dcm_paths,
#                              save_paths,
#                              chunk=64,
#                              batch_size = 1,
#                              num_threads=2,
#                              py_num_workers = 1,
#                              device_id=0):
#     #@TODO: fix StopIteration
#     raise NotImplementedError()

#     assert len(dcm_paths) == len(save_paths)

#     # dali to process with chunk in-memory
#     external_source = _DicomToJ2kExternalSourceV2(dcm_paths)
#     pipe = _j2k_decode_pipeline_ram_v2(external_source,
#                                       num_outputs=2,
#                                       batch_size=16,
#                                       num_threads=num_threads,
#                                       device_id=device_id,
#                                       debug=False)
#     pipe.build()

#     cur_idx = -1
#     while True:
#         try:
#             outs = pipe.run()
#         except StopIteration as e:
#             raise e
#             break
#         print('len:', len(outs[0]))
#         imgs = outs[0].as_cpu()
#         inverts = outs[1]
#         for j in range(len(inverts)):
#             cur_idx += 1
#             save_path = save_paths[cur_idx]
#             img = imgs.at(j).squeeze(-1)
#             invert = inverts.at(j)
#             img = any_to_fp32(img)
#             if invert:
#                 img = 1. - img
#             img = float_to_uint8(img)
#             # print(img.shape)
#             # cv2.imwrite(save_path, img)

#         print(len(imgs), len(inverts), cur_idx, len(save_paths))

#     assert cur_idx == len(
#         save_paths) - 1, f'{cur_idx} != {len(save_paths) - 1}'


class _DicomToJ2kExternalSourceV3:

    def __init__(self, dcm_paths, batch_size=1):
        self.dcm_paths = dcm_paths
        self.len = len(dcm_paths)
        self.batch_size = batch_size

    def __call__(self, batch_info):
        idx = batch_info.iteration
        start = idx * self.batch_size
        end = min(self.len, start + self.batch_size)
        if end <= start:
            raise StopIteration()
        # print('IDX:', batch_info.iteration, batch_info.epoch_idx)

        dcm_paths = self.dcm_paths[start:end]
        j2k_streams = []
        inverts = []
        nbits = []
        for dcm_path in dcm_paths:
            dcm = pydicom.dcmread(dcm_path)
#             assert dcm.file_meta.TransferSyntaxUID == J2K_SYNTAX_UID or 
#                     dcm.file_meta.TransferSyntaxUID == JLOSSLESS_SYNTAX_UID
            # # why need this ?
            ##################################
            # with open(file_path, 'rb') as fp:
            #     raw = DicomBytesIO(fp.read())
            #     dicom = pydicom.dcmread(raw)
            ##################################
            pixel_data = dcm.PixelData
            if dcm.file_meta.TransferSyntaxUID == J2K_SYNTAX_UID:
                header = J2K_HEADER
            elif dcm.file_meta.TransferSyntaxUID == JLOSSLESS_SYNTAX_UID:
                header = JLOSSLESS_HEADER
            offset = pixel_data.find(header)
            # @TODO: different size or not (for each dcm file) ?
            # if same size, buffering (np.copyto) for optimization
            # @ANS: no, different sizes
            j2k_stream = np.array(bytearray(pixel_data[offset:]), np.uint8)
            invert = (dcm.PhotometricInterpretation == 'MONOCHROME1')
            j2k_streams.append(j2k_stream)
            inverts.append(invert)
            nbits.append(dcm.BitsStored)
        return j2k_streams, np.array(inverts,
                                     dtype=np.bool_), np.array(nbits,
                                                               dtype=np.uint8)


@dali.pipeline_def
def _j2k_decode_pipeline_ram_v3(eii, resize=None, num_outputs=None):
    jpeg, invert, nbit = dali.fn.external_source(
        source=eii,
        num_outputs=num_outputs,
        dtype=[dali.types.UINT8, dali.types.BOOL, dali.types.UINT8],
        batch=True,
        batch_info=True,
        parallel=True)
    image = dali.fn.experimental.decoders.image(
        jpeg,
        device='mixed',
        output_type=dali.types.ANY_DATA,
        dtype=dali.types.UINT16)
    if resize is not None:
        image = dali.fn.resize(image, size=resize)
    return image, invert, nbit


def convert_with_dali_ram_v3(dcm_paths,
                             save_paths,
                             save_backend='cv2',
                             save_dtype='uint8',
                             chunk=-1,
                             batch_size=1,
                             num_threads=2,
                             py_num_workers=1,
                             device_id=0):
#     import pycuda.autoinit
    del chunk
    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']
    num_dcms = len(dcm_paths)

    # dali to process with chunk in-memory
    external_source = _DicomToJ2kExternalSourceV3(dcm_paths,
                                                  batch_size=batch_size)
    pipe = _j2k_decode_pipeline_ram_v3(
        external_source,
        num_outputs=3,
        py_num_workers=py_num_workers,
        py_start_method = 'spawn',
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        debug=False,
    )
    print('START BUILD!')
    try:
        pipe.build()
    except Exception as e:
        print('EXCEPT:', e)
    print('BUILD FIRST DONE!')
    
#     external_source = _DicomToJ2kExternalSourceV3(dcm_paths,
#                                                   batch_size=batch_size)
#     pipe = _j2k_decode_pipeline_ram_v3(
#         external_source,
#         num_outputs=3,
#         py_num_workers=py_num_workers,
#         py_start_method = 'fork',
#         batch_size=batch_size,
#         num_threads=num_threads,
#         device_id=device_id,
#         debug=False,
#     )
#     pipe.start_py_workers()
#     pipe.build()
    

    num_batchs = num_dcms // batch_size
    if num_dcms % batch_size > 0:
        num_batchs += 1

    cur_idx = -1
    for _batch_idx in tqdm(range(num_batchs)):
        try:
            outs = pipe.run()
        except StopIteration as e:
            raise AssertionError('This should not be the case.')
        imgs = outs[0].as_cpu()
        inverts = outs[1]
        nbits = outs[2]
        for j in range(len(inverts)):
            cur_idx += 1
            save_path = save_paths[cur_idx]
            img = imgs.at(j).squeeze(-1)  # uint16
            invert = inverts.at(j)
            nbit = nbits.at(j)
            img = from_uint16(img,
                              invert,
                              save_dtype,
                              bit_scale=False,
                              nbit=nbit)
            save_img_to_file(save_path, img, backend=save_backend)

    assert cur_idx == len(
        save_paths) - 1, f'{cur_idx} != {len(save_paths) - 1}'

    # del pipe
    # gc.collect()


@dali.pipeline_def
def j2k_decode_pipeline_disk(j2kfiles):
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
                           py_num_workers=1,
                           device_id=0):
    del py_num_workers
    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']

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
        inverts = []
        nbits = []
        for dcm_path, save_path in zip(chunk_dcm_paths, chunk_save_paths):
            dcm = pydicom.dcmread(dcm_path)
            assert dcm.file_meta.TransferSyntaxUID == J2K_SYNTAX_UID
            # # why need this ?
            ##################################
            # with open(file_path, 'rb') as fp:
            #     raw = DicomBytesIO(fp.read())
            #     dicom = pydicom.dcmread(raw)
            ##################################
            pixel_data = dcm.PixelData
            offset = pixel_data.find(J2K_HEADER)
            j2k_name = os.path.basename(save_path).replace('.png', '.temp')
            temp_j2k_path = os.path.join(j2k_temp_dir, j2k_name)
            temp_j2k_paths.append(temp_j2k_path)
            with open(temp_j2k_path, "wb") as temp_f:
                temp_f.write(bytearray(pixel_data[offset:]))
            inverts.append(dcm.PhotometricInterpretation == 'MONOCHROME1')
            nbits.append(dcm.BitsStored)
        # dali to process with chunk in-memory
        pipe = j2k_decode_pipeline_disk(temp_j2k_paths,
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
                break
            imgs = outs[0].as_cpu()
            for j in range(len(imgs)):
                idx_in_chunk += 1
                save_path = chunk_save_paths[idx_in_chunk]
                img = imgs.at(j).squeeze(-1)
                invert = inverts[idx_in_chunk]
                nbit = nbits[idx_in_chunk]
                img = from_uint16(img,
                                  invert,
                                  save_dtype,
                                  bit_scale=False,
                                  nbit=nbit)
                save_img_to_file(save_path, img, backend=save_backend)

        assert idx_in_chunk == chunk_size - 1, f'{idx_in_chunk} != {chunk_size - 1}'

        shutil.rmtree(j2k_temp_dir)


def convert_with_dali(
        dcm_paths,
        save_paths,
        save_backend='cv2',
        save_dtype='uint8',
        chunk=64,
        batch_size=1,  # for ram_v3 only
        num_threads=2,
        py_num_workers=1,  # for ram_v3 only
        device_id=0,
        cache='ram_v3',
        j2k_temp_dir=None):
    print('CONVERT WITH DALI...')
    if cache == 'ram_v1':
        convert_with_dali_ram_v1(dcm_paths,
                                 save_paths,
                                 save_backend=save_backend,
                                 save_dtype=save_dtype,
                                 chunk=chunk,
                                 batch_size=batch_size,
                                 num_threads=num_threads,
                                 py_num_workers=py_num_workers,
                                 device_id=device_id)
    # elif cache == 'ram_v2':
    #     convert_with_dali_ram_v2(dcm_paths,
    #                              save_paths,
    #                              chunk=chunk,
    #                              num_threads=num_threads,
    #                              device_id=device_id)
    elif cache == 'ram_v3':
        convert_with_dali_ram_v3(dcm_paths,
                                 save_paths,
                                 save_backend=save_backend,
                                 save_dtype=save_dtype,
                                 chunk=chunk,
                                 batch_size=batch_size,
                                 num_threads=num_threads,
                                 py_num_workers=py_num_workers,
                                 device_id=device_id)
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
                               py_num_workers=py_num_workers,
                               device_id=device_id)
    else:
        raise ValueError(f'Unsupported cache method {cache}')


def convert_with_dali_parallel(
        dcm_paths,
        save_paths,
        save_backend='cv2',
        save_dtype='uint8',
        chunk=64,
        batch_size=1,  # for ram_v3 only
        num_threads=2,
        py_num_workers=1,  # for ram_v3 only
        device_id=0,
        cache='ram_v3',
        j2k_temp_dir=None,
        parallel_n_jobs=2,
        parallel_backend='loky'):
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_dali(dcm_paths,
                                 save_paths,
                                 save_backend=save_backend,
                                 save_dtype=save_dtype,
                                 chunk=chunk,
                                 batch_size=batch_size,
                                 num_threads=num_threads,
                                 py_num_workers=py_num_workers,
                                 device_id=device_id,
                                 cache=cache,
                                 j2k_temp_dir=j2k_temp_dir)
    
#         _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
#             delayed(convert_with_dali)(
#                 dcm_paths,
#                 save_paths,
#                 save_backend=save_backend,
#                 save_dtype=save_dtype,
#                 chunk=chunk,  # disk
#                 batch_size=batch_size,  # disk, ram_v3
#                 num_threads=num_threads,
#                 py_num_workers=py_num_workers,  # ram_v3
#                 device_id=device_id,
#                 cache=cache,
#                 j2k_temp_dir=j2k_temp_dir,  # disk)
#             ) for i in range(1))
    else:
        assert cache != 'disk', 'Cache method `disk` can not be used in parallel.'
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
#         _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
#             delayed(convert_with_dali)(
#                 dcm_paths[start:end],
#                 save_paths[start:end],
#                 save_backend=save_backend,
#                 save_dtype=save_dtype,
#                 chunk=chunk,  # disk
#                 batch_size=batch_size,  # disk, ram_v3
#                 num_threads=num_threads,
#                 py_num_workers=py_num_workers,  # ram_v3
#                 device_id=worker_device_id,
#                 cache=cache,
#                 j2k_temp_dir=j2k_temp_dir,  # disk)
#             ) for start, end, worker_device_id in zip(starts, ends, device_id))

        
        workers = []
        for i in range(parallel_n_jobs):
            start = starts[i]
            end = ends[i]
            worker_device_id = device_id[i]
            worker = multiprocessing.Process(group = None,
                                             target = convert_with_dali,
                                            args = (dcm_paths[start:end], save_paths[start:end],),
                                            kwargs = {
                                                'save_backend': save_backend,
                                                'save_dtype': save_dtype,
                                                'chunk': chunk,
                                                'batch_size': batch_size,
                                                'num_threads': num_threads,
                                                'py_num_workers': py_num_workers,
                                                'device_id': worker_device_id,
                                                'cache': cache,
                                                'j2k_temp_dir': j2k_temp_dir,
                                            },
                                            daemon = False)
            workers.append(worker)
            
        for worker in workers:
            worker.start()
            print('Start worker!')
        for worker in workers:
            worker.join()


def load_img_dicomsdl(dcm_path, dtype='uint8', index=0, voilut = True):
    dcm = dicomsdl.open(dcm_path)
    info = dcm.getPixelDataInfo()
    ori_dtype = info['dtype']
    if info['SamplesPerPixel'] != 1:
        raise RuntimeError('SamplesPerPixel != 1')
    else:
        shape = [info['Rows'], info['Cols']]
    img = np.empty(shape, dtype=ori_dtype)
    assert img.dtype == np.uint16, f'{img.dtype}'
    dcm.copyFrameData(index, img)
    
    metadata = DicomsdlMetadata(dcm)
    if len(metadata.window_centers) == 0:
        print('No windows')
        voilut = False

    if voilut:
        assert dtype in ['uint8']
        
        # np v1
        start = time.time()
        img_np_v1 = apply_windowing_np(img,
            window_width=metadata.window_widths[0],
            window_center=metadata.window_centers[0],
            voi_func=metadata.voilut_func,
            y_min=0,
            y_max=255)
        end = time.time()
        take_np_v1 = round((end - start) * 1000, 2)
        
        # np v2
        start = time.time()
        img_np_v2 = apply_windowing_np_v2(img,
            window_width=metadata.window_widths[0],
            window_center=metadata.window_centers[0],
            voi_func=metadata.voilut_func,
            y_min=0,
            y_max=255)
        end = time.time()
        take_np_v2 = round((end - start) * 1000, 2)
        
        
        # np v2
        start = time.time()
        img_np_v3 = apply_windowing_np_v3(img,
            window_width=metadata.window_widths[0],
            window_center=metadata.window_centers[0],
            voi_func=metadata.voilut_func,
            y_min=0,
            y_max=255)
        end = time.time()
        take_np_v3 = round((end - start) * 1000, 2)
        
        
        img2 = np.empty(shape, dtype=np.int16)
        dcm.copyFrameData(index, img2)
        img2 = torch.from_numpy(img2).cuda()
    
        # torch
        t1 = time.time()
        img_torch = torch.from_numpy(img.astype(np.int16)).cuda()
        diff4 = torch.max(torch.abs(img_torch - img2))
        assert diff4.cpu().numpy() == 0
        print('???', diff4)
        t2 = time.time()
        img_torch = apply_windowing_torch(img_torch,
            window_width=metadata.window_widths[0],
            window_center=metadata.window_centers[0],
            voi_func=metadata.voilut_func,
            y_min=0,
            y_max=255)
        t3 = time.time()
        img_torch = img_torch.cpu().numpy()
        t4 = time.time()
        take_torch = f'[{round((t2 - t1) * 1000, 2)} {round((t3 - t2) * 1000, 2)} {round((t4 - t3) * 1000, 2)} {round((t4 - t1) * 1000, 2)}]'
        
        diff1, diff2, diff3 = np.max(np.abs(img_np_v1 - img_np_v2)), np.max(np.abs(img_np_v2 - img_np_v3)), np.max(np.abs(img_np_v2 - img_torch))
        
        print(f'{metadata.voilut_func} {diff1} {diff2} {diff3} with time np_v1 = {take_np_v1}, np_v2 = {take_np_v2}, np_v3 = {take_np_v3}, torch = {take_torch}')
        
        img = img_torch
        if metadata.invert:
            img = 255 - img
        img = img.astype(np.uint8)
    else:
        img = from_uint16(img, metadata.invert, dtype, bit_scale=False)
    return img


def _convert_single_with_dicomsdl(dcm_path,
                                  save_path,
                                  save_backend='cv2',
                                  save_dtype='uint8'):
    img = load_img_dicomsdl(dcm_path, dtype=save_dtype)
    save_img_to_file(save_path, img, backend=save_backend)


def convert_with_dicomsdl(dcm_paths,
                          save_paths,
                          save_backend='cv2',
                          save_dtype='uint8'):
    assert len(dcm_paths) == len(save_paths)
    for dcm_path, save_path in tqdm(zip(dcm_paths, save_paths)):
        _convert_single_with_dicomsdl(dcm_path, save_path, save_backend,
                                      save_dtype)


def convert_with_dicomsdl_parallel(dcm_paths,
                                   save_paths,
                                   save_backend='cv2',
                                   save_dtype='uint8',
                                   parallel_n_jobs=2,
                                   parallel_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_dicomsdl(dcm_paths, save_paths, save_backend,
                                     save_dtype)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(_convert_single_with_dicomsdl)(dcm_paths[j], save_paths[j],
                                                   save_backend, save_dtype)
            for j in tqdm(range(len(dcm_paths))))


def load_img_pydicom(dcm_path, dtype='uint8'):
    dcm = pydicom.dcmread(dcm_path)
    img = dcm.pixel_array
    # print(dcm.BitsStored, dcm.BitsAllocated, img.dtype, img.min(), img.max())
    assert img.dtype == np.uint16, f'{img.dtype}'
    invert = dcm.PhotometricInterpretation == 'MONOCHROME1'
    img = from_uint16(img, invert, dtype, bit_scale=False)
    return img


def _convert_single_with_pydicom(dcm_path,
                                 save_path,
                                 save_backend='cv2',
                                 save_dtype='uint8'):
    img = load_img_pydicom(dcm_path, dtype=save_dtype)
    save_img_to_file(save_path, img, backend=save_backend)


def convert_with_pydicom(dcm_paths,
                         save_paths,
                         save_backend='cv2',
                         save_dtype='uint8'):
    assert len(dcm_paths) == len(save_paths)
    for dcm_path, save_path in tqdm(zip(dcm_paths, save_paths)):
        _convert_single_with_pydicom(dcm_path, save_path, save_backend,
                                     save_dtype)


def convert_with_pydicom_parallel(dcm_paths,
                                  save_paths,
                                  save_backend='cv2',
                                  save_dtype='uint8',
                                  parallel_n_jobs=2,
                                  parallel_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_pydicom(dcm_paths, save_paths, save_backend,
                                    save_dtype)
    else:
        print(
            f'Starting {parallel_n_jobs} jobs with backend `{parallel_backend}`...'
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(_convert_single_with_pydicom)(dcm_paths[j], save_paths[j],
                                                  save_backend, save_dtype)
            for j in tqdm(range(len(dcm_paths))))
        
############################################        
##### NVJPEG2K

from nvidia.dali.experimental import eager
import nvidia.dali.types as types
from nvidia.dali.types import DALIDataType
        
def load_img_nvjpeg2k(j2k_decoder, dcm_path, dtype='uint8'):
    dcm = pydicom.dcmread(dcm_path)
    metadata  = PydicomMetadata(dcm)
#     assert dcm.file_meta.TransferSyntaxUID == J2K_SYNTAX_UID
    pixel_data = dcm.PixelData
    
    if dcm.file_meta.TransferSyntaxUID == J2K_SYNTAX_UID:
        header = J2K_HEADER
    elif dcm.file_meta.TransferSyntaxUID == JLOSSLESS_SYNTAX_UID:
        header = JLOSSLESS_HEADER
    offset = pixel_data.find(header)
    j2k_stream = bytearray(pixel_data[offset:])
    
#     img = j2k_decoder.decode(j2k_stream)

    j2k_stream = np.array(j2k_stream, dtype=np.uint8)
    output = eager.experimental.decoders.image([j2k_stream],
                                              device='gpu',
                                              output_type=types.ANY_DATA,
                                              dtype=DALIDataType.UINT16)
    img = output.as_cpu().at(0).squeeze(-1)
    
    assert img.dtype == np.uint16, f'Dtype is not uint16: {img.dtype}'
    invert = dcm.PhotometricInterpretation == 'MONOCHROME1'
    img = from_uint16(img, invert, dtype, bit_scale=False)
    return img


def _convert_single_with_nvjpeg2k(j2k_decoder,
                                dcm_path,
                                 save_path,
                                 save_backend='cv2',
                                 save_dtype='uint8'):
    img = load_img_nvjpeg2k(j2k_decoder, dcm_path, dtype=save_dtype)
    save_img_to_file(save_path, img, backend=save_backend)


def convert_with_nvjpeg2k(dcm_paths,
                         save_paths,
                         save_backend='cv2',
                         save_dtype='uint8'):
    assert len(dcm_paths) == len(save_paths)
#     j2k_decoder = nvjpeg2k.Decoder()
    j2k_decoder = None
    for i in tqdm(range(len(dcm_paths))):
        _convert_single_with_nvjpeg2k(j2k_decoder, dcm_paths[i], save_paths[i], save_backend,
                                     save_dtype)


def convert_with_nvjpeg2k_parallel(dcm_paths,
                                  save_paths,
                                  save_backend='cv2',
                                  save_dtype='uint8',
                                  parallel_n_jobs=2,
                                  parallel_backend='loky'):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print('No parralel. Starting the tasks within current process.')
        return convert_with_nvjpeg2k(dcm_paths, save_paths, save_backend,
                                    save_dtype)
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
            delayed(convert_with_nvjpeg2k)(dcm_paths[start:end],
                                           save_paths[start:end],
                                           save_backend,
                                           save_dtype)
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


def convert_all(
    df,
    dcm_root_dir,
    save_root_dir,
    dicomsdl_num_processes=2,
):
    os.makedirs(save_root_dir, exist_ok=True)
    machine_id_to_syntax_uid = make_uid_transfer_dict(df, dcm_root_dir)
    dali_dcm_paths = []
    dali_save_paths = []
    dicomsdl_dcm_paths = []
    dicomsdl_save_paths = []
    for i, row in df.iterrows():
        dcm_path = os.path.join(dcm_root_dir, str(row.patient_id),
                                f'{row.image_id}.dcm')
        save_path = os.path.join(save_root_dir,
                                 f'{row.patient_id}@{row.image_id}.png')
        syntax_uid = machine_id_to_syntax_uid[row.machine_id]
        if syntax_uid == J2K_SYNTAX_UID:
            dali_dcm_paths.append(dcm_path)
            dali_save_paths.append(save_path)
        else:
            dicomsdl_dcm_paths.append(dcm_path)
            dicomsdl_save_paths.append(save_path)

    # process with dali
    print('Convert with DALI:', len(dali_dcm_paths))
    start = time.time()
    j2k_temp_dir = os.path.join(save_root_dir, 'temp')
    convert_with_dali(
        dali_dcm_paths,
        dali_save_paths,
        chunk=64,  # disk
        batch_size=1,  # disk, ram_v3
        num_threads=2,
        py_num_workers=1,  # ram_v3
        device_id=0,
        cache='ram_v3',
        j2k_temp_dir=j2k_temp_dir,  # disk
    )

    end = time.time()
    print(f'\n---DALI done in {end - start} sec.\n')

    
    
    
    
    
class _Tester:

    def __init__(self):
        csv_path = '../../datasets/train.csv'
        dcm_root_dir = '../../datasets/train_images/'
        save_root_dir = 'temp_save'

        df = pd.read_csv(csv_path)
        from sklearn.utils import shuffle
        df = shuffle(df, random_state = 42).reset_index(drop = True)
        df = df[:2000]
        print('Total samples:', len(df))
        
        os.makedirs(save_root_dir, exist_ok=True)

        machine_id_to_syntax_uid = make_uid_transfer_dict(df, dcm_root_dir)
        dali_dcm_paths = []
        dali_save_paths = []
        dicomsdl_dcm_paths = []
        dicomsdl_save_paths = []
        for i, row in df.iterrows():
            dcm_path = os.path.join(dcm_root_dir, str(row.patient_id),
                                    f'{row.image_id}.dcm')
            save_path = os.path.join(save_root_dir,
                                     f'{row.patient_id}@{row.image_id}')
            syntax_uid = machine_id_to_syntax_uid[row.machine_id]
            if syntax_uid == J2K_SYNTAX_UID:
                dali_dcm_paths.append(dcm_path)
                dali_save_paths.append(save_path)
            else:
                dicomsdl_dcm_paths.append(dcm_path)
                dicomsdl_save_paths.append(save_path)

        self.save_root_dir = save_root_dir
        self.dali_dcm_paths = dali_dcm_paths
        self.dali_save_paths = dali_save_paths
        self.dicomsdl_dcm_paths = dicomsdl_dcm_paths
        self.dicomsdl_save_paths = dicomsdl_save_paths

    def speed_compare(self):
        dcm_paths = self.dali_dcm_paths
        save_paths = self.dali_save_paths
        save_root_dir = self.save_root_dir

        ########### PROCESS WITH DALI
        print('Convert with DALI:', len(dcm_paths))
        start = time.time()
        j2k_temp_dir = os.path.join(save_root_dir, 'temp')
        convert_with_dali_parallel(
            dcm_paths,
            save_paths,
            chunk=64,  # disk
            batch_size=1,  # disk, ram_v3
            num_threads=2,
            py_num_workers=1,  # ram_v3
            device_id=0,
            cache='ram_v3',
            j2k_temp_dir=j2k_temp_dir,  # disk
            parallel_n_jobs=4,
            parallel_backend='loky')
        end = time.time()
        print(f'\n---DALI done in {end - start} sec.\n')

        ############ PROCESS WITH DICOMSDL
        dicomsdl_dcm_paths = dcm_paths
        dicomsdl_save_paths = save_paths
        print('Convert with dicomsdl:', len(dicomsdl_dcm_paths))
        start = time.time()
        convert_with_dicomsdl_parallel(dicomsdl_dcm_paths,
                                       dicomsdl_save_paths,
                                       parallel_n_jobs=4,
                                       parallel_backend='loky')
        end = time.time()
        print(f'\n---Dicomsdl done in {end - start} sec.\n')

    def dali_parallel(self):
        dcm_paths = self.dali_dcm_paths
        save_paths = self.dali_save_paths
        save_root_dir = self.save_root_dir

        # process with dali
        print('Convert with DALI:', len(dcm_paths))
        start = time.time()
        j2k_temp_dir = os.path.join(save_root_dir, 'temp')
        convert_with_dali_parallel(
            dcm_paths,
            save_paths,
            chunk=64,  # disk
            batch_size=1,  # disk, ram_v3
            num_threads=2,
            py_num_workers=1,  # ram_v3
            device_id=0,
            cache='ram_v3',
            j2k_temp_dir=j2k_temp_dir,  # disk
            parallel_n_jobs=4,
            parallel_backend='loky')
        end = time.time()
        print(f'\n---DALI done in {end - start} sec.\n')

    def dicomsdl_parallel(self):
        dcm_paths = self.dali_dcm_paths
        save_paths = self.dali_save_paths

        ############ PROCESS WITH DICOMSDL
        dicomsdl_dcm_paths = dcm_paths
        dicomsdl_save_paths = save_paths
        print('Convert with dicomsdl:', len(dicomsdl_dcm_paths))
        start = time.time()
        convert_with_dicomsdl_parallel(dicomsdl_dcm_paths,
                                       dicomsdl_save_paths,
                                       parallel_n_jobs=4,
                                       parallel_backend='loky')
        end = time.time()
        print(f'\n---Dicomsdl done in {end - start} sec.\n')

    def compare_results(self, save_backend='cv2', save_dtype='uint8'):
        if save_backend == 'cv2':
            file_ext = 'png'
        elif save_backend == 'np':
            file_ext = 'npy'
            
#         dcm_paths = self.dali_dcm_paths[:50]
#         save_paths = self.dali_save_paths[:50]
        
        dcm_paths = self.dicomsdl_dcm_paths[:500]
        save_paths = self.dicomsdl_save_paths[:500]

        dali_dcm_paths = dcm_paths
        dicomsdl_dcm_paths = dcm_paths
        pydicom_dcm_paths = dcm_paths
        nvjpeg_dcm_paths = dcm_paths
        dali_save_paths = [
            f'{p}_{save_dtype}_dali.{file_ext}' for p in save_paths
        ]
        nvjpeg_save_paths = [
            f'{p}_{save_dtype}_nvjpeg.{file_ext}' for p in save_paths
        ]
        dicomsdl_save_paths = [
            f'{p}_{save_dtype}_dicomsdl.{file_ext}' for p in save_paths
        ]
        pydicom_save_paths = [
            f'{p}_{save_dtype}_pydicom.{file_ext}' for p in save_paths
        ]
        save_root_dir = self.save_root_dir

#         # process with dali
#         print('Convert with DALI:', len(dali_dcm_paths))
#         start = time.time()
#         j2k_temp_dir = os.path.join(save_root_dir, 'temp')
#         convert_with_dali_parallel(
#             dali_dcm_paths,
#             dali_save_paths,
#             save_backend = save_backend,
#             save_dtype = save_dtype,
#             chunk=64,  # disk
#             batch_size=1,  # disk, ram_v3
#             num_threads=1,
#             py_num_workers=1,  # ram_v3
#             device_id=0,
#             cache='ram_v3',
#             j2k_temp_dir=j2k_temp_dir,  # disk
#             parallel_n_jobs=4,
#             parallel_backend='multiprocessing')
#         end = time.time()
#         print(f'\n---DALI done in {end - start} sec.\n')
        
        
#         #### CONVERT WITH NVJPEG
#         print('Convert with nvjpeg2k:', len(nvjpeg_dcm_paths))
#         start = time.time()
#         convert_with_nvjpeg2k_parallel(nvjpeg_dcm_paths,
#                                        nvjpeg_save_paths,
#                                        save_backend = save_backend,
#                                        save_dtype = save_dtype,
#                                        parallel_n_jobs=4,
#                                        parallel_backend='loky')
#         end = time.time()
#         print(f'\n---NVJPEG2K done in {end - start} sec.\n')
        
        
        

        ############ PROCESS WITH DICOMSDL
        dicomsdl_dcm_paths = dcm_paths
        print('Convert with dicomsdl:', len(dicomsdl_dcm_paths))
        start = time.time()
        convert_with_dicomsdl_parallel(dicomsdl_dcm_paths,
                                       dicomsdl_save_paths,
                                       save_backend = save_backend,
                                       save_dtype = save_dtype,
                                       parallel_n_jobs=1,
                                       parallel_backend='loky')
        end = time.time()
        print(f'\n---Dicomsdl done in {end - start} sec.\n')

        ############ PROCESS WITH PYDICOM
        
        print('Convert with pydicom:', len(pydicom_dcm_paths))
        start = time.time()
        convert_with_pydicom_parallel(pydicom_dcm_paths,
                                       pydicom_save_paths,
                                       save_backend = save_backend,
                                       save_dtype = save_dtype,
                                       parallel_n_jobs=4,
                                       parallel_backend='loky')
        end = time.time()
        print(f'\n---Pydicom done in {end - start} sec.\n')

        print('Compare results:')
        for dali_save_path, nvjpeg_save_path, dicomsdl_save_path, pydicom_save_path in zip(
                dali_save_paths, nvjpeg_save_paths, dicomsdl_save_paths, pydicom_save_paths):
            print(dicomsdl_save_path)
            dali_img = load_img_from_file(dali_save_path, backend=save_backend)
            nvjpeg_img = load_img_from_file(nvjpeg_save_path, backend=save_backend)
            dicomsdl_img = load_img_from_file(dicomsdl_save_path,
                                              backend=save_backend)
            pydicom_img = load_img_from_file(pydicom_save_path,
                                             backend=save_backend)
            if dali_img is None or nvjpeg_img is None or dicomsdl_img is None or pydicom_img is None:
                continue
            print(dali_img.dtype, nvjpeg_img.dtype, dicomsdl_img.dtype, pydicom_img.dtype,
                  np.sum(dali_img - nvjpeg_img),
                  np.sum(nvjpeg_img - dicomsdl_img),
                  np.sum(dicomsdl_img - pydicom_img),
                  np.max(dicomsdl_img - pydicom_img),
                  np.mean(dicomsdl_img - pydicom_img))
            # np.testing.assert_allclose(dali_img, dicomsdl_img)
            
            
            
    def compare_dali_nvjpeg2k(self, save_backend='cv2', save_dtype='uint8'):
        if save_backend == 'cv2':
            file_ext = 'png'
        elif save_backend == 'np':
            file_ext = 'npy'
    
#         dcm_paths = self.dali_dcm_paths[:500]
#         save_paths = self.dali_save_paths[:500]
        
        dcm_paths = self.dicomsdl_dcm_paths[:500]
        save_paths = self.dicomsdl_save_paths[:500]

        dali_dcm_paths = dcm_paths
        dicomsdl_dcm_paths = dcm_paths
        dali_save_paths = [
            f'{p}_{save_dtype}_dali.{file_ext}' for p in save_paths
        ]
        nvjpeg2k_save_paths = [
            f'{p}_{save_dtype}_nvjpeg2k.{file_ext}' for p in save_paths
        ]
        
        save_root_dir = self.save_root_dir
        
#         # process with dali
#         print('Convert with DALI:', len(dali_dcm_paths))
#         start = time.time()
#         j2k_temp_dir = os.path.join(save_root_dir, 'temp')
#         convert_with_dali_parallel(
#             dali_dcm_paths,
#             dali_save_paths,
#             save_backend = save_backend,
#             save_dtype = save_dtype,
#             chunk=64,  # disk
#             batch_size=1,  # disk, ram_v3
#             num_threads=1, # default 2
#             py_num_workers=1,  # ram_v3
#             device_id=0,
#             cache='ram_v3',
#             j2k_temp_dir=j2k_temp_dir,  # disk
#             parallel_n_jobs=2,
#             parallel_backend='loky')
#         end = time.time()
#         print(f'\n---DALI done in {end - start} sec.\n')

        
        
        ############ PROCESS WITH NVJPEG2K
        nvjpeg2k_dcm_paths = dcm_paths
        print('Convert with nvjpeg2k:', len(nvjpeg2k_dcm_paths))
        start = time.time()
        convert_with_nvjpeg2k_parallel(nvjpeg2k_dcm_paths,
                                       nvjpeg2k_save_paths,
                                       save_backend = save_backend,
                                       save_dtype = save_dtype,
                                       parallel_n_jobs=4,
                                       parallel_backend='loky')
        end = time.time()
        print(f'\n---NVJPEG2K done in {end - start} sec.\n')
        

        print('Compare results:')
        for dali_save_path, nvjpeg2k_save_path in zip(
                dali_save_paths, nvjpeg2k_save_paths):
            print(dali_save_path, nvjpeg2k_save_path)
            dali_img = load_img_from_file(dali_save_path, backend=save_backend)
            nvjpeg2k_img = load_img_from_file(nvjpeg2k_save_path,
                                              backend=save_backend)
            if dali_img is None or nvjpeg2k_img is None:
                print('???')
                continue
            
            diff = np.sum(dali_img - nvjpeg2k_img)
            print(dali_img.dtype, nvjpeg2k_img.dtype,
                  diff)
            assert diff == 0
            # np.testing.assert_allclose(dali_img, nvjpeg2k_img)
