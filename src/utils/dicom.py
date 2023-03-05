# https://github.com/pydicom/pydicom/issues/1554
# https://github.com/pydicom/pydicom/issues/539

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
import gc
from pydicom.pixel_data_handlers.util import apply_voi_lut, pixel_dtype

J2K_SYNTAX_UID = '1.2.840.10008.1.2.4.90'
J2K_HEADER = b"\x00\x00\x00\x0C"

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
        self.metadatas = [None for _ in range(len(self.dcm_paths))]

    def __call__(self, batch_info):
        batch_idx = batch_info.iteration
        start = batch_idx * self.batch_size
        end = min(self.len, start + self.batch_size)
        if end <= start:
            raise StopIteration()
        # print('IDX:', batch_info.iteration, batch_info.epoch_idx)

        j2k_streams = []
        for idx in range(start, end):
            dcm_path = self.dcm_paths[idx]
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
            j2k_stream = np.array(bytearray(pixel_data[offset:]), np.uint8)
            j2k_streams.append(j2k_stream)
            self.metadatas[idx]  = PydicomMetadata(dcm)
        return (j2k_streams,)


@dali.pipeline_def
def _j2k_decode_pipeline_ram_v3(eii, resize=None, num_outputs=None):
    jpeg = dali.fn.external_source(
        source=eii,
        num_outputs=num_outputs,
        dtype=[dali.types.UINT8],
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
    return image


def convert_with_dali_ram_v3(dcm_paths,
                             save_paths,
                             save_backend='cv2',
                             save_dtype='uint8',
                             chunk=-1,
                             batch_size=1,
                             num_threads=2,
                             py_num_workers=1,
                             device_id=0,
                             voilut = True):
    del chunk
    assert len(dcm_paths) == len(save_paths)
    assert save_backend in ['cv2', 'np']
    num_dcms = len(dcm_paths)

    # dali to process with chunk in-memory
    external_source = _DicomToJ2kExternalSourceV3(dcm_paths,
                                                  batch_size=batch_size)
    pipe = _j2k_decode_pipeline_ram_v3(
        external_source,
        num_outputs=1,
        py_num_workers=py_num_workers,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        debug=False,
    )
    pipe.build()

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
        for j in range(len(imgs)):
            cur_idx += 1
            save_path = save_paths[cur_idx]
            img = imgs.at(j).squeeze(-1)  # uint16
            metadata = external_source.metadatas[cur_idx]

            if voilut:
                assert save_dtype in ['uint8']
                if save_dtype == 'uint8':
                    y_min = 0
                    y_max = 255
                img = apply_voilut(img,
                  window_width=metadata.window_widths[0],
                  window_center=metadata.window_centers[0],
                  voi_func=metadata.voilut_func,
                  y_min=y_min,
                  y_max=y_max)
                if metadata.invert:
                    img = y_max - img
                img = img.astype(np.uint8)
            else:
                img = from_uint16(img,
                                metadata.invert,
                                save_dtype,
                                bit_scale=False,
                                nbit=metadata.nbit)
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
        _ = Parallel(n_jobs=parallel_n_jobs, backend=parallel_backend)(
            delayed(convert_with_dali)(
                dcm_paths[start:end],
                save_paths[start:end],
                save_backend=save_backend,
                save_dtype=save_dtype,
                chunk=chunk,  # disk
                batch_size=batch_size,  # disk, ram_v3
                num_threads=num_threads,
                py_num_workers=py_num_workers,  # ram_v3
                device_id=worker_device_id,
                cache=cache,
                j2k_temp_dir=j2k_temp_dir,  # disk)
            ) for start, end, worker_device_id in zip(starts, ends, device_id))


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
        if dtype == 'uint8':
            y_min = 0
            y_max = 255
        img = apply_voilut(img,
            window_width=metadata.window_widths[0],
            window_center=metadata.window_centers[0],
            voi_func=metadata.voilut_func,
            y_min=y_min,
            y_max=y_max)
        if metadata.invert:
            img = y_max - img
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
        csv_path = '/home/dangnh36/datasets/.comp/rsna/train.csv'
        dcm_root_dir = '/home/dangnh36/datasets/.comp/rsna/train_images'
        save_root_dir = '/home/dangnh36/datasets/.comp/rsna/export/png_ori'

        df = pd.read_csv(csv_path)
        print('Total samples:', len(df))

        df = df[:1000]
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
        dcm_paths = self.dali_dcm_paths[:50]
        save_paths = self.dali_save_paths[:50]

        dali_dcm_paths = dcm_paths
        dicomsdl_dcm_paths = dcm_paths
        dali_save_paths = [
            f'{p}_{save_dtype}_dali.{file_ext}' for p in save_paths
        ]
        dicomsdl_save_paths = [
            f'{p}_{save_dtype}_dicomsdl.{file_ext}' for p in save_paths
        ]
        pydicom_save_paths = [
            f'{p}_{save_dtype}_pydicom.{file_ext}' for p in save_paths
        ]
        save_root_dir = self.save_root_dir

        # process with dali
        print('Convert with DALI:', len(dali_dcm_paths))
        start = time.time()
        j2k_temp_dir = os.path.join(save_root_dir, 'temp')
        convert_with_dali_parallel(
            dali_dcm_paths,
            dali_save_paths,
            save_backend = save_backend,
            save_dtype = save_dtype,
            chunk=64,  # disk
            batch_size=1,  # disk, ram_v3
            num_threads=2,
            py_num_workers=1,  # ram_v3
            device_id=0,
            cache='ram_v3',
            j2k_temp_dir=j2k_temp_dir,  # disk
            parallel_n_jobs=2,
            parallel_backend='loky')
        end = time.time()
        print(f'\n---DALI done in {end - start} sec.\n')

        ############ PROCESS WITH DICOMSDL
        dicomsdl_dcm_paths = dcm_paths
        print('Convert with dicomsdl:', len(dicomsdl_dcm_paths))
        start = time.time()
        convert_with_dicomsdl_parallel(dicomsdl_dcm_paths,
                                       dicomsdl_save_paths,
                                       save_backend = save_backend,
                                       save_dtype = save_dtype,
                                       parallel_n_jobs=2,
                                       parallel_backend='loky')
        end = time.time()
        print(f'\n---Dicomsdl done in {end - start} sec.\n')

        ############ PROCESS WITH PYDICOM
        pydicom_dcm_paths = dcm_paths
        print('Convert with pydicom:', len(pydicom_dcm_paths))
        start = time.time()
        convert_with_pydicom_parallel(pydicom_dcm_paths,
                                       pydicom_save_paths,
                                       save_backend = save_backend,
                                       save_dtype = save_dtype,
                                       parallel_n_jobs=2,
                                       parallel_backend='loky')
        end = time.time()
        print(f'\n---Pydicom done in {end - start} sec.\n')

        print('Compare results:')
        for dali_save_path, dicomsdl_save_path, pydicom_save_path in zip(
                dali_save_paths, dicomsdl_save_paths, pydicom_save_paths):
            print(dicomsdl_save_path)
            dali_img = load_img_from_file(dali_save_path, backend=save_backend)
            dicomsdl_img = load_img_from_file(dicomsdl_save_path,
                                              backend=save_backend)
            pydicom_img = load_img_from_file(pydicom_save_path,
                                             backend=save_backend)
            if dali_img is None or dicomsdl_img is None or pydicom_img is None:
                continue
            print(dali_img.dtype, dicomsdl_img.dtype, pydicom_img.dtype,
                  np.sum(dali_img - dicomsdl_img),
                  np.sum(dicomsdl_img - pydicom_img),
                  np.max(dicomsdl_img - pydicom_img),
                  np.mean(dicomsdl_img - pydicom_img))
            # np.testing.assert_allclose(dali_img, dicomsdl_img)


if __name__ == '__main__':
    tester = _Tester()
    # tester.speed_compare()
    # tester.dali_parallel()
    # tester.dicomsdl_parallel()
    tester.compare_results(save_backend='cv2', save_dtype='uint16')

# DALI
# chunk 1, 2 threads: 400-800 MB, 53 secs
# chunk 1, 1 threads: 400-800 MB, 51 secs
# chunk 16, 2 threads: 2500 MB, 27 secs
# chunk 16, 1 threads: 2500 MB, 26.8 secs
# chunk 32, 2 threads: 3800 MB, 25.5 secs
# chunk 32, 1 threads: 3800 MB, 33.3 secs
# Num threads improve speed when use larger chunk (ballance the GPU and CPU workload)
# For small chunk (e.g 1), use more threads only increase thread initialization time

# chuck 32, 2 threads, disk: 49.1 secs

# Test with :500
# ram_v3, batch 1: 50s
# ram_v1: 69s
# disk: 56s
