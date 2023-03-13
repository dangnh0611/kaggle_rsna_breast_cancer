import os

import cv2
import numpy as np
import shutil


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
    

def rm_and_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok= False)


def make_symlink(src, dst):
    abs_src = os.path.abspath(src)
    abs_dst = os.path.abspath(dst)
    assert os.path.exists(abs_src)
    os.symlink(abs_src, abs_dst)