"""
Prepair YOLOX detection dataset.
- Convert competition's raw dicom to png
- Convert YOLOv5 format --> COCO format
"""

import argparse
import os
import shutil
import sys

import cv2
from yolov5_2_coco import YOLOV5ToCOCO

from settings import SETTINGS
from src.utils import misc as misc_utils
from src.utils.dicom import convert_with_dicomsdl_parallel


def parse_args():
    parser = argparse.ArgumentParser('Prepair YOLOX ROI detection dataset in COCO format')
    parser.add_argument('--num-workers',
                        type=int,
                        default=4,
                        help='Number of workers for dicomsdl decoding.')
    args = parser.parse_args()
    return args


def main(args):
    ASSET_ROI_YOLOV5_DATA_DIR = os.path.join(SETTINGS.ASSETS_DIR, 'data', 'roi_det_yolov5_format')

    KAGGLE_DCM_DIR = os.path.join(SETTINGS.RAW_DATA_DIR,
                                  'rsna-breast-cancer-detection',
                                  'train_images')
    ROI_YOLOV5_DATA_DIR = os.path.join(SETTINGS.PROCESSED_DATA_DIR,
                                       'roi_det_yolox', 'yolov5_format')

    ROI_COCO_DATA_DIR = os.path.join(SETTINGS.PROCESSED_DATA_DIR,
                                     'roi_det_yolox', 'coco_format')
    
    # Copy manually annotated label
    misc_utils.rm_and_mkdir(os.path.dirname(ROI_YOLOV5_DATA_DIR))
    print(f'Copy from {ASSET_ROI_YOLOV5_DATA_DIR} --> {ROI_YOLOV5_DATA_DIR}')
    shutil.copytree(ASSET_ROI_YOLOV5_DATA_DIR,  ROI_YOLOV5_DATA_DIR)
    

    misc_utils.rm_and_mkdir(os.path.join(ROI_YOLOV5_DATA_DIR, 'images'))
    misc_utils.rm_and_mkdir(
        os.path.join(ROI_YOLOV5_DATA_DIR, 'background_images'))

    dcm_paths = []
    save_paths = []
    for split in ['train', 'val']:
        txt_list_path = os.path.join(ROI_YOLOV5_DATA_DIR, f'{split}.txt')
        with open(txt_list_path, 'r') as f:
            content = f.read()
        paths = [line for line in content.split('\n') if line]
        names = [os.path.basename(p) for p in paths]
        for name in names:
            patient_id, image_id = name.split('.')[0].split('@')
            dcm_path = os.path.join(KAGGLE_DCM_DIR, patient_id,
                                    f'{image_id}.dcm')
            save_path = os.path.join(ROI_YOLOV5_DATA_DIR, 'images', name)
            dcm_paths.append(dcm_path)
            save_paths.append(save_path)

    assert len(dcm_paths) == len(save_paths)
    print('Total:', len(dcm_paths))

    print('Converting dicom to png..')
    # convert dicom to png (full resolution)
    convert_with_dicomsdl_parallel(dcm_paths,
                                   save_paths,
                                   normalization='min_max',
                                   save_backend='cv2',
                                   save_dtype='uint8',
                                   parallel_n_jobs=args.num_workers,
                                   joblib_backend='loky',
                                   legacy=True)

    print('Converting YOLOv5 format to COCO format..')
    # YOLOv5 format to COCO format
    # https://github.com/RapidAI/YOLO2COCO/blob/main/yolov5_2_coco.py
    yolov5_to_coco_converter = YOLOV5ToCOCO(src_dir=ROI_YOLOV5_DATA_DIR,
                                            dst_dir=ROI_COCO_DATA_DIR)
    yolov5_to_coco_converter(mode_list=['train', 'val'])


if __name__ == '__main__':
    args = parse_args()
    main(args)