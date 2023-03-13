import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class RSNADataset(Dataset):

    def __init__(self,
                 csv_path,
                 img_dir,
                 augment_fn=None,
                 transform_fn=None,
                 n_channels=3,
                 subset='train'):
        assert subset in ['train', 'val']
        self.subset = subset
        self.img_paths = []
        self.labels = []
        self.augment_fn = augment_fn
        self.transform_fn = transform_fn
        self.n_channels = n_channels

        data_dict = {
            'kaggle': {
                'csv_path': csv_path,
                'img_dir': img_dir,
                'separator': '@'
            }
        }

        if subset == 'train':
            data_dict.update({
                'bmcd': {
                    'csv_path':
                    '/home/dangnh36/datasets/.comp/rsna/external/bmcd/label_v2.csv',
                    'img_dir':
                    '/raid/.comp/external/bmcd/uint8_voilut_png@yolox_nano_bre_416_datav2',
                    'separator': '@'
                },
                'cdd_cesm': {
                    'csv_path':
                    '/home/dangnh36/datasets/.comp/rsna/external/cdd_cesm/label_v2.csv',
                    'img_dir':
                    '/raid/.comp/external/cdd_cesm/uint8_voilut_png@yolox_nano_bre_416_datav2',
                    'separator': '@'
                },
                'cmmd': {
                    'csv_path':
                    '/home/dangnh36/datasets/.comp/rsna/external/cmmd/label_v2.csv',
                    'img_dir':
                    '/raid/.comp/external/cmmd/uint8_voilut_png@yolox_nano_bre_416_datav2',
                    'separator': '@'
                },
                'miniddsm': {
                    'csv_path':
                    '/home/dangnh36/datasets/.comp/rsna/external/miniddsm/label_v2.csv',
                    'img_dir':
                    '/raid/.comp/external/miniddsm/uint8_voilut_png@yolox_nano_bre_416_datav2',
                    'separator': '~'
                },
                'vindr': {
                    'csv_path':
                    '/home/dangnh36/datasets/.comp/rsna/external/vindr/all_labels.csv',
                    'img_dir':
                    '/raid/.comp/external/vindr/uint8_voilut_png@yolox_nano_bre_416_datav2',
                    'separator': '@'
                },
            })

        print('----------------------------')
        print(subset)
        print(data_dict)
        print('----------------------------')

        for data_name, data_info in data_dict.items():
            print('DATANAME:', data_name)
            data_csv_path = data_info['csv_path']
            data_img_dir = data_info['img_dir']
            data_sep = data_info['separator']
            df = pd.read_csv(data_csv_path)
            self.df = df
            for i in tqdm(range(len(df))):
                patient_id = df.at[i, 'patient_id']
                image_id = df.at[i, 'image_id']
                img_name = f'{patient_id}{data_sep}{image_id}.png'
                img_path = os.path.join(data_img_dir, img_name)
                if i == 0:
                    _tmp_img = cv2.imread(img_path)
                    assert _tmp_img is not None
                    del _tmp_img
                label = df.at[i, 'cancer']
                self.img_paths.append(img_path)
                self.labels.append(label)
            print(f'Done loading {data_name} with {len(df)} samples.')
        print(f'DATASET TOTAL LENGTH: {len(self.labels)} with positive percent = {sum(self.labels) / len(self.labels)}')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        if self.n_channels == 3:
            img = cv2.imread(img_path)
        elif self.n_channels == 1:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            raise AssertionError()
        if self.augment_fn:
            img = self.augment_fn(img)
        if self.transform_fn:
            img = self.transform_fn(img)
        return img, label

    def get_sampler_weights(self, pos_neg_ratio):
        assert pos_neg_ratio > 0
        labels = np.array(self.labels)
        num_pos = labels.sum()
        num_neg = len(labels) - num_pos
        ori_pos_neg_ratio = num_pos / num_neg
        pos_weight = pos_neg_ratio / ori_pos_neg_ratio
        print('Original pos/neg ratio:', ori_pos_neg_ratio)
        print('Expect pos/neg ratio:', pos_neg_ratio)
        print('Pos weight:', pos_weight)
        weights = np.ones_like(labels, dtype=np.float32)
        weights[labels == 1] = pos_weight
        return weights

    def get_labels(self):
        return np.array(self.labels)

    def get_df(self):
        assert self.subset == 'val'
        return self.df
