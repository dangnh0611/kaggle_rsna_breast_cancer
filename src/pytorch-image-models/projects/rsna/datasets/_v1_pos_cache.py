import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os


class RSNADataset(Dataset):
    def __init__(self, csv_path, img_dir, augment_fn = None, transform_fn = None):
        self.img_paths = []
        self.labels = []
        self.augment_fn = augment_fn
        self.transform_fn = transform_fn

        df = pd.read_csv(csv_path)
        self.df = df
        self.imgs = [None for i in range(len(df))]
        for i in tqdm(range(len(df))):
            patient_id = df.at[i, 'patient_id']
            image_id = df.at[i, 'image_id']
            img_name = f'{patient_id}@{image_id}.png'
            img_path = os.path.join(img_dir, img_name)
            label = df.at[i, 'cancer']
            self.img_paths.append(img_path)
            if label == 1:
                img = cv2.imread(img_path)
                self.imgs[i] = img
            self.labels.append(label)
        assert len(self.img_paths) == len(self.labels) == len(self.imgs)
        print(f'Done loading dataset with {len(self.labels)} samples.')

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]
        if label == 1:
            img = self.imgs[idx]
        else:
            img = cv2.imread(img_path)
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
        weights = np.ones_like(labels, dtype = np.float32)
        weights[labels==1] = pos_weight
        return weights

    def get_labels(self):
        return np.array(self.labels)

    def get_df(self):
        return self.df
    