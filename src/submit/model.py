import numpy as np
import torch
from timm.data import resolve_data_config
from timm.models import create_model
from torch import nn


class KFoldEnsembleModel(nn.Module):

    def __init__(self, model_info, ckpt_paths):
        super(KFoldEnsembleModel, self).__init__()
        fmodels = []
        for i, ckpt_path in enumerate(ckpt_paths):
            print(f'Loading model from {ckpt_path}')
            fmodel = create_model(
                model_info['model_name'],
                num_classes=model_info['num_classes'],
                in_chans=model_info['in_chans'],
                pretrained=False,
                checkpoint_path=ckpt_path,
                global_pool=model_info['global_pool'],
            ).eval()
            data_config = resolve_data_config({}, model=fmodel)
            print('Data config:', data_config)
            mean = np.array(data_config['mean']) * 255
            std = np.array(data_config['std']) * 255
            print(f'mean={mean}, std={std}')
            fmodels.append(fmodel)
        self.fmodels = nn.ModuleList(fmodels)

        self.register_buffer('mean',
                             torch.FloatTensor(mean).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor(std).reshape(1, 3, 1, 1))

    def forward(self, x):
        #         x = x.sub(self.mean).div(self.std)
        x = (x - self.mean) / self.std
        probs = []
        for fmodel in self.fmodels:
            logits = fmodel(x)
            #             prob = logits.softmax(dim=1)[:, 1]
            prob = logits.sigmoid()[:, 0]
            probs.append(prob)
        probs = torch.stack(probs, dim=1)
        return probs