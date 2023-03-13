import math
import torch
import numpy as np
from torch.utils.data import Sampler
import torch.distributed as dist


class BalanceSampler(Sampler):

    def __init__(self, dataset, ratio=8):
        self.r = ratio-1
        self.dataset = dataset
        labels = dataset.get_labels()
        self.pos_index = np.where(labels>0)[0]
        self.neg_index = np.where(labels==0)[0]
        print('Num pos:', len(self.pos_index))
        print('Num neg:', len(self.neg_index))

        self.neg_length = self.r*int(np.floor(len(self.neg_index)/self.r))
        self.len = self.neg_length + self.neg_length // self.r

    def __iter__(self):
        pos_index = self.pos_index.copy()
        neg_index = self.neg_index.copy()
        np.random.shuffle(pos_index)
        np.random.shuffle(neg_index)

        neg_index = neg_index[:self.neg_length].reshape(-1,self.r)
        pos_index = np.random.choice(pos_index, self.neg_length//self.r).reshape(-1,1)

        index = np.concatenate([pos_index,neg_index],-1).reshape(-1)
        return iter(index)

    def __len__(self):
        return self.len



class BalanceSamplerV2(Sampler):
    def __init__(self, dataset, batch_size, num_sched_epochs, num_epochs, start_ratio = 1/4, end_ratio = 1/8, one_pos_mode = True, seed = 42):
        np.random.seed(seed)
        self.dataset = dataset
        self.batch_size = batch_size
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.num_sched_epochs = num_sched_epochs
        self.num_epochs = num_epochs
        labels = dataset.get_labels()
        
        self.pos_idxs = np.where(labels>0)[0]
        self.neg_idxs = np.where(labels==0)[0]
        self.num_pos = len(self.pos_idxs)
        self.num_neg = len(self.neg_idxs)
        print(f'Num pos: {self.num_pos}, Num neg: {self.num_neg}')

        # percentage of pos samples per epoch
        if start_ratio != end_ratio:
            self.ratios = np.linspace(start_ratio, end_ratio, num_sched_epochs)
            self.ratios = np.concatenate([self.ratios, np.full((num_epochs - num_sched_epochs,), end_ratio)])
        else:
            self.ratios = np.full((num_epochs,), start_ratio)
        print('RATIO PER EPOCHS:', self.ratios)
        assert len(self.ratios) == num_epochs

        # pre-compute sampler indexs per epoch
        self.pre_compute_epoch_idxs = []
        for i, ratio in enumerate(self.ratios):
            epoch_idxs = self._pre_compute_epoch_idxs(ratio, one_pos_mode= one_pos_mode)
            self.pre_compute_epoch_idxs.append(epoch_idxs)
        self.cur_epoch = None

        self.set_epoch(0)


    def set_epoch(self, ep):
        assert ep < self.num_epochs, f'WARNING: Invalid set_epoch() with ep={ep} while max_epoch={self.num_epochs}'
        print(f'Set epoch to {ep} with sampler ratio = {self.ratios[ep]}')
        self.cur_epoch = ep
        self.len = len(self.pre_compute_epoch_idxs[ep])

    def _pre_compute_epoch_idxs(self, ratio, one_pos_mode = True):
        print(f'Pre-compute for ratio = {ratio}')
        epoch_num_pos = int(ratio * self.num_neg)
        epoch_num_iters = (epoch_num_pos + self.num_neg) // self.batch_size
        epoch_num_total = epoch_num_iters * self.batch_size
        # never downsampling neg samples if possible
        epoch_num_pos = epoch_num_total - self.num_neg
        
        # sampling pos idxs
        min_pos_per_iter = epoch_num_pos // epoch_num_iters
        if min_pos_per_iter < 1:
            if one_pos_mode:
                min_pos_per_iter = 1
                epoch_num_pos = epoch_num_iters
                print(f'ONE POS MODE: Switch num_pos_samples to {epoch_num_pos}')
            else:
                print(f"WARNING: At least one batch which has no positive sample: {epoch_num_pos}, {epoch_num_iters}.\n")
            
        ret_idxs = []
        pool_pos_idxs = []
        _count = 0
        while _count < epoch_num_pos:
            temp_pos_idxs = self.pos_idxs.copy()
            np.random.shuffle(temp_pos_idxs)
            pool_pos_idxs.append(temp_pos_idxs)
            _count += len(temp_pos_idxs)
        pool_pos_idxs = np.concatenate(pool_pos_idxs, axis = 0)
        assert len(pool_pos_idxs) >= epoch_num_pos

        _start = 0
        _end = 0
        for i in range(epoch_num_iters):
            _start = i * min_pos_per_iter
            _end = (i+1) * min_pos_per_iter
            ret_idxs.append(pool_pos_idxs[_start:_end].tolist())
        num_pos_remain = epoch_num_pos - _end
        assert num_pos_remain == epoch_num_pos % epoch_num_iters
        pool_remain_pos_idxs = pool_pos_idxs[_end:epoch_num_pos]
        assert len(pool_remain_pos_idxs) == num_pos_remain
        for i, j in enumerate(np.random.choice(np.arange(0, epoch_num_iters, 1), num_pos_remain, replace = False)):
            ret_idxs[j].append(pool_remain_pos_idxs[i])
            
        # sampling neg idxs
        pool_neg_idxs = self.neg_idxs.copy()
        np.random.shuffle(pool_neg_idxs)

        _cur = 0
        for i in range(epoch_num_iters):
            iter_idxs = ret_idxs[i]
            assert len(iter_idxs) - min_pos_per_iter <= 1
            _end = _cur + self.batch_size - len(iter_idxs)
            iter_idxs.extend(pool_neg_idxs[_cur: _end].tolist())
            _cur = _end
        if not one_pos_mode:
            assert _cur == len(pool_neg_idxs)

        ret_idxs = np.array(ret_idxs)
        assert ret_idxs.shape[0] == epoch_num_iters and ret_idxs.shape[1] == self.batch_size
        ret_idxs = ret_idxs.reshape(-1)
        return ret_idxs


    def __iter__(self):
        print(f'STARTING COMPUTE EPOCH {self.cur_epoch} SAMPLE INDEXS...')
        print('Current epoch sampler ratio:', self.ratios[self.cur_epoch])
        epoch_idxs = self.pre_compute_epoch_idxs[self.cur_epoch]
        print(f'{len(epoch_idxs) // self.batch_size} iters with {len(epoch_idxs)} samples')
        return iter(epoch_idxs)


    def __len__(self):
        return self.len
