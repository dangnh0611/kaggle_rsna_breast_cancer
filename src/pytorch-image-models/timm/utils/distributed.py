""" Distributed training/validation utils

Hacked together by / Copyright 2020 Ross Wightman
"""
import os

import torch
from torch import distributed as dist

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from .model import unwrap_model
import logging

logger = logging.getLogger(__name__)


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt


def gather_tensor(tensor, cur_rank, dst_rank, world_size, group = None):
    if cur_rank == dst_rank:
        tensor_list = [
            torch.zeros_like(tensor)
            for _ in range(world_size)
        ]
        dist.gather(tensor, tensor_list, dst=dst_rank, group=group)
        return tensor_list
    else:
        tensor_list = None
        dist.gather(tensor, tensor_list, dst=dst_rank, group=group)
        return tensor_list


def all_gather_tensor(tensor, world_size, group = None):
    tensor_list = [
        torch.zeros_like(tensor)
        for _ in range(world_size)
    ]
    dist.all_gather(tensor_list, tensor, group=group)
    return tensor_list


def all_gather_tensor_into_tensor(tensor, world_size, group = None):
    # concat along batch dim
    output_tensor = torch.zeros(tensor.size(0) * world_size, *tensor.size()[1:])
    dist.all_gather_into_tensor(output_tensor, tensor, group=group)
    return output_tensor



def distribute_bn(model, world_size, reduce=False):
    # ensure every node has the same running bn stats
    for bn_name, bn_buf in unwrap_model(model).named_buffers(recurse=True):
        if ('running_mean' in bn_name) or ('running_var' in bn_name):
            if reduce:
                # average bn stats across whole group
                torch.distributed.all_reduce(bn_buf, op=dist.ReduceOp.SUM)
                bn_buf /= float(world_size)
            else:
                # broadcast bn stats from rank 0 to whole group
                torch.distributed.broadcast(bn_buf, 0)


def is_global_primary(args):
    return args.rank == 0


def is_local_primary(args):
    return args.local_rank == 0


def is_primary(args, local=False):
    return is_local_primary(args) if local else is_global_primary(args)


def is_distributed_env():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) > 1
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) > 1
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    # TBD, support horovod?
    # if args.horovod:
    #     assert hvd is not None, "Horovod is not installed"
    #     hvd.init()
    #     args.local_rank = int(hvd.local_rank())
    #     args.rank = hvd.rank()
    #     args.world_size = hvd.size()
    #     args.distributed = True
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)
    #     os.environ['RANK'] = str(args.rank)
    #     os.environ['WORLD_SIZE'] = str(args.world_size)
    dist_backend = getattr(args, 'dist_backend', 'nccl')
    dist_url = getattr(args, 'dist_url', 'env://')
    if is_distributed_env():
        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=args.world_size,
                rank=args.rank,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

    if torch.cuda.is_available():
        if args.distributed:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    args.device = device
    device = torch.device(device)
    return device


# COPY FROM YOLOX
import functools
import os
import pickle
import time
from contextlib import contextmanager

import numpy as np

import torch
from torch import distributed as dist


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger.warning(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Returns:
        list[int]: size of the tensor, on each rank
        Tensor: padded tensor that has the max size
    """
    world_size = dist.get_world_size(group=group)
    assert (
        world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)

    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size,), dtype=torch.uint8, device=tensor.device
        )
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

# DEPRECATED
def _all_gather_obj(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
        for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list

# DEPRECATED
def _gather_obj(data, dst=0, group=None):
    """
    Run gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        dst (int): destination rank
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: on dst, a list of data gathered from each rank. Otherwise,
            an empty list.
    """
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving Tensor from all ranks
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [
            torch.empty((max_size,), dtype=torch.uint8, device=tensor.device)
            for _ in size_list
        ]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []
