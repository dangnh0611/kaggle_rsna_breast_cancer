#!/bin/bash
NUM_PROC=$1
shift
python3 -m torch.distributed.launch --master_port=$RANDOM --nproc_per_node=$NUM_PROC train_exp.py "$@"

