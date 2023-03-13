#! /bin/bash

# reproduce YOLOX nano 416 training
PYTHONPATH=$(pwd)/src/roi_det/YOLOX:$PYTHONPATH python3 src/roi_det/YOLOX/tools/train.py \
    -expn yolox_nano_416_reproduce \
    -f src/roi_det/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py \
    -d 0 -b 64 -o \
    -c assets/public_pretrains/yolox_nano.pth \
    seed 42

# validate (optional)
PYTHONPATH=$(pwd)/src/roi_det/YOLOX:$PYTHONPATH python3 src/roi_det/YOLOX/tools/eval.py \
    -expn yolox_nano_416_reproduce \
    -f src/roi_det/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py \
    -d 0 -b 64 \
    seed 42

# convert newly trained best checkpoint to TensorRT
# this may take a while
PYTHONPATH=$(pwd)/src/roi_det/YOLOX:$PYTHONPATH python3 src/roi_det/YOLOX/tools/trt.py \
    -expn yolox_nano_416_reproduce \
    -f src/roi_det/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py \
    -b 1

# select and copy best checkpoint to final selection model dir
python3 src/tools/select_yolox_best_ckpt.py

