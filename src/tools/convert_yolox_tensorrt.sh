

cd src/roi_det/YOLOX
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 tools/trt.py -f /kaggle/input/kaggle-rsna/yolox_nano_bre_416.py -c /kaggle/input/kaggle-rsna/yolox_nano_bre_416_v2.pth -b 1