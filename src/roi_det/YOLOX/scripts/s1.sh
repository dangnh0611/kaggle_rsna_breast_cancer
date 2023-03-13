python -m yolox.tools.train -f exps/example/custom/yolox_nano_bre_416.py -d 1 -b 64 -o -c pretrains/yolox_nano.pth

python -m yolox.tools.detect -f exps/example/custom/yolox_nano_bre_416.py \
    -c YOLOX_outputs/yolox_nano_bre_416/best_ckpt.pth -b 16 -d 1 --conf 0.5 --nms 0.9 \
    --tsize 416 --speed --fuse --workers 2 --input image_dir --output runs/detect/yolox_nano_bre_416