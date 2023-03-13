Reproduce solution from scratch

```bash
# Prepair ROI detection dataset
python3 src/tools/prepair_roi_det_dataset.py --num-workers 4

# Train and convert YOLOX to TensorRT as ROI detector
sh src/tools/train_and_convert_yolox_trt.sh

# Prepair classification dataset
python3 src/tools/prepair_classification_dataset.py --num-workers 8

# Train-val splitting
python3 src/tools/cv_split.py

# Train 4 x Convnext-small classification model
python3 src/tools/make_train_bash_script.py --mode fully_reproduce
sh ./_train_script_auto_generated.sh

# Select best checkpoints
python3 src/tools/select_classification_best_ckpts.py --mode fully_reproduce

# Convert trained 4 x Convnext-small to TensorRT
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/tools/convert_convnext_tensorrt.py --mode reproduce

# Inference on test data
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/submit/submit.py --mode reproduce --trt
```

For details, check [README.md](README.md)