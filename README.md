1st place solution for [RSNA Screening Mammography Breast Cancer Detection competition on Kaggle](https://www.kaggle.com/competitions/rsna-breast-cancer-detection)

Solution write up: https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449

![overall pipeline](docs/figs/preprocess.png)

**Notes:**
- Copy of the trained models can not be upload since the total size is > 2GB. So I create a kaggle dataset to store theme: https://www.kaggle.com/datasets/dangnh0611/rsna-breast-cancer-detection-best-ckpts

Please download those trained models and put in `assets/trained/`:
```bash
# this assume that kaggle api is installed: https://github.com/Kaggle/kaggle-api
kaggle datasets download -d dangnh0611/rsna-breast-cancer-detection-best-ckpts -p assets/trained
unzip rsna-breast-cancer-detection-best-ckpts.zip -d assets/trained/
rm assets/trained/rsna-breast-cancer-detection-best-ckpts.zip
```


# TABLE OF CONTENTS
- [TABLE OF CONTENTS](#table-of-contents)
- [1. ARCHIVE CONTENTS](#1-archive-contents)
- [2. HARDWARE](#2-hardware)
- [3. DATA SETUP](#3-data-setup)
- [4. SOLUTION PIPELINE](#4-solution-pipeline)
- [5. SOLUTION REPRODUCING](#5-solution-reproducing)
  - [5.1. Use trained models to make predictions](#51-use-trained-models-to-make-predictions)
    - [5.1.1. Convert trained YOLOX to TensorRT](#511-convert-trained-yolox-to-tensorrt)
    - [5.1.2. Convert trained 4 x Convnext-small models to TensorRT](#512-convert-trained-4-x-convnext-small-models-to-tensorrt)
    - [5.1.3. Submission](#513-submission)
  - [5.2. Keep trained YOLOX, re-train Convnext-small classification models](#52-keep-trained-yolox-re-train-convnext-small-classification-models)
    - [5.2.1. Convert trained YOLOX to TensorRT](#521-convert-trained-yolox-to-tensorrt)
    - [5.2.2. Prepair datasets to train classification models](#522-prepair-datasets-to-train-classification-models)
    - [5.2.3. Perform 4-folds splitting on competition data](#523-perform-4-folds-splitting-on-competition-data)
    - [5.2.4. Training 4 x Convnext-small classification models](#524-training-4-x-convnext-small-classification-models)
    - [5.2.5. Checkpoints selection](#525-checkpoints-selection)
    - [5.2.6. Convert selected best Convnext models to TensorRT](#526-convert-selected-best-convnext-models-to-tensorrt)
    - [5.2.7. Submission](#527-submission)
  - [5.3. Re-train all parts from scratch](#53-re-train-all-parts-from-scratch)
    - [5.3.1. Prepair dataset for training YOLOX ROI detector](#531-prepair-dataset-for-training-yolox-roi-detector)
    - [5.3.2. Retrain YOLOX for breast ROI detection](#532-retrain-yolox-for-breast-roi-detection)
    - [5.3.3. Prepair datasets to train classification models](#533-prepair-datasets-to-train-classification-models)
    - [5.3.4. Perform 4-folds splitting on competition data](#534-perform-4-folds-splitting-on-competition-data)
    - [5.3.5. Training 4 x Convnext-small classification models](#535-training-4-x-convnext-small-classification-models)
    - [5.3.6. Checkpoints selection](#536-checkpoints-selection)
    - [5.3.7. Convert selected best Convnext models to TensorRT](#537-convert-selected-best-convnext-models-to-tensorrt)
    - [5.3.8. Submission](#538-submission)



# 1. ARCHIVE CONTENTS
- `assets`: contain neccessary data files, trained models
    - `assets/data/`: csv label for external datasets (BMCD and CMMD), breast ROI box annotation in YOLOv5 format
    - `assets/public_pretrains/`: publicly available pretrains
    - `assets/trained/`: trained models, used for winning submission
- `datasets/`: where to store datasets (competition + external), expected to contain both raw and cleaned version.
    - `datasets/raw/`: raw version of competion data + all external datasets: BMCD, CDD-CESM, CMMD, MiniDDSM, Vindr. For how to correctly structure datasets, please refer to [docs/DATASETS.md](docs/DATASETS.md)
- `docker/`: Dockerfile
- `docs/`: documentations
- `src/`: contain almost source code for this project
    - `src/roi_det`: for training breast ROI detection model (YOLOX)
    - `src/pytorch-image-models`: for training classification model (Convnext-small)
    - `src/submit`: code to generate predictions (submission)
    - `src/tools`: contain python scripts, bash scripts to prepair datasets, training and convert models,..
    - `src/utils`: Utilities for dicom processing,..
- `SETTINGS.json`: define relative paths for IO


---


`SETTINGS.json` defines base paths for IO:

-  `RAW_DATA_DIR`: Where to store raw dataset, including both competition dataset and external datasets.
- `PROCESSED_DATA_DIR`: Where to store processed/cleaned datasets
- `MODEL_CHECKPOINT_DIR`: Store intermediate checkpoints during training
- `MODEL_FINAL_SELECTION_DIR`: Where to store final (best) models used for submission
- `SUBMISSION_DIR`: Where to store final submission/inference results
- `ASSETS_DIR`: Store trained models, manually annotated datasets/files. This must not be changed and define here for easier looking up only.
- `TEMP_DIR`: Where to store intermediate results/files


# 2. HARDWARE
The following machine were used to create the final solution: [NVIDIA DGX A100](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/nvidia-dgx-a100-datasheet.pdf). Most of my experiments can be done using 1-3 A100 GPUs.
However, final results can be easily reproduced using a single A100 GPU (40GB GPU Memory).
- OS: Ubuntu 18.04
- NVIDIA Driver version: 450.80.02
- CUDA 11.6, CUDNN 8.8
- Dependencies (recommended): see [docker/Dockerfile](docker/Dockerfile)
- Pip packages listed in [requirements.txt](requirements.txt)


# 3. DATA SETUP
Refer to [docs/DATASETS.md](docs/DATASETS.md) for details on how to correctly setup datasets.


# 4. SOLUTION PIPELINE
There are some stages to reproduce the entire solutions. I will briefly describe it for easier further understanding.
1. Train a YOLOX on some of competition images for breast ROI detection
    - Convert competition dicom files to 8-bits png images
    - Convert detection labels in YOLOv5 format to COCO format (YOLOX accepts COCO format without any modifications)
    - Train a YOLOX-nano 416x416 model on those images (521 train images, 50 val images)
    - Convert trained YOLOX model from Torch to TensorRT engine.
2. Using trained YOLOX TensorRT engine to crop breast ROI region, save to disk as 8-bits pngs
    - Clean and re-structure raw datasets (competition data + external data) in an unified way (standardize the format/structure)
    - Dicom decoding --> ROI detection (YOLOX) --> ROI crop --> normalization --> save to disk
3. Train Convnext-small model for classification using those saved ROI images
    - Do a 4-folds splits on competition data.
    - Train 4 Convnext-small model on each folds
    - Select best checkpoint for each fold
    - Convert those models from Torch to TensorRT
4. Inference on test data (submission)


# 5. SOLUTION REPRODUCING
All the following instructions assume that datasets (competition + external data) are all set up.
There are 4 options to reproduce the solutions:
1. Use trained models
    - No training, just use trained models in `assets/trained` to make predictions


2. Do not re-train YOLOX, fully reproduce Convnext-small classification models
    - Skip re-train the YOLOX part, use (my) trained YOLOX for further steps
    - Re-train 4x Convnext-small classification models. This part can be 100% reproduced (give you identical models/training log/result) without any randomness.
    - This method should give 100% identical score on both CV/LB/PB

3. Re-train all parts (reproduce from scratch)
    - Won't use any of (my) trained models in any parts, but re-train all of theme from scratch
    - This may not give 100% identical results/scores. The reason is that YOLOX can't be fully reproduced to get EXACTLY same model as used in winning submission. More details [here](docs/RANDOMNESS.md)
    - Note that dataset used for training Convnext-small classification models is generated base on YOLOX's prediction, so changes in YOLOX will cause changes in Convnext-small classification models --> Convnext-small classification models will also be unreproducible (in a 100% way).
    - But in general, it should give nearly identical results/scores within a reasonable margin.




------------------------------




## 5.1. Use trained models to make predictions

### 5.1.1. Convert trained YOLOX to TensorRT

A YOLOX-nano 416 engine which was optimized for NVIDIA A100 is provided at [assets/trained/yolox_nano_416_roi_trt_a100.pth](assets/trained/yolox_nano_416_roi_trt_a100.pth).
However, the recommended way is to convert it to TensorRT, optimized for your environment/hardware:
```bash
PYTHONPATH=$(pwd)/src/roi_det/YOLOX:$PYTHONPATH python3 src/roi_det/YOLOX/tools/trt.py \
    -expn trained_yolox_nano_416_to_tensorrt \
    -f src/roi_det/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py \
    -c assets/trained/yolox_nano_416_roi_torch.pth \
    --save-path assets/trained/yolox_nano_416_roi_trt.pth \
    -b 1
```
Behaviors:
- Create new directory `{MODEL_CHECKPOINT_DIR}/yolox_roi_det/trained_yolox_nano_416_to_tensorrt/`.
- The converted YOLOX TensorRT engine will also be saved to `./assets/trained/yolox_nano_416_roi_trt.pth`

### 5.1.2. Convert trained 4 x Convnext-small models to TensorRT
```bash
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/tools/convert_convnext_tensorrt.py --mode trained
```
Behaviours: Save a 4-folds combined TensorRT engine to `./assets/trained/best_ensemble_convnext_small_batch2_fp32.engine'`.

It takes 5-10 minutes for Kaggle's P100 GPU to finish, but take about 1 hour for A100 GPU (my case). 


### 5.1.3. Submission
```bash
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/submit/submit.py --mode trained --trt
```

Behaviours:
- Create a temporary directory storing 8-bits png images at `{TEMP_DIR}/pngs/` and expected to be removed once inference done. 
- Save submission csv result to `{SUBMISSION_DIR}/submission.csv`


------------------------------


## 5.2. Keep trained YOLOX, re-train Convnext-small classification models

### 5.2.1. Convert trained YOLOX to TensorRT

A YOLOX-nano 416 engine which was optimized for NVIDIA A100 is provided at [assets/trained/yolox_nano_416_roi_trt_a100.pth](assets/trained/yolox_nano_416_roi_trt_a100.pth).
However, the recommended way is to convert it to TensorRT, optimized for your environment/hardware:
```bash
PYTHONPATH=$(pwd)/src/roi_det/YOLOX:$PYTHONPATH python3 src/roi_det/YOLOX/tools/trt.py \
    -expn trained_yolox_nano_416_to_tensorrt \
    -f src/roi_det/YOLOX/exps/projects/rsna/yolox_nano_bre_416.py \
    -c assets/trained/yolox_nano_416_roi_torch.pth \
    --save-path assets/trained/yolox_nano_416_roi_trt.pth \
    -b 1
```
Behaviors:
- Create new directory `{MODEL_CHECKPOINT_DIR}/yolox_roi_det/trained_yolox_nano_416_to_tensorrt/`.
- The converted YOLOX TensorRT engine will also be saved to `./assets/trained/yolox_nano_416_roi_trt.pth`

### 5.2.2. Prepair datasets to train classification models
```bash
python3 src/tools/prepair_classification_dataset.py --num-workers 8 --roi-yolox-engine-path assets/trained/yolox_nano_416_roi_trt.pth
```
Behaviors:
- Create a `stage1_images` in each raw dataset directory: `{RAW_DATA_DIR}/{dataset_name}/stage1_images` for the intermediate stage.
- Create a new directory `{PROCESSED_DATA_DIR}/classification/` contains 8-bits png images `{PROCESSED_DATA_DIR}/classification/{dataset_name}/cleaned_images/` and cleaned label file `{PROCESSED_DATA_DIR}/classification/{dataset_name}/cleaned_label.csv` for each dataset.

### 5.2.3. Perform 4-folds splitting on competition data
```bash
python3 src/tools/cv_split.py
```
Behaviors: Create new directory and saving csv files in `{PROCESSED_DATA_DIR}/rsna-breast-cancer-detection/cv/v2/`

### 5.2.4. Training 4 x Convnext-small classification models
```bash
python3 src/tools/make_train_bash_script.py --mode fully_reproduce
```
This will save a file named `_train_script_auto_generated.sh` in current directory, which include commands and instructions to train Convnext-small classification models.
To reproduce using single GPU, simply run
```
sh ./_train_script_auto_generated.sh
```
This could take 8 days to finish training (around 2 days for each fold).

Or if you have multiple GPUs and want to speed up training, simply follow instructions in the generated train script `_train_script_auto_generated.sh` and run each command in parallel using different GPUs. For more details on the training process, take a look at [my write up](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449), part `4.3.Training`

Behaviours:
- This assumes that directory `{MODEL_CHECKPOINT_DIR}/timm_classification/` is empty before start any train commands
- Saving checkpoints/logs to `{MODEL_CHECKPOINT_DIR}/timm_classification/`, contains 6 sub-directories named 
    - `fully_reproduce_train_fold_2`
    - `fully_reproduce_train_fold_3`
    - `stage1_fully_reproduce_train_fold_0`
    - `stage1_fully_reproduce_train_fold_1`
    - `stage2_fully_reproduce_train_fold_0`
    - `stage2_fully_reproduce_train_fold_1`


### 5.2.5. Checkpoints selection
```bash
python3 src/tools/select_classification_best_ckpts.py --mode fully_reproduce
```

Behaviours: 
- This could overwrite convnext checkpoint files in `{MODEL_FINAL_SELECTION_DIR}/`
- Select and copy the 4 best checkpoints for each folds to `{MODEL_FINAL_SELECTION_DIR}/`:
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_0.pth.tar`
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_1.pth.tar`
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_2.pth.tar`
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_3.pth.tar`


### 5.2.6. Convert selected best Convnext models to TensorRT
```bash
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/tools/convert_convnext_tensorrt.py --mode reproduce
```
Behaviours: Save a 4-folds combined TensorRT engine to `{MODEL_FINAL_SELECTION_DIR}/best_ensemble_convnext_small_batch2_fp32.engine'`.

It takes 5-10 minutes for Kaggle's P100 GPU to finish, but take about 1 hour for A100 GPU (my case). 


### 5.2.7. Submission
```bash
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/submit/submit.py --mode partial_reproduce --trt
```

Behaviours:
- Create a temporary directory storing 8-bits png images at `{TEMP_DIR}/pngs/` and expected to be removed once inference done. 
- Save submission csv result to `{SUBMISSION_DIR}/submission.csv`
 


------------------------------




## 5.3. Re-train all parts from scratch


### 5.3.1. Prepair dataset for training YOLOX ROI detector
```bash
python3 src/tools/prepair_roi_det_dataset.py --num-workers 4
```
Behaviors:
- Copy mannual annotated breast ROI box in YOLOv5 format from `./assets/data/roi_det_yolov5_format/` to `{PROCESSED_DATA_DIR}/roi_det_yolox/yolov5_format/`
- Decode 571 dicom files in competition dataset to 8-bits png, stored at `{PROCESSED_DATA_DIR}/roi_det_yolox/yolov5_format/images/`
- Convert from YOLOv5 format to COCO format, stored at `{PROCESSED_DATA_DIR}/roi_det_yolox/coco_format/`

### 5.3.2. Retrain YOLOX for breast ROI detection
```bash
sh src/tools/train_and_convert_yolox_trt.sh
```
Behaviors:
- Train YOLOX, saving checkpoints to `{MODEL_CHECKPOINT_DIR}/yolox_roi_det/yolox_nano_416_reproduce/`
- (Optional) Perform evaluation on best checkpoint, print results
- Convert newly trained best checkpoint to TensorRT, stored in `{MODEL_CHECKPOINT_DIR}/yolox_roi_det/yolox_nano_416_reproduce/`
- Copy best Torch checkpoint to `{MODEL_FINAL_SELECTION_DIR}/yolox_nano_416_roi_torch.pth`
- Copy the converted best TensorRT engine in previous step to `{MODEL_FINAL_SELECTION_DIR}/yolox_nano_416_roi_trt.pth`


### 5.3.3. Prepair datasets to train classification models
This will use newly trained YOLOX in previous step as breast ROI extractor.
```bash
python3 src/tools/prepair_classification_dataset.py --num-workers 8
```
Behaviors:
- Create a `stage1_images` in each raw dataset directory: `{RAW_DATA_DIR}/{dataset_name}/stage1_images` for the intermediate stage.
- Create a new directory `{PROCESSED_DATA_DIR}/classification/` contains 8-bits png images `{PROCESSED_DATA_DIR}/classification/{dataset_name}/cleaned_images/` and cleaned label file `{PROCESSED_DATA_DIR}/classification/{dataset_name}/cleaned_label.csv` for each dataset.

### 5.3.4. Perform 4-folds splitting on competition data
```bash
python3 src/tools/cv_split.py
```
Behaviors: Create new directory and saving csv files in `{PROCESSED_DATA_DIR}/rsna-breast-cancer-detection/cv/v2/`

### 5.3.5. Training 4 x Convnext-small classification models
```bash
python3 src/tools/make_train_bash_script.py --mode fully_reproduce
```
This will save a file named `_train_script_auto_generated.sh` in current directory, which include commands and instructions to train Convnext-small classification models.
To reproduce using single GPU, simply run
```
sh ./_train_script_auto_generated.sh
```
This could take 8 days to finish training (around 2 days for each fold).

Or if you have multiple GPUs and want to speed up training, simply follow instructions in the generated train script `_train_script_auto_generated.sh` and run each command in parallel using different GPUs. For more details on the training process, take a look at [my write up](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/discussion/392449), part `4.3.Training`

Behaviours:
- This assumes that directory `{MODEL_CHECKPOINT_DIR}/timm_classification/` is empty before start any train commands
- Saving checkpoints/logs to `{MODEL_CHECKPOINT_DIR}/timm_classification/`, contains 6 sub-directories named 
    - `fully_reproduce_train_fold_2`
    - `fully_reproduce_train_fold_3`
    - `stage1_fully_reproduce_train_fold_0`
    - `stage1_fully_reproduce_train_fold_1`
    - `stage2_fully_reproduce_train_fold_0`
    - `stage2_fully_reproduce_train_fold_1`


### 5.3.6. Checkpoints selection
```bash
python3 src/tools/select_classification_best_ckpts.py --mode fully_reproduce
```
Behaviours: 
- This could overwrite convnext checkpoint files in `{MODEL_FINAL_SELECTION_DIR}/`
- Select and copy the 4 best checkpoints for each folds to `{MODEL_FINAL_SELECTION_DIR}/`:
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_0.pth.tar`
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_1.pth.tar`
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_2.pth.tar`
    - `{MODEL_FINAL_SELECTION_DIR}/best_convnext_fold_3.pth.tar`


### 5.3.7. Convert selected best Convnext models to TensorRT
```bash
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/tools/convert_convnext_tensorrt.py --mode reproduce
```
Behaviours: Save a 4-folds combined TensorRT engine to `{MODEL_FINAL_SELECTION_DIR}/best_ensemble_convnext_small_batch2_fp32.engine'`.

It takes 5-10 minutes for Kaggle's P100 GPU to finish, but take about 1 hour for A100 GPU (my case). 


### 5.3.8. Submission
```bash
PYTHONPATH=$(pwd)/src/pytorch-image-models/:$PYTHONPATH python3 src/submit/submit.py --mode reproduce --trt
```

Behaviours:
- Create a temporary directory storing 8-bits png images at `{TEMP_DIR}/pngs/` and expected to be removed once inference done. 
- Save submission csv result to `{SUBMISSION_DIR}/submission.csv`

