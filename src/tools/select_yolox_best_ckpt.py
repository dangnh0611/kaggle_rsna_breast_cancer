"""Simply copy best checkpoint to final selection models directory."""

import os
import shutil

from settings import SETTINGS

if __name__ == '__main__':
    SRC_DIR = os.path.join(SETTINGS.MODEL_CHECKPOINT_DIR, 'yolox_roi_det',
                        'yolox_nano_416_reproduce')
    DST_DIR = SETTINGS.MODEL_FINAL_SELECTION_DIR
    os.makedirs(DST_DIR, exist_ok=True)

    # Copy best torch ckpt
    SRC_BEST_TORCH_PATH = os.path.join(SRC_DIR, 'best_ckpt.pth')
    DST_BEST_TORCH_PATH = os.path.join(DST_DIR, 'yolox_nano_416_roi_torch.pth')
    print(f'Best Torch checkpoint:', SRC_BEST_TORCH_PATH)
    print(f'Copying {SRC_BEST_TORCH_PATH} to {DST_BEST_TORCH_PATH}')
    shutil.copy2(SRC_BEST_TORCH_PATH, DST_BEST_TORCH_PATH)

    # Copy best tensorrt ckpt
    SRC_BEST_TRT_PATH = os.path.join(SRC_DIR, 'model_trt.pth')
    DST_BEST_TRT_PATH = os.path.join(DST_DIR, 'yolox_nano_416_roi_trt.pth')
    print(f'Best TensorRT engine:', SRC_BEST_TRT_PATH)
    print(f'Copying {SRC_BEST_TRT_PATH} to {DST_BEST_TRT_PATH}')
    shutil.copy2(SRC_BEST_TRT_PATH, DST_BEST_TRT_PATH)
