FROM ultralytics/yolov5:v7.0

RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential make cmake
RUN apt install -y zsh

RUN pip install transformers tensorboard pandas pandas-profiling ipywidgets seaborn matplotlib scikit-learn scipy Pillow
RUN pip install yacs pyyaml Cython
RUN pip install cython_bbox 
RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110
RUN pip install fvcore sympy onnxoptimizer onnxsim
RUN pip install pydicom joblib dicomsdl python-gdcm pylibjpeg
RUN pip install pytorch-ignite exhaustive-weighted-random-sampler setproctitle
RUN pip install wandb numba tensorrt openpyxl onnxruntime-gpu
RUN pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com

#### EXTERNAL DEPENDENCIES #####
RUN mkdir /workspace/libs   
# YOLOX dependencies    
WORKDIR  /workspace/libs
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git
WORKDIR YOLOX
RUN pip install -v -e .  # or  python3 setup.py develop

# Torch2trt
WORKDIR  /workspace/libs
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
WORKDIR torch2trt
RUN python3 setup.py install

WORKDIR /workspace/
CMD ["/usr/bin/zsh"]