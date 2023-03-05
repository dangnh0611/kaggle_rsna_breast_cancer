FROM nvcr.io/nvidia/pytorch:22.08-py3 
LABEL maintainer="dangnh0611@gmail.com"
LABEL original_image="https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_20-12.html#rel_20-12"

RUN apt update && apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential python3-setuptools make cmake \
    ffmpeg libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev
RUN apt install -y zsh
RUN apt install -y libgl1


RUN pip install setuptools==59.5.0
RUN pip install --upgrade typing-extensions
RUN pip install opencv-python-headless==4.5.4.60
RUN pip install  albumentations transformers tensorboard pandas pandas-profiling ipywidgets seaborn matplotlib scikit-learn scipy Pillow
RUN pip install  yacs pyyaml Cython cython_bbox 

RUN pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/nightly --upgrade nvidia-dali-nightly-cuda110
RUN pip install 'pycuda<2021.1' fvcore sympy onnxoptimizer onnxsim
RUN pip install pydicom joblib dicomsdl python-gdcm pylibjpeg

##### EXTERNAL DEPENDENCIES #####
RUN mkdir /workspace/libs   
# YOLOX dependencies    
WORKDIR  /workspace/libs
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git
WORKDIR YOLOX
RUN pip install -v -e .  # or  python3 setup.py develop

WORKDIR /workspace/

CMD ["/usr/bin/zsh"]
