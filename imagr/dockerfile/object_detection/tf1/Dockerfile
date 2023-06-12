FROM australia-southeast1-docker.pkg.dev/ml-shared-c-c41d/ml/tf1_with_opencv:7f783eb

ARG DEBIAN_FRONTEND=noninteractive

# key rotation
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-get update && apt-get install -y --no-install-recommends wget

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    vim \
    tmux \ 
    ffmpeg \
    libsm6 \
    libxext6


# Add new user to avoid running as root
RUN mkdir -p /home/tensorflow
WORKDIR /home/tensorflow

COPY ./research/object_detection/packages/tf1/setup.py ./
RUN python -m pip install --user -U pip && \
    python -m pip install --user . && \
    python -m pip install --user google-api-python-client oauth2client keras==2.2.5

# Copy this version of of the model garden into the imagels
COPY . /home/tensorflow/models

# Compile protobuf configs
RUN (cd /home/tensorflow/models/research/ && protoc */protos/*.proto --python_out=.)

ENV TF_CPP_MIN_LOG_LEVEL 3
ENV PATH="/home/tensorflow/.local/bin:/home/tensorflow/models/:/home/tensorflow/.local/bin:${PATH}"
ENV PYTHONPATH="/home/tensorflow/models/research/:/home/tensorflow/models/research/slim:${PYTHONPATH}"
