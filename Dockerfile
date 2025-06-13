# Deeplabv3plus 项目 Dockerfile
# 用法：
#   1. 构建镜像（以本地CUDA 11.8为例）：
#      docker build -t deeplabv3plus:cuda118 --build-arg TORCH_VERSION=2.1.0 --build-arg TORCHVISION_VERSION=0.16.0 --build-arg CUDA_VERSION=11.8 .
#   2. 服务器（24G显存，CUDA 12.1）：
#      docker build -t deeplabv3plus:cuda121 --build-arg TORCH_VERSION=2.2.0 --build-arg TORCHVISION_VERSION=0.17.0 --build-arg CUDA_VERSION=12.1 .
#   3. 运行容器时挂载代码和结果目录，便于同步
#      docker run --gpus all -v /your/code:/workspace -v /your/results:/workspace/results -it deeplabv3plus:cuda121

ARG PYTHON_VERSION=3.10
ARG TORCH_VERSION=2.1.0
ARG TORCHVISION_VERSION=0.16.0
ARG CUDA_VERSION=11.8

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装基础依赖
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} python3-pip python3-dev \
    git wget unzip libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置python软链接
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python || true

# 安装pip
RUN python -m pip install --upgrade pip

# 安装PyTorch（根据参数自动适配）
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --extra-index-url https://download.pytorch.org/whl/cu${CUDA_VERSION/./}

# 安装项目依赖
COPY requirements.txt /workspace/requirements.txt
WORKDIR /workspace
RUN pip install -r requirements.txt

# 可选：安装Jupyter、TensorBoard等
RUN pip install jupyter tensorboard

# 复制项目代码（如需）
# COPY . /workspace

# 默认工作目录
WORKDIR /workspace

# 默认命令
CMD ["bash"]

# 说明：
# - 挂载本地代码和结果目录，便于同步和迁移
# - 训练/推理时建议用挂载方式操作数据和结果
# - 具体PyTorch/CUDA版本请根据硬件实际调整
