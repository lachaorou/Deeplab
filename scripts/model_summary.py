import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torchsummary import summary
from deeplab import DeepLab

#初始化模型
num_classes = 20  # 根据实际情况修改
model = DeepLab(num_classes=num_classes, backbone="xception", pretrained=True, downsample_factor=16)


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义输入尺寸
input_size = (3, 512, 512)  # 示例输入尺寸，你可根据实际情况修改

# 使用torchsummary打印模型参数
summary(model, input_size=input_size)

