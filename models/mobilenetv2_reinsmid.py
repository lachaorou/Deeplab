import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.xception import xception
from models.mobilenetv2 import mobilenetv2

class Reins(nn.Module):
    # ...existing code, 可直接复用你原有的 Reins 实现...
    pass

class MobileNetV2_ReinsMid(nn.Module):
    """
    MobileNetV2 主干，在 features[7] 后插入 Reins
    """
    def __init__(self, reins_params, downsample_factor=8, pretrained=True):
        super().__init__()
        from functools import partial
        model = mobilenetv2(pretrained)
        self.features = model.features
        self.reins = Reins(**reins_params)
        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # 前4层提取低层特征
        low_level_features = self.features[:4](x)
        # 4~7层
        x = self.features[4:8](low_level_features)
        # 在主干中间插入 Reins
        x = self.reins(x, layer=0)
        # 8~最后
        x = self.features[8:](x)
        return low_level_features, x

# 你可以在主模型中替换 backbone 为 MobileNetV2_ReinsMid
# 训练脚本中通过参数选择 backbone 实现即可。
