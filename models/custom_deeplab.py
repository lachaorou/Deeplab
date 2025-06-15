import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDeeplab(nn.Module):
    def __init__(self, backbone, aspp, use_reins=False, reins_params=None, reins_insert_pos='aspp', multi_reins_points=None, num_classes=21):
        super().__init__()
        self.backbone = backbone
        self.aspp = aspp
        self.use_reins = use_reins
        self.reins_insert_pos = reins_insert_pos
        self.multi_reins_points = multi_reins_points or []
        self.num_classes = num_classes
        if use_reins:
            from models.reins import Reins
            self.reins = Reins(**reins_params)
        # 其它可选模块可在此扩展
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(24, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        low_level, x = self.backbone(x)
        # 支持多点插入Reins
        if self.use_reins and self.reins_insert_pos == 'backbone':
            x = self.reins(x)
        x = self.aspp(x)
        if self.use_reins and self.reins_insert_pos == 'aspp':
            x = self.reins(x)
        # 可扩展多点插入
        for point in self.multi_reins_points:
            if point == 'backbone':
                x = self.reins(x)
            elif point == 'aspp':
                x = self.reins(x)
        low_level = self.shortcut_conv(low_level)
        x = F.interpolate(x, size=(low_level.size(2), low_level.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
