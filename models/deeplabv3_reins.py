import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul
from torch import Tensor
from models.xception import xception
from models.mobilenetv2 import mobilenetv2

class Reins(nn.Module):
    def __init__(
        self,
        num_layers: int,  # 模型的层数
        embed_dims: int,  # 嵌入维度，决定了特征向量的维度，影响模型对特征的表示能力
        patch_size: int,  # 图像分块的大小，在处理图像数据时用于将图像划分为多个小块
        token_length: int = 120,  # 科学系令牌 token 的长度，默认为 120，令牌在 rein 方法中用于实例级别的特征细化
        use_softmax: bool = True,  # 决定是否使用 softmax 函数将注意力分数进行归一化，默认 true
        scale_init: float = 0.001,  # 缩放因子的初始值，默认为 0.001，用于调整特征的变化幅度
    ) -> None:
        # 调用父类 nn.Module 的构造函数，进行必要的初始化。调用 self.create_model() 方法，用于创建模型的参数和模块
        super().__init__()
        print( f"Reins embed_dims: {embed_dims}" )  # 添加此行
        self.num_layers = num_layers
        self.embed_dims = embed_dims  # 嵌入维度，通常指特征数量或者特征向量的维度
        self.patch_size = patch_size
        self.token_length = token_length
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.create_model()

    def create_model(self):
        self.learnable_tokens = nn.Parameter(  # nn.Parameter 表示这是一个需要训练的参数
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )
        self.scale = nn.Parameter(torch.tensor(self.scale_init))

        # 定义一个线性层，用于将令牌特征转换为与输入特征相同的维度，输入和输出维度也是 embed_dims
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)

        # 另一个线性层，用于对特征的调整量进行进一步处理，输入和输出的维度也是 embed_dims
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)

        # 计算 val，用于初始化 learnable_tokens 的均匀分布范围
        val = math.sqrt(
            6.0 / float(3 * reduce(mul, (self.patch_size, self.patch_size), 1) + self.embed_dims)
        )

        # 使用 nn.init.uniform_ 对 learnable_tokens 进行均匀分布初始化，范围是 [-val, val]
        nn.init.uniform_(self.learnable_tokens.data, -val, val)

        # 使用 nn.init.kaiming_uniform_ 对 mlp_delta_f 和 mlp_token2feat 的权重进行 Kaiming 均匀分布初始化，a=math.sqrt(5) 是 Kaiming 初始化的一个参数。
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

    def get_tokens(self, layer: int) -> Tensor:
        if layer == -1:
            # return all
            return self.learnable_tokens
        else:
            return self.learnable_tokens[layer]

    def forward(
        # feats: 输入的特征张量
        # batch_first: 布尔值，指示特征张量的维度顺序是否为 [batch_size, sequence_length, feature_dim]，如果是则为 true，否则 false
        # has_cls_token：布尔值，指示特征张量中是否包含分类令牌 class token，默认 true
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=False
    ) -> Tensor:
        # print(f"Input feats shape: {feats.shape}")

        if feats.dim() == 3:
            batch_size, num_patches, feature_dim = feats.shape
            height = width = int(num_patches ** 0.5)  # 假设 height 和 width 是 sqrt(sequence_length)
            feats = feats.view(batch_size, feature_dim, height, width)  # 转换为四维

        # 如果 feats 是四维的 [batch_size, channels, height, width]，则不需要 permute
        if feats.dim() == 4 :
            feats = feats

        # 将三维的 feats 转换为四维的 [batch_size, feature_dim, height, width]
        # feats = feats.view(batch_size, feature_dim, height, width)

        tokens = self.get_tokens(layer)  # 获取当前层的令牌
        delta_feat = self.forward_delta_feat(  # 计算特征的调整量
            feats,
            tokens,
            layer,
        )
        delta_feat = delta_feat * self.scale  # 调整量 delta_feat 乘以缩放因子 self.scale
        feats = feats + delta_feat
        return feats  # 返回最终的特征张量

    def forward_delta_feat(self, feats: torch.Tensor, tokens: torch.Tensor, layers: int) -> torch.Tensor:
        
        batch_size, channels, height, width = feats.shape

        feats = feats.view(batch_size, channels, -1).permute(0, 2, 1)  # 转换为 [batch_size, num_patches, channels]

        attn = torch.einsum("nbc,mc->nbm", feats, tokens)
        if self.use_softmax:
            attn = attn * (self.embed_dims ** -0.5)
            attn = F.softmax(attn, dim=-1)

        delta_f = torch.einsum(
            "nbm,mc->nbc",
            attn[:, :, 1:],
            self.mlp_token2feat(tokens[1:, :]),
        )
        delta_f = self.mlp_delta_f(delta_f + feats)

        # 将结果转换回原始的形状
        delta_f = delta_f.permute(0, 2, 1).view(batch_size, channels, height, width)
        return delta_f

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

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
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


# ASPP 特征提取模块
# 利用不同膨胀率的膨胀卷积进行特征提取
class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 一共五个分支
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # 第五个分支，全局平均池化 + 卷积
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        # 将五个分支的内容堆叠起来
        # 然后 1x1 卷积整合特征。
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone, pretrained=True, downsample_factor=16, token_length=120, num_layers=6, embed_dims=None):
        print(f'[Debug] DeepLab __init__ 参数: num_classes={num_classes}, backbone={backbone}, pretrained={pretrained}, downsample_factor={downsample_factor}, token_length={token_length}, num_layers={num_layers}, embed_dims={embed_dims}')
        super(DeepLab, self).__init__()

        if backbone=="xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet.'.format(backbone))

        # Debug: 打印 token_length，确保传递正确
        print(f'[Debug] DeepLab __init__ token_length={token_length}')
        # 灵活化embed_dims
        if embed_dims is None:
            embed_dims = in_channels
        # 初始化 Reins 模型，参数可控
        self.reins = Reins(num_layers=num_layers, embed_dims=embed_dims, patch_size=32, token_length=token_length)

        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
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
        low_level_features, x = self.backbone(x)
        # 自动适配reins tokens维度
        if x.shape[1] != self.reins.embed_dims:
            self.reins = Reins(num_layers=self.reins.num_layers, embed_dims=x.shape[1], patch_size=32, token_length=self.reins.token_length)
            self.reins = self.reins.to(x.device)
        x = self.reins.forward(
            x,
            layer=0,
            batch_first=True,
            has_cls_token=False
        )
        x = self.aspp(x)
        low_level_features = self.shortcut_conv(low_level_features)
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), mode='bilinear', align_corners=True)
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x



if __name__ == "__main__":
    model = DeepLab(num_classes=20 , backbone= "xception")
    input_tensor = torch.randn(4, 3, 512, 512)
    output = model(input_tensor)
