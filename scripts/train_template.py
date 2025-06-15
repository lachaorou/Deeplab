"""
通用训练主入口模板，支持灵活选择deeplabv3_plus、deeplabv3_reins、custom_deeplab等模型结构，并自动加载config.py参数。
"""
import argparse
from config import config
from models.deeplabv3_plus import DeepLab as DeepLabV3Plus
from models.deeplabv3_reins import DeepLab as DeepLabV3Reins
from models.custom_deeplab import CustomDeeplab
from models.xception import xception
from models.mobilenetv2 import mobilenetv2
from models.deeplabv3_plus import ASPP
from models.reins import Reins

# 1. 解析命令行参数（可覆盖config.py默认值）
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='custom', choices=['deeplabv3plus','deeplabv3reins','custom'], help='模型结构')
    parser.add_argument('--backbone', type=str, default=config.backbone)
    parser.add_argument('--use_reins', action='store_true')
    parser.add_argument('--reins_mode', type=str, default='single')
    parser.add_argument('--multi_reins_points', type=str, default='')
    parser.add_argument('--token_length', type=int, default=64)
    parser.add_argument('--embed_dims', type=int, default=320)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=config.num_classes)
    # ...可继续添加其它参数
    return parser.parse_args()

# 2. 构建模型

def build_model(args):
    if args.model == 'deeplabv3plus':
        model = DeepLabV3Plus(num_classes=args.num_classes, backbone=args.backbone)
    elif args.model == 'deeplabv3reins':
        model = DeepLabV3Reins(num_classes=args.num_classes, backbone=args.backbone, token_length=args.token_length, num_layers=args.num_layers, embed_dims=args.embed_dims)
    elif args.model == 'custom':
        # 以MobileNetV2为例
        if args.backbone == 'mobilenet':
            backbone = mobilenetv2(pretrained=True)
            embed_dims = 320
        elif args.backbone == 'xception':
            backbone = xception(downsample_factor=16, pretrained=True)
            embed_dims = 2048
        else:
            raise ValueError('Unsupported backbone')
        aspp = ASPP(dim_in=embed_dims, dim_out=256)
        reins_params = dict(num_layers=args.num_layers, embed_dims=embed_dims, patch_size=32, token_length=args.token_length)
        multi_reins_points = args.multi_reins_points.split('/') if args.multi_reins_points else []
        model = CustomDeeplab(backbone=backbone, aspp=aspp, use_reins=args.use_reins, reins_params=reins_params, reins_insert_pos=args.reins_mode, multi_reins_points=multi_reins_points, num_classes=args.num_classes)
    else:
        raise ValueError('Unknown model type')
    return model

if __name__ == '__main__':
    args = parse_args()
    model = build_model(args)
    print(model)
    # ...后续训练流程
