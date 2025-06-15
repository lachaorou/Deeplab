"""
train.py - DeeplabV3+ 语义分割主训练脚本

本脚本为项目主训练入口，支持参数交互、权重归档、模型结构灵活配置、断点续训、可视化等。
适用于Cityscapes/VOC等数据集，支持多主干、多消融实验。

主要功能：
- 参数集中管理与交互式修改
- 自动生成结果归档目录
- 支持Debug模式快速调试
- 支持tokens机制与多种损失函数
- 训练过程可视化与日志归档
- 断点续训与权重自动加载

使用方法：
    python scripts/train.py [--debug]

Author: 团队协作
Date: 2025-06-15
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import datetime
from functools import partial
from typing import Any

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from numpy import ndarray, dtype, floating
from numpy._typing import _32Bit
from torch.utils.data import DataLoader

from models.deeplabv3_reins import DeepLab
from models.deeplabv3_training import (get_lr_scheduler, set_optimizer_lr,
                                     weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader0 import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch
from utils.colorize import colorize_mask  # 统一Cityscapes配色
import subprocess
import random
import shutil
from datetime import datetime
from utils.visualize_training_progress import visualize_train_samples, visualize_val_compare, batch_update_all_markdown
import csv
import time
from config import config, save_config, load_config

# 训练结果自动归档目录生成函数
from datetime import datetime

def get_result_dirs(model_name):
    """
    自动生成训练结果归档目录（日志、mIoU、可视化等），并返回各目录路径和时间戳。
    Args:
        model_name (str): 当前模型名称
    Returns:
        logs_dir, mious_dir, visual_dir, time_str (str): 各目录绝对路径及时间戳
    """
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    results_dir = os.path.join(base, 'Results')
    logs_dir = os.path.join(results_dir, f'logs_{model_name}', time_str)
    mious_dir = os.path.join(results_dir, f'mious_{model_name}', time_str)
    visual_dir = os.path.join(results_dir, f'visual_{model_name}', time_str)
    for d in [logs_dir, mious_dir, visual_dir]:
        os.makedirs(d, exist_ok=True)
    return logs_dir, mious_dir, visual_dir, time_str


def str2bool(v):
    """
    字符串转布尔类型，兼容命令行参数解析。
    Args:
        v (str|bool): 输入字符串或布尔值
    Returns:
        bool: 转换后的布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



# ===== Debug模式支持 =====
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='启用调试模式，仅用10张图片、2轮epoch')
    return parser.parse_args()

args = parse_args()
debug = getattr(args, 'debug', False)

# 提前定义VOCdevkit_path，兼容debug和正式模式
VOCdevkit_path = 'data/voc/VOCdevkit'

if debug:
    # [Debug] 启用调试模式：仅用前10张图片，epoch=2
    EPOCH = 2
    # 只用前10张图片
    train_txt = os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt")
    val_txt = os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt")
    with open(train_txt, 'r') as f:
        train_lines = [x.strip() for x in f.readlines()[:10] if x.strip()]
    with open(val_txt, 'r') as f:
        val_lines = [x.strip() for x in f.readlines()[:10] if x.strip()]
else:
    # Normal mode
    pass

def print_param_table(param_dict, explain_dict=None):
    """
    以表格形式打印训练参数及说明。
    Args:
        param_dict (dict): 参数字典
        explain_dict (dict): 参数说明字典（可选）
    """
    print("\n========== 训练参数表 ==========")
    max_key_len = max(len(k) for k in param_dict)
    for k, v in param_dict.items():
        explain = f"（{explain_dict[k]}）" if explain_dict and k in explain_dict else ''
        print(f"{k.ljust(max_key_len)} {explain}: {v}")
    print("================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='外部参数文件路径（支持一键复现）')
    args, unknown = parser.parse_known_args()
    # 优先加载外部参数文件
    if args.config:
        config = load_config(args.config)
        print(f"[Info] 已从 {args.config} 加载参数，支持一键复现！")
    # 其余参数解析和主流程...
    # =============================
    # 主流程入口 Main Training Entry
    # =============================
    # 1. 参数说明与交互
    # 2. 权重选择与参数同步
    # 3. 模型定义与损失函数选择
    # 4. 数据加载与训练主循环
    # 5. 日志与结果归档
    # 6. 支持断点续训与可视化

    # 参数说明（中英文对照）
    param_explain = {
        'init_epoch': '起始轮数',
        'epoch': '总训练轮数',
        'save_dir': '结果保存目录',
        'eval_period': '评估周期',
        'save_period': '权重保存周期',
        'batch_size': '批量大小',
        'backbone': '主干网络',
        'num_classes': '类别数',
        'model_path': '权重文件路径',
        'input_shape': '输入分辨率',
        'use_tokens': '是否启用tokens机制',
        'token_length': 'tokens长度',
        'num_layers': 'tokens层数',
        'embed_dims': 'tokens嵌入维度',
        'patch_size': 'tokens patch size',
        'use_softmax': 'tokens softmax',
        'scale_init': 'tokens缩放初值',
        'loss_type': '损失函数类型',
        'init_lr': '初始学习率',
        'lr_decay_type': '学习率衰减策略',
    }
    # 提前定义一些默认参数
    default_args = {
        'init_epoch': 0,
        'epoch': 150,
        'save_dir': 'results/logs_deeplab/logs_mobilenetv2rein',
        'eval_period': 5,
        'save_period': 5,
        'batch_size': 6,  # 升级为6，适配24G显存
        'backbone': 'mobilenet',
        'num_classes': 19,
        'model_path': 'results/logs_deeplab/logs_mobilenetv2rein/best_epoch_weights.pth',
        'input_shape': [1536, 1024],  # 升级分辨率，宽高顺序
        'use_tokens': True,  # tokens机制默认开启
        'token_length': 100,  # 新增：tokens长度
        'num_layers': 6,     # 新增：tokens层数
        'embed_dims': None,  # 新增：tokens嵌入维度，None自动适配
        'patch_size': 32,    # 新增：tokens patch size
        'use_softmax': True, # 新增：tokens softmax
        'scale_init': 0.001, # 新增：tokens缩放初值
        'loss_type': 'ce',  # 损失函数类型
        'init_lr': 1e-4,  # 初始学习率
        'lr_decay_type': 'cos',  # 学习率衰减策略
        # 可补充其它常用参数
    }
    print("\n========== 当前训练默认配置如下 ==========")
    for k, v in default_args.items():
        print(f"{k} = {v}")
    print("========================================")
    print("如需更改参数，请输入参数名=值，多个参数用分号分隔。如无需求请输入 n 或直接回车：")
    print("[提示] tokens机制相关参数包括：token_length, num_layers, embed_dims, patch_size, use_softmax, scale_init，均可在此处自定义！")
    user_input = input().strip()
    if user_input and user_input.lower() != 'n':
        for pair in user_input.split(';'):
            pair = pair.strip()
            if '=' in pair:
                k, v = pair.split('=', 1)
                k = k.strip()
                v = v.strip()
                if k in default_args:
                    # 自动类型转换
                    if isinstance(default_args[k], int):
                        v = int(v)
                    elif isinstance(default_args[k], float):
                        v = float(v)
                    elif isinstance(default_args[k], bool):
                        v = str2bool(v)
                    default_args[k] = v
        for k, v in default_args.items():
            if hasattr(args, k):
                setattr(args, k, v)
    print("\n[Info] 最终训练参数:")
    print_param_table(default_args, param_explain)
    print("[提示] tokens机制相关参数已全部显示，如需自定义请在上方输入。embed_dims=None时会自动适配主干输出通道。")
    print("[权重选择说明]：如需强制加载预训练权重，请直接输入True或回车（mobilenet_v2.pth.tar/xception_pytorch_imagenet.pth），如需加载last/best权重请手动输入权重路径。");
    print(f"token_length = {default_args['token_length']}")
    print(f"loss_type = {default_args['loss_type']}")
    print(f"init_lr = {default_args['init_lr']}")
    print(f"lr_decay_type = {default_args['lr_decay_type']}")
    print(f"当前加载权重文件(model_path): {default_args.get('model_path', '')}")  # 明确显示权重文件

    # 参数交互和权重选择后，立即同步default_args到args对象，保证后续args.xxx访问不报错
    for k, v in default_args.items():
        setattr(args, k, v)

    # === 权重加载交互和参数同步后，定义模型 ===
    # 这里假设你有类似如下的模型定义代码：
    model = DeepLab(
        num_classes=default_args['num_classes'],
        backbone=default_args['backbone'],
        pretrained=True,
        downsample_factor=8,
        token_length=default_args['token_length'],
        num_layers=default_args['num_layers'],
        embed_dims=default_args['embed_dims']
    )

    # 优化损失函数选择
    if default_args['loss_type'] == 'ce':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    elif default_args['loss_type'] == 'dice':
        from utils.train_utils import DiceLoss
        criterion = DiceLoss(num_classes=default_args['num_classes'])
    elif default_args['loss_type'] == 'focal':
        from utils.train_utils import FocalLoss
        criterion = FocalLoss(num_classes=default_args['num_classes'])
    else:
        raise ValueError('不支持的损失函数类型')

    # 优化学习率设置
    optimizer = torch.optim.Adam(model.parameters(), lr=default_args['init_lr'])
    if default_args['lr_decay_type'] == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=default_args['epoch'], eta_min=1e-6)
    elif default_args['lr_decay_type'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        scheduler = None

    parser = argparse.ArgumentParser(description="Deeplabv3+ Cityscapes Training")
    parser.add_argument('--Cuda', type=str2bool, default=True, help='是否使用CUDA')
    parser.add_argument('--seed', type=int, default=11, help='随机种子')
    parser.add_argument('--distributed', type=str2bool, default=False, help='是否分布式训练')
    parser.add_argument('--sync_bn', type=str2bool, default=False, help='是否使用sync_bn')
    parser.add_argument('--fp16', type=str2bool, default=True, help='是否使用混合精度')
    parser.add_argument('--num_classes', type=int, default=19, help='类别数')
    parser.add_argument('--backbone', type=str, default="mobilenet", choices=["mobilenet", "xception"], help='主干网络')
    parser.add_argument('--pretrained', type=str2bool, default=True, help='是否加载主干预训练权重')
    parser.add_argument('--weight_type', type=str, default='last', choices=['last', 'best', 'pretrain'], help='加载权重类型')
    parser.add_argument('--batch_size', type=int, default=6, help='批量大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--epoch', type=int, default=150, help='训练总轮数')
    parser.add_argument('--init_epoch', type=int, default=0, help='起始轮数')
    parser.add_argument('--log_dir', type=str, default='results/logs_deeplab/logs_mobilenetv2rein', help='训练日志和权重保存目录')
    parser.add_argument('--model_path', type=str, default='results/logs_deeplab/logs_mobilenetv2rein/best_epoch_weights.pth', help='初始权重路径')
    parser.add_argument('--use_tokens', type=str2bool, default=True, help='是否启用tokens机制')
    parser.add_argument('--debug', action='store_true', help='启用调试模式，仅用10张图片、2轮epoch')
    parser.add_argument('--input_shape', type=int, nargs=2, default=[1536, 1024], help='输入图片分辨率')
    parser.add_argument('--token_length', type=int, default=120, help='reins tokens长度')
    parser.add_argument('--num_layers', type=int, default=6, help='reins tokens层数')
    parser.add_argument('--embed_dims', type=int, default=None, help='reins tokens嵌入维度，None自动适配')
    parser.add_argument('--patch_size', type=int, default=32, help='reins tokens patch size')
    parser.add_argument('--use_softmax', type=str2bool, default=True, help='reins tokens softmax')
    parser.add_argument('--scale_init', type=float, default=0.001, help='reins tokens缩放初值')
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'dice', 'focal'], help='损失函数类型')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='初始学习率')
    parser.add_argument('--lr_decay_type', type=str, default='cos', choices=['cos', 'step'], help='学习率衰减策略')
    args = parser.parse_args()

    debug = getattr(args, 'debug', False)
    VOCdevkit_path = 'data/voc/VOCdevkit'

    # 训练参数
    EPOCH = args.epoch
    INIT_EPOCH = args.init_epoch
    MODEL_PATH = args.model_path
    LOG_DIR = args.log_dir
    input_shape = args.input_shape
    token_length = args.token_length

    # 权重加载逻辑优化
    print("\n是否使用默认预训练权重？[True/False/自定义权重绝对路径]（回车默认True）：")
    user_weight_input = input().strip()
    weight_path = None
    LOG_DIR = default_args['save_dir']
    backbone = default_args['backbone']
    # 优先使用根目录 model_data 下的权重
    model_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_data'))
    if user_weight_input == '' or user_weight_input.lower() == 'true':
        if backbone == 'mobilenet':
            weight_path = os.path.join(model_data_dir, 'mobilenet_v2.pth.tar')
        elif backbone == 'xception':
            weight_path = os.path.join(model_data_dir, 'xception_pytorch_imagenet.pth')
        else:
            weight_path = ''
        print(f"[Info] 加载预训练权重文件: {weight_path}")
    elif user_weight_input.lower() == 'false':
        print("请输入自定义权重绝对路径：")
        custom_path = input().strip()
        weight_path = custom_path
        print(f"[Info] 加载自定义权重文件: {weight_path}")
    else:
        weight_path = user_weight_input
        print(f"[Info] 加载指定权重文件: {weight_path}")
    default_args['model_path'] = weight_path
    if hasattr(args, 'model_path'):
        args.model_path = weight_path

    if debug:
        print("[Debug] 启用调试模式：仅用前10张图片，epoch=2")
        EPOCH = 2
        INIT_EPOCH = 0
        train_txt = os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt")
        val_txt = os.path.join(VOCdevkit_path, "ImageSets/Segmentation/val.txt")
        with open(train_txt, 'r') as f:
            train_lines = [x.strip() for x in f.readlines()[:10] if x.strip()]
        with open(val_txt, 'r') as f:
            val_lines = [x.strip() for x in f.readlines()[:10] if x.strip()]
    else:
        # Normal mode
        pass

    # 用命令行参数覆盖原有变量
    Cuda = args.Cuda
    seed = args.seed
    distributed = args.distributed
    sync_bn = args.sync_bn
    fp16 = args.fp16
    num_classes = args.num_classes
    backbone = args.backbone
    pretrained = args.pretrained
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epoch = args.epoch
    use_tokens = args.use_tokens

    
    #   一般来讲，网络从0开始的训练效果会很差，因为权值太过随机，特征提取效果不明显，因此非常、非常、非常不建议大家从0开始训练！
    #   如果一定要从0开始，可以了解imagenet数据集，首先训练分类模型，获得网络的主干部分权值，分类模型的 主干部分 和该模型通用，基于此进行训练。
    #----------------------------------------------------------------------------------------------------------------------------#
    # 断点续训，加载第45轮权重
    model_path      = ''  # 从头训练，不加载旧权重
    #---------------------------------------------------------#
    #   downsample_factor   下采样的倍数8、16 
    #                       8下采样的倍数较小、理论上效果更好。
    #                       但也要求更大的显存
    #---------------------------------------------------------#
    downsample_factor   = 8
    #------------------------------#
    #   输入图片的大小
    #------------------------------#
    # input_shape         = [1280, 960]  # 适配8G显存
    input_shape         = [1536, 1024]  # 升级分辨率，适配24G显存
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的同学的训练需求。
    #   冻结训练需要的显存较小，显卡非常差的情况下，可设置Freeze_Epoch等于UnFreeze_Epoch，此时仅仅进行冻结训练。
    #      
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从整个模型的预训练权重开始训练： 
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（不冻结）
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）从主干网络的预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False，optimizer_type = 'adam'，Init_lr = 5e-4，weight_decay = 0。（不冻结）
    #       SGD：
    #           Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 120，Freeze_Train = True，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（冻结）
    #           Init_Epoch = 0，UnFreeze_Epoch = 120，Freeze_Train = False，optimizer_type = 'sgd'，Init_lr = 7e-3，weight_decay = 1e-4。（不冻结）
    #       其中：由于从主干网络的预训练权重开始训练，主干的权值不一定适合语义分割，需要更多的训练跳出局部最优解。
    #             UnFreeze_Epoch可以在120-300之间调整。
    #             Adam相较于SGD收敛的快一些。因此UnFreeze_Epoch理论上可以小一点，但依然推荐更多的Epoch。
    #   （三）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   冻结阶段训练参数此时模型的主干被冻结了，特征提取网络不发生改变.用的显存较小，仅对网络进行微调
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch(当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size(当Freeze_Train=False时失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0  # 从0开始
    Freeze_Epoch        = 0
    Freeze_batch_size   = 4  # 恢复为4，适配8G显存

    #------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   此时模型的主干不被冻结了，特征提取网络会发生改变
    #   占用的显存较大，网络所有的参数都会发生改变
    #   UnFreeze_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 150  # 总训练轮数150轮
    Unfreeze_batch_size = 4  # 恢复为4，适配8G显存
    #------------------------------------------------------------------#
    #   Freeze_Train    是否进行冻结训练
    #                   默认先冻结主干训练后解冻训练。
    #------------------------------------------------------------------#
    Freeze_Train        = False

    #------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 7e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    #------------------------------------------------------------------#
    optimizer_type      = "sgd"
    momentum            = 0.9
    weight_decay        = 1e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    #------------------------------------------------------------------#
    save_period         = 5  # 每5轮保存一次权值
    #------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    #------------------------------------------------------------------#
    save_dir            = LOG_DIR  # mobilenetv2 主干下权重保存路径
    #------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    #                   评估需要消耗较多的时间，频繁评估会导致训练非常慢
    #   此处获得的mAP会与get_map.py获得的会有所不同，原因有二：
    #   （一）此处获得的mAP为验证集的mAP。
    #   （二）此处设置评估参数较为保守，目的是加快评估速度。
    #------------------------------------------------------------------#
    eval_flag           = True  # 开启训练时自动评估
    eval_period         = 5    # 每5轮评估一次 mIoU
#    eval_flag           = False  # 若想关闭评估，取消上面注释并启用本行
#    eval_period         = 99999  # 关闭评估时period设大即可

    #------------------------------------------------------------------#
    #   VOCdevkit_path  数据集路径
    #------------------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #------------------------------------------------------------------#
    dice_loss       = True  # 是否使用dice loss
    #------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    #------------------------------------------------------------------#
    focal_loss      = True
    #------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。
    #   设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：
    #   num_classes = 3
    #   cls_weights = np.array([1, 2, 3], np.float32)
    #   num_classes = 20
    #   cls_weights = np.array([])
    #   loss(x, y)=∑i max(0, w[y]∗(margin−x[y]+x[i])) / px.size(0)
    #------------------------------------------------------------------#
    # 自动统计类别权重，优先使用stat_label_distribution.py输出
    try:
        result = subprocess.run([
            sys.executable, os.path.join(os.path.dirname(__file__), '../utils/stat_label_distribution.py')
        ], capture_output=True, text=True, check=True)
        import re
        match = re.search(r'cls_weights = np.array\(\[([\s\S]*?)\], np.float32\)', result.stdout)
        if match:
            weights_str = match.group(1)
            weights = [float(x.strip().strip(',')) for x in weights_str.strip().split('\n') if x.strip()]
            cls_weights = np.array(weights, np.float32)
            # print(f"[Info] 已自动加载类别权重: {cls_weights}")
        else:
            raise ValueError('未找到类别权重输出')
    except Exception as e:
        # print(f"[Warning] 自动统计类别权重失败，使用默认权重。原因: {e}")
        # 建议的类别权重cls_weights（归一化，提升小众类表现，可直接替换）
        cls_weights = np.array([
            0.833209,
            0.909937,
            0.852506,
            1.023590,
            1.008330,
            0.987683,
            1.096596,
            1.032917,
            0.866557,
            0.994650,
            0.929913,
            0.989200,
            1.120306,
            0.903261,
            1.078538,
            1.082251,
            1.093792,
            1.150533,
            1.046229,
        ], np.float32)
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   keras里开启多线程有些时候速度反而慢了许多
    #                   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    #------------------------------------------------------------------#
    num_workers         = 4

    seed_everything(seed)
    #------------------------------------------------------#
    #   设置用到的显卡
    #------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            # print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            # print("Gpu Device Count : ", ngpus_per_node)
            pass
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    #----------------------------------------------------#
    #   下载预训练权重（本地优先，无需重复下载）
    #----------------------------------------------------#
    # 固定预训练权重路径，按主干自动选择
    if backbone == "mobilenet":
        pretrained_weights = os.path.join(model_data_dir, 'mobilenet_v2.pth.tar')
    elif backbone == "xception":
        pretrained_weights = os.path.join(model_data_dir, 'xception.pth.tar')
    else:
        pretrained_weights = None

    if pretrained and pretrained_weights and os.path.exists(pretrained_weights):
        # print(f"[Info] 使用固定预训练权重: {pretrained_weights}")
        pass
    else:
        pass

    # 模型初始化时传递token_length参数
    model = DeepLab(num_classes=num_classes, backbone=backbone, downsample_factor=8, pretrained=pretrained, token_length=token_length)
    if not pretrained:
        weights_init(model)
    # 不再加载全模型权重，避免类别数不一致问题

    #----------------------#
    #   记录Loss
    #----------------------#
    if local_rank == 0:
        time_str        = datetime.strftime(datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history    = None

    #------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    #------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    #----------------------------#
    #   多卡同步Bn
    #----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
    
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    # 修正VOCdevkit_path为新结构下的默认路径
    voc_candidates = [
        VOCdevkit_path,
        os.path.join("data", "voc", "VOCdevkit"),
        os.path.join("Dataset", "Voc", "VOCdevkit"),
        os.path.join("dataset", "voc", "VOCdevkit"),
    ]
    found = False
    for cand in voc_candidates:
        if os.path.exists(os.path.join(cand, "ImageSets/Segmentation/train.txt")):
            VOCdevkit_path = cand
            found = True
            break
    if not found:
        raise FileNotFoundError(f"未找到 train.txt，请检查 VOCdevkit 路径！\n已尝试: " + " | ".join(voc_candidates))
    with open(os.path.join(VOCdevkit_path,"ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path,"ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )
        #---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数 
        #----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
#            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
#            print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
#            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
        
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度.也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代,Interval_Epoch为冻结训练的世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        UnFreeze_flag = False
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False

        #-------------------------------------------------------------------#
        #   如果不冻结训练的话，直接设置batch_size为Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        #-------------------------------------------------------------------#
        nbs             = 16
        lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
        if backbone == "xception":
            lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
            lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        #---------------------------------------#
        #   根据optimizer_type选择优化器
        #---------------------------------------#
        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay)
        }[optimizer_type]

        #---------------------------------------#
        #   获得学习率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        #---------------------------------------#
        #   判断每一个世代的长度
        #---------------------------------------#
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset   = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
        val_dataset     = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        # 验证集应不打乱顺序且不丢弃最后一个batch
        gen_val         = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                    drop_last=False, collate_fn=deeplab_dataset_collate, sampler=val_sampler,
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        #----------------------#
        #   记录eval的map曲线
        #----------------------#
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        #---------------------------------------#
        #   开始模型训练
        #---------------------------------------#
        sample_ids = None  # 只在第一次抽取
        for epoch in range(INIT_EPOCH, UnFreeze_Epoch):
            #---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            #---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                #-------------------------------------------------------------------#
                #   判断当前batch_size，自适应调整学习率
                #-------------------------------------------------------------------#
                nbs             = 16
                lr_limit_max    = 5e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 3e-4 if optimizer_type == 'adam' else 5e-4
                if backbone == "xception":
                    lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                    lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                #---------------------------------------#
                #   获得学习率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                # 正确解冻主干参数，兼容DataParallel/DistributedDataParallel
                backbone = model_train.module.backbone if hasattr(model_train, "module") else model_train.backbone
                for param in backbone.parameters():
                    param.requires_grad = True
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                if distributed:
                    batch_size = batch_size // ngpus_per_node
                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = deeplab_dataset_collate, sampler=train_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, 
                                            drop_last=False, collate_fn=deeplab_dataset_collate, sampler=val_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                UnFreeze_flag = True
            if distributed:
                train_sampler.set_epoch(epoch)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank)
            # === 新增：每5轮自动可视化10个训练样本 ===
            if (epoch + 1) % 5 == 0 and local_rank == 0:
                pred_dir = os.path.join('miou_out_mobilenetv2_rein', 'detection-results_fixed')
                label_dir = os.path.join(VOCdevkit_path, 'SegmentationClass')
                if sample_ids is None:
                    sample_ids = visualize_train_samples(train_lines, pred_dir, label_dir, epoch+1, sample_ids=None)
                # 可视化后批量更新Markdown表格
                batch_update_all_markdown()
            # === 新增：每5轮自动可视化10个验证样本 ===
            if (epoch + 1) % 5 == 0 and local_rank == 0:
                pred_dir = os.path.join('miou_out_mobilenetv2_rein', 'detection-results_fixed')
                label_dir = os.path.join(VOCdevkit_path, 'SegmentationClass')
                visualize_val_compare(val_lines, pred_dir, label_dir, epoch+1)
            # === 新增：每轮评估后自动检测mIoU ===
            if local_rank == 0 and eval_callback is not None and hasattr(eval_callback, 'miou_history'):
                if len(eval_callback.miou_history) > 0:
                    miou = eval_callback.miou_history[-1]
                    if miou > 1.0:
                        print("\n[Error] 检测到mIoU异常暴增(mIoU=%.4f > 1)，训练已自动终止！" % miou)
                        print("【mIoU异常/暴增常见原因及修复方法】\n1. 检查标签与预测类别空间是否一致（如类别数、像素值分布、类别顺序）。\n2. 检查标签与预测文件名、shape是否一一对应。\n3. 检查dataloader和评估脚本是否有多余的类别映射或二次处理。\n4. 用check_label_values.py、check_pred_unique.py、check_label_pred_pair.py等脚本批量排查。\n5. 用visualize_label_pred.py批量可视化分析模型输出结构。\n6. 检查训练/评估参数是否同步。\n7. 若为类别错位或标签异常，需修正后重新训练。\n")
                        raise RuntimeError("mIoU异常，训练终止。请根据提示排查数据和流程！")
                abnormal_samples = []
                abnormal_types = {}
                if eval_callback is not None and hasattr(eval_callback, 'abnormal_pairs'):
                    abnormal_samples = eval_callback.abnormal_pairs
                if hasattr(eval_callback, 'miou_abnormal_pairs'):
                    abnormal_samples = eval_callback.miou_abnormal_pairs
                # 扩展：支持异常类型统计（如有类型信息）
                if hasattr(eval_callback, 'abnormal_types'):
                    abnormal_types = eval_callback.abnormal_types  # dict: {type: [samples]}
                abnormal_count = len(abnormal_samples)
                sample_show = abnormal_samples[:5] if abnormal_count > 0 else []
                # 终端输出摘要
                if abnormal_count > 0:
                    print(f"[Epoch {epoch+1}] 异常样本数: {abnormal_count}，示例: {', '.join([str(x) for x in sample_show])}")
                    if abnormal_types:
                        print("  异常类型分布:")
                        for t, lst in abnormal_types.items():
                            print(f"    {t}: {len(lst)}")
                # 详细内容写入日志
                log_path = os.path.join('results', 'abnormal_samples_log.txt')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[Epoch {epoch+1}] 异常样本数: {abnormal_count}\n")
                    if abnormal_count > 0:
                        f.write("示例: " + ', '.join([str(x) for x in sample_show]) + "\n")
                        if abnormal_types:
                            f.write("异常类型分布:\n")
                            for t, lst in abnormal_types.items():
                                f.write(f"  {t}: {len(lst)}\n")
                        if abnormal_count > 5:
                            f.write("更多异常样本: " + ', '.join([str(x) for x in abnormal_samples[5:15]]) + "\n")
            if distributed:
                dist.barrier()
        if local_rank == 0:
            loss_history.writer.close()

    # 注释掉冗余debug输出
    # print(f"[Debug] 当前类别权重 cls_weights:")
    # print(cls_weights)
    # print("[Debug] 损失函数: CrossEntropyLoss, ignore_index=255, 权重 shape:", cls_weights.shape)

    # === 权重文件自动检测与友好提示 ===
    def auto_find_weight_file(log_dir, prefer='last'):
        """
        自动查找log_dir下的last或best权重文件。
        prefer: 'last' 或 'best'，优先查找类型。
        返回权重文件路径或''。
        """
        import glob
        import re
        if not os.path.isdir(log_dir):
            return ''
        weight_files = glob.glob(os.path.join(log_dir, '*.pth'))
        if not weight_files:
            return ''
        # 匹配best/last/epoch等
        best_files = [f for f in weight_files if 'best' in os.path.basename(f).lower()]
        last_files = [f for f in weight_files if 'last' in os.path.basename(f).lower()]
        epoch_files = [f for f in weight_files if re.search(r'epoch(\d+)', os.path.basename(f))]
        if prefer == 'best' and best_files:
            return sorted(best_files)[-1]
        if prefer == 'last' and last_files:
            return sorted(last_files)[-1]
        if epoch_files:
            # 取最大epoch号
            epoch_files = sorted(epoch_files, key=lambda x: int(re.search(r'epoch(\d+)', os.path.basename(x)).group(1)), reverse=True)
            return epoch_files[0]
        # 兜底返回最新修改的
        return max(weight_files, key=os.path.getmtime)

    # 检查权重文件是否存在，否则自动查找
    model_path = default_args.get('model_path', '')
    if not model_path or not os.path.isfile(model_path):
        print(f"[Warning] 指定权重文件不存在: {model_path}")
        auto_weight = auto_find_weight_file(default_args['save_dir'], prefer='last')
        if auto_weight:
            print(f"[Info] 自动查找到可用权重文件: {auto_weight}")
            default_args['model_path'] = auto_weight
        else:
            print(f"[Warning] 未在 {default_args['save_dir']} 下找到可用权重文件，将从头训练！")
            default_args['model_path'] = ''
    else:
        print(f"[Info] 已检测到权重文件: {model_path}")

    def get_weight_path(log_dir, backbone, prefer='last'):
        import os
        # 优先last_epoch_weights.pth
        last_path = os.path.join(log_dir, 'last_epoch_weights.pth')
        if os.path.exists(last_path):
            return last_path
        # 其次best_epoch_weights.pth
        best_path = os.path.join(log_dir, 'best_epoch_weights.pth')
        if os.path.exists(best_path):
            return best_path
        # 否则加载主干预训练权重
        if backbone == 'mobilenet':
            return os.path.join('model_data', 'mobilenet_v2.pth.tar')
        elif backbone == 'xception':
            return os.path.join('model_data', 'xception_pytorch_imagenet.pth')
        else:
            return ''

    # 权重加载逻辑
    weight_path = get_weight_path(LOG_DIR, backbone, prefer=args.weight_type)
    print(f"[Info] 加载权重文件: {weight_path}")
    default_args['model_path'] = weight_path

    # 继续训练时，自动调整Init_Epoch
    if default_args['model_path'] and os.path.isfile(default_args['model_path']):
        import re
        # 提取当前权重文件的epoch信息
        match = re.search(r'epoch(\d+)', os.path.basename(default_args['model_path']))
        if match:
            latest_epoch = int(match.group(1))
            if latest_epoch > default_args['init_epoch']:
                print(f"[Info] 检测到权重文件中包含的最新epoch({latest_epoch}) > 当前设置的起始epoch({default_args['init_epoch']}), 自动调整为{latest_epoch}。")
                default_args['init_epoch'] = latest_epoch
        else:
            print(f"[Warning] 未能从权重文件名中识别出epoch信息，继续使用默认的init_epoch={default_args['init_epoch']}。")

    # 冻结阶段设置
    if Freeze_Train:
        if debug:
            Freeze_Epoch = 2  # Debug模式下，冻结2轮
            UnFreeze_Epoch = 4  # Debug模式下，总共训练4轮
        else:
            Freeze_Epoch = 50  # 通常设置，冻结前50轮
            UnFreeze_Epoch = 150  # 通常设置，总共训练150轮
        print(f"[Info] 冻结训练设置：Freeze_Epoch={Freeze_Epoch}, UnFreeze_Epoch={UnFreeze_Epoch}")

    # 兼容旧版配置
    if isinstance(default_args['input_shape'], int):
        default_args['input_shape'] = [default_args['input_shape'], default_args['input_shape']]

    # 记录最终配置
    if local_rank == 0:
        config_path = os.path.join(LOG_DIR, 'config.txt')
        with open(config_path, 'w') as f:
            for k, v in default_args.items():
                f.write(f"{k}={v}\n")
        print(f"[Info] 配置已保存至: {config_path}")

# === 保存最终参数快照到实验目录 ===
def save_config(final_args, save_dir):
    """
    保存最终参数配置到指定目录（config.txt）。
    Args:
        final_args (dict): 参数字典
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, 'config.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        for k, v in final_args.items():
            f.write(f"{k}={v}\n")

# 合并命令行参数、交互输入、默认值，生成final_args
final_args = default_args.copy()
# 用命令行参数覆盖
for k in vars(args):
    v = getattr(args, k)
    if v is not None:
        final_args[k] = v
# 用交互输入覆盖（已在default_args中处理）
# 保存到实验目录
save_config(final_args, save_dir=final_args['save_dir'])

import subprocess

def run_evaluation_and_save(logs_dir, mious_dir, param_dict):
    """
    自动调用评估脚本，解析mIoU/mPA等指标并写入csv。
    Args:
        logs_dir (str): 日志目录
        mious_dir (str): mIoU结果目录
        param_dict (dict): 当前实验参数
    """
    eval_script = os.path.join(os.path.dirname(__file__), 'get_miou.py')
    # 假设 get_miou.py 支持命令行参数 --log_dir --output_dir
    result = subprocess.run([
        sys.executable, eval_script,
        '--log_dir', logs_dir,
        '--output_dir', mious_dir
    ], capture_output=True, text=True)
    # 简单解析输出中的 mIoU、mPA
    import re
    miou = None
    mpa = None
    for line in result.stdout.splitlines():
        if 'mIoU' in line:
            miou = re.findall(r"[\d.]+", line)[-1]
        if 'mPA' in line:
            mpa = re.findall(r"[\d.]+", line)[-1]
    metrics = {'mIoU': miou or '', 'mPA': mpa or ''}
    # 保存到csv
    csv_path = os.path.join(mious_dir, 'exp_results.csv')
    save_exp_result_csv(csv_path, param_dict, metrics)
    print(f"[Info] 评估结果已保存: {csv_path}")

# 训练主循环结束后自动评估
run_evaluation_and_save(logs_dir, mious_dir, default_args)

def generate_visual_reports(visual_dir, by_sample_dir):
    """
    自动生成Word和PDF可视化报告，图片按样本为主嵌入表格，报告归档到visual_dir。
    Args:
        visual_dir (str): 可视化报告保存目录
        by_sample_dir (str): 按样本可视化图片目录
    """
    import shutil
    import sys
    # 复制并调用 export_visualization_to_word.py 和 export_visualization_to_pdf.py
    word_script = os.path.join(os.path.dirname(__file__), 'export_visualization_to_word.py')
    pdf_script = os.path.join(os.path.dirname(__file__), 'export_visualization_to_pdf.py')
    # 运行Word报告生成
    subprocess.run([
        sys.executable, word_script,
        '--input_dir', by_sample_dir,
        '--output', os.path.join(visual_dir, 'visualization_report.docx')
    ], check=False)
    # 运行PDF报告生成
    subprocess.run([
        sys.executable, pdf_script,
        '--input_dir', by_sample_dir,
        '--output', os.path.join(visual_dir, 'visualization_report.pdf')
    ], check=False)
    print(f"[Info] 可视化报告已生成: {visual_dir}")

if __name__ == "__main__":
    # 训练主循环结束后自动生成可视化报告
    by_sample_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Inspect/VisualizeCompare/by_sample'))
    generate_visual_reports(visual_dir, by_sample_dir)

import csv

def save_experiment_record(param_dict, best_weight_path):
    """
    保存当前实验参数到config.txt和Documents/experiment_records.csv。
    Args:
        param_dict (dict): 当前训练参数
        best_weight_path (str): best权重保存路径
    """
    import os
    # 保存 config.txt
    log_dir = param_dict.get('save_dir', '')
    config_path = os.path.join(log_dir, 'config.txt')
    with open(config_path, 'w', encoding='utf-8') as f:
        for k, v in param_dict.items():
            f.write(f"{k}={v}\n")
        f.write(f"best_weight_path={best_weight_path}\n")
    # 追加/更新 experiment_records.csv
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Documents/experiment_records.csv'))
    fieldnames = ['model_name', 'train_time', 'backbone', 'input_shape', 'num_classes', 'weight_path', 'voc_path', 'token_length']
    # 读取已有内容，避免重复
    records = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    # 构造新记录
    new_record = {
        'model_name': param_dict.get('backbone', ''),
        'train_time': param_dict.get('train_time', ''),
        'backbone': param_dict.get('backbone', ''),
        'input_shape': str(param_dict.get('input_shape', '')),
        'num_classes': param_dict.get('num_classes', ''),
        'weight_path': best_weight_path,
        'voc_path': param_dict.get('voc_path', param_dict.get('VOCdevkit_path', '')),
        'token_length': param_dict.get('token_length', ''),
    }
    # 检查是否已存在相同权重路径，若有则更新，否则追加
    updated = False
    for i, row in enumerate(records):
        if row['weight_path'] == best_weight_path:
            records[i] = new_record
            updated = True
            break
    if not updated:
        records.append(new_record)
    # 写回csv
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    print(f"[Info] 实验参数已同步至: {config_path} 和 {csv_path}")

