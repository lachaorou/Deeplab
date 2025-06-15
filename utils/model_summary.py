import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.deeplabv3_reins import DeepLab
import time

def get_user_input(prompt, default=None, type_cast=str):
    try:
        value = input(prompt)
        if value.strip() == '' and default is not None:
            return default
        return type_cast(value)
    except Exception:
        return default

def parse_tuple(s):
    s = s.strip().replace('(','').replace(')','')
    return tuple(map(int, s.split(',')))

# 默认配置（与train.py保持一致）
DEFAULT_CONFIG = {
    'backbone': {'type': 'mobilenet', 'pretrained': True, 'downsample_factor': 8},
    'reins': {'token_length': 120, 'num_layers': 2, 'embed_dims': None},
    'aspp': {'rate': 2, 'dim_out': 256},
    'num_classes': 19,
    'input_shape': (3, 512, 512),
    'epochs': 150
}

print("【模块化参数配置】支持交互式自定义所有功能块参数（无死板限制）")
print("直接回车使用默认值，或输入自定义数值。\n")

# 交互式参数选择（无choices限制）
config = {}
# backbone
config['backbone'] = {}
config['backbone']['type'] = get_user_input(f"backbone.type (mobilenet/xception, 默认: {DEFAULT_CONFIG['backbone']['type']}): ", default=DEFAULT_CONFIG['backbone']['type'])
config['backbone']['pretrained'] = get_user_input(f"backbone.pretrained (True/False, 默认: {DEFAULT_CONFIG['backbone']['pretrained']}): ", default=DEFAULT_CONFIG['backbone']['pretrained'], type_cast=lambda x: str(x).lower() in ['true','1','yes','y'])
config['backbone']['downsample_factor'] = get_user_input(f"backbone.downsample_factor (如8/16, 默认: {DEFAULT_CONFIG['backbone']['downsample_factor']}): ", default=DEFAULT_CONFIG['backbone']['downsample_factor'], type_cast=int)
# reins
config['reins'] = {}
config['reins']['token_length'] = get_user_input(f"reins.token_length (如100/120等, 默认: {DEFAULT_CONFIG['reins']['token_length']}): ", default=DEFAULT_CONFIG['reins']['token_length'], type_cast=int)
config['reins']['num_layers'] = get_user_input(f"reins.num_layers (如2/4/6等, 默认: {DEFAULT_CONFIG['reins']['num_layers']}): ", default=DEFAULT_CONFIG['reins']['num_layers'], type_cast=int)
config['reins']['embed_dims'] = get_user_input(f"reins.embed_dims (None或整数, 默认: {DEFAULT_CONFIG['reins']['embed_dims']}): ", default=DEFAULT_CONFIG['reins']['embed_dims'], type_cast=lambda x: None if x in ['None','none','null',''] else int(x))
# aspp
config['aspp'] = {}
config['aspp']['rate'] = get_user_input(f"aspp.rate (如2/4/8/16, 默认: {DEFAULT_CONFIG['aspp']['rate']}): ", default=DEFAULT_CONFIG['aspp']['rate'], type_cast=int)
config['aspp']['dim_out'] = get_user_input(f"aspp.dim_out (如256/512, 默认: {DEFAULT_CONFIG['aspp']['dim_out']}): ", default=DEFAULT_CONFIG['aspp']['dim_out'], type_cast=int)
# 其它
config['num_classes'] = get_user_input(f"num_classes (如19, 默认: {DEFAULT_CONFIG['num_classes']}): ", default=DEFAULT_CONFIG['num_classes'], type_cast=int)
config['input_shape'] = get_user_input(f"input_shape (如3,512,512，默认: {DEFAULT_CONFIG['input_shape']}): ", default=DEFAULT_CONFIG['input_shape'], type_cast=parse_tuple)
config['epochs'] = get_user_input(f"epochs (训练轮数，默认: {DEFAULT_CONFIG['epochs']}): ", default=DEFAULT_CONFIG['epochs'], type_cast=int)

INPUT_SHAPE = config['input_shape']
NUM_CLASSES = config['num_classes']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 构建模型，传递各功能块参数
model = DeepLab(
    num_classes=NUM_CLASSES,
    backbone=config['backbone']['type'],
    pretrained=config['backbone']['pretrained'],
    downsample_factor=config['backbone']['downsample_factor'],
    token_length=config['reins']['token_length'],
    num_layers=config['reins']['num_layers'],
    embed_dims=config['reins']['embed_dims']
    # ASPP等其它模块参数如需深度定制，可在模型定义中进一步开放
)
model = model.to(DEVICE)
model.eval()

# 统计参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")

# 统计各模块参数量
print("\n各模块参数量:")
for name, module in model.named_children():
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}: {params:,}")

# 显存与 batch_size 粗略估算
try:
    dummy = torch.randn(1, *INPUT_SHAPE).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)
    print("\n单张输入推理成功，可用于显存测试。")
except Exception as e:
    print(f"推理失败: {e}")
    out = None

# 自动推荐batch_size（经验值，24GB显存，分辨率512/768）
def recommend_batch_size(input_shape, total_params, gpu_mem=24):
    c, h, w = input_shape
    # 经验公式：batch_size ≈ (gpu_mem * 1024 - total_params*4/1024**2) // (h*w*c*4/1024**2*2.5)
    # 2.5为经验系数，考虑激活/显存碎片等
    if h*w <= 512*512:
        return 12
    elif h*w <= 768*768:
        return 6
    else:
        return 2

if out is not None:
    batch_size = recommend_batch_size(INPUT_SHAPE, total_params)
    dummy = torch.randn(batch_size, *INPUT_SHAPE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    start = time.time()
    for _ in range(3):
        optimizer.zero_grad()
        output = model(dummy)
        loss = criterion(output, torch.zeros(batch_size, INPUT_SHAPE[1], INPUT_SHAPE[2], dtype=torch.long, device=DEVICE))
        loss.backward()
        optimizer.step()
    elapsed = (time.time() - start) / 3
    print(f"\n推荐 batch_size: {batch_size}（24GB显存经验值）")
    print(f"单 batch 前向+反向平均耗时: {elapsed:.3f} 秒")
    num_samples = 1000  # 可根据实际训练集大小调整
    steps_per_epoch = num_samples // batch_size
    print(f"单 epoch 预计耗时: {steps_per_epoch * elapsed / 60:.2f} 分钟 (batch_size={batch_size}, 样本数={num_samples})")
    print(f"{config['epochs']} 轮训练预计总耗时: {steps_per_epoch * elapsed * config['epochs'] / 3600:.2f} 小时")
    print("可根据实际训练集大小和 batch_size 调整预估。")

# 推荐配置输出
print("\n【推荐配置】（以4张4090显卡，每张24GB显存为例）")
print("- 推荐输入分辨率：512x512 或 768x768")
print("- 推荐单卡 batch_size：8~16（512x512），4~8（768x768）")
print("- 推荐总 batch_size：32~64（4卡512），16~32（4卡768）")
print("- 实际可根据 nvidia-smi 监控微调\n")
