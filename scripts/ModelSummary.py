import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.deeplabv3_reins import DeepLab
import time

# 配置
INPUT_SHAPE = (3, 512, 512)  # 可根据实际输入尺寸调整
NUM_CLASSES = 19
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 实例化模型（补全所有必需参数）
model = DeepLab(
    num_classes=NUM_CLASSES,
    backbone="mobilenet",
    pretrained=True,
    downsample_factor=8,
    token_length=120,
    num_layers=2
)
model = model.to(DEVICE)

# 统计参数量
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"模型总参数量: {total_params:,}")
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

print("\n建议: 可用 nvidia-smi 监控实际显存占用，逐步增大 batch_size 直至显存溢出，取最大安全值。")

# 训练时间预估（以单次前向+反向为例）
if out is not None:
    batch_size = 2
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
    print(f"\n单 batch 前向+反向平均耗时: {elapsed:.3f} 秒")
    # 预估单 epoch 时间
    num_samples = 1000  # 可根据实际训练集大小调整
    steps_per_epoch = num_samples // batch_size
    epoch_time = steps_per_epoch * elapsed
    print(f"单 epoch 预计耗时: {epoch_time/60:.2f} 分钟 (batch_size={batch_size}, 样本数={num_samples})")
    print("可根据实际训练集大小和 batch_size 调整预估。")
else:
    print("未能完成推理，无法预估训练时间。")
